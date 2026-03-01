#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text

import chromadb
from openai import OpenAI

# Importar config centralizado
from tfm_match.config import get_env


# -----------------------
# Utils
# -----------------------

def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def sanitize_text(text: str) -> str:
    """Normaliza texto, reduce ruido y PII superficial sin perder semántica."""
    if not text:
        return ""
    t = str(text).strip().lower()

    # Normaliza tildes
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])

    # PII heurística
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
    t = re.sub(r"\b(\+?\d[\d\s\-\(\)]{7,}\d)\b", " ", t)

    # Separadores y espacios
    t = t.replace("|", " ").replace("•", " ").replace("·", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def upsert_collection(collection, ids, embeddings, documents, metadatas):
    """Compat: upsert si existe; si no, delete+add."""
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return
    try:
        collection.delete(ids=ids)
    except Exception:
        pass
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


# -----------------------
# Canonicalización de "cargo"
# -----------------------

DEFAULT_JOB_TITLE_COLS = [
    # Títulos de cargos por experiencia (muy probable en candidates_prepared)
    "job_name1", "job_name2", "job_name3", "job_name4", "job_name5",
    # Opcionales (si quieres enriquecer)
    "brief_description", "profile_text",
]

MAX_TITLES_PER_CANDIDATE = 5

def dedup_job_titles(job_titles: List[str], max_titles: int = MAX_TITLES_PER_CANDIDATE) -> List[str]:
    """Normaliza, deduplica y limita lista de cargos preservando orden."""
    titles = [sanitize_text(x) for x in job_titles if sanitize_text(x)]
    seen = set()
    dedup: List[str] = []
    for t in titles:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
        if len(dedup) >= max_titles:
            break
    return dedup

def build_job_title_doc(job_titles: List[str], extra: str = "") -> str:
    """
    Documento canónico para embedding.
    Formato simplificado para mejor matching.
    """
    # Nota: `extra` se mantiene por compatibilidad; actualmente no se incorpora al doc
    dedup = dedup_job_titles(job_titles)
    if not dedup:
        return ""

    # Formato simplificado: solo los títulos separados por punto y coma
    # Esto mejora el matching directo
    return "; ".join(dedup)


def main():
    parser = argparse.ArgumentParser(description="Indexador de Cargo/Nombre del cargo (MySQL via SQLAlchemy -> OpenAI embeddings -> Chroma)")
    parser.add_argument("--table", default="candidates_prepared", help="Tabla MySQL origen")
    parser.add_argument("--id-col", default="id_candidate", help="Columna ID del candidato")
    parser.add_argument("--job-cols", default="", help="CSV columnas para cargo/títulos (forzar)")
    parser.add_argument("--extra-cols", default="", help="CSV columnas extra (headline/resumen) opcionales")
    parser.add_argument("--where", default="", help="Filtro SQL opcional (sin 'WHERE')")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch embeddings/upsert")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep entre batches")
    parser.add_argument("--read-chunk-rows", type=int, default=2000, help="Filas por chunk al leer desde MySQL")
    args = parser.parse_args()

    # Env
    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    mysql_url = get_env("MYSQL_URL", required=True)

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_JOB_TITLE", "candidates_job_title")

    from datetime import datetime
    def ts():
        return datetime.now().strftime("%H:%M:%S")

    print(f"[{ts()}] START index_job_title")
    print(f"[{ts()}] TABLE={args.table} | ID_COL={args.id_col}")
    print(f"[{ts()}] CHROMA_DIR={chroma_dir} | COLLECTION={collection_name}")
    print(f"[{ts()}] EMBEDDING_MODEL={embedding_model} | batch_size={args.batch_size}")

    client = OpenAI(api_key=openai_api_key)
    engine = create_engine(mysql_url, pool_pre_ping=True)

    # Chroma
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Detecta columnas
    cols_df = pd.read_sql(f"SHOW COLUMNS FROM {args.table}", engine)
    existing_cols = set(cols_df["Field"].tolist())

    if args.id_col not in existing_cols:
        raise ValueError(f"La columna id '{args.id_col}' no existe en {args.table}. Columnas: {sorted(existing_cols)}")

    if args.job_cols.strip():
        job_cols = [c.strip() for c in args.job_cols.split(",") if c.strip()]
    else:
        job_cols = [c for c in DEFAULT_JOB_TITLE_COLS if c in existing_cols and c.startswith("job_name")]

    # extra cols opcionales
    extra_cols = []
    if args.extra_cols.strip():
        extra_cols = [c.strip() for c in args.extra_cols.split(",") if c.strip() and c in existing_cols]
    else:
        for c in ["brief_description", "profile_text"]:
            if c in existing_cols:
                extra_cols.append(c)

    if not job_cols and not extra_cols:
        raise ValueError(
            f"No se encontraron columnas para cargo en {args.table}. "
            f"Define --job-cols job_name1,job_name2,... o --extra-cols ..."
        )

    select_cols = [args.id_col] + job_cols + extra_cols
    select_cols_sql = ", ".join(select_cols)

    sql = f"SELECT {select_cols_sql} FROM {args.table}"
    if args.where.strip():
        sql += f" WHERE {args.where}"

    print(f"[{ts()}] Ejecutando query SQL y leyendo filas...")

    buffer_ids: List[str] = []
    buffer_docs: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []
    buffer_delete_ids: List[str] = []

    total_rows = 0
    total_docs = 0
    indexed_candidates = 0

    def flush():
        nonlocal buffer_ids, buffer_docs, buffer_metas, buffer_delete_ids, total_docs
        if not buffer_ids:
            return

        # Limpieza: evita documentos "stale" cuando un candidato cambia de títulos entre indexaciones.
        # Borramos en batch los IDs que vamos a re-escribir.
        if buffer_delete_ids:
            try:
                collection.delete(ids=list(set(buffer_delete_ids)))
            except Exception:
                pass
            buffer_delete_ids = []

        print(f"[{ts()}] FLUSH: generando embeddings para {len(buffer_docs)} docs...")
        t0 = time.time()

        # reintentos simples
        max_retries = 3
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.embeddings.create(model=embedding_model, input=buffer_docs)
                embeddings = [d.embedding for d in resp.data]
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[{ts()}] WARN embeddings attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(2 * attempt)

        if last_err is not None:
            raise last_err

        upsert_collection(
            collection=collection,
            ids=buffer_ids,
            embeddings=embeddings,
            documents=buffer_docs,
            metadatas=buffer_metas
        )

        total_docs += len(buffer_ids)
        buffer_ids, buffer_docs, buffer_metas = [], [], []

        print(f"[{ts()}] FLUSH OK: upsert {len(embeddings)} docs en {time.time() - t0:.2f}s")

    # Leer por chunks
    for chunk_df in pd.read_sql(sql, engine, chunksize=args.read_chunk_rows):
        for _, row in chunk_df.iterrows():
            total_rows += 1
            cand_id = str(row[args.id_col])

            titles = []
            for c in job_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                v = str(v).strip()
                # Filtrar valores basura: "0", "nan", "None", "null"
                if v and v not in ["0", "nan", "None", "null"]:
                    titles.append(v)

            extra_parts = []
            for c in extra_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                v = str(v).strip()
                if v:
                    extra_parts.append(v)
            extra = "\n".join(extra_parts)

            dedup_titles = dedup_job_titles(titles, max_titles=MAX_TITLES_PER_CANDIDATE)
            if not dedup_titles:
                continue

            indexed_candidates += 1

            # IDs a limpiar (combinado + slots individuales fijos)
            buffer_delete_ids.append(f"{cand_id}::job_title")
            for j in range(1, MAX_TITLES_PER_CANDIDATE + 1):
                buffer_delete_ids.append(f"{cand_id}::job_title::{j}")

            base_meta: Dict[str, Any] = {
                "id_candidate": cand_id,
                "dimension": "job_title",
                "source_table": args.table,
                "source_cols": ",".join(select_cols),
                "titles_count": len(dedup_titles),
                "titles": ";".join(dedup_titles)[:1500],
            }

            # Doc combinado (backward compatible con id histórico)
            combined_doc = "; ".join(dedup_titles)
            buffer_ids.append(f"{cand_id}::job_title")
            buffer_docs.append(combined_doc)
            buffer_metas.append({**base_meta, "doc_type": "combined"})

            # Docs individuales (1 embedding por cargo) -> mejora el score cuando el query coincide con 1 cargo específico
            for j, title in enumerate(dedup_titles, start=1):
                buffer_ids.append(f"{cand_id}::job_title::{j}")
                buffer_docs.append(title)
                buffer_metas.append({**base_meta, "doc_type": "single", "job_title": title, "job_title_idx": j})

            if total_rows % 2000 == 0:
                print(f"[{ts()}] Leídas {total_rows} filas | buffer_docs={len(buffer_docs)} | total_docs={total_docs}")

            if len(buffer_ids) >= args.batch_size:
                flush()
                time.sleep(args.sleep)

    flush()

    print("Indexación finalizada (cargo/job_title).")
    print(f"- Filas leídas: {total_rows}")
    print(f"- Candidatos indexados (con doc): {indexed_candidates}")
    print(f"- Documentos indexados: {total_docs}")
    print(f"- Colección Chroma: {collection_name}")
    print(f"- Directorio Chroma: {chroma_dir}")
    print(f"- Columnas usadas: {select_cols}")


if __name__ == "__main__":
    main()
