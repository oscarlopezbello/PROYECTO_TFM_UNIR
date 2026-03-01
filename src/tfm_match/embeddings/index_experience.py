#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from sqlalchemy import create_engine

import chromadb
from openai import OpenAI

# Importar config centralizado (carga .env automáticamente)
from tfm_match.config import get_env


# -----------------------
# Utilidades
# -----------------------

def sanitize_text(text: str) -> str:
    """Normaliza texto y reduce PII superficial (móvil/email), manteniendo semántica."""
    if not text:
        return ""

    t = text.strip().lower()

    # Normaliza tildes
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])

    # Elimina emails/teléfonos (heurístico)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
    t = re.sub(r"\b(\+?\d[\d\s\-\(\)]{7,}\d)\b", " ", t)

    # Normaliza separadores
    t = t.replace("|", " ").replace("•", " ").replace("·", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_for_duration(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])
    return t


def parse_duration_to_months(text: str) -> Optional[int]:
    """
    Heurística simple para convertir duraciones a meses.
    Soporta variantes comunes ES/EN:
      - "2 años 3 meses", "2 anos y 3 meses", "2y 3m"
      - "18 meses"
      - "3 years 2 months", "3y2m"
    """
    t = _normalize_for_duration(text)
    if not t:
        return None

    # Unifica separadores
    t = t.replace(",", " ").replace(";", " ").replace("|", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # patrón compacto: 2y3m / 2a3m
    m = re.search(r"\b(\d+)\s*(?:y|a|yr|yrs|year|years)\s*(\d+)\s*(?:m|mo|mos|month|months|mes|meses)\b", t)
    if m:
        try:
            return int(m.group(1)) * 12 + int(m.group(2))
        except Exception:
            return None

    # patrón explícito: X años y Y meses
    m = re.search(r"\b(\d+)\s*(?:ano|anos|year|years|yr|yrs)\s*(?:y|and)?\s*(\d+)\s*(?:mes|meses|month|months|mo|mos)\b", t)
    if m:
        try:
            return int(m.group(1)) * 12 + int(m.group(2))
        except Exception:
            return None

    # solo años
    m = re.search(r"\b(\d+)\s*(?:ano|anos|year|years|yr|yrs)\b", t)
    if m:
        try:
            return int(m.group(1)) * 12
        except Exception:
            return None

    # solo meses
    m = re.search(r"\b(\d+)\s*(?:mes|meses|month|months|mo|mos)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    return None


def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Divide texto en chunks (aprox por caracteres) para evitar inputs demasiado largos."""
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"(?:\n+|\. )", text)
    chunks, buff = [], ""

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buff) + len(p) + 1 <= max_chars:
            buff = (buff + " " + p).strip()
        else:
            if buff:
                chunks.append(buff)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                buff = ""
            else:
                buff = p

    if buff:
        chunks.append(buff)

    return chunks


def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def upsert_collection(collection, ids, embeddings, documents, metadatas):
    """Compatibilidad: usa upsert si existe; si no, delete+add."""
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return
    try:
        collection.delete(ids=ids)
    except Exception:
        pass
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


# -----------------------
# Main
# -----------------------

DEFAULT_EXPERIENCE_COLS = [
  "profile_text", "brief_description",
  "job_name1","job_description1","job_duration1","job_place1",
  "job_name2","job_description2","job_duration2","job_place2",
  "job_name3","job_description3","job_duration3","job_place3",
  "job_name4","job_description4","job_duration4","job_place4",
  "job_name5","job_description5","job_duration5","job_place5",
]

DURATION_COLS = [f"job_duration{i}" for i in range(1, 6)]

def main():
    parser = argparse.ArgumentParser(description="Indexador de experiencia (MySQL via SQLAlchemy -> OpenAI embeddings -> Chroma)")
    parser.add_argument("--table", default="candidates_prepared", help="Tabla MySQL origen")
    parser.add_argument("--id-col", default="id_candidate", help="Columna ID del candidato")
    parser.add_argument("--experience-cols", default="", help="Lista CSV de columnas de experiencia (si quieres forzar)")
    parser.add_argument("--where", default="", help="Filtro SQL opcional (sin 'WHERE')")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de batch para embeddings/upsert")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Máx chars por chunk de experiencia")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep entre batches (seg)")
    parser.add_argument("--read-chunk-rows", type=int, default=2000, help="Filas por chunk al leer desde MySQL")
    args = parser.parse_args()

    # Config (env)
    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")

    mysql_url = get_env("MYSQL_URL", required=True)

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_EXPERIENCE", "candidates_experience")

    from datetime import datetime

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    print(f"[{ts()}] START index_experience")
    print(f"[{ts()}] TABLE={args.table} | ID_COL={args.id_col}")
    print(f"[{ts()}] CHROMA_DIR={chroma_dir} | COLLECTION={collection_name}")
    print(f"[{ts()}] EMBEDDING_MODEL={embedding_model} | batch_size={args.batch_size} | chunk_size={args.chunk_size}")

    client = OpenAI(api_key=openai_api_key)

    # SQLAlchemy engine
    engine = create_engine(mysql_url, pool_pre_ping=True)

    # Chroma
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Detecta columnas disponibles
    cols_df = pd.read_sql(f"SHOW COLUMNS FROM {args.table}", engine)
    existing_cols = set(cols_df["Field"].tolist())

    if args.id_col not in existing_cols:
        raise ValueError(f"La columna id '{args.id_col}' no existe en {args.table}. Columnas: {sorted(existing_cols)}")

    if args.experience_cols.strip():
        exp_cols = [c.strip() for c in args.experience_cols.split(",") if c.strip()]
    else:
        exp_cols = [c for c in DEFAULT_EXPERIENCE_COLS if c in existing_cols]

    if not exp_cols:
        raise ValueError(
            f"No se encontraron columnas de experiencia en {args.table}. "
            f"Encontradas: {sorted(existing_cols)}. "
            f"Define --experience-cols col1,col2,..."
        )

    print(f"[{ts()}] Ejecutando query SQL y leyendo filas...")

    select_cols_sql = ", ".join([args.id_col] + exp_cols)
    sql = f"SELECT {select_cols_sql} FROM {args.table}"
    if args.where.strip():
        sql += f" WHERE {args.where}"

    buffer_ids: List[str] = []
    buffer_docs: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []

    total_rows = 0
    total_docs = 0

    def flush():
        nonlocal buffer_ids, buffer_docs, buffer_metas, total_docs

        if not buffer_ids:
            return

        print(f"[{ts()}] FLUSH: generando embeddings para {len(buffer_docs)} docs...")
        t0 = time.time()

        # (opcional) reintentos ante fallos de red
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

        print(f"[{ts()}] FLUSH OK: upsert {len(embeddings)} docs en {time.time()-t0:.2f}s")

    # Leer MySQL por chunks
    for chunk_df in pd.read_sql(sql, engine, chunksize=args.read_chunk_rows):
        for _, row in chunk_df.iterrows():

            if total_rows % 200 == 0:
                print(f"[{ts()}] Leídas {total_rows} filas | buffer_docs={len(buffer_docs)} | total_docs={total_docs}")


            total_rows += 1
            cand_id = str(row[args.id_col])

            # (opcional) meses totales inferidos desde job_duration1..5 para scoring estable en queries
            dur_months: List[int] = []
            for c in DURATION_COLS:
                if c not in exp_cols:
                    continue
                v = row.get(c)
                if pd.isna(v):
                    continue
                mm = parse_duration_to_months(str(v))
                if mm is not None and mm > 0:
                    dur_months.append(mm)
            exp_months_total: Optional[int] = sum(dur_months) if dur_months else None

            exp_text_parts = []
            for c in exp_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                v = str(v).strip()
                if v:
                    exp_text_parts.append(v)

            exp_text = "\n".join(exp_text_parts)
            exp_text = sanitize_text(exp_text)

            if not exp_text:
                continue

            chunks = chunk_text(exp_text, max_chars=args.chunk_size)
            for idx, ch in enumerate(chunks):
                doc_id = f"{cand_id}::exp::{idx}"
                buffer_ids.append(doc_id)
                buffer_docs.append(ch)
                meta: Dict[str, Any] = {
                    "id_candidate": cand_id,
                    "chunk_idx": idx,
                    "source_table": args.table,
                    "source_cols": ",".join(exp_cols),
                    "text_len": len(ch),
                    "dimension": "experience"
                }
                if exp_months_total is not None:
                    meta["exp_months"] = int(exp_months_total)
                    meta["exp_years"] = float(exp_months_total) / 12.0
                    meta["exp_months_source"] = "sum_job_duration_1_5"
                buffer_metas.append(meta)

                if total_rows % 200 == 0:
                    print(f"[{ts()}] Leídas {total_rows} filas | buffer_docs={len(buffer_docs)} | total_docs={total_docs}")

                if len(buffer_ids) >= args.batch_size:
                    flush()
                    time.sleep(args.sleep)

    flush()

    print("Indexación finalizada.")
    print(f"- Filas leídas: {total_rows}")
    print(f"- Documentos/chunks indexados: {total_docs}")
    print(f"- Colección Chroma: {collection_name}")
    print(f"- Directorio Chroma: {chroma_dir}")
    print(f"- Tabla: {args.table}")
    print(f"- Columnas usadas: {exp_cols}")


if __name__ == "__main__":
    main()
