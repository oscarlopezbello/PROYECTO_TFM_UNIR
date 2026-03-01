#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine

import chromadb
from openai import OpenAI

from tfm_match.config import get_env


# -----------------------
# Utils
# -----------------------

def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def sanitize_text(text: str) -> str:
    """Normaliza texto y reduce ruido; mantiene semántica."""
    if not text:
        return ""
    t = str(text).strip()

    # Separar CamelCase antes de bajar a minúsculas:
    # "EducaciónBásicaSecundaria" -> "Educación Básica Secundaria"
    t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)

    t = t.lower()

    # Normaliza tildes
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])

    # Reduce PII superficial (heurístico)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
    t = re.sub(r"\b(\+?\d[\d\s\-\(\)]{7,}\d)\b", " ", t)

    t = t.replace("|", " ").replace("•", " ").replace("·", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


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
# Mapeo de formación (ordinal)
# -----------------------

EDU_LEVELS = {
    "none": 0,
    "bachiller": 1,
    "tecnico": 2,
    "tecnologo": 3,
    "profesional": 4,
    "posgrado": 5,
}

PATTERNS = [
    (r"\b(doctorado|phd|d\.?phil|maestria|master|magister|m\.?sc|mba|especializacion|posgrado)\b", "posgrado"),
    (r"\b(profesional|pregrado|grado|universitario|ingenier[ia]|ingeniero|licenciad[oa]|administrador|abogad[oa]|contador|economista|psicolog[oa])\b", "profesional"),
    (r"\b(tecnolog[oa]|tecnologia)\b", "tecnologo"),
    (r"\b(tecnic[oa]|tecnico\s+profesional|tecnico\s+laboral)\b", "tecnico"),
    (r"\b(bachiller|secundaria|media)\b", "bachiller"),
]


def detect_education_level(text: str) -> Tuple[str, int, str]:
    """
    Retorna: (nivel_canonico, rank, evidencia_match)
    """
    t = sanitize_text(text)
    if not t:
        return "none", 0, ""

    for pat, canon in PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return canon, EDU_LEVELS[canon], m.group(0)

    return "none", 0, ""


def build_education_doc(canon_level: str, raw_text: str) -> str:
    """
    Documento canónico para embeddings (mezcla nivel + detalle).
    """
    raw = sanitize_text(raw_text)
    if len(raw) > 800:
        raw = raw[:800]
    return f"education_level: {canon_level}; details: {raw}" if raw else f"education_level: {canon_level}"


# -----------------------
# Main indexer
# -----------------------

# Ajustado a tu tabla candidates_prepared
DEFAULT_EDU_COLS = [
    "last_grade", "last_grade_ordinal",
    "institution_of_education1", "institution_of_education2", "institution_of_education3",
    "institution_of_education4", "institution_of_education5",
    "study_area1", "study_area2", "study_area3", "study_area4", "study_area5",
    "study_description1", "study_description2", "study_description3", "study_description4", "study_description5",
    "study_time1", "study_time2", "study_time3", "study_time4", "study_time5",
]

def main():
    parser = argparse.ArgumentParser(description="Indexador de Formación (MySQL via SQLAlchemy -> embeddings -> Chroma)")
    parser.add_argument("--table", default="candidates_prepared", help="Tabla MySQL origen")
    parser.add_argument("--id-col", default="id_candidate", help="Columna ID del candidato")
    parser.add_argument("--edu-cols", default="", help="Lista CSV de columnas para formación (forzar)")
    parser.add_argument("--where", default="", help="Filtro SQL opcional (sin 'WHERE')")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de batch para embeddings/upsert")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep entre batches (seg)")
    parser.add_argument("--read-chunk-rows", type=int, default=2000, help="Filas por chunk al leer desde MySQL")
    args = parser.parse_args()

    # Env
    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")

    mysql_url = get_env("MYSQL_URL", required=True)

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_EDUCATION", "candidates_education")

    client = OpenAI(api_key=openai_api_key)

    # SQLAlchemy
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
        raise ValueError(f"La columna id '{args.id_col}' no existe en {args.table}.")

    if args.edu_cols.strip():
        edu_cols = [c.strip() for c in args.edu_cols.split(",") if c.strip()]
    else:
        edu_cols = [c for c in DEFAULT_EDU_COLS if c in existing_cols]

    if not edu_cols:
        raise ValueError(
            f"No se encontraron columnas de formación en {args.table}. "
            f"Encontradas: {sorted(existing_cols)}. "
            f"Define --edu-cols col1,col2,..."
        )

    select_cols_sql = ", ".join([args.id_col] + edu_cols)
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

        # reintentos simples ante fallos transitorios
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
                print(f"WARN embeddings attempt {attempt}/{max_retries} failed: {e}")
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

    # Lectura por chunks
    for chunk_df in pd.read_sql(sql, engine, chunksize=args.read_chunk_rows):
        for _, row in chunk_df.iterrows():
            total_rows += 1
            cand_id = str(row[args.id_col])

            pieces = []
            for c in edu_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                v = str(v).strip()
                if v and v not in ["0", "nan", "None", "null"]:
                    pieces.append(v)

            raw_edu = "\n".join(pieces)
            canon, rank, evidence = detect_education_level(raw_edu)
            if canon == "none":
                continue

            doc = build_education_doc(canon, raw_edu)
            doc_id = f"{cand_id}::edu"

            meta: Dict[str, Any] = {
                "id_candidate": cand_id,
                "dimension": "education",
                "source_table": args.table,
                "source_cols": ",".join(edu_cols),
                "edu_level": canon,
                "edu_rank": rank,
                "evidence": evidence
            }

            for lvl, r in EDU_LEVELS.items():
                meta[f"has_{lvl}"] = 1 if canon == lvl else 0

            buffer_ids.append(doc_id)
            buffer_docs.append(doc)
            buffer_metas.append(meta)

            if len(buffer_ids) >= args.batch_size:
                flush()
                time.sleep(args.sleep)

    flush()

    print("Indexación finalizada (formación).")
    print(f"- Filas leídas: {total_rows}")
    print(f"- Documentos indexados: {total_docs}")
    print(f"- Colección Chroma: {collection_name}")
    print(f"- Directorio Chroma: {chroma_dir}")
    print(f"- Tabla: {args.table}")
    print(f"- Columnas usadas: {edu_cols}")


if __name__ == "__main__":
    main()
