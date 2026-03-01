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
# Utils
# -----------------------

def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])
    t = t.replace("|", " ").replace("•", " ").replace("·", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def upsert_collection(collection, ids, embeddings, documents, metadatas):
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return
    try:
        collection.delete(ids=ids)
    except Exception:
        pass
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


# -----------------------
# Sector / Área canonización
# -----------------------

CANONICAL_SECTORS = [
    "bpo_contact_center",
    "retail_comercio",
    "logistica_transporte",
    "hospitalidad_turismo",
    "telecom",
    "manufactura",
    "finanzas_banca",
    "salud",
    "tecnologia_it",
    "educacion",
    "servicios_generales"
]

SECTOR_SYNONYMS = [
    (r"\b(bpo|contact\s*center|call\s*center|atencion\s+al\s+cliente|customer\s+service)\b", "bpo_contact_center"),
    (r"\b(retail|comercio|tienda|ventas\s+tienda|punto\s+de\s+venta|pos)\b", "retail_comercio"),
    (r"\b(logistica|logistico|transporte|distribucion|bodega|almacen|wms|last\s*mile)\b", "logistica_transporte"),
    (r"\b(hotel|hoteleria|hospitalidad|turismo|restaurante|gastronomia)\b", "hospitalidad_turismo"),
    (r"\b(telecom|telecomunicaciones|operador|telefonia|internet)\b", "telecom"),
    (r"\b(manufactura|produccion|planta|ensamble|fabrica)\b", "manufactura"),
    (r"\b(finanzas|banca|banco|seguros|riesgo|credito)\b", "finanzas_banca"),
    (r"\b(salud|hospital|clinica|enfermeria)\b", "salud"),
    (r"\b(it|ti\b|tecnologia|software|desarrollo|data|datos|devops|cloud)\b", "tecnologia_it"),
    (r"\b(educacion|docencia|colegio|universidad|formacion)\b", "educacion"),
]

SPLIT_RE = re.compile(r"[;,/|\n]+")


def extract_sectors(raw_text: str) -> List[str]:
    t = sanitize_text(raw_text)
    if not t:
        return []

    sectors = set()

    # patrones sobre texto completo
    for pat, canon in SECTOR_SYNONYMS:
        if re.search(pat, t, flags=re.IGNORECASE):
            sectors.add(canon)

    # tokens
    tokens = [sanitize_text(tok) for tok in SPLIT_RE.split(raw_text) if sanitize_text(tok)]
    for tok in tokens:
        for pat, canon in SECTOR_SYNONYMS:
            if re.search(pat, tok, flags=re.IGNORECASE):
                sectors.add(canon)
        if tok in CANONICAL_SECTORS:
            sectors.add(tok)

    return sorted(sectors)


def canonical_sector_doc(sectors: List[str]) -> str:
    if not sectors:
        return ""
    return "; ".join([f"sector: {s}" for s in sectors])


# -----------------------
# Main indexer
# -----------------------

# Ajustado a tu tabla candidates_prepared (proxy “sector/área” desde formación/área de estudio + perfil)
DEFAULT_SECTOR_COLS = [
    "study_area1", "study_area2", "study_area3", "study_area4", "study_area5",
    "profile_text", "brief_description",
    # opcionales si existen:
    "job_name1", "job_name2", "job_name3", "job_name4", "job_name5",
    "job_description1", "job_description2", "job_description3", "job_description4", "job_description5",
]

def main():
    parser = argparse.ArgumentParser(description="Indexador de Área/Sector (MySQL via SQLAlchemy -> embeddings -> Chroma)")
    parser.add_argument("--table", default="candidates_prepared", help="Tabla MySQL origen")
    parser.add_argument("--id-col", default="id_candidate", help="Columna ID del candidato")
    parser.add_argument("--sector-cols", default="", help="Lista CSV de columnas para sector/area (forzar)")
    parser.add_argument("--where", default="", help="Filtro SQL opcional (sin 'WHERE')")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de batch para embeddings/upsert")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep entre batches (seg)")
    parser.add_argument("--read-chunk-rows", type=int, default=2000, help="Filas por chunk al leer desde MySQL")
    args = parser.parse_args()

    # Env
    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")

    mysql_url = get_env("MYSQL_URL", required=True)  # mysql+pymysql://user:pass@host:3306/db

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_SECTOR", "candidates_sector")

    client = OpenAI(api_key=openai_api_key)

    # SQLAlchemy
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
        raise ValueError(f"La columna id '{args.id_col}' no existe en {args.table}.")

    if args.sector_cols.strip():
        sector_cols = [c.strip() for c in args.sector_cols.split(",") if c.strip()]
    else:
        sector_cols = [c for c in DEFAULT_SECTOR_COLS if c in existing_cols]

    if not sector_cols:
        raise ValueError(
            f"No se encontraron columnas de sector/área en {args.table}. "
            f"Encontradas: {sorted(existing_cols)}. "
            f"Define --sector-cols col1,col2,..."
        )

    select_cols_sql = ", ".join([args.id_col] + sector_cols)
    sql = f"SELECT {select_cols_sql} FROM {args.table}"
    if args.where.strip():
        sql += f" WHERE {args.where}"

    buffer_ids: List[str] = []
    buffer_docs: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []
    buffer_delete_ids: List[str] = []

    total_rows = 0
    total_docs = 0

    def flush():
        nonlocal buffer_ids, buffer_docs, buffer_metas, buffer_delete_ids, total_docs
        if not buffer_ids:
            return

        # Limpieza: evita docs "stale" si cambia el set de sectores de un candidato.
        if buffer_delete_ids:
            try:
                collection.delete(ids=list(set(buffer_delete_ids)))
            except Exception:
                pass
            buffer_delete_ids = []

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
            for c in sector_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                v = str(v).strip()
                if v:
                    pieces.append(v)

            raw_sector_text = "\n".join(pieces)
            sectors = extract_sectors(raw_sector_text)
            doc = canonical_sector_doc(sectors)

            if not doc:
                continue

            doc_id = f"{cand_id}::sector"

            # IDs a limpiar (combinado + slots individuales fijos)
            buffer_delete_ids.append(f"{cand_id}::sector")
            for j in range(1, 6):
                buffer_delete_ids.append(f"{cand_id}::sector::{j}")

            base_meta: Dict[str, Any] = {
                "id_candidate": cand_id,
                "dimension": "sector",
                "source_table": args.table,
                "source_cols": ",".join(sector_cols),
                "sector_count": len(sectors),
                "sectors": ",".join(sectors)
            }

            for s in CANONICAL_SECTORS:
                base_meta[f"has_{s}"] = 1 if s in sectors else 0

            # Doc combinado (backward compatible con id histórico)
            buffer_ids.append(doc_id)
            buffer_docs.append(doc)
            buffer_metas.append({**base_meta, "doc_type": "combined"})

            # Docs individuales (1 embedding por sector) -> mejora matching cuando query pide un solo sector
            for j, s in enumerate(sectors[:5], start=1):
                buffer_ids.append(f"{cand_id}::sector::{j}")
                buffer_docs.append(f"sector: {s}")
                buffer_metas.append({**base_meta, "doc_type": "single", "sector": s, "sector_idx": j})

            if len(buffer_ids) >= args.batch_size:
                flush()
                time.sleep(args.sleep)

    flush()

    print("Indexación finalizada (área/sector).")
    print(f"- Filas leídas: {total_rows}")
    print(f"- Documentos indexados: {total_docs}")
    print(f"- Colección Chroma: {collection_name}")
    print(f"- Directorio Chroma: {chroma_dir}")
    print(f"- Tabla: {args.table}")
    print(f"- Columnas usadas: {sector_cols}")


if __name__ == "__main__":
    main()
