#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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

# Importar config centralizado (carga .env automáticamente)
from tfm_match.config import get_env


# -----------------------
# Utilidades generales
# -----------------------

def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()

    # Normaliza tildes
    t = unicodedata.normalize("NFKD", t)
    t = "".join([c for c in t if not unicodedata.combining(c)])

    # Reduce PII superficial
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
    t = re.sub(r"\b(\+?\d[\d\s\-\(\)]{7,}\d)\b", " ", t)

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
# Parsing de idiomas + nivel
# -----------------------

LANG_PATTERNS = {
    "english": r"\b(ingles|ingl[eé]s|english)\b",
    "spanish": r"\b(espanol|espa[nñ]ol|spanish|castellano)\b",
    "portuguese": r"\b(portugues|portugu[eê]s|portuguese)\b",
    "french": r"\b(frances|franc[eê]s|french)\b",
    "german": r"\b(aleman|alem[aá]n|german)\b",
    "italian": r"\b(italiano|italian)\b",
}

CEFR_RE = re.compile(r"\b(a1|a2|b1|b2|c1|c2)\b", re.IGNORECASE)
CEFR_RANK = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}

KW_LEVEL_RANK = {
    "basic": 2,         # ~A2
    "intermediate": 4,  # ~B2
    "advanced": 5,      # ~C1
    "fluent": 6,        # ~C2
    "native": 7,        # >C2 para priorizar
}

LEVEL_KW = {
    "basic": r"\b(basico|basica|basic|elemental)\b",
    "intermediate": r"\b(intermedio|intermedia|intermediate|conversacional)\b",
    "advanced": r"\b(avanzado|avanzada|advanced)\b",
    "fluent": r"\b(fluido|fluida|fluent)\b",
    "native": r"\b(nativo|nativa|native|bilingue)\b",
}


def detect_level(text: str) -> Tuple[str, int]:
    if not text:
        return "none", 0

    m = CEFR_RE.search(text)
    if m:
        lvl = m.group(1).lower()
        return lvl.upper(), CEFR_RANK.get(lvl, 0)

    for key, pat in LEVEL_KW.items():
        if re.search(pat, text, re.IGNORECASE):
            return key, KW_LEVEL_RANK[key]

    return "none", 0


def parse_languages(text: str) -> List[Dict[str, Any]]:
    """
    Extrae idiomas presentes en texto y un nivel aproximado (si aparece).
    Retorna: [{language, level, rank}, ...]
    """
    t = sanitize_text(text)
    if not t:
        return []

    found = []
    for lang, pat in LANG_PATTERNS.items():
        if re.search(pat, t, re.IGNORECASE):
            lvl, rank = detect_level(t)
            found.append({"language": lang, "level": lvl, "rank": rank})

    # Dedup por idioma conservando el mayor rank
    best = {}
    for item in found:
        l = item["language"]
        if l not in best or item["rank"] > best[l]["rank"]:
            best[l] = item
    return list(best.values())


def parse_languages_from_pieces(pieces: List[str]) -> List[Dict[str, Any]]:
    """
    Parseo más preciso cuando tenemos items separados (ej: "Inglés B2", "Francés A2").
    Evita asignar el mismo nivel a todos los idiomas al mirar el texto completo.
    """
    if not pieces:
        return []

    found: List[Dict[str, Any]] = []
    for p in pieces:
        t = sanitize_text(p)
        if not t:
            continue
        lvl, rank = detect_level(t)
        for lang, pat in LANG_PATTERNS.items():
            if re.search(pat, t, re.IGNORECASE):
                found.append({"language": lang, "level": lvl, "rank": rank})

    # Dedup por idioma conservando el mayor rank
    best: Dict[str, Dict[str, Any]] = {}
    for item in found:
        l = item["language"]
        if l not in best or item["rank"] > best[l]["rank"]:
            best[l] = item
    return list(best.values())


def canonical_language_doc(lang_items: List[Dict[str, Any]]) -> str:
    """
    Documento canónico para embedding.
    Ej: "english: B2; spanish: native"
    """
    if not lang_items:
        return ""

    order = ["english", "spanish", "portuguese", "french", "german", "italian"]
    lang_items_sorted = sorted(
        lang_items,
        key=lambda x: order.index(x["language"]) if x["language"] in order else 999
    )

    parts = []
    for it in lang_items_sorted:
        parts.append(f"{it['language']}: {it['level']}")
    return "; ".join(parts)


# -----------------------
# Main indexer
# -----------------------

# Ajustado a tu tabla candidates_prepared
DEFAULT_LANGUAGE_COLS = [
    "language1", "language2", "language3", "language4", "language5",
    "language_level1", "language_level2", "language_level3", "language_level4", "language_level5",
]

def main():
    parser = argparse.ArgumentParser(description="Indexador de idiomas (MySQL via SQLAlchemy -> embeddings -> Chroma)")
    parser.add_argument("--table", default="candidates_prepared", help="Tabla MySQL origen")
    parser.add_argument("--id-col", default="id_candidate", help="Columna ID del candidato")
    parser.add_argument("--language-cols", default="", help="Lista CSV de columnas para idiomas (forzar)")
    parser.add_argument("--where", default="", help="Filtro SQL opcional (sin 'WHERE')")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de batch para embeddings/upsert")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep entre batches (seg)")
    parser.add_argument("--read-chunk-rows", type=int, default=2000, help="Filas por chunk al leer desde MySQL")
    args = parser.parse_args()

    # Env
    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")

    # DB por URL única
    mysql_url = get_env("MYSQL_URL", required=True)  # mysql+pymysql://user:pass@host:3306/db

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_LANGUAGE", "candidates_language")

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

    if args.language_cols.strip():
        lang_cols = [c.strip() for c in args.language_cols.split(",") if c.strip()]
    else:
        lang_cols = [c for c in DEFAULT_LANGUAGE_COLS if c in existing_cols]

    if not lang_cols:
        raise ValueError(
            f"No se encontraron columnas de idioma en {args.table}. "
            f"Encontradas: {sorted(existing_cols)}. "
            f"Define --language-cols col1,col2,..."
        )

    select_cols_sql = ", ".join([args.id_col] + lang_cols)
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

        resp = client.embeddings.create(model=embedding_model, input=buffer_docs)
        embeddings = [d.embedding for d in resp.data]

        upsert_collection(
            collection=collection,
            ids=buffer_ids,
            embeddings=embeddings,
            documents=buffer_docs,
            metadatas=buffer_metas
        )

        total_docs += len(buffer_ids)
        buffer_ids, buffer_docs, buffer_metas = [], [], []

    # Lectura por chunks desde MySQL
    for chunk_df in pd.read_sql(sql, engine, chunksize=args.read_chunk_rows):
        for _, row in chunk_df.iterrows():
            total_rows += 1
            cand_id = str(row[args.id_col])

            # Construye texto unificado idioma + nivel (si viene separado)
            # Ejemplo: "Inglés B2\nFrancés A2"
            pieces = []
            # Caso típico en tu tabla: language1..5 y language_level1..5
            for i in range(1, 6):
                lang_col = f"language{i}"
                lvl_col = f"language_level{i}"
                lang_val = row.get(lang_col) if lang_col in lang_cols else None
                lvl_val = row.get(lvl_col) if lvl_col in lang_cols else None

                if pd.isna(lang_val):
                    lang_val = None
                if pd.isna(lvl_val):
                    lvl_val = None

                if lang_val:
                    s = str(lang_val).strip()
                    if lvl_val:
                        s = f"{s} {str(lvl_val).strip()}"
                    pieces.append(s)

            # Si forzaste columnas distintas, adicionalmente concatena lo que exista
            if not pieces:
                for c in lang_cols:
                    v = row.get(c)
                    if pd.isna(v):
                        continue
                    v = str(v).strip()
                    if v:
                        pieces.append(v)

            lang_text = "\n".join(pieces).strip()
            # Preferimos parsear por piezas para asignar niveles correctamente
            lang_items = parse_languages_from_pieces(pieces) if pieces else parse_languages(lang_text)
            doc = canonical_language_doc(lang_items)

            if not doc:
                continue

            doc_id = f"{cand_id}::lang"

            meta: Dict[str, Any] = {
                "id_candidate": cand_id,
                "dimension": "language",
                "source_table": args.table,
                "source_cols": ",".join(lang_cols),
            }

            # Flags por idioma (para filtros/analítica posterior)
            for it in lang_items:
                lang = it["language"]
                meta[f"has_{lang}"] = 1
                meta[f"lvl_{lang}_rank"] = int(it["rank"])
                meta[f"lvl_{lang}"] = it["level"]

            buffer_ids.append(doc_id)
            buffer_docs.append(doc)
            buffer_metas.append(meta)

            if len(buffer_ids) >= args.batch_size:
                flush()
                time.sleep(args.sleep)

    flush()

    print("Indexación finalizada (idiomas).")
    print(f"- Filas leídas: {total_rows}")
    print(f"- Documentos indexados: {total_docs}")
    print(f"- Colección Chroma: {collection_name}")
    print(f"- Directorio Chroma: {chroma_dir}")
    print(f"- Tabla: {args.table}")
    print(f"- Columnas usadas: {lang_cols}")


if __name__ == "__main__":
    main()
