
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
from typing import Optional, Tuple

import chromadb
from openai import OpenAI

from tfm_match.config import get_env


CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
KW_RANK = {"basic": 2, "intermediate": 4, "advanced": 5, "fluent": 6, "native": 7}

# Mapeo UI -> canonical (igual que core/filters.py)
LANG_UI_TO_CANON = {
    "ingles": "english",
    "inglés": "english",
    "english": "english",
    "frances": "french",
    "francés": "french",
    "french": "french",
    "portugues": "portuguese",
    "portugués": "portuguese",
    "portuguese": "portuguese",
    "espanol": "spanish",
    "español": "spanish",
    "spanish": "spanish",
}

def parse_required_level(level: str) -> Tuple[str, int]:
    if not level:
        return "none", 0
    lv = level.strip().upper()
    if lv in CEFR_RANK:
        return lv, CEFR_RANK[lv]
    lv2 = level.strip().lower()
    if lv2 in KW_RANK:
        return lv2, KW_RANK[lv2]
    return level.strip(), 0


def build_query(lang: str, level: str) -> str:
    # Canonical string compatible con el indexer
    if not lang:
        return ""
    l0 = lang.strip().lower()
    l = LANG_UI_TO_CANON.get(l0, l0)
    lvl, _ = parse_required_level(level)
    if lvl and lvl != "none":
        return f"{l}: {lvl}"
    return f"{l}: none"


def main():
    parser = argparse.ArgumentParser(description="Consulta índice de idiomas en Chroma")
    parser.add_argument("--q", default="", help="Texto libre de consulta (ej: 'Inglés B2')")
    parser.add_argument("--lang", default="", help="Idioma requerido (ej: english, ingles)")
    parser.add_argument("--level", default="", help="Nivel requerido (ej: B2, C1, intermediate, advanced)")
    parser.add_argument("--topk", type=int, default=20, help="Top-K resultados (antes de filtros)")
    parser.add_argument("--min-level", action="store_true", help="Aplicar filtro por nivel mínimo (post-filter)")
    parser.add_argument("--show-doc", action="store_true", help="Mostrar documento canónico recuperado")
    args = parser.parse_args()

    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_dir = get_env("CHROMA_DIR", "./data/chroma")
    collection_name = get_env("CHROMA_COLLECTION_LANGUAGE", "candidates_language")

    client = OpenAI(api_key=openai_api_key)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_collection(name=collection_name)

    # Construye query
    if args.lang:
        q = build_query(args.lang, args.level)
        req_lang = LANG_UI_TO_CANON.get(args.lang.strip().lower(), args.lang.strip().lower())
        req_level_str, req_rank = parse_required_level(args.level)
    else:
        q = (args.q or "").strip()
        if not q:
            q = "english: B2"
        # Heurística para extraer idioma principal si quieres filtrar
        m = re.match(r"^\s*([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\s*[: ]\s*([A-Za-z0-9]+)\s*$", q)
        req_lang_raw = m.group(1).strip().lower() if m else ""
        req_lang = LANG_UI_TO_CANON.get(req_lang_raw, req_lang_raw)
        req_level_str, req_rank = parse_required_level(m.group(2)) if m else ("none", 0)

    emb = client.embeddings.create(model=embedding_model, input=[q]).data[0].embedding

    res = collection.query(
        query_embeddings=[emb],
        n_results=args.topk,
        include=["metadatas", "documents", "distances"]
    )

    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]

    print(f"Consulta: {q}")
    print(f"Colección: {collection_name}")
    print("Nota: en cosine distance, menor es mejor (aprox_sim ~ 1 - dist)\n")

    results = []
    for i, doc_id in enumerate(ids):
        meta = metas[i] or {}
        dist = float(dists[i])
        approx_sim = 1.0 - dist

        # Post-filter por idioma + mínimo nivel
        if args.min_level and req_lang:
            has_flag = meta.get(f"has_{req_lang}", 0) == 1
            cand_rank = int(meta.get(f"lvl_{req_lang}_rank", 0) or 0)
            if not has_flag:
                continue
            if req_rank > 0 and cand_rank < req_rank:
                continue

        results.append((doc_id, meta, dist, approx_sim, docs[i]))

    if not results:
        print("Sin resultados tras aplicar filtros.")
        return

    for rank, (doc_id, meta, dist, sim, doc) in enumerate(results, start=1):
        cand = str(meta.get("id_candidate", "NA"))
        print(f"{rank:02d}. id_candidate={cand} | dist={dist:.4f} | ~sim={sim:.4f} | doc_id={doc_id}")

        if req_lang:
            cand_lvl = meta.get(f"lvl_{req_lang}", "none")
            cand_rank = meta.get(f"lvl_{req_lang}_rank", 0)
            print(f"    {req_lang} nivel={cand_lvl} (rank={cand_rank})")

        if args.show_doc and doc:
            snippet = (doc[:220] + "...") if len(doc) > 220 else doc
            print(f"    doc: {snippet}")


if __name__ == "__main__":
    main()
