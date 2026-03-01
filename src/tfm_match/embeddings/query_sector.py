#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List

import chromadb
from openai import OpenAI

from tfm_match.config import get_env

def build_query(sectors: List[str]) -> str:
    # Canonical string compatible con el indexer
    sectors = [s.strip().lower() for s in sectors if s.strip()]
    if not sectors:
        return ""
    return "; ".join([f"sector: {s}" for s in sectors])


def main():
    parser = argparse.ArgumentParser(description="Consulta índice de Área/Sector en Chroma")
    parser.add_argument("--q", default="", help="Texto libre (ej: 'BPO contact center' / 'logistica')")
    parser.add_argument("--sector", default="", help="Sector canónico (ej: bpo_contact_center). Puedes repetir separado por coma.")
    parser.add_argument("--topk", type=int, default=20, help="Top-K resultados")
    parser.add_argument("--dedup", action="store_true", help="Deduplicar por candidato (útil si hay múltiples docs por candidato)")
    parser.add_argument("--must-have", action="store_true", help="Post-filter: el candidato debe tener el/los sectores canónicos")
    parser.add_argument("--show-doc", action="store_true", help="Mostrar documento canónico recuperado")
    args = parser.parse_args()

    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_dir = get_env("CHROMA_DIR", "./data/chroma")
    collection_name = get_env("CHROMA_COLLECTION_SECTOR", "candidates_sector")

    client = OpenAI(api_key=openai_api_key)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_collection(name=collection_name)

    sectors = []
    if args.sector.strip():
        sectors = [s.strip() for s in args.sector.split(",") if s.strip()]

    if sectors:
        q = build_query(sectors)
    else:
        q = (args.q or "").strip()
        if not q:
            q = "bpo contact center"

    emb = client.embeddings.create(model=embedding_model, input=[q]).data[0].embedding

    n_fetch = args.topk
    if args.dedup:
        n_fetch = min(max(args.topk * 6, args.topk), 200)

    res = collection.query(
        query_embeddings=[emb],
        n_results=n_fetch,
        include=["metadatas", "documents", "distances"]
    )

    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]

    print(f"Consulta: {q}")
    print(f"Colección: {collection_name}")
    print("Nota: en cosine distance, menor es mejor (aprox_sim ~ 1 - dist)\n")

    out = []
    for i, doc_id in enumerate(ids):
        meta = metas[i] or {}
        dist = float(dists[i])
        approx_sim = 1.0 - dist

        # Post-filter: debe contener sectores canónicos
        if args.must_have and sectors:
            ok = True
            for s in sectors:
                if meta.get(f"has_{s}", 0) != 1:
                    ok = False
                    break
            if not ok:
                continue

        out.append((doc_id, meta, dist, approx_sim, docs[i]))

    if not out:
        print("Sin resultados tras aplicar filtros.")
        return

    if args.dedup:
        best = {}
        for (doc_id, meta, dist, sim, doc) in out:
            cand = str(meta.get("id_candidate", "NA"))
            if cand not in best or dist < best[cand]["dist"]:
                best[cand] = {"dist": dist, "sim": sim, "doc_id": doc_id, "meta": meta, "doc": doc}
        sorted_items = sorted(best.items(), key=lambda x: x[1]["dist"])
        for rank, (cand, item) in enumerate(sorted_items[: args.topk], start=1):
            meta = item["meta"] or {}
            sects = meta.get("sectors", "")
            print(f"{rank:02d}. id_candidate={cand} | dist={item['dist']:.4f} | ~sim={item['sim']:.4f} | doc_id={item['doc_id']}")
            if sects:
                print(f"    sectors: {sects}")
            if args.show_doc and item["doc"]:
                doc = item["doc"]
                snippet = (doc[:220] + "...") if len(doc) > 220 else doc
                print(f"    doc: {snippet}")
    else:
        for rank, (doc_id, meta, dist, sim, doc) in enumerate(out[: args.topk], start=1):
            cand = str(meta.get("id_candidate", "NA"))
            sects = meta.get("sectors", "")
            print(f"{rank:02d}. id_candidate={cand} | dist={dist:.4f} | ~sim={sim:.4f} | doc_id={doc_id}")
            if sects:
                print(f"    sectors: {sects}")
            if args.show_doc and doc:
                snippet = (doc[:220] + "...") if len(doc) > 220 else doc
                print(f"    doc: {snippet}")


if __name__ == "__main__":
    main()
