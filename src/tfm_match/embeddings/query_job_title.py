#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Optional

import chromadb
from openai import OpenAI


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise ValueError(f"Falta variable de entorno requerida: {name}")
    return v


def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def main():
    parser = argparse.ArgumentParser(description="Consulta índice de cargo/job_title en Chroma")
    parser.add_argument("--q", default="", help="Texto de consulta (nombre del cargo)")
    parser.add_argument("--topk", type=int, default=10, help="Top-K resultados")
    parser.add_argument("--dedup", action="store_true", help="Deduplicar por candidato (si hubiese múltiples docs)")
    parser.add_argument("--show-doc", action="store_true", help="Mostrar snippet del documento recuperado")
    args = parser.parse_args()

    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_JOB_TITLE", "candidates_job_title")

    q = args.q.strip()
    if not q:
        q = "Científico de datos"

    client = OpenAI(api_key=openai_api_key)

    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_collection(name=collection_name)

    emb = client.embeddings.create(model=embedding_model, input=[q]).data[0].embedding

    res = collection.query(
        query_embeddings=[emb],
        # Si hay múltiples docs por candidato (job_title::1..N), pedimos más y luego deduplicamos.
        n_results=min(args.topk * 6, 200) if args.dedup else args.topk,
        include=["metadatas", "documents", "distances"]
    )

    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]

    print(f"Consulta: {q}")
    print(f"Colección: {collection_name}")
    print("Resultados (distancia coseno ~ 1 - similitud; menor es mejor):\n")

    if not ids:
        print("Sin resultados.")
        return

    if args.dedup:
        best = {}
        for i, _id in enumerate(ids):
            cand = str(metas[i].get("id_candidate"))
            dist = float(dists[i])
            if cand not in best or dist < best[cand]["dist"]:
                best[cand] = {"dist": dist, "id": _id, "meta": metas[i], "doc": docs[i]}
        sorted_items = sorted(best.items(), key=lambda x: x[1]["dist"])

        for rank, (cand, item) in enumerate(sorted_items[: args.topk], start=1):
            dist = item["dist"]
            approx_sim = 1.0 - dist
            print(f"{rank:02d}. id_candidate={cand} | dist={dist:.4f} | ~sim={approx_sim:.4f} | doc_id={item['id']}")
            if args.show_doc:
                snippet = (item["doc"][:220] + "...") if item["doc"] and len(item["doc"]) > 220 else item["doc"]
                print(f"    snippet: {snippet}")
                jt = (item.get("meta") or {}).get("job_title")
                if jt:
                    print(f"    job_title(meta): {jt}")
    else:
        for i, _id in enumerate(ids):
            cand = str(metas[i].get("id_candidate"))
            dist = float(dists[i])
            approx_sim = 1.0 - dist
            print(f"{i+1:02d}. doc_id={_id} | id_candidate={cand} | dist={dist:.4f} | ~sim={approx_sim:.4f}")
            if args.show_doc:
                snippet = (docs[i][:220] + "...") if docs[i] and len(docs[i]) > 220 else docs[i]
                print(f"    snippet: {snippet}")


if __name__ == "__main__":
    main()
