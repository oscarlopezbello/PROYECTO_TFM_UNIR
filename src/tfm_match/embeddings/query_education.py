#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import chromadb
from openai import OpenAI


EDU_LEVELS = {
    "none": 0,
    "bachiller": 1,
    "tecnico": 2,
    "tecnologo": 3,
    "profesional": 4,
    "posgrado": 5,
}

# Alias para que tu UI/usuarios escriban con tildes o variantes
ALIASES = {
    "técnico": "tecnico",
    "tecnico": "tecnico",
    "tecnólogo": "tecnologo",
    "tecnologo": "tecnologo",
    "profesional": "profesional",
    "posgrado": "posgrado",
    "postgrado": "posgrado",
    "maestria": "posgrado",
    "especializacion": "posgrado",
    "doctorado": "posgrado",
    "bachiller": "bachiller",
}

def get_env(name: str, default=None, required: bool = False):
    v = os.getenv(name, default)
    if required and not v:
        raise ValueError(f"Falta variable de entorno requerida: {name}")
    return v


def canonicalize_level(x: str) -> str:
    if not x:
        return "none"
    k = x.strip().lower()
    return ALIASES.get(k, k)


def build_query(level: str) -> str:
    # Debe ser compatible con el indexer
    return f"education_level: {level}"


def main():
    parser = argparse.ArgumentParser(description="Consulta índice de Formación en Chroma")
    parser.add_argument("--level", default="", help="Nivel requerido: tecnico, tecnologo, profesional, posgrado")
    parser.add_argument("--q", default="", help="Texto libre alternativo (si no usas --level)")
    parser.add_argument("--topk", type=int, default=50, help="Top-K resultados (antes de filtros)")
    parser.add_argument("--min-level", action="store_true", help="Hard filter: candidato debe tener nivel >= requerido")
    parser.add_argument("--show-doc", action="store_true", help="Mostrar documento recuperado")
    args = parser.parse_args()

    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_dir = get_env("CHROMA_DIR", "./data/chroma")
    collection_name = get_env("CHROMA_COLLECTION_EDUCATION", "candidates_education")

    client = OpenAI(api_key=openai_api_key)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_collection(name=collection_name)

    if args.level.strip():
        req_level = canonicalize_level(args.level)
        if req_level not in EDU_LEVELS:
            raise ValueError(f"Nivel no soportado: {args.level}. Usa: {list(EDU_LEVELS.keys())}")
        req_rank = EDU_LEVELS[req_level]
        q = build_query(req_level)
    else:
        q = (args.q or "").strip()
        if not q:
            q = "education_level: profesional"
        # Si usas texto libre, no aplicamos filtro por ranking a menos que el usuario lo dé con --level
        req_level, req_rank = "", 0

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

    out = []
    for i, doc_id in enumerate(ids):
        meta = metas[i] or {}
        dist = float(dists[i])
        approx_sim = 1.0 - dist

        # Hard filter por mínimo nivel
        if args.min_level and req_rank > 0:
            cand_rank = int(meta.get("edu_rank", 0) or 0)
            if cand_rank < req_rank:
                continue

        out.append((doc_id, meta, dist, approx_sim, docs[i]))

    if not out:
        print("Sin resultados tras aplicar filtros.")
        return

    for rank, (doc_id, meta, dist, sim, doc) in enumerate(out, start=1):
        cand = str(meta.get("id_candidate", "NA"))
        cand_level = meta.get("edu_level", "none")
        cand_rank = meta.get("edu_rank", 0)
        ev = meta.get("evidence", "")

        print(f"{rank:02d}. id_candidate={cand} | edu={cand_level} (rank={cand_rank}) | dist={dist:.4f} | ~sim={sim:.4f} | doc_id={doc_id}")
        if ev:
            print(f"    evidence: {ev}")
        if args.show_doc and doc:
            snippet = (doc[:220] + "...") if len(doc) > 220 else doc
            print(f"    doc: {snippet}")


if __name__ == "__main__":
    main()
