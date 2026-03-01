#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import chromadb
from openai import OpenAI

from tfm_match.gold.text_sanitizer import sanitize_text
from tfm_match.config import get_env

# Opcional: lookup a MySQL para enriquecer salida
try:
    import pandas as pd
    from sqlalchemy import create_engine, text
except Exception:
    pd = None
    create_engine = None
    text = None

# =========================================================
# RANGOS (UI) + penalización "después de"
# - 0-1   penaliza después de 3 años
# - 1-3   penaliza después de 5 años
# - 3-5   penaliza después de 8 años
# - 5+    penaliza después de 12 años
#
# Interpretación (0-1/1-3/3-5):
#   1) x < m            => 0
#   2) m <= x <= upper  => rampa lineal 0..1
#   3) upper < x <= P   => 1 (plateau)
#   4) x > P            => exp(-k(x-P))
#
# Interpretación (5+):
#   - x < 5      => x/5
#   - 5 <= x<=12 => 1
#   - x > 12     => exp(-k(x-12))
#
# sim se usa como filtro (sim > tau) y como desempate en el ordenamiento,
# NO se multiplica por el score_experiencia.
# =========================================================

RANGE_CFG: Dict[str, Dict[str, float]] = {
    "0-1": {"m": 0.0, "upper": 1.0, "penalize_after": 3.0},
    "1-3": {"m": 1.0, "upper": 3.0, "penalize_after": 5.0},
    "3-5": {"m": 3.0, "upper": 5.0, "penalize_after": 8.0},
    "5+":  {"m": 5.0, "upper": 5.0, "penalize_after": 12.0},
}


def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def range_to_text(rng: str) -> str:
    rng = (rng or "").strip()
    if not rng:
        return ""
    if rng == "5+":
        return "Más de 5 años de experiencia"
    if rng == "0-1":
        return "Entre 0 y 1 años de experiencia"
    a, b = rng.split("-")
    return f"Entre {a} y {b} años de experiencia"


def extract_months_from_text(text: str) -> Optional[int]:
    if not text:
        return None

    t = text.lower()
    t = t.replace("años", "anos").replace("año", "ano")
    t = t.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

    months: List[int] = []

    # X anos y Y meses
    for y, m in re.findall(r"(\\d+)\\s*anos?\\s*y\\s*(\\d+)\\s*mes(?:es)?", t):
        try:
            months.append(int(y) * 12 + int(m))
        except Exception:
            pass

    # Solo años: X anos
    for y in re.findall(r"(\\d+)\\s*anos?", t):
        try:
            months.append(int(y) * 12)
        except Exception:
            pass

    # Solo meses: X meses
    for m in re.findall(r"(\\d+)\\s*mes(?:es)?", t):
        try:
            months.append(int(m))
        except Exception:
            pass

    return max(months) if months else None


def extract_months(meta: Dict[str, Any], doc: str) -> Optional[int]:
    # Meses (si el indexador los guardó)
    for k in ["exp_months", "experience_months", "months_experience", "total_months", "exp_total_months"]:
        if k in meta and meta[k] is not None and str(meta[k]).strip() != "":
            try:
                return int(float(meta[k]))
            except Exception:
                pass

    # Años (si el indexador los guardó)
    for k in ["exp_years", "experience_years", "years_experience", "total_years", "exp_total_years"]:
        if k in meta and meta[k] is not None and str(meta[k]).strip() != "":
            try:
                return int(float(meta[k])) * 12
            except Exception:
                pass

    return extract_months_from_text(doc or "")


def f_bucket(x_years: float, m: float, upper: float, penalize_after: float, k: float) -> float:
    """
    Para rangos 0-1, 1-3, 3-5:
      - x < m                       => 0
      - m <= x <= upper             => (x-m)/(upper-m)
      - upper < x <= penalize_after => 1
      - x > penalize_after          => exp(-k(x-penalize_after))
    """
    if x_years < m:
        return 0.0

    # rampa
    if x_years <= upper:
        if upper == m:
            return 1.0
        return (x_years - m) / (upper - m)

    # plateau
    if x_years <= penalize_after:
        return 1.0

    # penalización
    return math.exp(-k * (x_years - penalize_after))


def f_5plus(x_years: float, k: float, penalize_after: float = 12.0) -> float:
    """
    5+:
      - x < 5        => x/5
      - 5 <= x <= 12 => 1
      - x > 12       => exp(-k(x-12))
    """
    if x_years < 0:
        return 0.0
    if x_years < 5.0:
        return max(0.0, min(1.0, x_years / 5.0))
    if x_years <= penalize_after:
        return 1.0
    return math.exp(-k * (x_years - penalize_after))


def fetch_candidate_info(engine, candidate_ids: List[str], table: str, cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Devuelve un dict: {id_candidate: {col: value, ...}}
    """
    if not candidate_ids:
        return {}

    # MySQL: IN con bind params seguros
    placeholders = ", ".join([f":id{i}" for i in range(len(candidate_ids))])
    sql = f"SELECT {', '.join(cols)}, id_candidate FROM {table} WHERE id_candidate IN ({placeholders})"

    params = {f"id{i}": int(cid) if str(cid).isdigit() else cid for i, cid in enumerate(candidate_ids)}
    df = pd.read_sql(text(sql), engine, params=params)

    out = {}
    for _, r in df.iterrows():
        cid = str(r["id_candidate"])
        out[cid] = {c: (None if pd.isna(r[c]) else r[c]) for c in cols if c in r}
    return out


def main():
    parser = argparse.ArgumentParser(description="Consulta índice de experiencia en Chroma (orden: score_experiencia, luego sim)")
    parser.add_argument("--q", default="", help="Texto de consulta (rol/experiencia deseada)")
    parser.add_argument("--range", default="", choices=["", "0-1", "1-3", "3-5", "5+"],
                        help="Rango de experiencia requerido: 0-1, 1-3, 3-5, 5+")
    parser.add_argument("--tau", type=float, default=0.6,
                        help="Filtro por similitud: solo deja pasar hits con sim > tau (default 0.6)")
    parser.add_argument("--k", type=float, default=0.3,
                        help="Parámetro k del decaimiento exp(-k*Δ) (default 0.3)")
    parser.add_argument("--hard-range", action="store_true",
                        help="Filtro duro: exige poder inferir x. (En 1-3/3-5 exige x>=m; en 0-1/5+ no descarta por m)")
    parser.add_argument("--topk", type=int, default=10, help="Top-K resultados (chunks)")
    parser.add_argument("--dedup", action="store_true", help="Agrupar por candidato (deduplicar chunks)")
    parser.add_argument("--show-doc", action="store_true", help="Mostrar snippet del documento recuperado")

    # Enriquecimiento opcional con MySQL (SQLAlchemy)
    parser.add_argument("--mysql", action="store_true", help="Enriquecer resultados consultando MySQL (requiere MYSQL_URL)")
    parser.add_argument("--mysql-table", default="candidates_prepared", help="Tabla MySQL para lookup")
    parser.add_argument("--mysql-cols", default="experience,skills", help="Columnas CSV a mostrar desde MySQL (además de id_candidate)")

    args = parser.parse_args()

    openai_api_key = get_env("OPENAI_API_KEY", required=True)
    embedding_model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")

    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_EXPERIENCE", "candidates_experience")

    q_text = (args.q or "").strip()
    rng_text = range_to_text(args.range)

    if rng_text and q_text:
        q = f"{rng_text}. {q_text}"
    elif rng_text:
        q = rng_text
    elif q_text:
        q = q_text
    else:
        q = "Entre 1 y 3 años de experiencia. supervisor de operaciones"

    # Consistencia con el resto del sistema: sanitiza el texto antes de embebido
    q = sanitize_text(q)

    client = OpenAI(api_key=openai_api_key)

    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # Validación: la colección existe
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo abrir la colección '{collection_name}' en CHROMA_DIR='{chroma_dir}'. "
            f"¿Ya indexaste experience? Error: {e}"
        )

    if collection.count() == 0:
        raise RuntimeError(
            f"La colección '{collection_name}' existe pero está vacía (count=0). "
            f"Ejecuta el indexador de experience contra CHROMA_DIR='{chroma_dir}'."
        )

    emb = client.embeddings.create(model=embedding_model, input=[q]).data[0].embedding

    # Oversampling si vamos a deduplicar por candidato (pueden existir múltiples chunks/docs por candidato)
    n_fetch = args.topk
    if args.dedup:
        max_fetch = max(args.topk, 200)
        n_fetch = min(args.topk * 6, max_fetch)

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
    print(f"CHROMA_DIR: {chroma_dir}")
    print(f"Colección: {collection_name} (count={collection.count()})")
    if args.range:
        cfg = RANGE_CFG[args.range]
        print(f"Rango: {args.range} | m={cfg['m']} upper={cfg['upper']} penalize_after={cfg['penalize_after']} | tau={args.tau} | k={args.k} | hard_range={bool(args.hard_range)}")
    else:
        print(f"Sin rango | tau={args.tau} | k={args.k} | hard_range={bool(args.hard_range)}")
    print("Filtro: sim > tau. Orden: score_experiencia desc, luego sim desc (cuando hay rango). Sin rango: sim desc.\n")

    # Preparar lookup MySQL si aplica
    mysql_map: Dict[str, Dict[str, Any]] = {}
    if args.mysql:
        if pd is None or create_engine is None:
            raise RuntimeError("Faltan dependencias para MySQL lookup. Instala: poetry add sqlalchemy pymysql pandas")

        mysql_url = get_env("MYSQL_URL", required=True)
        engine = create_engine(mysql_url, pool_pre_ping=True)

        mysql_cols = [c.strip() for c in args.mysql_cols.split(",") if c.strip()]
        # candidate ids dedup para query
        cand_ids = []
        for i in range(len(ids)):
            if metas[i] and metas[i].get("id_candidate") is not None:
                cand_ids.append(str(metas[i]["id_candidate"]))
        cand_ids = sorted(set(cand_ids))

        mysql_map = fetch_candidate_info(engine, cand_ids, args.mysql_table, mysql_cols)

    def print_mysql(cid: str):
        if not args.mysql:
            return
        info = mysql_map.get(cid)
        if not info:
            print("    mysql: (sin datos)")
            return
        # imprime de forma compacta
        for k, v in info.items():
            if v is None:
                continue
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"    mysql.{k}: {s}")

    def sim_from_dist(dist: float) -> float:
        return max(0.0, min(1.0, 1.0 - float(dist)))

    def pass_hard_range(months: Optional[int]) -> bool:
        if not args.hard_range or not args.range:
            return True
        if months is None:
            return False
        if args.range in ("0-1", "5+"):
            return True
        m = RANGE_CFG[args.range]["m"]
        return (months / 12.0) >= m

    def compute_score(meta: Dict[str, Any], doc_txt: str, dist: float):
        sim = sim_from_dist(dist)
        if not args.range:
            return None, None, sim

        months = extract_months(meta, doc_txt)
        if months is None:
            return None, None, sim

        x_years = months / 12.0
        cfg = RANGE_CFG[args.range]

        if args.range == "5+":
            score = f_5plus(x_years=x_years, k=args.k, penalize_after=cfg["penalize_after"])
        else:
            score = f_bucket(
                x_years=x_years,
                m=cfg["m"],
                upper=cfg["upper"],
                penalize_after=cfg["penalize_after"],
                k=args.k
            )
        return score, months, sim

    if args.dedup:
        best_by_candidate: Dict[str, Dict[str, Any]] = {}

        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            cand = str(meta.get("id_candidate", ""))
            if not cand:
                continue

            dist = float(dists[i])
            sim = sim_from_dist(dist)
            if not (sim > args.tau):
                continue

            doc_txt = docs[i] or ""
            score, months, _ = compute_score(meta, doc_txt, dist)

            if not pass_hard_range(months):
                continue

            # Sin rango: dedup por sim (dist asc)
            if not args.range:
                if cand not in best_by_candidate or dist < best_by_candidate[cand]["dist"]:
                    best_by_candidate[cand] = {
                        "score": None,
                        "sim": sim,
                        "dist": dist,
                        "id": _id,
                        "meta": meta,
                        "doc": doc_txt,
                        "months": months
                    }
                continue

            # Con rango: si no pudimos inferir score, no entra
            if score is None:
                continue

            # Mejor chunk por candidato: score desc, luego sim desc, luego dist asc
            if cand not in best_by_candidate:
                best_by_candidate[cand] = {
                    "score": score,
                    "sim": sim,
                    "dist": dist,
                    "id": _id,
                    "meta": meta,
                    "doc": doc_txt,
                    "months": months
                }
            else:
                cur = best_by_candidate[cand]
                better = (
                    (score > cur.get("score", -1.0)) or
                    (score == cur.get("score") and sim > cur["sim"]) or
                    (score == cur.get("score") and sim == cur["sim"] and dist < cur["dist"])
                )
                if better:
                    best_by_candidate[cand] = {
                        "score": score,
                        "sim": sim,
                        "dist": dist,
                        "id": _id,
                        "meta": meta,
                        "doc": doc_txt,
                        "months": months
                    }

        if not best_by_candidate:
            print("Sin resultados tras aplicar filtros.")
            return

        if args.range:
            # Orden final: score desc, sim desc, dist asc
            sorted_items = sorted(best_by_candidate.items(), key=lambda x: (-float(x[1]["score"]), -x[1]["sim"], x[1]["dist"]))
        else:
            # Sin rango: sim desc, dist asc
            sorted_items = sorted(best_by_candidate.items(), key=lambda x: (-x[1]["sim"], x[1]["dist"]))

        for rank, (cand, item) in enumerate(sorted_items[: args.topk], start=1):
            months = item.get("months")
            years_txt = f"{months//12}y{months%12}m" if isinstance(months, int) else "NA"
            if args.range:
                print(f"{rank:02d}. id_candidate={cand} | exp={years_txt} | score_experiencia={float(item['score']):.4f} | doc_id={item['id']}")
            else:
                print(f"{rank:02d}. id_candidate={cand} | ~sim={item['sim']:.4f} | doc_id={item['id']}")
            print_mysql(cand)
            if args.show_doc:
                doc = item.get("doc") or ""
                snippet = (doc[:220] + "...") if len(doc) > 220 else doc
                print(f"    snippet: {snippet}")
    else:
        rows: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            cand = str(meta.get("id_candidate", ""))
            dist = float(dists[i])
            sim = sim_from_dist(dist)
            if not (sim > args.tau):
                continue

            doc_txt = docs[i] or ""
            score, months, _ = compute_score(meta, doc_txt, dist)

            if not pass_hard_range(months):
                continue

            # Sin rango: rank por sim
            if not args.range:
                rows.append({"id": _id, "cand": cand, "months": months, "score": None, "sim": sim, "dist": dist, "doc": doc_txt})
                continue

            # Con rango: si no hay score, no entra
            if score is None:
                continue

            rows.append({"id": _id, "cand": cand, "months": months, "score": score, "sim": sim, "dist": dist, "doc": doc_txt})

        if not rows:
            print("Sin resultados tras aplicar filtros.")
            return

        if args.range:
            rows.sort(key=lambda r: (-float(r["score"]), -r["sim"], r["dist"]))
        else:
            rows.sort(key=lambda r: (-r["sim"], r["dist"]))

        for idx, r in enumerate(rows[: args.topk], start=1):
            months = r["months"]
            years_txt = f"{months//12}y{months%12}m" if isinstance(months, int) else "NA"
            if args.range:
                print(f"{idx:02d}. doc_id={r['id']} | id_candidate={r['cand']} | exp={years_txt} | score_experiencia={float(r['score']):.4f}")
            else:
                print(f"{idx:02d}. doc_id={r['id']} | id_candidate={r['cand']} | ~sim={r['sim']:.4f}")
            if r["cand"]:
                print_mysql(r["cand"])
            if args.show_doc:
                snippet = (r["doc"][:220] + "...") if len(r["doc"]) > 220 else r["doc"]
                print(f"    snippet: {snippet}")


if __name__ == "__main__":
    main()
