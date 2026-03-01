from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

import re
import math
import unicodedata
import time
import threading
from collections import deque
import statistics
import pandas as pd
import chromadb
from openai import OpenAI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from tfm_match.config import (
    get_env,
    OPENAI_API_KEY as CFG_OPENAI_API_KEY,
    MYSQL_URL as CFG_MYSQL_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_DIR,
    CHROMA_COLLECTION_SKILLS,
    CHROMA_COLLECTION_EXPERIENCE,
    CHROMA_COLLECTION_EDUCATION,
    CHROMA_COLLECTION_LANGUAGE,
    CHROMA_COLLECTION_SECTOR,
    CHROMA_COLLECTION_JOB_TITLE,
    CHROMA_COLLECTION_CITY,
    CANDIDATES_TABLE,
)
from tfm_match.core.embeddings_manager import EmbeddingsManager
from tfm_match.core.dimension_matcher import DimensionMatcher
from tfm_match.core.result_aggregator import ResultAggregator
from tfm_match.core.filters import HardFilters
from tfm_match.core.persistence import PersistenceManager
from tfm_match.llm.client import invoke_match_orchestrator
from tfm_match.api.reranking_rules import apply_reranking_rules


app = FastAPI(title="TFM Match API", version="1.0.0")


def _normalize_match_payload(payload: dict) -> dict:
    """Normaliza payload del LLM a estructura MatchRequest."""
    w = payload.get("weights") or {}
    # Default 0 para dimensiones no especificadas: solo cuentan las que el usuario definió
    weights = {
        "skills": w.get("skills", 0),
        "experience": w.get("experience", 0),
        "education": w.get("education", 0),
        "language": w.get("language", 0),
        "sector": w.get("sector", 0),
        "job_title": w.get("job_title", 0),
        "city": w.get("city", 0),
    }
    hf = payload.get("hard_filters") or {}
    hard_filters = {
        "education_min": hf.get("education_min"),
        "language_required": hf.get("language_required"),
        "language_min_level": hf.get("language_min_level"),
    }
    return {
        "skills": payload.get("skills") or "",
        "experience": payload.get("experience") or "",
        "education": payload.get("education") or "",
        "language": payload.get("language") or "",
        "sector": payload.get("sector") or "",
        "job_title": payload.get("job_title") or "",
        "city": payload.get("city") or "",
        "top_k": payload.get("top_k", 10),
        "weights": weights,
        "hard_filters": hard_filters,
    }


# ============================================================
# Config / Globals (via ENV)
# ============================================================

# Validar variables requeridas para la API
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
MYSQL_URL = get_env("MYSQL_URL", required=True)
EMBED_MODEL = EMBEDDING_MODEL

# Nombres de colecciones (defaults coherentes con tus scripts)
COLLECTIONS = {
    "skills": CHROMA_COLLECTION_SKILLS,
    "experience": CHROMA_COLLECTION_EXPERIENCE,
    "education": CHROMA_COLLECTION_EDUCATION,
    "language": CHROMA_COLLECTION_LANGUAGE,
    "sector": CHROMA_COLLECTION_SECTOR,
    "job_title": CHROMA_COLLECTION_JOB_TITLE,
    "city": CHROMA_COLLECTION_CITY,
}


# Inicialización (se setea en startup)
oa_client: Optional[OpenAI] = None
engine = None
chroma_client = None
chroma_collections: Dict[str, Any] = {}

def _percentile(sorted_vals: List[float], p: float) -> float:
    """
    Percentil simple (nearest-rank con interpolación lineal) sobre lista ordenada.
    p en [0,1].
    """
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


class LatencyTracker:
    """
    Tracker en memoria para latencias (últimas N requests).
    """
    def __init__(self, maxlen: int = 500):
        self._lock = threading.Lock()
        self._samples = deque(maxlen=maxlen)  # cada sample: dict[str,float]

    def record(self, timings_ms: Dict[str, float]) -> None:
        with self._lock:
            # Copia defensiva
            self._samples.append(dict(timings_ms))

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {"count": 0, "stages": {}, "note": "Sin muestras aún."}

        # union de keys
        keys = set()
        for s in samples:
            keys.update(s.keys())

        stages = {}
        for k in sorted(keys):
            vals = [float(s.get(k, 0.0) or 0.0) for s in samples]
            vals_sorted = sorted(vals)
            stages[k] = {
                "count": len(vals),
                "mean_ms": round(statistics.mean(vals), 2),
                "p50_ms": round(_percentile(vals_sorted, 0.50), 2),
                "p95_ms": round(_percentile(vals_sorted, 0.95), 2),
                "min_ms": round(vals_sorted[0], 2),
                "max_ms": round(vals_sorted[-1], 2),
            }

        return {
            "count": len(samples),
            "stages": stages,
            "last": samples[-1],
        }


LATENCY_TRACKER = LatencyTracker(maxlen=500)


@app.on_event("startup")
def startup():
    global oa_client, engine, chroma_client, chroma_collections

    oa_client = OpenAI(api_key=OPENAI_API_KEY)
    engine = create_engine(MYSQL_URL, pool_pre_ping=True)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    chroma_collections = {}
    for dim, name in COLLECTIONS.items():
        try:
            chroma_collections[dim] = chroma_client.get_collection(name=name)
        except Exception:
            # Si aún no existe la colección, simplemente queda deshabilitada
            chroma_collections[dim] = None


# ============================================================
# Schemas
# ============================================================

class Weights(BaseModel):
    skills: int = Field(5, ge=0, le=10)
    experience: int = Field(0, ge=0, le=10)
    education: int = Field(0, ge=0, le=10)
    language: int = Field(0, ge=0, le=10)
    sector: int = Field(0, ge=0, le=10)
    job_title: int = Field(0, ge=0, le=10)
    city: int = Field(0, ge=0, le=10)


class HardFiltersRequest(BaseModel):
    # Formación mínima: "Técnico", "Tecnólogo", "Profesional", "Posgrado"
    education_min: Optional[str] = None

    # Idioma requerido: "Inglés", "Francés", "Portugués", "Español", etc.
    language_required: Optional[str] = None

    # Nivel mínimo opcional: "A1","A2","B1","B2","C1","C2"
    language_min_level: Optional[str] = None


class MatchRequest(BaseModel):
    # Dimensiones 
    skills: Optional[str] = None
    experience: Optional[str] = None
    education: Optional[str] = None
    language: Optional[str] = None
    sector: Optional[str] = None
    job_title: Optional[str] = None
    city: Optional[str] = None

    top_k: int = Field(10, ge=1, le=100)
    weights: Weights = Weights()
    hard_filters: HardFiltersRequest = HardFiltersRequest()


# ============================================================
# Helpers: Ahora en core/
# ============================================================
# Las funciones embed_text, query_dim, apply_hard_filters, etc.
# han sido movidas a módulos core/ para reutilización


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chroma_dir": CHROMA_DIR,
        "collections": {k: (v is not None) for k, v in chroma_collections.items()},
        "embed_model": EMBED_MODEL,
    }


@app.get("/latency")
def get_latency():
    """
    Devuelve métricas de latencia en memoria (últimas N requests).
    """
    return LATENCY_TRACKER.summary()


@app.get("/cities")
def get_cities():
    """
    Retorna lista de ciudades únicas desde la base de datos.
    """
    if engine is None:
        raise HTTPException(500, "DB engine no inicializado")
    
    try:
        sql = text(f"""
            SELECT DISTINCT TRIM(location) as city
            FROM {CANDIDATES_TABLE}
            WHERE location IS NOT NULL 
              AND TRIM(location) != ''
            ORDER BY city
        """)
        df = pd.read_sql(sql, engine)
        cities = df["city"].tolist()
        return {"cities": cities}
    except Exception as e:
        raise HTTPException(500, f"Error obteniendo ciudades: {str(e)}")


@app.get("/job_requests")
def list_job_requests():
    if engine is None:
        raise HTTPException(500, "DB engine no inicializado")

    query = """
        SELECT id_request, query_text, top_k, created_at
        FROM job_requests
        ORDER BY created_at DESC
        LIMIT 50
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")


@app.get("/job_requests/{id_request}")
def get_job_request_results(id_request: int):
    if engine is None:
        raise HTTPException(500, "DB engine no inicializado")

    meta_df = pd.read_sql(
        text("""
            SELECT id_request, query_text, top_k, created_at
            FROM job_requests
            WHERE id_request = :id_request
        """),
        engine,
        params={"id_request": id_request}
    )

    if meta_df.empty:
        return {"error": "job_request no encontrado"}

    results_df = pd.read_sql(
        text("""
            SELECT candidate_id, affinity, rank_position
            FROM job_request_results
            WHERE id_request = :id_request
            ORDER BY rank_position ASC
        """),
        engine,
        params={"id_request": id_request}
    )

    return {
        "job_request": meta_df.iloc[0].to_dict(),
        "results": results_df.to_dict(orient="records")
    }


def _run_match(request: MatchRequest) -> dict:
    """Ejecuta el flujo completo de matching. Usado por la tool execute_match."""
    if engine is None or oa_client is None:
        raise HTTPException(500, "Servicio no inicializado correctamente")

    t_start = time.perf_counter()
    timings_ms: Dict[str, float] = {}

    # Inicializar componentes del core
    embeddings_mgr = EmbeddingsManager(oa_client, EMBED_MODEL)
    matcher = DimensionMatcher(chroma_collections, embeddings_mgr)
    aggregator = ResultAggregator()
    filters = HardFilters(chroma_collections)
    persistence = PersistenceManager(engine, CANDIDATES_TABLE)

    # 1) Normaliza textos (no obligatorios)
    q_skills = (request.skills or "").strip()
    q_experience = (request.experience or "").strip()
    q_education = (request.education or "").strip()
    q_language = (request.language or "").strip()
    q_sector = (request.sector or "").strip()
    q_job_title = (request.job_title or "").strip()
    q_city = (request.city or "").strip()

    # Compatibilidad con tu UI actual: si solo manda skills, funciona igual.
    if not any([q_skills, q_experience, q_education, q_language, q_sector, q_job_title, q_city]):
        raise HTTPException(422, "Debes enviar al menos una dimensión (skills/experience/education/language/sector/job_title/city).")

    # 2) Preparar query_payload
    query_payload = {
        "skills": q_skills,
        "experience": q_experience,
        "education": q_education,
        "language": q_language,
        "sector": q_sector,
        "job_title": q_job_title,
        "city": q_city,
        "hard_filters": request.hard_filters.model_dump(),
    }

    # 3) Retrieval por dimensión (usando DimensionMatcher)
    per_dim_k = max(request.top_k * 10, 50)

    t_retrieval = time.perf_counter()
    hits_by_dim: Dict[str, List[Dict[str, Any]]] = {
        "skills": matcher.query_dimension("skills", q_skills, per_dim_k, dedup_by_candidate=True),
        "education": matcher.query_dimension("education", q_education, per_dim_k, dedup_by_candidate=True),
        "language": matcher.query_dimension("language", q_language, per_dim_k, dedup_by_candidate=True),
        "sector": matcher.query_dimension("sector", q_sector, per_dim_k, dedup_by_candidate=True),
        "job_title": matcher.query_dimension("job_title", q_job_title, per_dim_k, dedup_by_candidate=True),
    }
    
    # Ciudad: match directo en MySQL (sin embeddings)
    if q_city and chroma_collections.get("city") is None:
        city_hits = persistence.match_city_direct(q_city)
        hits_by_dim["city"] = city_hits
    else:
        hits_by_dim["city"] = matcher.query_dimension("city", q_city, per_dim_k, dedup_by_candidate=True)
    
    # Experiencia: match directo con lógica de rangos (sin embeddings)
    if q_experience and chroma_collections.get("experience"):
        # Si existe colección de embeddings, usarla
        hits_by_dim["experience"] = matcher.query_dimension("experience", q_experience, per_dim_k, dedup_by_candidate=True)
    elif q_experience:
        # Si no, usar match directo con lógica de rangos
        experience_hits = persistence.match_experience_direct(q_experience)
        hits_by_dim["experience"] = experience_hits
    else:
        hits_by_dim["experience"] = []

    timings_ms["retrieval_ms"] = round((time.perf_counter() - t_retrieval) * 1000.0, 2)

    # 4) Unión de candidatos (usando ResultAggregator)
    t_collect = time.perf_counter()
    candidate_ids = aggregator.collect_candidates(hits_by_dim)
    timings_ms["collect_candidates_ms"] = round((time.perf_counter() - t_collect) * 1000.0, 2)

    # 5) Hard filters opcionales (usando HardFilters)
    t_filters = time.perf_counter()
    candidate_ids = filters.apply(candidate_ids, request.hard_filters)
    timings_ms["hard_filters_ms"] = round((time.perf_counter() - t_filters) * 1000.0, 2)

    if not candidate_ids:
        # Guardar job_request vacío
        job_request_id = persistence.save_job_request_and_results(
            query_payload,
            request.weights.model_dump(),
            request.top_k,
            []
        )
        return {
            "job_request_id": job_request_id,
            "query": query_payload,
            "results": [],
            "note": "Sin candidatos tras aplicar hard filters."
        }

    # 6) Score ponderado + breakdown (usando ResultAggregator)
    weight_map = {
        "skills": request.weights.skills,
        "experience": request.weights.experience,
        "education": request.weights.education,
        "language": request.weights.language,
        "sector": request.weights.sector,
        "job_title": request.weights.job_title,
        "city": request.weights.city,
    }

    t_agg = time.perf_counter()
    scored = aggregator.combine_and_rank(
        hits_by_dim,
        weight_map,
        candidate_ids,
        request.top_k
    )
    timings_ms["combine_and_rank_ms"] = round((time.perf_counter() - t_agg) * 1000.0, 2)

    # 7) Enriquecimiento MySQL (usando PersistenceManager)
    t_enrich = time.perf_counter()
    scored = persistence.enrich_candidates(scored)
    timings_ms["enrich_mysql_ms"] = round((time.perf_counter() - t_enrich) * 1000.0, 2)

    # 7.x) Reglas / reranking (skills/sector/experience/education/city/language)
    t_rules = time.perf_counter()

    # 7.05) Similitud de idioma (rule-based) para el breakdown:
    # Para el caso de uso de la UI (idioma requerido + nivel opcional), es más coherente usar una regla discreta:
    # - cumple (>=N) -> 1
    # - a 1 nivel por debajo -> 0.5
    # - 2+ niveles por debajo / no tiene idioma -> 0
    #
    # Esto evita resultados confusos del coseno (p.ej. "Inglés" vs "Español (Nativo)" dando 33%).
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
        "alemán": "german",
        "aleman": "german",
        "german": "german",
        "italiano": "italian",
        "italian": "italian",
    }
    CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    KW_RANK = {"basic": 2, "intermediate": 4, "advanced": 5, "fluent": 6, "native": 7, "none": 0}

    def _parse_lang_reqs(q: str) -> List[Dict[str, Any]]:
        if not q:
            return []
        txt = str(q).strip()
        parts = [p.strip() for p in re.split(r"[;\n,|]+", txt) if p.strip()]
        reqs: Dict[str, int] = {}
        for p in parts:
            m = re.match(r"^\s*([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\s*[: ]?\s*([A-Za-z0-9]+)?\s*$", p)
            lang_raw = m.group(1).strip().lower() if m else ""
            lvl_raw = (m.group(2) or "").strip() if m else ""
            if not lang_raw:
                continue
            canon = LANG_UI_TO_CANON.get(lang_raw, lang_raw)
            rr = 0
            if lvl_raw:
                lv_up = lvl_raw.upper()
                if lv_up in CEFR_RANK:
                    rr = CEFR_RANK[lv_up]
                else:
                    rr = KW_RANK.get(lvl_raw.lower(), 0)
            reqs[canon] = max(reqs.get(canon, 0), rr)
        return [{"language": k, "req_rank": v} for k, v in reqs.items()]

    if q_language and chroma_collections.get("language"):
        col_lang = chroma_collections["language"]
        w_lang = int(weight_map.get("language", 0) or 0)
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)

        reqs = _parse_lang_reqs(q_language)
        if reqs:
            doc_ids = [f"{r.get('candidate_id')}::lang" for r in scored if r.get("candidate_id") is not None]
            metas_by_id: Dict[str, Dict[str, Any]] = {}
            try:
                res_get = col_lang.get(ids=doc_ids, include=["metadatas"])
                ids_out = res_get.get("ids", [])
                metas_out = res_get.get("metadatas", [])
                for i in range(len(ids_out)):
                    metas_by_id[str(ids_out[i])] = metas_out[i] or {}
            except Exception:
                metas_by_id = {}

            for r in scored:
                cid = r.get("candidate_id")
                if cid is None:
                    continue
                doc_id = f"{cid}::lang"
                meta = metas_by_id.get(doc_id, {}) or {}

                scores: List[float] = []
                for req in reqs:
                    canon = req["language"]
                    req_rank = int(req["req_rank"] or 0)
                    has_flag = meta.get(f"has_{canon}", 0)
                    try:
                        has_flag = int(has_flag)
                    except Exception:
                        has_flag = 0
                    if has_flag != 1:
                        scores.append(0.0)
                        continue

                    if req_rank <= 0:
                        scores.append(1.0)
                        continue

                    cand_rank = meta.get(f"lvl_{canon}_rank", 0)
                    try:
                        cand_rank = int(cand_rank)
                    except Exception:
                        cand_rank = 0

                    if cand_rank >= req_rank:
                        scores.append(1.0)
                    elif cand_rank == req_rank - 1:
                        scores.append(0.5)
                    else:
                        scores.append(0.0)

                lang_sim = float(sum(scores) / len(scores)) if scores else 0.0

                bd = r.get("breakdown") or {}
                cosine_original = float((bd.get("language") or {}).get("score_0_1", 0.0) or 0.0)
                bd["language"] = {
                    "score_0_1": round(float(lang_sim), 4),
                    "score_pct": round(float(lang_sim) * 100, 2),
                    "weight": w_lang,
                    "contribution": round(float(lang_sim) * float(w_lang), 4),
                    "rule": "rule_based_language (>=req=1; 1-below=0.5; else=0)",
                    "requirements": reqs,
                    "cosine_sim_original": round(float(cosine_original), 4),
                    "doc_id": doc_id,
                }
                r["breakdown"] = bd

                if w_lang > 0 and total_weight > 0:
                    weighted_sum = 0.0
                    for dim, ww in weight_map.items():
                        if ww <= 0:
                            continue
                        if dim == "language":
                            dim_score = float(lang_sim)
                        else:
                            dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                        weighted_sum += dim_score * float(ww)
                    r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

            scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
            scored = scored[: request.top_k]

    # 7.1) Similitud de skills (fallback léxico) para evitar 0% en casos obvios (ej: "Excel" vs "microsoftexcel").
    # Esto NO reemplaza retrieval por embeddings; solo corrige el breakdown (y affinity si weight>0) post-enrichment.
    STOPWORDS_SKILLS = {
        "y", "de", "la", "el", "los", "las", "en", "con", "sin", "para", "por", "a", "al", "del", "o", "u",
        "un", "una", "unos", "unas",
    }

    def _skills_tokens(text_val: str) -> List[str]:
        if not text_val:
            return []
        t = str(text_val)
        # separa CamelCase
        t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)
        # quita tildes
        t = unicodedata.normalize("NFKD", t)
        t = "".join([c for c in t if not unicodedata.combining(c)])
        t = t.lower()
        # separadores comunes
        t = re.sub(r"[,\|\;/•·\-]+", " ", t)
        # deja letras/números/espacios
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        toks = [x for x in t.split(" ") if x and x not in STOPWORDS_SKILLS]
        return toks

    def _office_suite_present(cand_tokens: List[str]) -> bool:
        # Cubre "office" / "ofimatica" o productos Microsoft pegados
        office_markers = {
            "office", "ofimatica", "ofimatica", "ofimaticas", "microsoftoffice",
            "excel", "microsoftexcel",
            "word", "microsoftword",
            "powerpoint", "microsoftpowerpoint",
            "outlook", "microsoftoutlook",
        }
        for ct in cand_tokens:
            if ct in office_markers:
                return True
            if ct.startswith("microsoft") and any(x in ct for x in ("excel", "word", "powerpoint", "outlook", "office")):
                return True
        return False

    def _skills_lex_sim(query_text: str, cand_text: str) -> Dict[str, Any]:
        q_toks = _skills_tokens(query_text)
        c_toks = _skills_tokens(cand_text)
        if not q_toks:
            return {"sim": 0.0, "matched": [], "missing": [], "query_tokens": [], "cand_tokens": []}

        # Dedup preservando orden
        seen = set()
        q_uniq: List[str] = []
        for t in q_toks:
            if t not in seen:
                q_uniq.append(t)
                seen.add(t)

        c_set = set(c_toks)
        matched: List[str] = []
        missing: List[str] = []

        for qt in q_uniq:
            ok = False
            # "office" como suite
            if qt in ("office", "ofimatica", "ofimatica", "microsoftoffice"):
                ok = _office_suite_present(c_toks)
            else:
                # match exacto
                if qt in c_set:
                    ok = True
                else:
                    # match por substring para tokens suficientemente largos (y algunos especiales)
                    special_sub = qt in ("excel", "word", "powerpoint", "outlook", "atencion", "cliente")
                    if len(qt) >= 5 or special_sub:
                        for ct in c_toks:
                            if qt in ct:
                                ok = True
                                break

            if ok:
                matched.append(qt)
            else:
                missing.append(qt)

        sim = float(len(matched)) / float(len(q_uniq)) if q_uniq else 0.0
        return {"sim": sim, "matched": matched, "missing": missing, "query_tokens": q_uniq, "cand_tokens": c_toks[:40]}

    if q_skills:
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_sk = int(weight_map.get("skills", 0) or 0)
        for r in scored:
            cand_sk_txt = r.get("skills", "") or ""
            bd = r.get("breakdown") or {}
            # coseno original (si existe) para diagnóstico
            cos_sim = float((bd.get("skills") or {}).get("score_0_1", 0.0) or 0.0)
            lex = _skills_lex_sim(q_skills, cand_sk_txt)
            sk_sim = float(lex["sim"])

            bd["skills"] = {
                "score_0_1": round(sk_sim, 4),
                "score_pct": round(sk_sim * 100, 2),
                "weight": w_sk,
                "contribution": round(sk_sim * float(w_sk), 4),
                "rule": "lexical_token_match (supports microsoftexcel/excel and office suite)",
                "matched": lex["matched"],
                "missing": lex["missing"],
                "cosine_sim_original": round(cos_sim, 4),
            }
            r["breakdown"] = bd

            if w_sk > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "skills":
                        dim_score = sk_sim
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[: request.top_k]

    # 7.15) Similitud de sector (fallback léxico) para evitar 0% en casos obvios
    # Ej: query "moto, mantenimiento" vs candidato "TecnologíaenMantenimientoMecánicoIndustrial"
    # (Este texto suele venir de study_area en MySQL.)
    STOPWORDS_SECTOR = {
        "y", "de", "la", "el", "los", "las", "en", "con", "sin", "para", "por", "a", "al", "del", "o", "u",
        "un", "una", "unos", "unas",
        "tecnologia", "tecnologia", "tecnica", "tecnico", "profesional", "industrial",
    }

    def _sector_tokens(text_val: str) -> List[str]:
        if not text_val:
            return []
        t = str(text_val)
        t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)  # CamelCase
        t = unicodedata.normalize("NFKD", t)
        t = "".join([c for c in t if not unicodedata.combining(c)])
        t = t.lower()
        t = re.sub(r"[,\|\;/•·\-]+", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        toks = [x for x in t.split(" ") if x and x not in STOPWORDS_SECTOR]
        return toks

    def _sector_lex_sim(query_text: str, cand_text: str) -> Dict[str, Any]:
        q_toks = _sector_tokens(query_text)
        c_toks = _sector_tokens(cand_text)
        if not q_toks:
            return {"sim": 0.0, "matched": [], "missing": [], "query_tokens": [], "cand_tokens": []}

        # Dedup preservando orden
        seen = set()
        q_uniq: List[str] = []
        for t in q_toks:
            if t not in seen:
                q_uniq.append(t)
                seen.add(t)

        matched: List[str] = []
        missing: List[str] = []
        for qt in q_uniq:
            ok = False
            # match exacto o substring (términos >=4)
            if qt in c_toks:
                ok = True
            elif len(qt) >= 4:
                for ct in c_toks:
                    if qt in ct or ct in qt:
                        ok = True
                        break
            if ok:
                matched.append(qt)
            else:
                missing.append(qt)

        sim = float(len(matched)) / float(len(q_uniq)) if q_uniq else 0.0
        return {"sim": sim, "matched": matched, "missing": missing, "query_tokens": q_uniq, "cand_tokens": c_toks[:40]}

    if q_sector:
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_sec = int(weight_map.get("sector", 0) or 0)
        for r in scored:
            cand_sec_txt = r.get("sector", "") or ""
            bd = r.get("breakdown") or {}
            cos_sim = float((bd.get("sector") or {}).get("score_0_1", 0.0) or 0.0)
            lex = _sector_lex_sim(q_sector, cand_sec_txt)
            sec_sim = float(lex["sim"])

            bd["sector"] = {
                "score_0_1": round(sec_sim, 4),
                "score_pct": round(sec_sim * 100, 2),
                "weight": w_sec,
                "contribution": round(sec_sim * float(w_sec), 4),
                "rule": "lexical_token_match (CamelCase + accents normalization)",
                "matched": lex["matched"],
                "missing": lex["missing"],
                "cosine_sim_original": round(cos_sim, 4),
            }
            r["breakdown"] = bd

            if w_sec > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "sector":
                        dim_score = sec_sim
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[: request.top_k]

    # 7.2) Similitud de experiencia (rule-based por rango) para que la UI muestre % correcto en "Experiencia"
    # Si q_experience es un rango ("0-1","1-3","3-5","5+"), calculamos x en años y aplicamos una versión
    # con "límite de caída" (penalize_after) por sobrecalificación:
    #
    # Para 0-1 / 1-3 / 3-5:
    #   - x < m                 => x/m
    #   - m <= x <= upper        => 1
    #   - upper < x <= P         => 1
    #   - x > P                  => exp(-k*(x-P))
    #
    # Para 5+:
    #   - x < 5                  => x/5
    #   - 5 <= x <= 12            => 1
    #   - x > 12                  => exp(-k*(x-12))
    #
    # Donde P es el "máximo" antes de empezar a caer:
    #   0-1 -> P=2 años
    #   1-3 -> P=5 años
    #   3-5 -> P=8 años
    #   5+  -> P=12 años
    #
    # El ordenamiento se hace SOLO por este score (desc).
    # Para desempate, se usa la similitud coseno (score original de embeddings) (desc).
    RANGE_MAP_MONTHS = {
        "0-1": (0, 12),
        "1-3": (13, 36),
        "3-5": (37, 60),
        "5+": (61, 10_000),
    }

    def _parse_duration_piece_to_months(piece: str) -> int:
        if not piece:
            return 0
        t = str(piece).strip().lower()
        t = (
            t.replace("años", "anos")
            .replace("año", "ano")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        # "X anos y Y meses"
        m = re.search(r"\b(\d+)\s*anos?\s*y\s*(\d+)\s*mes(?:es)?\b", t)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        # "X anos"
        m = re.search(r"\b(\d+)\s*anos?\b", t)
        if m:
            return int(m.group(1)) * 12
        # "X meses"
        m = re.search(r"\b(\d+)\s*mes(?:es)?\b", t)
        if m:
            return int(m.group(1))
        return 0

    def _total_months_from_experience_text(exp_text: str) -> int:
        if not exp_text:
            return 0
        parts = [p.strip() for p in re.split(r"[;,\n|/]+", str(exp_text)) if p.strip()]
        return sum(_parse_duration_piece_to_months(p) for p in parts)

    EXP_RANGE_CFG_YEARS = {
        "0-1": {"m": 0.0, "upper": 1.0, "penalize_after": 2.0},
        "1-3": {"m": 1.0, "upper": 3.0, "penalize_after": 5.0},
        "3-5": {"m": 3.0, "upper": 5.0, "penalize_after": 8.0},
        "5+": {"m": 5.0, "upper": 5.0, "penalize_after": 12.0},
    }

    def _f_years_bucket(x_years: float, m: float, upper: float, penalize_after: float, k: float) -> float:
        """
        Para 0-1 / 1-3 / 3-5:
          - x < m                  => x/m
          - m <= x <= upper        => 1
          - upper < x <= P         => 1
          - x > P                  => exp(-k*(x-P))
        """
        if x_years < 0:
            return 0.0
        if x_years < m:
            # Si m==0 (solo podría pasar en 0-1), entonces cualquier x>=0 no está "por debajo".
            if m <= 0:
                return 1.0
            return max(0.0, min(1.0, x_years / m))
        if x_years <= penalize_after:
            return 1.0
        return float(math.exp(-float(k) * (x_years - penalize_after)))

    def _f_years_5plus(x_years: float, penalize_after: float, k: float) -> float:
        """
        Para 5+:
          - x < 5        => x/5
          - 5 <= x <= P  => 1
          - x > P        => exp(-k*(x-P))
        """
        if x_years < 0:
            return 0.0
        if x_years < 5.0:
            return max(0.0, min(1.0, x_years / 5.0))
        if x_years <= penalize_after:
            return 1.0
        return float(math.exp(-float(k) * (x_years - penalize_after)))

    if q_experience and q_experience.strip() in RANGE_MAP_MONTHS:
        rng = q_experience.strip()
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_exp = int(weight_map.get("experience", 0) or 0)
        # k sugerido por tu documento: [0.2, 0.5]
        # Usamos default 0.3 (configurable si luego lo expones en request/env).
        k_exp = 0.3
        cfg = EXP_RANGE_CFG_YEARS.get(rng, {"m": 0.0, "upper": 1.0, "penalize_after": 2.0})
        m_req = float(cfg["m"])
        upper_req = float(cfg["upper"])
        p_req = float(cfg["penalize_after"])

        for r in scored:
            cand_exp_txt = r.get("experience", "") or ""
            total_m = _total_months_from_experience_text(cand_exp_txt)
            x_years = float(total_m) / 12.0 if total_m and total_m > 0 else 0.0
            if rng == "5+":
                exp_score = _f_years_5plus(x_years=x_years, penalize_after=p_req, k=k_exp)
            else:
                exp_score = _f_years_bucket(x_years=x_years, m=m_req, upper=upper_req, penalize_after=p_req, k=k_exp)

            bd = r.get("breakdown") or {}
            # conservar similitud coseno original (embeddings) para desempate/diagnóstico
            cos_sim = float((bd.get("experience") or {}).get("score_0_1", 0.0) or 0.0)
            bd["experience"] = {
                "score_0_1": round(float(exp_score), 4),
                "score_pct": round(float(exp_score) * 100, 2),
                "weight": w_exp,
                "contribution": round(float(exp_score) * float(w_exp), 4),
                "rule": "bucket w/ penalize_after: x<m => x/m; x in [m,upper] => 1; x in (upper,P] => 1; x>P => exp(-k(x-P))",
                "query_experience": rng,
                "candidate_experience": cand_exp_txt,
                "total_months": int(total_m),
                "x_years": round(float(x_years), 4),
                "m": m_req,
                "upper": upper_req,
                "penalize_after": p_req,
                "k": k_exp,
                "cosine_sim_tiebreak": round(float(cos_sim), 4),
            }
            r["breakdown"] = bd

            # Recalcular affinity solo si el peso de experience > 0
            if w_exp > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "experience":
                        dim_score = float(exp_score)
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        # Ordenamiento: SOLO por score de experiencia (desc). Desempate por coseno (desc).
        def _exp_sort_key(item: dict):
            bd = item.get("breakdown") or {}
            exp_bd = bd.get("experience") or {}
            score = float(exp_bd.get("score_0_1", 0.0) or 0.0)
            tie = float(exp_bd.get("cosine_sim_tiebreak", 0.0) or 0.0)
            return (score, tie)

        scored.sort(key=_exp_sort_key, reverse=True)
        scored = scored[: request.top_k]

    # 7.25) Similitud de educación (rule-based) para evitar 0% cuando el candidato no aparece en top hits de embeddings
    # Regla simple:
    # - Si el candidato cumple o supera el nivel requerido -> 1.00
    # - Si está a 1 nivel por debajo -> 0.50
    # - 2+ niveles por debajo / no detectable -> 0.00
    #
    # Nota: esto alimenta `breakdown["education"]` para que la UI muestre el valor correcto.
    EDU_RANK_UI = {
        "none": 0,
        "bachiller": 1,
        "tecnico": 2,
        "técnico": 2,
        "tecnologo": 3,
        "tecnólogo": 3,
        "profesional": 4,
        "posgrado": 5,
    }

    def _edu_rank_from_text(text_val: str) -> int:
        if not text_val:
            return 0
        t = str(text_val)
        # separar CamelCase
        t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)
        t = t.lower()
        # sin tildes (simple)
        t = (
            t.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )
        t = re.sub(r"\s+", " ", t).strip()

        # patrones (alineados con index_education.py)
        if re.search(r"\b(doctorado|phd|d\.?phil|maestria|master|magister|m\.?sc|mba|especializacion|posgrado)\b", t):
            return 5
        if re.search(r"\b(profesional|pregrado|grado|universitario|ingenier[ia]|ingeniero|licenciad[oa]|administrador|abogad[oa]|contador|economista|psicolog[oa])\b", t):
            return 4
        if re.search(r"\b(tecnolog[oa]|tecnologia)\b", t):
            return 3
        if re.search(r"\b(tecnic[oa]|tecnico\s+profesional|tecnico\s+laboral)\b", t):
            return 2
        if re.search(r"\b(bachiller|secundaria|media|bachillerato)\b", t):
            return 1
        return 0

    if q_education:
        q_edu_rank = _edu_rank_from_text(q_education)
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_edu = int(weight_map.get("education", 0) or 0)

        for r in scored:
            cand_edu_txt = r.get("education", "") or ""
            c_rank = _edu_rank_from_text(cand_edu_txt)
            edu_sim = 0.0
            if q_edu_rank > 0 and c_rank > 0:
                if c_rank >= q_edu_rank:
                    edu_sim = 1.0
                elif c_rank == q_edu_rank - 1:
                    edu_sim = 0.5
                else:
                    edu_sim = 0.0

            bd = r.get("breakdown") or {}
            bd["education"] = {
                "score_0_1": round(float(edu_sim), 4),
                "score_pct": round(float(edu_sim) * 100, 2),
                "weight": w_edu,
                "contribution": round(float(edu_sim) * float(w_edu), 4),
                "rule": ">=req:1.0; 1-below:0.5; else:0.0",
                "query_education": q_education,
                "candidate_education": cand_edu_txt,
                "req_rank": q_edu_rank,
                "cand_rank": c_rank,
            }
            r["breakdown"] = bd

            # Recalcular affinity solo si el peso de education > 0
            if w_edu > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "education":
                        dim_score = float(edu_sim)
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[: request.top_k]

    # 7.5) Similitud de ubicación (para breakdown de "city")
    # Regla:
    # - Mismo municipio => 1.00
    # - Mismo departamento (municipio distinto o no especificado) => 0.85
    # - Distinto departamento/indeterminado => 0.00
    #
    # Nota: esto alimenta `breakdown["city"]` para que la UI muestre 85% cuando aplica.

    def _split_location(loc: str):
        if not loc:
            return "", ""
        t = str(loc).strip()
        # separadores comunes en location: "Bogotá, Cundinamarca", "Medellín - Antioquia", "Antioquia/Medellín"
        parts = [p.strip() for p in re.split(r"[,;\-\|/]+", t) if p.strip()]
        if not parts:
            return "", ""

        # Heurística Colombia: detectar departamento para asignar (depto, municipio) de forma robusta.
        # Soporta strings sin espacios como "ValledelCauca/Cali".
        BASE_CO_DEPARTMENTS = [
            "Amazonas",
            "Antioquia",
            "Arauca",
            "Atlántico",
            "Bolívar",
            "Boyacá",
            "Caldas",
            "Caquetá",
            "Casanare",
            "Cauca",
            "Cesar",
            "Chocó",
            "Córdoba",
            "Cundinamarca",
            "Guainía",
            "Guaviare",
            "Huila",
            "La Guajira",
            "Magdalena",
            "Meta",
            "Nariño",
            "Norte de Santander",
            "Putumayo",
            "Quindío",
            "Risaralda",
            "San Andrés y Providencia",
            "Santander",
            "Sucre",
            "Tolima",
            "Valle del Cauca",
            "Vaupés",
            "Vichada",
            # Distrito capital (a veces se usa como "departamento")
            "Bogotá",
            "Bogotá D.C.",
            "Bogota DC",
            "Distrito Capital",
        ]

        def _strip_accents(s: str) -> str:
            s2 = unicodedata.normalize("NFKD", s)
            return "".join([c for c in s2 if not unicodedata.combining(c)])

        def _canon_key(s: str) -> str:
            # separa CamelCase -> "ValledelCauca" => "Valledel Cauca"
            s = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", str(s))
            s = _strip_accents(s).lower()
            s = re.sub(r"[^a-z\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _nospace_key(s: str) -> str:
            return _canon_key(s).replace(" ", "")

        # keys de departamento: con espacios y sin espacios (para detectar "valledelcauca")
        CO_DEPT_KEYS = set()
        for dname in BASE_CO_DEPARTMENTS:
            CO_DEPT_KEYS.add(_canon_key(dname))
            CO_DEPT_KEYS.add(_nospace_key(dname))

        # Normaliza valores basura de UI/DB
        if len(parts) == 1:
            one = parts[0].strip()
            if one.lower() in ("no aplica", "na", "n/a", "0"):
                return "", ""
            # si es un departamento, lo colocamos como depto (municipio vacío)
            one_c = _canon_key(one)
            if one_c in CO_DEPT_KEYS or one_c.replace(" ", "") in CO_DEPT_KEYS:
                return "", one_c
            return one_c, ""

        # Si hay más de 2 partes, tomamos primera y última para depto/muni candidates.
        a = parts[0].strip().lower()
        b = parts[-1].strip().lower()

        a_n = _canon_key(a)
        b_n = _canon_key(b)
        a_ns = a_n.replace(" ", "")
        b_ns = b_n.replace(" ", "")

        # Caso "Departamento/Municipio" (UI) -> a es depto
        if (a_n in CO_DEPT_KEYS or a_ns in CO_DEPT_KEYS) and not (b_n in CO_DEPT_KEYS or b_ns in CO_DEPT_KEYS):
            depto = a_n
            municipio = b_n
            return municipio, depto

        # Caso "Municipio/Departamento" (otros) -> b es depto
        if (b_n in CO_DEPT_KEYS or b_ns in CO_DEPT_KEYS) and not (a_n in CO_DEPT_KEYS or a_ns in CO_DEPT_KEYS):
            depto = b_n
            municipio = a_n
            return municipio, depto

        # Ambiguo: por defecto asumimos "municipio, depto" (último como depto)
        municipio = a_n
        depto = b_n
        return municipio, depto

    if q_city:
        q_mun, q_depto = _split_location(q_city)
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)

        for r in scored:
            cand_city = r.get("city", "") or ""
            c_mun, c_depto = _split_location(cand_city)

            # city_sim
            city_sim = 0.0
            if q_mun and c_mun and q_mun == c_mun:
                city_sim = 1.0
            elif q_depto and c_depto and q_depto == c_depto:
                city_sim = 0.85

            # Inyectar/actualizar breakdown["city"] aunque el peso sea 0 (para que la UI lo muestre bien)
            bd = r.get("breakdown") or {}
            w_city = int(weight_map.get("city", 0) or 0)
            bd["city"] = {
                "score_0_1": round(float(city_sim), 4),
                "score_pct": round(float(city_sim) * 100, 2),
                "weight": w_city,
                "contribution": round(float(city_sim) * float(w_city), 4),
                "rule": "same_muni=1.0; same_depto=0.85; else=0.0",
                "query_city": q_city,
                "candidate_city": cand_city,
            }
            r["breakdown"] = bd

            # Recalcular affinity SOLO si el peso de city > 0 (si es 0, es informativo)
            if w_city > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    dim_score = 0.0
                    if dim == "city":
                        dim_score = float(city_sim)
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        # Re-ordenar por afinidad luego del ajuste y recortar
        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[: request.top_k]

    timings_ms["rules_reranking_ms"] = round((time.perf_counter() - t_rules) * 1000.0, 2)

    # 8) Guardar resultados en job_requests + job_request_results
    t_persist = time.perf_counter()
    job_request_id = persistence.save_job_request_and_results(
        query_payload,
        request.weights.model_dump(),
        request.top_k,
        scored
    )
    timings_ms["persist_mysql_ms"] = round((time.perf_counter() - t_persist) * 1000.0, 2)

    timings_ms["end_to_end_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
    LATENCY_TRACKER.record(timings_ms)

    return {
        "job_request_id": job_request_id,
        "query": query_payload,
        "results": scored,
        "collections_enabled": {k: (v is not None) for k, v in chroma_collections.items()},
        "timings_ms": timings_ms,
    }


@app.post("/match")
def match_candidates(request: MatchRequest, background_tasks: BackgroundTasks):
    """Endpoint de match orquestado por LLM con tools por dimensión (mismas que MCP)."""
    if engine is None or oa_client is None:
        raise HTTPException(500, "Servicio no inicializado correctamente")

    t_start = time.perf_counter()
    timings_ms: Dict[str, float] = {}

    payload = request.model_dump()

    embeddings_mgr = EmbeddingsManager(oa_client, EMBED_MODEL)
    matcher = DimensionMatcher(chroma_collections, embeddings_mgr)
    aggregator = ResultAggregator()
    filters_obj = HardFilters(chroma_collections)
    persistence = PersistenceManager(engine, CANDIDATES_TABLE)

    try:
        t_orch = time.perf_counter()
        response = invoke_match_orchestrator(
            payload, oa_client, LLM_MODEL,
            matcher, aggregator, filters_obj, persistence, chroma_collections
        )
        timings_ms["orchestrator_ms"] = round((time.perf_counter() - t_orch) * 1000, 2)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error en orquestación LLM: {str(e)}")

    scored = response.get("results", [])

    if scored:
        t_rules = time.perf_counter()

        q_skills = (request.skills or "").strip()
        q_experience = (request.experience or "").strip()
        q_education = (request.education or "").strip()
        q_language = (request.language or "").strip()
        q_sector = (request.sector or "").strip()
        q_city = (request.city or "").strip()

        weight_map = {
            "skills": request.weights.skills,
            "experience": request.weights.experience,
            "education": request.weights.education,
            "language": request.weights.language,
            "sector": request.weights.sector,
            "job_title": request.weights.job_title,
            "city": request.weights.city,
        }

        scored = apply_reranking_rules(
            scored, q_skills, q_experience, q_education, q_language,
            q_sector, q_city, weight_map, request.top_k, chroma_collections
        )
        response["results"] = scored
        timings_ms["rules_reranking_ms"] = round((time.perf_counter() - t_rules) * 1000, 2)

    query_payload = response.get("query", {})
    weight_map_save = response.get("weights_used", {})
    job_request_id = persistence.save_job_request_and_results(
        query_payload, weight_map_save, request.top_k, scored
    )
    response["job_request_id"] = job_request_id
    response["collections_enabled"] = {k: (v is not None) for k, v in chroma_collections.items()}
    timings_ms["end_to_end_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    response["timings_ms"] = timings_ms
    LATENCY_TRACKER.record(timings_ms)

    background_tasks.add_task(
        persistence.save_match_execution,
        payload,
        response,
    )
    return response
