#!/usr/bin/env python3
"""
Test de integracion para la orquestacion LLM del sistema TFM Match.

Valida que GPT-4o-mini, dado un payload de match, invoque las tools
de dimension correctas y termine llamando combine_and_rank_candidates.

Requisitos: MySQL, ChromaDB, OPENAI_API_KEY (genera costo ~$0.01-0.05).

Ejecucion:
    poetry run python tests/test_llm_orchestration.py
"""

import json
import sys
import time
from unittest.mock import patch
from typing import Dict, Any, List

import chromadb
from openai import OpenAI
from sqlalchemy import create_engine

from tfm_match.config import (
    OPENAI_API_KEY, MYSQL_URL, EMBEDDING_MODEL, LLM_MODEL, CHROMA_DIR,
    CHROMA_COLLECTION_SKILLS, CHROMA_COLLECTION_EXPERIENCE,
    CHROMA_COLLECTION_EDUCATION, CHROMA_COLLECTION_LANGUAGE,
    CHROMA_COLLECTION_SECTOR, CHROMA_COLLECTION_JOB_TITLE,
    CANDIDATES_TABLE,
)
from tfm_match.core.embeddings_manager import EmbeddingsManager
from tfm_match.core.dimension_matcher import DimensionMatcher
from tfm_match.core.result_aggregator import ResultAggregator
from tfm_match.core.filters import HardFilters
from tfm_match.core.persistence import PersistenceManager
from tfm_match.llm.client import invoke_match_orchestrator, _handle_dimension_call


passed = 0
failed = 0
tool_call_log: List[str] = []


def report(test_name: str, ok: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    msg = f"  [{status}] {test_name}"
    if detail:
        msg += f" -- {detail}"
    print(msg, flush=True)


def init_components():
    """Inicializa componentes core/ igual que api/main.py."""
    oa_client = OpenAI(api_key=OPENAI_API_KEY)
    engine = create_engine(MYSQL_URL, pool_pre_ping=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection_names = {
        "skills": CHROMA_COLLECTION_SKILLS,
        "experience": CHROMA_COLLECTION_EXPERIENCE,
        "education": CHROMA_COLLECTION_EDUCATION,
        "language": CHROMA_COLLECTION_LANGUAGE,
        "sector": CHROMA_COLLECTION_SECTOR,
        "job_title": CHROMA_COLLECTION_JOB_TITLE,
    }

    chroma_collections = {}
    for dim, name in collection_names.items():
        try:
            chroma_collections[dim] = chroma_client.get_collection(name=name)
        except Exception:
            chroma_collections[dim] = None

    embeddings_mgr = EmbeddingsManager(oa_client, EMBEDDING_MODEL)
    matcher = DimensionMatcher(chroma_collections, embeddings_mgr)
    aggregator = ResultAggregator()
    filters_obj = HardFilters(chroma_collections)
    persistence = PersistenceManager(engine, CANDIDATES_TABLE)

    return oa_client, matcher, aggregator, filters_obj, persistence, chroma_collections


def tracked_handle_dimension_call(dim, args, matcher, persistence, chroma_collections,
                                  dimension_results, per_dim_k):
    """Wrapper que registra cada tool call de dimension antes de ejecutarla."""
    tool_call_log.append(f"query_{dim}_dimension")
    return _handle_dimension_call(
        dim, args, matcher, persistence, chroma_collections,
        dimension_results, per_dim_k,
    )


def run_orchestrator_with_tracking(payload, oa_client, matcher, aggregator,
                                   filters_obj, persistence, chroma_collections):
    """Ejecuta el orquestador interceptando las llamadas a tools."""
    global tool_call_log
    tool_call_log = []

    with patch("tfm_match.llm.client._handle_dimension_call", tracked_handle_dimension_call):
        result = invoke_match_orchestrator(
            payload, oa_client, LLM_MODEL,
            matcher, aggregator, filters_obj, persistence, chroma_collections,
        )

    tool_call_log.append("combine_and_rank_candidates")
    return result


# ── Test cases ──────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "TC_LLM_01",
        "description": "Payload con 2 dimensiones (skills + sector)",
        "payload": {
            "skills": "atencion al cliente, CRM",
            "experience": "",
            "education": "",
            "language": "",
            "sector": "Call center, BPO",
            "job_title": "",
            "city": "",
            "top_k": 5,
            "weights": {
                "skills": 5, "experience": 0, "education": 0,
                "language": 0, "sector": 4, "job_title": 0, "city": 0,
            },
        },
        "expected_tools": {"query_skills_dimension", "query_sector_dimension"},
        "unexpected_tools": {"query_education_dimension", "query_language_dimension"},
    },
    {
        "id": "TC_LLM_02",
        "description": "Payload completo con 5 dimensiones activas",
        "payload": {
            "skills": "liderazgo, coaching, KPIs",
            "experience": "3-5",
            "education": "Profesional",
            "language": "Ingles B1",
            "sector": "Call center",
            "job_title": "Supervisor",
            "city": "",
            "top_k": 5,
            "weights": {
                "skills": 6, "experience": 5, "education": 4,
                "language": 3, "sector": 4, "job_title": 5, "city": 0,
            },
            "hard_filters": {
                "education_min": "Profesional",
                "language_required": "Ingles",
                "language_min_level": "B1",
            },
        },
        "expected_tools": {
            "query_skills_dimension", "query_experience_dimension",
            "query_education_dimension", "query_language_dimension",
            "query_sector_dimension", "query_job_title_dimension",
        },
        "unexpected_tools": set(),
    },
    {
        "id": "TC_LLM_03",
        "description": "Payload minimo: solo skills",
        "payload": {
            "skills": "ventas, merchandising",
            "experience": "",
            "education": "",
            "language": "",
            "sector": "",
            "job_title": "",
            "city": "",
            "top_k": 5,
            "weights": {
                "skills": 7, "experience": 0, "education": 0,
                "language": 0, "sector": 0, "job_title": 0, "city": 0,
            },
        },
        "expected_tools": {"query_skills_dimension"},
        "unexpected_tools": {
            "query_education_dimension", "query_language_dimension",
            "query_sector_dimension",
        },
    },
]


def run_test_case(case, oa_client, matcher, aggregator, filters_obj,
                  persistence, chroma_collections):
    """Ejecuta un caso de prueba y valida las tools invocadas."""
    case_id = case["id"]
    desc = case["description"]
    print(f"\n--- {case_id}: {desc} ---", flush=True)

    try:
        t0 = time.perf_counter()
        result = run_orchestrator_with_tracking(
            case["payload"], oa_client, matcher, aggregator,
            filters_obj, persistence, chroma_collections,
        )
        elapsed = time.perf_counter() - t0
    except Exception as e:
        report(f"{case_id}: ejecucion", False, str(e))
        return

    report(f"{case_id}: ejecucion completada", True, f"{elapsed:.2f}s")

    called_set = set(tool_call_log)

    for expected in case["expected_tools"]:
        was_called = expected in called_set
        report(f"{case_id}: llamo {expected}", was_called)

    for unexpected in case.get("unexpected_tools", set()):
        was_not_called = unexpected not in called_set
        report(f"{case_id}: no llamo {unexpected}", was_not_called)

    report(
        f"{case_id}: llamo combine_and_rank_candidates",
        "combine_and_rank_candidates" in called_set,
    )

    results = result.get("results", [])
    report(
        f"{case_id}: retorno candidatos",
        len(results) > 0,
        f"{len(results)} candidatos",
    )

    if results:
        first = results[0]
        has_affinity = "affinity" in first
        has_breakdown = "breakdown" in first
        report(f"{case_id}: formato correcto (affinity + breakdown)", has_affinity and has_breakdown)

    print(f"  Tools invocadas: {tool_call_log}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    global passed, failed

    print("=" * 60)
    print("TEST LLM ORCHESTRATION - TFM Match")
    print(f"Modelo: {LLM_MODEL}")
    print("=" * 60)

    print("\nInicializando componentes...", flush=True)
    try:
        oa_client, matcher, aggregator, filters_obj, persistence, chroma_collections = init_components()
        print("Componentes inicializados.\n", flush=True)
    except Exception as e:
        print(f"\nError al inicializar: {e}")
        print("Verifica que MySQL, ChromaDB y OPENAI_API_KEY esten configurados.")
        sys.exit(1)

    for case in TEST_CASES:
        run_test_case(
            case, oa_client, matcher, aggregator,
            filters_obj, persistence, chroma_collections,
        )

    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
