#!/usr/bin/env python3
"""
Test de integracion para las tools MCP del servidor TFM Match.

Prueba las 9 tools y 3 resources directamente (sin levantar el servidor MCP),
inicializando los mismos componentes core/ que usa server.py.

Requisitos: MySQL corriendo, ChromaDB indexado, OPENAI_API_KEY configurada.

Ejecucion:
    poetry run python tests/test_mcp_tools.py
"""

import asyncio
import json
import sys
import time

import chromadb
from openai import OpenAI
from sqlalchemy import create_engine

from tfm_match.config import (
    OPENAI_API_KEY, MYSQL_URL, EMBEDDING_MODEL, CHROMA_DIR,
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
from tfm_match.mcp.tools.dimension_tools import (
    handle_dimension_tool_call, get_dimension_tools_list,
)
from tfm_match.mcp.tools.aggregation_tools import (
    handle_aggregation_tool_call, get_aggregation_tools_list,
)


passed = 0
failed = 0


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
    """Inicializa los componentes core/ igual que mcp/server.py."""
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

    return matcher, aggregator, filters_obj, persistence


# ── Dimension tool tests ────────────────────────────────────────────────

DIMENSION_QUERIES = {
    "skills": "atencion al cliente, CRM",
    "experience": "2 anos en call center",
    "education": "Profesional",
    "language": "Espanol nativo",
    "sector": "Call center, BPO",
    "job_title": "Asesor de servicio al cliente",
}


async def test_dimension_tools(matcher):
    """Prueba las 6 tools de dimension."""
    print("\n--- Tools de dimension ---", flush=True)

    for dim, query in DIMENSION_QUERIES.items():
        tool_name = f"query_{dim}_dimension"
        try:
            t0 = time.perf_counter()
            result = await handle_dimension_tool_call(
                tool_name,
                {"query_text": query, "weight": 5, "top_k": 10},
                matcher,
            )
            elapsed = time.perf_counter() - t0

            text = result[0].text
            has_candidates = "id_candidate" in text or "Candidato ID" in text
            report(
                tool_name,
                has_candidates,
                f"{elapsed:.2f}s" + ("" if has_candidates else " (sin candidatos)"),
            )
        except Exception as e:
            report(tool_name, False, str(e))


async def test_dimension_tool_empty_query(matcher):
    """Prueba que una query vacia retorna error controlado."""
    print("\n--- Dimension tool: query vacia ---", flush=True)
    result = await handle_dimension_tool_call(
        "query_skills_dimension",
        {"query_text": "", "weight": 5},
        matcher,
    )
    text = result[0].text
    is_error = "Error" in text or "vac" in text.lower()
    report("query_skills_dimension (vacia)", is_error, "Maneja query vacia correctamente")


# ── Aggregation tool tests ──────────────────────────────────────────────

async def test_combine_and_rank(matcher, aggregator, filters_obj, persistence):
    """Prueba combine_and_rank_candidates con resultados reales de 2 dimensiones."""
    print("\n--- Tool: combine_and_rank_candidates ---", flush=True)

    skills_result = await handle_dimension_tool_call(
        "query_skills_dimension",
        {"query_text": "atencion al cliente, CRM", "weight": 5, "top_k": 20},
        matcher,
    )
    sector_result = await handle_dimension_tool_call(
        "query_sector_dimension",
        {"query_text": "Call center, BPO", "weight": 4, "top_k": 20},
        matcher,
    )

    skills_text = skills_result[0].text
    sector_text = sector_result[0].text

    skills_json = json.loads(skills_text.split("```json\n")[1].split("\n```")[0])
    sector_json = json.loads(sector_text.split("```json\n")[1].split("\n```")[0])

    dimension_results = [
        {
            "dimension": "skills",
            "candidates": skills_json["candidates"],
            "weight": 5,
        },
        {
            "dimension": "sector",
            "candidates": sector_json["candidates"],
            "weight": 4,
        },
    ]

    try:
        t0 = time.perf_counter()
        result = await handle_aggregation_tool_call(
            "combine_and_rank_candidates",
            {"dimension_results": dimension_results, "top_k": 5},
            aggregator, filters_obj, persistence,
        )
        elapsed = time.perf_counter() - t0

        text = result[0].text
        has_ranking = "Afinidad Total" in text or "affinity" in text
        report(
            "combine_and_rank_candidates",
            has_ranking,
            f"{elapsed:.2f}s, ranking generado" if has_ranking else "sin ranking",
        )
    except Exception as e:
        report("combine_and_rank_candidates", False, str(e))


async def test_get_candidate_details(persistence):
    """Prueba get_candidate_details con un ID valido."""
    print("\n--- Tool: get_candidate_details ---", flush=True)
    try:
        result = await handle_aggregation_tool_call(
            "get_candidate_details",
            {"candidate_id": "1"},
            ResultAggregator(), HardFilters({}), persistence,
        )
        text = result[0].text
        has_profile = "Candidato" in text or "id_candidate" in text.lower() or "Perfil" in text
        report("get_candidate_details", has_profile)
    except Exception as e:
        report("get_candidate_details", False, str(e))


async def test_explain_match_breakdown():
    """Prueba explain_match_breakdown con un breakdown simulado."""
    print("\n--- Tool: explain_match_breakdown ---", flush=True)
    breakdown = {
        "skills": {"score_pct": 85.0, "weight": 5, "contribution": 0.35},
        "sector": {"score_pct": 72.0, "weight": 4, "contribution": 0.24},
    }
    try:
        result = await handle_aggregation_tool_call(
            "explain_match_breakdown",
            {"candidate_id": "1", "breakdown": breakdown},
            ResultAggregator(), HardFilters({}), PersistenceManager(None, ""),
        )
        text = result[0].text
        has_explanation = "skills" in text.lower() and "contribuci" in text.lower()
        report("explain_match_breakdown", has_explanation)
    except Exception as e:
        report("explain_match_breakdown", False, str(e))


# ── Tool listing tests ──────────────────────────────────────────────────

def test_tool_listings():
    """Verifica que se listen las 9 tools con schemas validos."""
    print("\n--- Listado de tools ---", flush=True)

    dim_tools = get_dimension_tools_list()
    report(
        "get_dimension_tools_list",
        len(dim_tools) == 6,
        f"{len(dim_tools)} tools (esperadas: 6)",
    )

    for tool in dim_tools:
        has_schema = "query_text" in json.dumps(tool.inputSchema)
        report(f"  schema: {tool.name}", has_schema)

    agg_tools = get_aggregation_tools_list()
    report(
        "get_aggregation_tools_list",
        len(agg_tools) == 3,
        f"{len(agg_tools)} tools (esperadas: 3)",
    )

    total = len(dim_tools) + len(agg_tools)
    report("total tools MCP", total == 9, f"{total} tools")


# ── Resource tests ──────────────────────────────────────────────────────

async def test_resources():
    """Prueba los 3 resources MCP importando read_resource de server.py."""
    print("\n--- Resources MCP ---", flush=True)

    from tfm_match.mcp.server import read_resource

    resources = {
        "tfm://collections/stats": "collections",
        "tfm://schema/dimensions": "dimensions",
        "tfm://config/weights": "weights",
    }

    for uri, label in resources.items():
        try:
            data = await read_resource(uri)
            parsed = json.loads(data)
            report(f"resource: {label}", isinstance(parsed, dict), f"{len(parsed)} keys")
        except Exception as e:
            report(f"resource: {label}", False, str(e))


# ── Main ────────────────────────────────────────────────────────────────

async def run_all():
    global passed, failed

    print("=" * 60)
    print("TEST MCP TOOLS - TFM Match")
    print("=" * 60)
    print("\nInicializando componentes core/...", flush=True)

    try:
        matcher, aggregator, filters_obj, persistence = init_components()
        print("Componentes inicializados.\n", flush=True)
    except Exception as e:
        print(f"\nError al inicializar: {e}")
        print("Verifica que MySQL y ChromaDB esten disponibles y .env configurado.")
        sys.exit(1)

    test_tool_listings()
    await test_dimension_tools(matcher)
    await test_dimension_tool_empty_query(matcher)
    await test_combine_and_rank(matcher, aggregator, filters_obj, persistence)
    await test_get_candidate_details(persistence)
    await test_explain_match_breakdown()
    await test_resources()

    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all())
    sys.exit(0 if success else 1)
