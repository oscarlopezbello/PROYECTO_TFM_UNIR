"""
Cliente LLM - Orquesta el match invocando tools por dimensión (mismas que MCP).

GPT-4o-mini decide qué dimensiones consultar según el payload,
llama cada tool individualmente, y luego combina los resultados.
"""

import json
import time
from typing import Dict, Any, List, Optional

from openai import OpenAI

from tfm_match.llm.spec import MATCH_ORCHESTRATOR_SPEC
from tfm_match.core.dimension_matcher import DimensionMatcher
from tfm_match.core.result_aggregator import ResultAggregator
from tfm_match.core.filters import HardFilters
from tfm_match.core.persistence import PersistenceManager

MAX_ITERATIONS = 15

DIMENSIONS_CONFIG = {
    "skills": "Busca candidatos por habilidades técnicas y blandas (ej: atención al cliente, CRM, ventas)",
    "experience": "Busca candidatos por experiencia laboral (ej: 0-1, 1-3, 3-5, 5+)",
    "education": "Busca candidatos por nivel educativo (ej: Profesional, Tecnólogo, Técnico)",
    "language": "Busca candidatos por idiomas (ej: Inglés B2, Francés básico)",
    "sector": "Busca candidatos por sector o área laboral (ej: Call center, Retail, BPO)",
    "job_title": "Busca candidatos por cargo o título (ej: Asesor, Cajero, Supervisor)",
    "city": "Busca candidatos por ubicación/ciudad (ej: Bogotá, Medellín, Cali)",
}


def _build_dimension_tools() -> List[dict]:
    """Construye las 7 tools de dimensión en formato OpenAI Function Calling."""
    tools = []
    for dim, description in DIMENSIONS_CONFIG.items():
        tools.append({
            "type": "function",
            "function": {
                "name": f"query_{dim}_dimension",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": f"Texto de búsqueda para {dim}"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número máximo de candidatos a retornar",
                            "default": 100
                        },
                        "weight": {
                            "type": "integer",
                            "description": f"Peso/importancia de {dim} (0-10)",
                            "default": 5
                        }
                    },
                    "required": ["query_text"]
                }
            }
        })
    return tools


COMBINE_TOOL = {
    "type": "function",
    "function": {
        "name": "combine_and_rank_candidates",
        "description": (
            "Combina resultados de todas las dimensiones consultadas, aplica filtros obligatorios, "
            "calcula scoring ponderado y genera ranking final de candidatos. "
            "Debe llamarse DESPUÉS de consultar las dimensiones individuales."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hard_filters": {
                    "type": "object",
                    "description": "Filtros obligatorios opcionales",
                    "properties": {
                        "education_min": {
                            "type": "string",
                            "description": "Nivel educativo mínimo (Técnico, Tecnólogo, Profesional, Posgrado)"
                        },
                        "language_required": {
                            "type": "string",
                            "description": "Idioma requerido (ej: Inglés)"
                        },
                        "language_min_level": {
                            "type": "string",
                            "description": "Nivel mínimo CEFR (A1, A2, B1, B2, C1, C2)"
                        }
                    }
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de top candidatos a retornar",
                    "default": 10
                }
            },
            "required": []
        }
    }
}

ALL_TOOLS = _build_dimension_tools() + [COMBINE_TOOL]


def _handle_dimension_call(
    dim: str,
    args: dict,
    matcher: DimensionMatcher,
    persistence: PersistenceManager,
    chroma_collections: Dict[str, Any],
    dimension_results: Dict[str, dict],
    per_dim_k: int,
) -> str:
    """
    Ejecuta la búsqueda de una dimensión usando core/ (misma lógica que MCP tools).
    Almacena los resultados en dimension_results y retorna texto resumen para el LLM.
    """
    query_text = args.get("query_text", "")
    weight = args.get("weight", 5)

    if not query_text or not query_text.strip():
        return f"Dimensión '{dim}': sin texto de búsqueda, omitida."

    # Fallbacks idénticos a _run_match en main.py
    if dim == "city" and chroma_collections.get("city") is None:
        hits = persistence.match_city_direct(query_text)
    elif dim == "experience" and not chroma_collections.get("experience"):
        hits = persistence.match_experience_direct(query_text)
    else:
        hits = matcher.query_dimension(dim, query_text, per_dim_k, dedup_by_candidate=True)

    dimension_results[dim] = {"hits": hits, "weight": weight}

    n = len(hits)
    top_ids = [str(h["id_candidate"]) for h in hits[:5]]
    return (
        f"Dimensión '{dim}': {n} candidatos encontrados. Peso: {weight}/10. "
        f"Top IDs: {', '.join(top_ids)}"
    )


def _handle_combine_call(
    args: dict,
    dimension_results: Dict[str, dict],
    aggregator: ResultAggregator,
    filters: HardFilters,
    persistence: PersistenceManager,
    original_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combina resultados de dimensiones acumuladas usando core/ (misma lógica que MCP).
    NO guarda en MySQL (eso lo hace match_candidates después del reranking).
    """
    all_dims = list(DIMENSIONS_CONFIG.keys())

    hits_by_dim: Dict[str, list] = {}
    for dim in all_dims:
        if dim in dimension_results:
            hits_by_dim[dim] = dimension_results[dim]["hits"]
        else:
            hits_by_dim[dim] = []

    w = original_payload.get("weights", {})
    weight_map = {dim: int(w.get(dim, 0) or 0) for dim in all_dims}

    candidate_ids = aggregator.collect_candidates(hits_by_dim)

    query_payload = {dim: (original_payload.get(dim, "") or "").strip() for dim in all_dims}
    hf = args.get("hard_filters") or original_payload.get("hard_filters") or {}
    query_payload["hard_filters"] = hf

    if not candidate_ids:
        return {
            "query": query_payload,
            "results": [],
            "weights_used": weight_map,
            "note": "Sin candidatos tras búsqueda."
        }

    if hf and any(v for v in hf.values() if v):
        class _FiltersConfig:
            def __init__(self, config):
                self.education_min = config.get("education_min")
                self.language_required = config.get("language_required")
                self.language_min_level = config.get("language_min_level")
        candidate_ids = filters.apply(candidate_ids, _FiltersConfig(hf))

    top_k = args.get("top_k") or original_payload.get("top_k", 10)

    if not candidate_ids:
        return {
            "query": query_payload,
            "results": [],
            "weights_used": weight_map,
            "note": "Sin candidatos tras hard filters."
        }

    scored = aggregator.combine_and_rank(hits_by_dim, weight_map, candidate_ids, top_k)

    if not scored:
        return {
            "query": query_payload,
            "results": [],
            "weights_used": weight_map,
            "note": "Sin candidatos tras ranking."
        }

    scored = persistence.enrich_candidates(scored)

    return {
        "query": query_payload,
        "results": scored,
        "weights_used": weight_map,
    }


def _ensure_all_dimensions_queried(
    payload: Dict[str, Any],
    dimension_results: Dict[str, dict],
    matcher: DimensionMatcher,
    persistence: PersistenceManager,
    chroma_collections: Dict[str, Any],
    per_dim_k: int,
) -> None:
    """
    Safety net: si el LLM omitió alguna dimensión con texto y peso > 0,
    la consulta aquí para garantizar resultados completos.
    """
    w = payload.get("weights", {})
    for dim in DIMENSIONS_CONFIG:
        text = (payload.get(dim, "") or "").strip()
        weight = int(w.get(dim, 0) or 0)

        if text and weight > 0 and dim not in dimension_results:
            _handle_dimension_call(
                dim,
                {"query_text": text, "weight": weight},
                matcher, persistence, chroma_collections,
                dimension_results, per_dim_k
            )


def invoke_match_orchestrator(
    payload: Dict[str, Any],
    client: OpenAI,
    model: str,
    matcher: DimensionMatcher,
    aggregator: ResultAggregator,
    filters: HardFilters,
    persistence: PersistenceManager,
    chroma_collections: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orquesta el match invocando tools por dimensión (misma lógica que MCP).
    GPT-4o-mini decide qué dimensiones consultar y luego combina resultados.

    Args:
        payload: Payload del match (skills, weights, top_k, etc.)
        client: Cliente OpenAI
        model: Modelo a usar (ej: gpt-4o-mini)
        matcher: DimensionMatcher inicializado
        aggregator: ResultAggregator
        filters: HardFilters
        persistence: PersistenceManager
        chroma_collections: Dict con colecciones ChromaDB

    Returns:
        Response del match (query, results, weights_used)
    """
    top_k = payload.get("top_k", 10)
    per_dim_k = max(top_k * 10, 50)

    user_message = f"Match request:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"

    messages: List[dict] = [
        {"role": "system", "content": MATCH_ORCHESTRATOR_SPEC},
        {"role": "user", "content": user_message},
    ]

    dimension_results: Dict[str, dict] = {}
    combine_result: Optional[Dict[str, Any]] = None

    for _iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=ALL_TOOLS,
            temperature=0.1,
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            break

        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if name.startswith("query_") and name.endswith("_dimension"):
                dim = name.replace("query_", "").replace("_dimension", "")
                result_text = _handle_dimension_call(
                    dim, args, matcher, persistence, chroma_collections,
                    dimension_results, per_dim_k
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

            elif name == "combine_and_rank_candidates":
                _ensure_all_dimensions_queried(
                    payload, dimension_results, matcher, persistence,
                    chroma_collections, per_dim_k
                )

                combine_result = _handle_combine_call(
                    args, dimension_results, aggregator, filters,
                    persistence, payload
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "status": "ok",
                        "candidates_ranked": len(combine_result.get("results", []))
                    }, ensure_ascii=False),
                })

            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Tool desconocida: {name}",
                })

        if combine_result is not None:
            break

    if combine_result is None:
        raise ValueError(
            "El LLM no completó el proceso de matching. "
            f"Dimensiones consultadas: {list(dimension_results.keys())}. "
            "No se llamó combine_and_rank_candidates."
        )

    return combine_result
