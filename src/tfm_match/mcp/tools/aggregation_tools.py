"""
Aggregation Tools - Tool MCP para combinar resultados de múltiples dimensiones.
"""

import json
from typing import Any, Dict
from mcp.server import Server
from mcp.types import Tool, TextContent

from tfm_match.core.result_aggregator import ResultAggregator
from tfm_match.core.filters import HardFilters
from tfm_match.core.persistence import PersistenceManager


def format_final_results(job_request_id: int, scored_results: list) -> str:
    """Formatea resultados finales ranqueados para el LLM."""
    if not scored_results:
        return "No se encontraron candidatos que cumplan con los criterios."
    
    output = f"# Resultados del Matching\n\n"
    output += f"**Job Request ID**: {job_request_id}\n"
    output += f"**Candidatos encontrados**: {len(scored_results)}\n\n"
    output += "---\n\n"
    
    for i, candidate in enumerate(scored_results, 1):
        output += f"## {i}. Candidato ID: {candidate['candidate_id']}\n\n"
        output += f"### Afinidad Total: **{candidate['affinity']}%**\n\n"
        
        # Skills
        if candidate.get('skills'):
            skills_preview = candidate['skills'][:150] + "..." if len(candidate['skills']) > 150 else candidate['skills']
            output += f"**Habilidades**: {skills_preview}\n\n"
        
        # Brief description
        if candidate.get('brief_description'):
            brief_preview = candidate['brief_description'][:200] + "..." if len(candidate['brief_description']) > 200 else candidate['brief_description']
            output += f"**Perfil**: {brief_preview}\n\n"
        
        # Breakdown por dimensión
        if candidate.get('breakdown'):
            output += "### Breakdown por Dimensión:\n\n"
            for dim, scores in candidate['breakdown'].items():
                output += f"- **{dim.capitalize()}**: "
                output += f"{scores['score_pct']}% "
                output += f"(peso: {scores['weight']}, "
                output += f"contribución: {scores['contribution']:.1%})\n"
            output += "\n"
        
        output += "---\n\n"
    
    # JSON estructurado
    output += "\n## Datos Estructurados\n\n```json\n"
    output += json.dumps({
        "job_request_id": job_request_id,
        "total_results": len(scored_results),
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "affinity": c["affinity"],
                "breakdown": c.get("breakdown", {}),
                "skills": c.get("skills", "")[:100],
                "brief_description": c.get("brief_description", "")[:100]
            }
            for c in scored_results
        ]
    }, indent=2, ensure_ascii=False)
    output += "\n```\n"
    
    return output


def get_aggregation_tools_list() -> list[Tool]:
    """
    Retorna la lista de 3 tools de agregación.
    """
    return [
            Tool(
                name="combine_and_rank_candidates",
                description=(
                    "Combina resultados de múltiples dimensiones, aplica filtros obligatorios, "
                    "calcula scoring ponderado y genera ranking final de candidatos. "
                    "Esta tool debe llamarse DESPUÉS de consultar las dimensiones individuales."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dimension_results": {
                            "type": "array",
                            "description": "Resultados de cada dimensión consultada",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dimension": {
                                        "type": "string",
                                        "enum": ["skills", "experience", "education", "language", "sector", "job_title"],
                                        "description": "Nombre de la dimensión"
                                    },
                                    "candidates": {
                                        "type": "array",
                                        "description": "Lista de candidatos retornados por query_X_dimension",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id_candidate": {"type": "string"},
                                                "similarity": {"type": "number"},
                                                "distance": {"type": "number"}
                                            }
                                        }
                                    },
                                    "weight": {
                                        "type": "integer",
                                        "description": "Peso de esta dimensión (0-10)",
                                        "minimum": 0,
                                        "maximum": 10
                                    }
                                },
                                "required": ["dimension", "candidates", "weight"]
                            }
                        },
                        "hard_filters": {
                            "type": "object",
                            "description": "Filtros obligatorios a aplicar",
                            "properties": {
                                "education_min": {
                                    "type": "string",
                                    "enum": ["Técnico", "Tecnólogo", "Profesional", "Posgrado"],
                                    "description": "Nivel educativo mínimo requerido"
                                },
                                "language_required": {
                                    "type": "string",
                                    "description": "Idioma requerido (ej: Inglés, Francés)"
                                },
                                "language_min_level": {
                                    "type": "string",
                                    "enum": ["A1", "A2", "B1", "B2", "C1", "C2"],
                                    "description": "Nivel mínimo del idioma (CEFR)"
                                }
                            }
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número de top candidatos a retornar",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["dimension_results"]
                }
            ),
            Tool(
                name="get_candidate_details",
                description="Obtiene el perfil completo de un candidato específico por su ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "description": "ID del candidato a consultar"
                        }
                    },
                    "required": ["candidate_id"]
                }
            ),
            Tool(
                name="explain_match_breakdown",
                description=(
                    "Explica detalladamente por qué un candidato tiene cierto porcentaje de afinidad, "
                    "desglosando la contribución de cada dimensión"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "description": "ID del candidato"
                        },
                        "breakdown": {
                            "type": "object",
                            "description": "Breakdown del candidato (obtenido de combine_and_rank_candidates)"
                        }
                    },
                    "required": ["candidate_id", "breakdown"]
                }
            )
        ]


def register_aggregation_tools(
    server: Server,
    aggregator: ResultAggregator,
    filters: HardFilters,
    persistence: PersistenceManager
):
    """
    Registra las 3 tools de agregación en el servidor MCP.
    DEPRECATED: Ya no se usa, el handler está centralizado en server.py
    """
    pass  # Ya no necesario, handler centralizado


async def handle_aggregation_tool_call(
    name: str,
    arguments: dict,
    aggregator: ResultAggregator,
    filters: HardFilters,
    persistence: PersistenceManager
) -> list[TextContent]:
    """
    Maneja las llamadas a tools de agregación.
    Esta función es llamada por el handler centralizado en server.py
    """
    
    if name == "combine_and_rank_candidates":
        try:
            # 1. Parsear dimension_results
            dimension_results = arguments.get("dimension_results", [])
            
            if not dimension_results:
                return [TextContent(
                    type="text",
                    text="Error: No se proporcionaron resultados de dimensiones. Debe llamar primero a query_X_dimension para cada dimensión relevante."
                )]
            
            # Reconstruir formato hits_by_dim
            hits_by_dim = {}
            weight_map = {}
            
            for dim_result in dimension_results:
                dimension = dim_result["dimension"]
                candidates = dim_result["candidates"]
                weight = dim_result["weight"]
                
                # Convertir candidatos al formato esperado por ResultAggregator
                hits = []
                for cand in candidates:
                    hits.append({
                        "id_candidate": cand["id_candidate"],
                        "sim": cand["similarity"],
                        "dist": cand.get("distance", 1.0 - cand["similarity"]),
                        "meta": {},
                        "doc": ""
                    })
                
                hits_by_dim[dimension] = hits
                weight_map[dimension] = weight
            
            # Asegurar que todas las dimensiones estén presentes (aunque vacías)
            all_dims = ["skills", "experience", "education", "language", "sector", "job_title"]
            for dim in all_dims:
                if dim not in hits_by_dim:
                    hits_by_dim[dim] = []
                if dim not in weight_map:
                    weight_map[dim] = 0
            
            # 2. Recolectar candidatos
            candidate_ids = aggregator.collect_candidates(hits_by_dim)
            
            if not candidate_ids:
                return [TextContent(
                    type="text",
                    text="No se encontraron candidatos en las dimensiones consultadas."
                )]
            
            # 3. Aplicar hard filters si existen
            hard_filters_config = arguments.get("hard_filters", {})
            if hard_filters_config:
                # Crear objeto con atributos para compatibilidad
                class FiltersConfig:
                    def __init__(self, config):
                        self.education_min = config.get("education_min")
                        self.language_required = config.get("language_required")
                        self.language_min_level = config.get("language_min_level")
                
                filters_obj = FiltersConfig(hard_filters_config)
                candidate_ids = filters.apply(candidate_ids, filters_obj)
                
                if not candidate_ids:
                    return [TextContent(
                        type="text",
                        text="No se encontraron candidatos que cumplan con los filtros obligatorios (education_min, language_required)."
                    )]
            
            # 4. Scoring y ranking
            top_k = arguments.get("top_k", 10)
            scored = aggregator.combine_and_rank(
                hits_by_dim,
                weight_map,
                candidate_ids,
                top_k
            )
            
            if not scored:
                return [TextContent(
                    type="text",
                    text="No se pudieron rankear candidatos (posiblemente pesos=0 en todas las dimensiones)."
                )]
            
            # 5. Enriquecer con MySQL
            scored = persistence.enrich_candidates(scored)
            
            # 6. Guardar en MySQL
            query_payload = {
                "dimensions_used": list(weight_map.keys()),
                "weights": weight_map,
                "hard_filters": hard_filters_config,
                "mcp_generated": True
            }
            
            job_request_id = persistence.save_job_request_and_results(
                query_payload,
                weight_map,
                top_k,
                scored
            )
            
            # 7. Formatear y retornar
            formatted_output = format_final_results(job_request_id, scored)
            
            return [TextContent(
                type="text",
                text=formatted_output
            )]
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return [TextContent(
                type="text",
                text=f"Error al combinar y rankear candidatos:\n{str(e)}\n\n{error_trace}"
            )]
    
    elif name == "get_candidate_details":
        try:
            candidate_id = arguments.get("candidate_id", "")
            
            if not candidate_id:
                return [TextContent(
                    type="text",
                    text="Error: candidate_id es requerido"
                )]
            
            # Obtener detalles del candidato
            details = persistence.fetch_candidates_from_mysql([candidate_id])
            
            if not details or candidate_id not in details:
                return [TextContent(
                    type="text",
                    text=f"No se encontró el candidato con ID: {candidate_id}"
                )]
            
            cand_data = details[candidate_id]
            
            output = f"# Perfil del Candidato {candidate_id}\n\n"
            output += f"**ID**: {candidate_id}\n\n"
            
            if cand_data.get('skills'):
                output += f"## Habilidades\n{cand_data['skills']}\n\n"
            
            if cand_data.get('brief_description'):
                output += f"## Descripción\n{cand_data['brief_description']}\n\n"
            
            if cand_data.get('profile_text'):
                output += f"## Perfil Completo\n{cand_data['profile_text'][:500]}...\n\n"
            
            output += "\n```json\n"
            output += json.dumps(cand_data, indent=2, ensure_ascii=False)
            output += "\n```\n"
            
            return [TextContent(
                type="text",
                text=output
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error al obtener detalles del candidato: {str(e)}"
            )]
    
    elif name == "explain_match_breakdown":
        try:
            candidate_id = arguments.get("candidate_id", "")
            breakdown = arguments.get("breakdown", {})
            
            if not candidate_id or not breakdown:
                return [TextContent(
                    type="text",
                    text="Error: candidate_id y breakdown son requeridos"
                )]
            
            output = f"# Explicación del Match - Candidato {candidate_id}\n\n"
            output += "## Desglose por Dimensión:\n\n"
            
            total_contribution = 0.0
            
            for dim, scores in breakdown.items():
                output += f"### {dim.capitalize()}\n"
                output += f"- **Score**: {scores['score_pct']}%\n"
                output += f"- **Peso asignado**: {scores['weight']}/10\n"
                output += f"- **Contribución al total**: {scores['contribution']:.1%}\n"
                output += f"- **Interpretación**: "
                
                if scores['score_pct'] >= 80:
                    output += "Excelente match en esta dimensión \n"
                elif scores['score_pct'] >= 60:
                    output += "Buen match \n"
                elif scores['score_pct'] >= 40:
                    output += "Match moderado\n"
                else:
                    output += "Match bajo\n"
                
                total_contribution += scores['contribution']
                output += "\n"
            
            output += f"\n**Total acumulado**: {total_contribution:.1%}\n"
            
            return [TextContent(
                type="text",
                text=output
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error al explicar breakdown: {str(e)}"
            )]
    
    else:
        return [TextContent(
            type="text",
            text=f"Tool desconocida: {name}"
        )]
