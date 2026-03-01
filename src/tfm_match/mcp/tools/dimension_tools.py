"""
Dimension Tools - Tools MCP para consultar cada dimensión individualmente.
"""

import json
from typing import Any, Dict
from mcp.server import Server
from mcp.types import Tool, TextContent

from tfm_match.core.dimension_matcher import DimensionMatcher


# Definición de dimensiones (global para reutilización)
DIMENSIONS_CONFIG = {
    "skills": {
        "description": "Busca candidatos por habilidades técnicas y blandas (ej: atención al cliente, CRM, ventas)",
        "examples": ["atención al cliente, CRM", "ventas, negociación", "Excel avanzado, análisis de datos"]
    },
    "experience": {
        "description": "Busca candidatos por experiencia laboral (ej: 2 años en call center, vendedor senior)",
        "examples": ["2 años en call center", "vendedor con experiencia", "supervisor de equipo"]
    },
    "education": {
        "description": "Busca candidatos por nivel educativo (ej: Profesional, Tecnólogo, Técnico)",
        "examples": ["Profesional en Administración", "Tecnólogo", "Técnico con especialización"]
    },
    "language": {
        "description": "Busca candidatos por idiomas (ej: Inglés B2, Francés básico)",
        "examples": ["Inglés avanzado", "Portugués B1", "Español nativo"]
    },
    "sector": {
        "description": "Busca candidatos por sector o área (ej: Call center, Retail, Consumo masivo)",
        "examples": ["Call center, BPO", "Retail, supermercados", "Consumo masivo, FMCG"]
    },
    "job_title": {
        "description": "Busca candidatos por título o cargo (ej: Asesor de servicio, Cajero, Supervisor)",
        "examples": ["Asesor de servicio al cliente", "Cajero principal", "Supervisor de operaciones"]
    }
}


def format_dimension_results(dimension: str, results: list, weight: int) -> str:
    """Formatea resultados de una dimensión para el LLM."""
    if not results:
        return f"No se encontraron candidatos para la dimensión '{dimension}'."
    
    output = f"## Resultados de dimensión: {dimension}\n"
    output += f"**Peso configurado**: {weight}/10\n"
    output += f"**Candidatos encontrados**: {len(results)}\n\n"
    
    # Top 10 para no saturar el contexto
    for i, hit in enumerate(results[:10], 1):
        output += f"### {i}. Candidato ID: {hit['id_candidate']}\n"
        output += f"- **Similitud**: {hit['sim']:.4f} ({hit['sim']*100:.2f}%)\n"
        output += f"- **Distancia**: {hit['dist']:.4f}\n"
        if hit.get('doc'):
            doc_preview = hit['doc'][:200] + "..." if len(hit['doc']) > 200 else hit['doc']
            output += f"- **Contenido**: {doc_preview}\n"
        output += "\n"
    
    if len(results) > 10:
        output += f"*(Mostrando 10 de {len(results)} resultados)*\n"
    
    # Retornar también como JSON parseable
    output += "\n---\n**Datos estructurados**:\n```json\n"
    output += json.dumps({
        "dimension": dimension,
        "weight": weight,
        "total_candidates": len(results),
        "candidates": [
            {
                "id_candidate": r["id_candidate"],
                "similarity": round(r["sim"], 4),
                "distance": round(r["dist"], 4)
            }
            for r in results
        ]
    }, indent=2, ensure_ascii=False)
    output += "\n```\n"
    
    return output


def get_dimension_tools_list() -> list[Tool]:
    """
    Retorna la lista de 6 tools de dimensiones.
    Cada tool busca candidatos en una dimensión específica.
    """
    
    tools = []
    
    for dim, config in DIMENSIONS_CONFIG.items():
        tools.append(Tool(
            name=f"query_{dim}_dimension",
            description=f"{config['description']}. Ejemplos: {', '.join(config['examples'][:2])}",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": f"Texto describiendo {dim} buscados. Ejemplos: {', '.join(config['examples'])}"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número máximo de candidatos a retornar",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "weight": {
                        "type": "integer",
                        "description": f"Peso/importancia de {dim} en el matching final (0-10)",
                        "default": 5,
                        "minimum": 0,
                        "maximum": 10
                    }
                },
                "required": ["query_text"]
            }
        ))
    
    return tools


def register_dimension_tools(server: Server, matcher: DimensionMatcher):
    """
    Registra las 6 tools de dimensiones en el servidor MCP.
    DEPRECATED: Ya no se usa, el handler está centralizado en server.py
    """
    pass  # Ya no necesario, handler centralizado


async def handle_dimension_tool_call(
    name: str,
    arguments: dict,
    matcher: DimensionMatcher
) -> list[TextContent]:
    """
    Maneja las llamadas a tools de dimensión.
    Esta función es llamada por el handler centralizado en server.py
    """
    # Extraer dimensión del nombre de la tool
    if not name.startswith("query_") or not name.endswith("_dimension"):
        raise ValueError(f"Tool desconocida: {name}")
    
    dimension = name.replace("query_", "").replace("_dimension", "")
    
    if dimension not in DIMENSIONS_CONFIG:
        raise ValueError(f"Dimensión desconocida: {dimension}")
    
    # Extraer parámetros (soporta tanto 'query_text' como 'query' para flexibilidad)
    query_text = arguments.get("query_text") or arguments.get("query", "")
    top_k = arguments.get("top_k", 50)
    weight = arguments.get("weight", 5)
    
    if not query_text or not query_text.strip():
        return [TextContent(
            type="text",
            text=f"Error: query_text o query está vacío para dimensión '{dimension}'. Proporciona un texto de búsqueda."
        )]
    
    # Ejecutar búsqueda usando DimensionMatcher
    try:
        results = matcher.query_dimension(
            dim=dimension,
            query_text=query_text,
            n_results=top_k,
            dedup_by_candidate=True
        )
        
        formatted_output = format_dimension_results(dimension, results, weight)
        
        return [TextContent(
            type="text",
            text=formatted_output
        )]
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f"Error al consultar dimensión '{dimension}':\n{str(e)}\n\n{error_trace}"
        return [TextContent(
            type="text",
            text=error_msg
        )]
