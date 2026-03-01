"""
TFM Match MCP Server

Servidor MCP que expone funcionalidad de matching de candidatos
a través de Model Context Protocol.

Expone:
- 6 tools de consulta por dimensión (skills, experience, education, language, sector, job_title)
- 1 tool de agregación y ranking
- 2 tools auxiliares (detalles de candidato, explicación de match)
"""

import asyncio
import sys
from typing import Optional

import chromadb
from openai import OpenAI
from sqlalchemy import create_engine
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from tfm_match.config import (
    get_env,
    OPENAI_API_KEY,
    MYSQL_URL,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    CHROMA_COLLECTION_SKILLS,
    CHROMA_COLLECTION_EXPERIENCE,
    CHROMA_COLLECTION_EDUCATION,
    CHROMA_COLLECTION_LANGUAGE,
    CHROMA_COLLECTION_SECTOR,
    CHROMA_COLLECTION_JOB_TITLE,
    CANDIDATES_TABLE,
)
from tfm_match.core.embeddings_manager import EmbeddingsManager
from tfm_match.core.dimension_matcher import DimensionMatcher
from tfm_match.core.result_aggregator import ResultAggregator
from tfm_match.core.filters import HardFilters
from tfm_match.core.persistence import PersistenceManager
from tfm_match.mcp.tools.dimension_tools import register_dimension_tools, get_dimension_tools_list, handle_dimension_tool_call
from tfm_match.mcp.tools.aggregation_tools import register_aggregation_tools, get_aggregation_tools_list, handle_aggregation_tool_call


# Validar configuración
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY no configurada", file=sys.stderr)
    sys.exit(1)

if not MYSQL_URL:
    print("ERROR: MYSQL_URL no configurada", file=sys.stderr)
    sys.exit(1)


# Crear servidor MCP
app = Server("tfm-match-mcp")

# Variables globales (se inicializan en startup)
matcher: Optional[DimensionMatcher] = None
aggregator: Optional[ResultAggregator] = None
filters: Optional[HardFilters] = None
persistence: Optional[PersistenceManager] = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Lista todas las tools disponibles (9 total)."""
    dimension_tools = get_dimension_tools_list()
    aggregation_tools = get_aggregation_tools_list()
    return dimension_tools + aggregation_tools


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handler centralizado para todas las tool calls."""
    # Verificar que los componentes estén inicializados
    if matcher is None or aggregator is None:
        return [TextContent(
            type="text",
            text="Error: El servidor MCP aún no ha terminado de inicializar. Por favor, intenta de nuevo en unos segundos."
        )]
    
    # Determinar si es dimension tool o aggregation tool
    if name.endswith("_dimension"):
        return await handle_dimension_tool_call(name, arguments, matcher)
    elif name in ["combine_and_rank_candidates", "get_candidate_details", "explain_match_breakdown"]:
        return await handle_aggregation_tool_call(name, arguments, aggregator, filters, persistence)
    else:
        return [TextContent(
            type="text",
            text=f"Tool desconocida: {name}"
        )]


@app.list_resources()
async def list_resources() -> list[Resource]:
    """Lista recursos disponibles (información estática del sistema)."""
    return [
        Resource(
            uri="tfm://collections/stats",
            name="ChromaDB Collections Status",
            mimeType="application/json",
            description="Estado actual de las colecciones ChromaDB indexadas"
        ),
        Resource(
            uri="tfm://schema/dimensions",
            name="Available Dimensions Schema",
            mimeType="application/json",
            description="Información sobre las 6 dimensiones disponibles para matching"
        ),
        Resource(
            uri="tfm://config/weights",
            name="Weights Configuration",
            mimeType="application/json",
            description="Guía sobre cómo configurar pesos para cada dimensión"
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Lee un recurso por su URI."""
    
    if uri == "tfm://collections/stats":
        import json
        
        # Obtener stats de colecciones (si están disponibles)
        stats = {
            "chroma_dir": CHROMA_DIR,
            "collections": {
                "skills": {"name": CHROMA_COLLECTION_SKILLS, "indexed": True},
                "experience": {"name": CHROMA_COLLECTION_EXPERIENCE, "indexed": True},
                "education": {"name": CHROMA_COLLECTION_EDUCATION, "indexed": True},
                "language": {"name": CHROMA_COLLECTION_LANGUAGE, "indexed": True},
                "sector": {"name": CHROMA_COLLECTION_SECTOR, "indexed": True},
                "job_title": {"name": CHROMA_COLLECTION_JOB_TITLE, "indexed": True},
            },
            "embedding_model": EMBEDDING_MODEL,
            "status": "operational"
        }
        
        return json.dumps(stats, indent=2)
    
    elif uri == "tfm://schema/dimensions":
        import json
        
        dimensions_info = {
            "available_dimensions": [
                {
                    "name": "skills",
                    "description": "Habilidades técnicas y blandas del candidato",
                    "examples": [
                        "atención al cliente, CRM, manejo de objeciones",
                        "ventas, negociación, merchandising",
                        "Excel avanzado, análisis de datos, SQL"
                    ],
                    "typical_weight": 5,
                    "tool": "query_skills_dimension"
                },
                {
                    "name": "experience",
                    "description": "Experiencia laboral previa en cargos similares",
                    "examples": [
                        "2 años en call center",
                        "vendedor con 3+ años de experiencia",
                        "supervisor de equipos"
                    ],
                    "typical_weight": 4,
                    "tool": "query_experience_dimension"
                },
                {
                    "name": "education",
                    "description": "Nivel educativo del candidato",
                    "examples": [
                        "Profesional en Administración",
                        "Tecnólogo en Gestión",
                        "Técnico con especialización"
                    ],
                    "typical_weight": 3,
                    "tool": "query_education_dimension"
                },
                {
                    "name": "language",
                    "description": "Idiomas que maneja el candidato y su nivel",
                    "examples": [
                        "Inglés B2",
                        "Portugués avanzado",
                        "Francés básico"
                    ],
                    "typical_weight": 3,
                    "tool": "query_language_dimension"
                },
                {
                    "name": "sector",
                    "description": "Sectores o áreas donde ha trabajado",
                    "examples": [
                        "Call center, BPO",
                        "Retail, supermercados, consumo masivo",
                        "Logística, bodegaje, distribución"
                    ],
                    "typical_weight": 4,
                    "tool": "query_sector_dimension"
                },
                {
                    "name": "job_title",
                    "description": "Cargos o títulos de puestos que ha ocupado",
                    "examples": [
                        "Asesor de servicio al cliente",
                        "Cajero principal",
                        "Supervisor de operaciones"
                    ],
                    "typical_weight": 4,
                    "tool": "query_job_title_dimension"
                }
            ],
            "total_dimensions": 6,
            "matching_approach": "multicriteria_weighted"
        }
        
        return json.dumps(dimensions_info, indent=2)
    
    elif uri == "tfm://config/weights":
        import json
        
        weights_guide = {
            "weight_system": {
                "min": 0,
                "max": 10,
                "description": "Los pesos determinan la importancia relativa de cada dimensión en el matching final"
            },
            "recommendations": {
                "critical_requirement": {
                    "weight": 7-10,
                    "description": "Use peso alto (7-10) para requisitos críticos del cargo"
                },
                "important_but_flexible": {
                    "weight": 4-6,
                    "description": "Use peso medio (4-6) para requisitos importantes pero no eliminatorios"
                },
                "nice_to_have": {
                    "weight": 1-3,
                    "description": "Use peso bajo (1-3) para características deseables pero no esenciales"
                },
                "not_relevant": {
                    "weight": 0,
                    "description": "Use peso 0 si la dimensión no es relevante para el cargo"
                }
            },
            "examples_by_role": {
                "call_center_agent": {
                    "skills": 7,
                    "experience": 4,
                    "education": 2,
                    "language": 3,
                    "sector": 5,
                    "job_title": 5,
                    "rationale": "Skills es crítico (atención al cliente), sector y job_title importantes para experiencia similar"
                },
                "senior_supervisor": {
                    "skills": 6,
                    "experience": 8,
                    "education": 5,
                    "language": 4,
                    "sector": 4,
                    "job_title": 6,
                    "rationale": "Experiencia es crítica para rol senior, educación importante para management"
                },
                "entry_level": {
                    "skills": 6,
                    "experience": 2,
                    "education": 3,
                    "language": 2,
                    "sector": 4,
                    "job_title": 4,
                    "rationale": "Foco en skills, experiencia menos crítica para entry-level"
                }
            },
            "total_weight_calculation": "La afinidad final se calcula como: sum(score_dimension * weight_dimension) / sum(weights_used)"
        }
        
        return json.dumps(weights_guide, indent=2)
    
    else:
        raise ValueError(f"Recurso desconocido: {uri}")


async def main():
    """Función principal que inicializa y corre el servidor MCP."""
    
    global matcher, aggregator, filters, persistence
    
    print("Iniciando TFM Match MCP Server...", file=sys.stderr)
    print(f"ChromaDB: {CHROMA_DIR}", file=sys.stderr)
    print(f" MySQL: {MYSQL_URL[:50]}...", file=sys.stderr)
    print(f"Embedding Model: {EMBEDDING_MODEL}", file=sys.stderr)
    
    # Inicializar clientes
    print("Inicializando componentes...", file=sys.stderr)
    
    oa_client = OpenAI(api_key=OPENAI_API_KEY)
    engine = create_engine(MYSQL_URL, pool_pre_ping=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Cargar colecciones
    collections = {
        "skills": CHROMA_COLLECTION_SKILLS,
        "experience": CHROMA_COLLECTION_EXPERIENCE,
        "education": CHROMA_COLLECTION_EDUCATION,
        "language": CHROMA_COLLECTION_LANGUAGE,
        "sector": CHROMA_COLLECTION_SECTOR,
        "job_title": CHROMA_COLLECTION_JOB_TITLE,
    }
    
    chroma_collections = {}
    for dim, name in collections.items():
        try:
            chroma_collections[dim] = chroma_client.get_collection(name=name)
            print(f" Colección '{dim}' cargada", file=sys.stderr)
        except Exception as e:
            print(f"  Colección '{dim}' no disponible: {e}", file=sys.stderr)
            chroma_collections[dim] = None
    
    # Inicializar componentes core
    embeddings_mgr = EmbeddingsManager(oa_client, EMBEDDING_MODEL)
    matcher = DimensionMatcher(chroma_collections, embeddings_mgr)
    aggregator = ResultAggregator()
    filters = HardFilters(chroma_collections)
    persistence = PersistenceManager(engine, CANDIDATES_TABLE)
    
    print("Componentes core inicializados", file=sys.stderr)
    
    # Registrar tools
    print("Registrando tools...", file=sys.stderr)
    register_dimension_tools(app, matcher)
    register_aggregation_tools(app, aggregator, filters, persistence)
    print("Tools registradas", file=sys.stderr)
    
    print("TFM Match MCP Server listo", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Correr servidor
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
