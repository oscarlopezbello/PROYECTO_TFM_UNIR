# TFM Match - Sistema de Matching de Vacantes Laborales

## Descripcion

Sistema de matching de vacantes laborales en sectores de consumo masivo en Colombia. Utiliza embeddings semanticos, busqueda vectorial (ChromaDB) y orquestacion LLM (GPT-4o-mini) con tools por dimension basadas en Model Context Protocol (MCP) para encontrar los mejores candidatos.

**Autores:** Oscar Ivan Lopez Bello & Jorge Andres Rojas  
**Institucion:** Universidad Internacional de La Rioja (UNIR)  
**Programa:** Master en Inteligencia Artificial  
**Año:** 2026

## Arquitectura

El proyecto implementa una arquitectura de **doble interfaz** que comparte el mismo codigo base (`core/`):

```
┌──────────────────────────────────────────────────────────────┐
│                       INTERFACES                             │
├──────────────────────────────────────────────────────────────┤
│  FastAPI + Streamlit (Web)     │   MCP Server (Conversacional)│
│  api/main.py + front-end/     │   mcp/server.py              │
│       ↓                       │        ↓                     │
│  LLM orquesta tools           │   LLM invoca tools via MCP   │
│  por dimension (Function      │   protocolo estandar          │
│  Calling OpenAI)              │   (Cursor, Claude, etc.)      │
└──────────────┬────────────────┴──────────────┬───────────────┘
               │                               │
               └───────────┬───────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│                CORE BUSINESS LOGIC (shared)                  │
├──────────────────────────────────────────────────────────────┤
│  dimension_matcher.py    - Busqueda por dimension            │
│  result_aggregator.py    - Scoring ponderado y ranking       │
│  embeddings_manager.py   - Generacion de embeddings          │
│  filters.py              - Filtros obligatorios (hard)       │
│  persistence.py          - Almacenamiento MySQL              │
└──────────────┬───────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
├──────────────────────────────────────────────────────────────┤
│  ChromaDB   - ~22k candidatos indexados (6 dimensiones)      │
│  MySQL      - Datos maestros + historico de busquedas        │
│  OpenAI     - Embeddings (text-embedding-3-small)            │
└──────────────────────────────────────────────────────────────┘
```

### Flujo de matching (desde el frontend)

```
1. Usuario llena formulario en Streamlit (skills, experiencia, educacion, etc.)
2. Streamlit envia POST /match a FastAPI
3. FastAPI invoca GPT-4o-mini con 8 tools disponibles (7 dimension + 1 agregacion)
4. GPT-4o-mini analiza el payload y llama las tools de dimension relevantes:
   - query_skills_dimension, query_experience_dimension, etc.
5. Cada tool ejecuta DimensionMatcher.query_dimension() contra ChromaDB
6. GPT-4o-mini llama combine_and_rank_candidates para combinar y rankear
7. Se aplica reranking rule-based (idioma CEFR, skills lexico, experiencia por rango, etc.)
8. Resultados se guardan en MySQL y se retornan al frontend
```

## Dimensiones de Matching

| Dimension | Descripcion | Ejemplo |
|-----------|-------------|---------|
| **Skills** | Habilidades tecnicas y blandas | atencion al cliente, CRM, Excel |
| **Experience** | Experiencia laboral | 0-1, 1-3, 3-5, 5+ años |
| **Education** | Nivel educativo | Tecnico, Tecnologo, Profesional |
| **Language** | Idiomas y niveles | Ingles B2, Frances basico |
| **Sector** | Sector o area laboral | Call center, BPO, Retail |
| **Job Title** | Cargo o titulo | Asesor de servicio al cliente |
| **City** | Ubicacion/ciudad | Bogota, Medellin |

## Instalacion

### Requisitos previos

- Python >= 3.12
- MySQL 8.0+
- Poetry (gestor de dependencias)
- API Key de OpenAI

### Setup

```bash
# 1. Clonar repositorio
git clone https://github.com/oscarlopezbello/PROYECTO_TFM_UNIR.git
cd PROYECTO_TFM_UNIR

# 2. Instalar dependencias
poetry install

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales reales

# 4. Verificar que MySQL esta corriendo y la base de datos existe
# La tabla principal es: candidates_prepared

# 5. Verificar colecciones ChromaDB (deben estar indexadas previamente)
# Las colecciones se almacenan en el directorio configurado en CHROMA_DIR
```

### Variables de entorno

Copiar `.env.example` a `.env` y completar:

```env
OPENAI_API_KEY=<tu-api-key-de-openai>
MYSQL_URL=mysql+pymysql://usuario:contraseña@localhost:3306/analytics_db
CHROMA_DIR=./data/chroma
```

### Indexacion de ChromaDB

Las colecciones necesarias son:

- `candidates_skills`
- `candidates_experience`
- `candidates_education`
- `candidates_language`
- `candidates_sector`
- `candidates_job_title`

Para indexar (si las colecciones no existen):

```bash
poetry run python -m tfm_match.embeddings.index_skills
poetry run python -m tfm_match.embeddings.index_experience
poetry run python -m tfm_match.embeddings.index_education
poetry run python -m tfm_match.embeddings.index_language
poetry run python -m tfm_match.embeddings.index_sector
poetry run python -m tfm_match.embeddings.index_job_title
```

## Uso

### Opcion 1: Interfaz Web (Streamlit + FastAPI)

```bash
# Terminal 1: Iniciar backend FastAPI
poetry run uvicorn tfm_match.api.main:app --reload

# Terminal 2: Iniciar frontend Streamlit
poetry run streamlit run src/tfm_match/front-end/app.py
```

Abrir http://localhost:8501 en el navegador.

### Opcion 2: API REST directa

```bash
# Iniciar servidor
poetry run uvicorn tfm_match.api.main:app --reload

# Ejemplo de request
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "skills": "atencion al cliente, CRM",
    "experience": "1-3",
    "education": "Tecnico",
    "language": "Espanol",
    "sector": "Call center, BPO",
    "job_title": "Asesor de servicio al cliente",
    "city": "Bogota",
    "top_k": 10,
    "weights": {
      "skills": 5, "experience": 3, "education": 2,
      "language": 2, "sector": 4, "job_title": 4, "city": 2
    }
  }'
```

### Opcion 3: MCP (Conversacional desde IDE)

El servidor MCP permite interactuar con el sistema desde un IDE compatible (Cursor, etc.):

```bash
# Iniciar servidor MCP directamente
poetry run python -m tfm_match.mcp.server
```

O configurar en Cursor usando el archivo `mcp-config.json` incluido.

Ejemplo de interaccion conversacional:

```
Usuario: "Busco un supervisor de call center con experiencia en liderazgo,
         nivel profesional y ingles B2 minimo"

LLM (usando MCP):
1. query_skills_dimension("liderazgo de equipos, KPIs, coaching")
2. query_experience_dimension("5+")
3. query_education_dimension("Profesional")
4. query_language_dimension("Ingles B2")
5. query_sector_dimension("Call center")
6. query_job_title_dimension("Supervisor de operaciones")
7. combine_and_rank_candidates(hard_filters={education_min: "Profesional", ...}, top_k=10)
```

## Tools MCP

### Tools por Dimension (6 tools)

| Tool | Descripcion |
|------|-------------|
| `query_skills_dimension` | Busca candidatos por habilidades tecnicas y blandas |
| `query_experience_dimension` | Busca por experiencia laboral |
| `query_education_dimension` | Busca por nivel educativo |
| `query_language_dimension` | Busca por idiomas y niveles |
| `query_sector_dimension` | Busca por sector o area laboral |
| `query_job_title_dimension` | Busca por cargo o titulo |

### Tools de Agregacion y Utilidades (3 tools)

| Tool | Descripcion |
|------|-------------|
| `combine_and_rank_candidates` | Combina resultados de dimensiones, aplica scoring ponderado y filtros |
| `get_candidate_details` | Obtiene perfil completo de un candidato |
| `explain_match_breakdown` | Explica el desglose de afinidad de un candidato |

### Resources MCP (3 resources)

| Resource | URI | Descripcion |
|----------|-----|-------------|
| ChromaDB Collections Status | `tfm://collections/stats` | Estado de las colecciones indexadas |
| Available Dimensions Schema | `tfm://schema/dimensions` | Descripcion de las 6 dimensiones |
| Weights Configuration | `tfm://config/weights` | Guia de configuracion de pesos |

## Endpoints API

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/health` | Estado del servicio y colecciones |
| POST | `/match` | Ejecuta matching de candidatos |
| GET | `/cities` | Lista de ciudades disponibles |
| GET | `/job_requests` | Historial de busquedas |
| GET | `/job_requests/{id}` | Resultados de una busqueda especifica |
| GET | `/latency` | Metricas de latencia |

## Sistema de Pesos

Los pesos (0-10) determinan la importancia de cada dimension:

- **0**: Dimension no relevante (no se considera)
- **1-3**: Deseable pero no critico
- **4-6**: Importante (contribuye significativamente)
- **7-10**: Critico (requisito esencial del cargo)

Calculo del score final:

```
affinity = (score_skills * w_skills + score_experience * w_experience + ...) / sum(weights)
```

## Pipeline de Datos

El proyecto sigue una arquitectura de capas (Bronze -> Silver -> Gold) para preparar los datos antes de indexarlos:

### Bronze (extraccion)

Extrae datos crudos desde MySQL.

```bash
# El modulo bronze/mysql.py proporciona get_engine() y load_table()
# Se usa internamente por los scripts de las capas superiores
```

### Silver (limpieza y normalizacion)

Limpia, normaliza y estandariza los datos de candidatos.

```bash
# Construir la capa silver desde candidates_clean
poetry run python -m tfm_match.scripts.build_silver
```

### Gold (textos para embeddings)

Construye los textos optimizados que se usaran como input para generar embeddings.

```bash
# Construir candidates_prepared con profile_text
poetry run python -m tfm_match.scripts.build_candidates_prepared

# Actualizar profile_text si candidates_clean cambio
poetry run python -m tfm_match.scripts.populate_profile_text
```

### Indexacion en ChromaDB

Una vez preparados los datos, se indexan en ChromaDB por dimension:

```bash
# Indexar todas las colecciones (script batch en PowerShell)
./scripts/index_all_collections.ps1

# O indexar una por una
poetry run python -m tfm_match.embeddings.index_skills
poetry run python -m tfm_match.embeddings.index_experience
poetry run python -m tfm_match.embeddings.index_education
poetry run python -m tfm_match.embeddings.index_language
poetry run python -m tfm_match.embeddings.index_sector
poetry run python -m tfm_match.embeddings.index_job_title

# Verificar que las colecciones se crearon correctamente
poetry run python -m tfm_match.embeddings.test_chroma
```

### Scripts de utilidad

| Script | Comando | Descripcion |
|--------|---------|-------------|
| Inspeccionar datos crudos | `poetry run python -m tfm_match.scripts.inspect_data` | Muestra estructura de la tabla candidates |
| Inspeccionar silver | `poetry run python -m tfm_match.scripts.inspect_silver` | Muestra datos de la capa silver |
| Inspeccionar gold text | `poetry run python -m tfm_match.scripts.inspect_gold_text` | Muestra los textos construidos para embeddings |
| Query manual skills | `poetry run python -m tfm_match.embeddings.query_skills` | Prueba manual de busqueda en coleccion skills |
| Query manual experience | `poetry run python -m tfm_match.embeddings.query_experience` | Prueba manual de busqueda en coleccion experience |
| Query manual education | `poetry run python -m tfm_match.embeddings.query_education` | Prueba manual de busqueda en coleccion education |
| Query manual language | `poetry run python -m tfm_match.embeddings.query_language` | Prueba manual de busqueda en coleccion language |
| Query manual sector | `poetry run python -m tfm_match.embeddings.query_sector` | Prueba manual de busqueda en coleccion sector |
| Query manual job_title | `poetry run python -m tfm_match.embeddings.query_job_title` | Prueba manual de busqueda en coleccion job_title |
| Eliminar coleccion | `poetry run python -m tfm_match.embeddings.delete_education_collection` | Elimina una coleccion para re-indexar (existen para education, experience, job_title, sector) |
| Crear tabla MySQL | `scripts/create_match_executions.sql` | SQL para crear la tabla match_executions |
| MCP Inspector | `./run_mcp_inspector.ps1` | Lanza el inspector oficial de MCP para debug |

## Estructura del Proyecto

```
PROYECTO_TFM_UNIR/
├── src/tfm_match/
│   ├── api/                           # API REST (FastAPI)
│   │   ├── main.py                    # Endpoints + orquestacion LLM
│   │   └── reranking_rules.py         # Reglas de reranking post-scoring
│   ├── llm/                           # Modulo de orquestacion LLM
│   │   ├── client.py                  # Multi-tool orchestration con GPT-4o-mini
│   │   └── spec.py                    # System prompt del orquestador
│   ├── mcp/                           # Servidor MCP
│   │   ├── server.py                  # Servidor principal
│   │   └── tools/
│   │       ├── dimension_tools.py     # 6 tools por dimension
│   │       └── aggregation_tools.py   # Agregacion + utilidades
│   ├── core/                          # Logica de negocio compartida
│   │   ├── dimension_matcher.py       # Busqueda por dimension en ChromaDB
│   │   ├── result_aggregator.py       # Scoring ponderado y ranking
│   │   ├── embeddings_manager.py      # Generacion de embeddings OpenAI
│   │   ├── filters.py                 # Hard filters (educacion, idioma)
│   │   └── persistence.py            # Persistencia MySQL
│   ├── embeddings/                    # Indexacion y consultas ChromaDB
│   │   ├── index_skills.py            # Indexador de skills
│   │   ├── index_experience.py        # Indexador de experience
│   │   ├── index_education.py         # Indexador de education
│   │   ├── index_language.py          # Indexador de language
│   │   ├── index_sector.py            # Indexador de sector
│   │   ├── index_job_title.py         # Indexador de job_title
│   │   ├── query_*.py                 # Scripts de consulta manual por dimension
│   │   ├── delete_*_collection.py     # Scripts para eliminar colecciones
│   │   └── test_chroma.py            # Verificacion de colecciones
│   ├── gold/                          # Construccion de textos para embeddings
│   │   ├── text_builder.py            # Genera profile_text por candidato
│   │   └── text_sanitizer.py          # Sanitizacion de textos
│   ├── silver/                        # Limpieza y normalizacion de datos
│   │   ├── clean_candidates.py        # Limpieza de datos de candidatos
│   │   └── schema.py                 # Esquemas estandarizados
│   ├── bronze/                        # Extraccion de datos desde MySQL
│   │   └── mysql.py                   # Conexion y carga de tablas
│   ├── scripts/                       # Scripts de ETL y utilidades
│   │   ├── build_candidates_prepared.py  # Construye tabla candidates_prepared
│   │   ├── build_silver.py            # Construye capa silver
│   │   ├── populate_profile_text.py   # Actualiza profile_text
│   │   ├── inspect_data.py            # Inspeccion de datos crudos
│   │   ├── inspect_silver.py          # Inspeccion de capa silver
│   │   └── inspect_gold_text.py       # Inspeccion de textos gold
│   ├── front-end/
│   │   ├── app.py                     # Frontend Streamlit
│   │   └── styles.css                 # Estilos del frontend
│   └── config.py                      # Configuracion centralizada (.env)
├── scripts/
│   ├── create_match_executions.sql    # SQL para tabla de ejecuciones
│   └── index_all_collections.ps1      # Script batch de indexacion
├── tests/
│   ├── test_mcp_tools.py              # Test de las 9 tools + 3 resources MCP
│   ├── test_llm_orchestration.py      # Test de orquestacion LLM con GPT-4o-mini
│   ├── baseline_cases.json            # Casos de prueba para regresion API
│   ├── capture_baseline.py            # Captura snapshot de resultados API
│   └── validate_consistency.py        # Valida consistencia entre versiones
├── .env.example                       # Plantilla de variables de entorno
├── .gitignore
├── mcp-config.json                    # Configuracion MCP para Cursor
├── pyproject.toml                     # Dependencias (Poetry)
├── run_mcp_inspector.ps1              # Lanzador de MCP Inspector
└── README.md                          # Este archivo
```

## Metricas del Sistema

- **Candidatos indexados**: ~22,000
- **Dimensiones de matching**: 7 (skills, experience, education, language, sector, job_title, city)
- **Colecciones ChromaDB**: 6 (city usa fallback MySQL)
- **Modelo de embeddings**: text-embedding-3-small (1536 dimensiones)
- **LLM orquestador**: GPT-4o-mini
- **Base de datos**: MySQL 8.0
- **Trazabilidad**: 100% (todas las busquedas se guardan en MySQL)

## Testing

El proyecto incluye tres tipos de tests:

### 1. Tests de MCP Tools

Prueba las 9 tools y 3 resources del servidor MCP directamente (sin levantar el servidor completo). Valida que cada tool retorna candidatos en el formato esperado.

```bash
poetry run python tests/test_mcp_tools.py
```

Requiere: MySQL, ChromaDB y OPENAI_API_KEY.

### 2. Tests de orquestacion LLM

Prueba que GPT-4o-mini invoca las tools de dimension correctas segun el payload y termina llamando `combine_and_rank_candidates`. Ejecuta 3 casos con diferentes combinaciones de dimensiones.

```bash
poetry run python tests/test_llm_orchestration.py
```

Requiere: MySQL, ChromaDB y OPENAI_API_KEY (genera costo ~$0.01-0.05 por ejecucion).

### 3. Tests de regresion API

Captura y compara resultados del endpoint `/match` para detectar cambios no deseados.

```bash
# Con FastAPI corriendo en otra terminal
poetry run python tests/capture_baseline.py

# Despues de hacer cambios, capturar nuevos resultados y comparar
poetry run python tests/capture_baseline.py --output refactored_results.json
poetry run python tests/validate_consistency.py \
  --baseline baseline_snapshot.json \
  --new refactored_results.json
```

## Sectores Objetivo

- Call centers y BPO
- Retail y supermercados
- Consumo masivo (FMCG)
- Logistica y distribucion

## Licencia

Proyecto academico - Trabajo Fin de Master, UNIR 2026

## Contacto

- **Oscar Ivan Lopez Bello** - oscarivan.lopez141@comunidadunir.net
- **Jorge Andres Rojas** - jorgeandres.rojas142@comunidadunir.net
