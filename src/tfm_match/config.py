#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de configuración centralizado para TFM Match.
Carga variables de entorno desde .env automáticamente.
"""

import os
from typing import Optional
from pathlib import Path

# Cargar .env desde la raíz del proyecto
# El archivo config.py está en: src/tfm_match/config.py
# Necesitamos subir 3 niveles: config.py -> tfm_match -> src -> PROYECTO_TFM
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

# Función para cargar .env manualmente (sin dependencia de python-dotenv)
def _load_env_file(env_path: Path):
    """Carga variables de entorno desde un archivo .env manualmente."""
    if not env_path.exists():
        print(f"[!] No se encontro archivo .env en: {env_path}")
        print("   Las variables se leeran del sistema operativo.")
        return
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Ignorar líneas vacías y comentarios
                if not line or line.startswith('#'):
                    continue
                
                # Parsear línea KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remover comillas si existen
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Solo establecer si no existe ya en el entorno
                    if key and key not in os.environ:
                        os.environ[key] = value
        
        print(f"[OK] Variables de entorno cargadas desde: {env_path}")
    except Exception as e:
        print(f"[!] Error al cargar .env: {e}")

# Cargar el archivo .env
_load_env_file(DOTENV_PATH)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Lee una variable de entorno.
    
    Args:
        name: Nombre de la variable
        default: Valor por defecto si no existe
        required: Si True, lanza error si la variable no existe
        
    Returns:
        Valor de la variable de entorno
        
    Raises:
        ValueError: Si required=True y la variable no existe
    """
    v = os.getenv(name, default)
    if required and not v:
        raise ValueError(f"Falta variable de entorno requerida: {name}")
    return v


# Variables comunes pre-cargadas
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=False)  # No requerida para scripts que no la necesitan
MYSQL_URL = get_env("MYSQL_URL", required=False)
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = get_env("LLM_MODEL", "gpt-4o-mini")
CHROMA_DIR = get_env("CHROMA_DIR", "./data/chroma")

# Nombres de colecciones
CHROMA_COLLECTION_SKILLS = get_env("CHROMA_COLLECTION_SKILLS", "candidates_skills")
CHROMA_COLLECTION_EXPERIENCE = get_env("CHROMA_COLLECTION_EXPERIENCE", "candidates_experience")
CHROMA_COLLECTION_EDUCATION = get_env("CHROMA_COLLECTION_EDUCATION", "candidates_education")
CHROMA_COLLECTION_LANGUAGE = get_env("CHROMA_COLLECTION_LANGUAGE", "candidates_language")
CHROMA_COLLECTION_SECTOR = get_env("CHROMA_COLLECTION_SECTOR", "candidates_sector")
CHROMA_COLLECTION_JOB_TITLE = get_env("CHROMA_COLLECTION_JOB_TITLE", "candidates_job_title")
CHROMA_COLLECTION_CITY = get_env("CHROMA_COLLECTION_CITY", "candidates_city")

CANDIDATES_TABLE = get_env("CANDIDATES_TABLE", "candidates_prepared")
