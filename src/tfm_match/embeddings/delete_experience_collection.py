#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para eliminar la colección de experience en ChromaDB.
Úsalo antes de re-indexar para asegurar datos limpios.
"""

import os
from pathlib import Path
import chromadb


def get_env(name: str, default: str = None) -> str:
    return os.getenv(name, default)


def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def main():
    chroma_dir = resolve_path(get_env("CHROMA_DIR", "./data/chroma"))
    collection_name = get_env("CHROMA_COLLECTION_EXPERIENCE", "candidates_experience")

    print(f"CHROMA_DIR: {chroma_dir}")
    print(f"Intentando eliminar coleccion: {collection_name}")

    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"OK - Coleccion '{collection_name}' eliminada exitosamente")
    except Exception as e:
        print(f"WARN - No se pudo eliminar (puede que no exista): {e}")

    print("\nListo. Ahora puedes ejecutar index_experience.py para recrear la coleccion limpia.")


if __name__ == "__main__":
    main()


