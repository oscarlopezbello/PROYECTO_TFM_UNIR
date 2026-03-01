import chromadb
from pathlib import Path

CHROMA_DIR = "./data/chroma"

def main():
    # Normaliza el path para evitar confusiones de cwd
    chroma_path = str(Path(CHROMA_DIR).resolve())
    print(f"CHROMA_DIR (resolved): {chroma_path}")

    client = chromadb.PersistentClient(path=chroma_path)
    cols = client.list_collections()

    if not cols:
        print("No hay colecciones. Revisa si indexaste y/o si CHROMA_DIR coincide con el del backend.")
        return

    print("\nColecciones encontradas:")
    for c in cols:
        name = getattr(c, "name", str(c))
        col = client.get_collection(name)
        print(f"- {name} | count={col.count()}")

if __name__ == "__main__":
    main()
