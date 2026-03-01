import chromadb
from openai import OpenAI

from tfm_match.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    CHROMA_COLLECTION_SKILLS,
)

# Validar variables requeridas
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY es requerida para este script")

# CONFIG
COLLECTION_NAME = CHROMA_COLLECTION_SKILLS
EMBED_MODEL = EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma persistente
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_collection(COLLECTION_NAME)

def embed_query(text: str):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def main():
    query_text = "habilidades varias ,ofimatica atencionalcliente,servicioalcliente,ventas,microsoftexcel"

    query_embedding = embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    print("Query:", query_text)
    print("IDs candidatos:", results["ids"][0])
    print("Distancias:", results["distances"][0])

if __name__ == "__main__":
    main()
