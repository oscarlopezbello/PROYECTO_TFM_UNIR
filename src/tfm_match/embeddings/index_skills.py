from openai import OpenAI
import chromadb
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import time
from pathlib import Path

from tfm_match.gold.text_sanitizer import sanitize_text
from tfm_match.config import (
    get_env,
    OPENAI_API_KEY,
    MYSQL_URL,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    CHROMA_COLLECTION_SKILLS,
)

# Validar variables requeridas
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY es requerida para este script")
if not MYSQL_URL:
    raise ValueError("MYSQL_URL es requerida para este script")

print("Chroma persistirá en:", CHROMA_DIR)

# ========= CONFIG =========
EMBED_MODEL = EMBEDDING_MODEL
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 0.2
COLLECTION_NAME = CHROMA_COLLECTION_SKILLS

# ========= CLIENTS =========
client = OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.Settings(
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

engine = create_engine(MYSQL_URL)

# ========= MAIN =========
def main():
    print("Cargando skills desde MySQL...")

    df = pd.read_sql(
        "SELECT id_candidate, skills FROM candidates_prepared",
        engine
    )

    # Limpieza
    df["clean_skills"] = df["skills"].fillna("").apply(sanitize_text)
    df = df[df["clean_skills"].str.len() > 5]
    df.reset_index(drop=True, inplace=True)

    texts = df["clean_skills"].tolist()
    ids = df["id_candidate"].astype(str).tolist()

    print(f"Registros a indexar: {len(texts)}")

    for i in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Indexando embeddings"
    ):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch_texts
                )

                embeddings = [e.embedding for e in response.data]

                # Crear metadatos para cada documento
                batch_metadatas = [
                    {
                        "id_candidate": id_cand,
                        "dimension": "skills",
                        "source_table": "candidates_prepared",
                        "source_cols": "skills"
                    }
                    for id_cand in batch_ids
                ]

                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
                break

            except Exception as e:
                print(f"Error batch {i}-{i+BATCH_SIZE}: {e}")
                time.sleep(2 * (attempt + 1))

        time.sleep(SLEEP_BETWEEN_BATCHES)

    print("Embeddings de SKILLS indexados correctamente")
    print(f"Colección: {COLLECTION_NAME}")
    print(f"Persistido en: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
