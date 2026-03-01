import pandas as pd
from sqlalchemy import text

from tfm_match.gold.text_builder import build_candidate_text
from tfm_match.bronze.mysql import get_engine, load_table

def main():
    engine = get_engine()

    print("Cargando candidates_clean...")
    df = load_table("candidates_clean")

    print("Construyendo profile_text...")
    df["profile_text"] = build_candidate_text(df)

    print("Actualizando candidates_prepared...")
    with engine.begin() as conn:
        for _, row in df[["id_candidate", "profile_text"]].iterrows():
            conn.execute(
                text("""
                    UPDATE candidates_prepared
                    SET profile_text = :profile_text
                    WHERE id_candidate = :id_candidate
                """),
                {
                    "id_candidate": int(row["id_candidate"]),
                    "profile_text": row["profile_text"]
                }
            )

    print("candidates_prepared actualizado correctamente")

if __name__ == "__main__":
    main()
