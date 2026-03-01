import pandas as pd
from sqlalchemy import text

from tfm_match.bronze.mysql import get_engine, load_table
from tfm_match.silver.clean_candidates import clean_candidates
from tfm_match.gold.text_builder import build_candidate_text

SRC = "candidates_clean"
DST = "candidates_prepared"

DROP_COLS = (
    ["age"]
    + [f"start_time{i}" for i in range(1, 6)]
    + [f"end_time{i}" for i in range(1, 6)]
)

FINAL_COLS = [
    "id_candidate",
    "location",
    "profile_text",
    "description",
    "brief_description",
    "skills",
    "last_grade",
    "last_grade_ordinal",
    "study_area1","study_description1","study_time1","institution_of_education1",
    "study_area2","study_description2","study_time2","institution_of_education2",
    "study_area3","study_description3","study_time3","institution_of_education3",
    "study_area4","study_description4","study_time4","institution_of_education4",
    "study_area5","study_description5","study_time5","institution_of_education5",
    "language1","language_level1","language2","language_level2",
    "language3","language_level3","language4","language_level4",
    "language5","language_level5",
    "job_name1","job_place1","job_description1","job_duration1",
    "job_name2","job_place2","job_description2","job_duration2",
    "job_name3","job_place3","job_description3","job_duration3",
    "job_name4","job_place4","job_description4","job_duration4",
    "job_name5","job_place5","job_description5","job_duration5",
]

def main():
    # 1) Leer fuente
    df = load_table(SRC, limit=1_000_000)

    # 2) Limpieza estructural
    df = clean_candidates(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # 3) Texto unificado
    df["profile_text"] = build_candidate_text(df)

    # 4) Selección final
    df = df[FINAL_COLS]

    # 5) Carga MySQL
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {DST}"))
    df.to_sql(DST, engine, if_exists="append", index=False, chunksize=2000)

    print(f"Cargados {len(df)} registros en {DST}")

if __name__ == "__main__":
    main()
