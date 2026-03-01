import pandas as pd
from tfm_match.bronze.mysql import load_table
from tfm_match.silver.clean_candidates import clean_candidates

def main():
    df_raw = load_table("candidates_clean", limit=1000000)  # ajusta si es muy grande
    df_silver = clean_candidates(df_raw)

    df_silver.to_parquet("data_silver_candidates.parquet", index=False)
    print("Silver guardado en data_silver_candidates.parquet")
    print("Filas:", len(df_silver), "Columnas:", df_silver.shape[1])

if __name__ == "__main__":
    main()
