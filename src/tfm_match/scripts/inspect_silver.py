from tfm_match.bronze.mysql import load_table
from tfm_match.silver.clean_candidates import clean_candidates

df_raw = load_table("candidates", limit=50)
df_clean = clean_candidates(df_raw)

print(df_clean.head())
print(df_clean.isna().sum())
