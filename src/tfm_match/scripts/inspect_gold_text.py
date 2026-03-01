import pandas as pd
from tfm_match.gold.text_builder import build_candidate_text

df = pd.read_parquet("data_silver_candidates.parquet")
texts = build_candidate_text(df)

print("Ejemplo de texto candidato:\n")
print(texts.iloc[0][:1000])
