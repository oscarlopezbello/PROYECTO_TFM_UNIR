from tfm_match.bronze.mysql import load_table


df = load_table("candidates", limit=10)

print(df.head())
print("\nColumnas:", df.columns.tolist())
print("\nTipos:\n", df.dtypes)
print("\nNulos:\n", df.isna().sum())
