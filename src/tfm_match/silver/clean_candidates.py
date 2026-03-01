import pandas as pd
from tfm_match.silver.schema import TEXT_COLUMNS, NUMERIC_COLUMNS, DROP_COLUMNS

def clean_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.strip()
            )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
