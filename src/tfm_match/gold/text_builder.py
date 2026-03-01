import pandas as pd
from tfm_match.gold.text_sanitizer import sanitize_text

TEXT_FIELDS = [
    "skills",
    "skills_body",
    "brief_description",
    "job_description1",
    "job_description2",
    "job_description3",
]

def build_candidate_text(df: pd.DataFrame) -> pd.Series:
    texts = []

    for _, row in df.iterrows():
        parts = []
        for col in TEXT_FIELDS:
            if col in df.columns and row[col]:
                parts.append(str(row[col]))

        clean_text = sanitize_text(" ".join(parts))
        texts.append(clean_text)

    return pd.Series(texts, index=df.index)
