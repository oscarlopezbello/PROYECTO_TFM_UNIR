from sqlalchemy import create_engine
import pandas as pd
from tfm_match.config import MYSQL_URL


def get_engine():
    if not MYSQL_URL:
        raise ValueError("MYSQL_URL no configurada. Revisa el archivo .env")
    return create_engine(MYSQL_URL)


def load_table(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)
