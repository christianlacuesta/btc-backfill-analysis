import os
import json
import mysql.connector
import pandas as pd
from google.cloud import bigquery

# ------------------ CONFIG ------------------

# MySQL connection settings
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",   # change if needed
    "database": "bitcoin",
    "port": 3306,
}

MYSQL_TABLE = "btc_pattern_1h"

# BigQuery settings
BQ_PROJECT = "bitcoin-480204"
BQ_DATASET = "crypto"
BQ_TABLE = "btc_pattern_1h"   # full id: bitcoin-480204.crypto.btc_pattern_1h


# ------------------ MAIN SCRIPT ------------------

def main():
    # Optional: make sure project is visible to libraries that read this env var
    os.environ["GOOGLE_CLOUD_PROJECT"] = BQ_PROJECT

    # ----- STEP 1: READ FROM MYSQL -----
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)

    query = f"""
        SELECT
            id,
            signal_ts,
            window_start_ts,
            delta,
            pattern
        FROM {MYSQL_TABLE}
    """

    # This may log "using SQLAlchemy" but works fine with the connector
    df = pd.read_sql(query, mysql_conn)
    mysql_conn.close()

    # ----- STEP 1b: NORMALIZE pattern COLUMN -----
    # Ensure pattern is valid JSON text for BigQuery (stored as STRING)
    def normalize_pattern(v):
        if v is None:
            return None

        # If already a dict/list, we keep it and serialize
        if isinstance(v, (dict, list)):
            obj = v
        else:
            # Try to parse JSON from string; if it fails, just cast to string
            try:
                obj = json.loads(v)
            except Exception:
                return str(v)

        # Compact JSON string (no spaces) to save space
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    df["pattern"] = df["pattern"].apply(normalize_pattern)

    print("Loaded rows from MySQL:", len(df))

    # ----- STEP 2: LOAD INTO BIGQUERY -----
    client = bigquery.Client(project=BQ_PROJECT)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    # BigQuery schema: pattern stored as STRING (JSON text)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # use WRITE_APPEND if you want to append
        schema=[
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("signal_ts", "TIMESTAMP"),
            bigquery.SchemaField("window_start_ts", "TIMESTAMP"),
            bigquery.SchemaField("delta", "FLOAT64"),
            bigquery.SchemaField("pattern", "STRING"),
        ],
    )

    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    load_job.result()  # wait for completion

    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}")


if __name__ == "__main__":
    main()
