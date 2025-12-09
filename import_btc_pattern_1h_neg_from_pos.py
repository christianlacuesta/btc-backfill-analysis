import os
import json
import mysql.connector
import pandas as pd
from google.cloud import bigquery

# --------------- CONFIG ---------------

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",   # change if needed
    "database": "bitcoin",
    "port": 3306,
}

MYSQL_TABLE = "btc_pattern_1h_neg_from_pos"

BQ_PROJECT = "bitcoin-480204"
BQ_DATASET = "crypto"
BQ_TABLE   = "btc_pattern_1h_neg_from_pos"


# --------------- MAIN SCRIPT ---------------

def main():

    # ----- STEP 1: READ FROM MYSQL -----
    conn = mysql.connector.connect(**MYSQL_CONFIG)

    query = f"""
        SELECT
            id,
            pattern_id,
            pattern_hash,
            signal_ts,
            window_start_ts,
            delta,
            candles
        FROM {MYSQL_TABLE}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print("Loaded rows from MySQL:", len(df))

    # ----- STEP 2: NORMALIZE JSON COLUMN -----
    def normalize_json(v):
        if v is None:
            return None
        if isinstance(v, (dict, list)):
            obj = v
        else:
            try:
                obj = json.loads(v)
            except Exception:
                return str(v)
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    df["candles"] = df["candles"].apply(normalize_json)

    # ----- STEP 3: LOAD INTO BIGQUERY -----
    client = bigquery.Client(project=BQ_PROJECT)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("pattern_id", "INT64"),
            bigquery.SchemaField("pattern_hash", "STRING"),
            bigquery.SchemaField("signal_ts", "TIMESTAMP"),
            bigquery.SchemaField("window_start_ts", "TIMESTAMP"),
            bigquery.SchemaField("delta", "FLOAT64"),
            bigquery.SchemaField("candles", "STRING"),  # JSON stored as string
        ],
    )

    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    load_job.result()

    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}")


if __name__ == "__main__":
    main()
