# import_btc_pattern_self_matches.py

import mysql.connector
import pandas as pd
from google.cloud import bigquery

# --------------- CONFIG ---------------

# MySQL connection
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",   # change if needed
    "database": "bitcoin",
    "port": 3306,
}

MYSQL_TABLE = "btc_pattern_self_matches"

# BigQuery destination
BQ_PROJECT = "bitcoin-480204"
BQ_DATASET = "crypto"
BQ_TABLE   = "btc_pattern_self_matches"   # full id: bitcoin-480204.crypto.btc_pattern_self_matches


def main():
    # ----- STEP 1: READ FROM MYSQL -----
    conn = mysql.connector.connect(**MYSQL_CONFIG)

    query = f"""
        SELECT
            id,
            pattern_id_1,
            pattern_id_2,
            sim_score,
            created_ts
        FROM {MYSQL_TABLE}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print("Loaded rows from MySQL:", len(df))

    # ----- STEP 2: LOAD INTO BIGQUERY -----
    client = bigquery.Client(project=BQ_PROJECT)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    # Schema mapping (MySQL → BigQuery)
    # id / pattern_id_*  : BIGINT/INT  -> INT64
    # sim_score          : DOUBLE      -> FLOAT64
    # created_ts         : DATETIME    -> TIMESTAMP
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",   # overwrite; use WRITE_APPEND to append
        schema=[
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("pattern_id_1", "INT64"),
            bigquery.SchemaField("pattern_id_2", "INT64"),
            bigquery.SchemaField("sim_score", "FLOAT64"),
            bigquery.SchemaField("created_ts", "TIMESTAMP"),
        ],
    )

    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    load_job.result()  # wait for completion

    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}")


if __name__ == "__main__":
    main()
