import mysql.connector
import pandas as pd
from datetime import timedelta

# ---------- DB CONFIG ----------
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",
    "database": "bitcoin",
    "port": 3306,
}

TABLE = "btc_ohlc_1m"
PATTERN_TABLE = "btc_pattern_1h"
PRODUCT = "BTC-USD"


# ---------- LOAD CANDLES ----------
def load_candles():
    conn = mysql.connector.connect(**DB_CONFIG)
    query = f"""
        SELECT ts, open_price, high_price, low_price, close_price, volume
        FROM {TABLE}
        WHERE product_id = %s
        ORDER BY ts
    """
    df = pd.read_sql(query, conn, params=(PRODUCT,), parse_dates=["ts"])
    conn.close()
    df = df.sort_values("ts").set_index("ts")
    return df


# ---------- FIND ONLY move_500 = 1 SIGNALS ----------
def find_patterns_and_insert(df):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    insert_sql = f"""
        INSERT INTO {PATTERN_TABLE}
        (signal_ts, window_start_ts, delta, pattern)
        VALUES (%s, %s, %s, %s)
    """

    # compute future 1..5 minutes
    for k in range(1, 6):
        df[f"c_plus_{k}m"] = df["close_price"].shift(-k)

    df["max_future_close"] = df[[f"c_plus_{k}m" for k in range(1, 6)]].max(axis=1)
    df["delta"] = df["max_future_close"] - df["close_price"]

    # filter ONLY positive signals
    signal_df = df[df["delta"] >= 500]

    print(f"Found {len(signal_df)} move_500 = 1 patterns. Inserting...")

    batch = []

    for ts, row in signal_df.iterrows():
        window_start = ts - timedelta(minutes=60)

        # get last 60 minutes of history
        window = df.loc[window_start:ts][
            ["open_price", "high_price", "low_price", "close_price", "volume"]
        ]

        if len(window) < 60:
            continue  # incomplete hour

        pattern_json = (
            window.reset_index()
                  .rename(columns={"ts": "timestamp"})
                  .to_json(orient="records")
        )

        batch.append((
            ts.to_pydatetime(),
            window_start.to_pydatetime(),
            float(row["delta"]),
            pattern_json
        ))

        # Insert every 300 rows
        if len(batch) >= 300:
            cursor.executemany(insert_sql, batch)
            conn.commit()
            print(f"Inserted {len(batch)} patterns...")
            batch = []

    # Insert leftover rows
    if batch:
        cursor.executemany(insert_sql, batch)
        conn.commit()
        print(f"Inserted final {len(batch)} patterns.")

    cursor.close()
    conn.close()
    print("Done.")


# ---------- MAIN ----------
if __name__ == "__main__":
    print("Loading candles...")
    df = load_candles()

    print("Scanning for move_500 = 1 patterns...")
    find_patterns_and_insert(df)
