import os
import json
import hashlib
import numpy as np
from datetime import timedelta
from google.cloud import bigquery

# ---- CONFIG -----------------------------------------------------------------

PROJECT_ID = os.getenv("PROJECT_ID", "bitcoin-480204")
DATASET = os.getenv("DATASET", "crypto")

PRODUCT_ID      = os.getenv("PRODUCT_ID", "BTC-USD")
SIM_THRESHOLD   = float(os.getenv("SIM_THRESHOLD", "0.75"))   # min similarity
MOVE_THRESHOLD  = float(os.getenv("MOVE_THRESHOLD", "500.0")) # +$500 move
WINDOW_MINUTES  = int(os.getenv("WINDOW_MINUTES", "60"))
FUTURE_MINUTES  = int(os.getenv("FUTURE_MINUTES", "5"))

OHLC_TABLE       = f"{PROJECT_ID}.{DATASET}.btc_ohlc_1m"
POS_PATTERN_TBL  = f"{PROJECT_ID}.{DATASET}.btc_pattern_1h"
NEG_PATTERN_TBL  = f"{PROJECT_ID}.{DATASET}.btc_pattern_1h_neg_from_pos"

client = bigquery.Client(project=PROJECT_ID)


# ---- HELPER FUNCTIONS -------------------------------------------------------

def get_next_id(table_fqn: str) -> int:
    """Return max(id)+1 for the given table (BigQuery has no auto-increment)."""
    query = f"SELECT IFNULL(MAX(id), 0) + 1 AS next_id FROM `{table_fqn}`"
    rows = list(client.query(query).result())
    return rows[0].next_id if rows else 1


def hash_seq(closes: np.ndarray) -> str:
    """Create a hash from closes normalized vs first close."""
    closes = np.asarray(closes, dtype=float)
    if closes.size == 0:
        return ""
    base = closes[0] if closes[0] != 0 else 1.0
    norm = [(c - base) / base for c in closes]
    s = ",".join(f"{x:.6f}" for x in norm)
    return hashlib.sha256(s.encode()).hexdigest()


def normalize(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return arr
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def shape_similarity(series_a, series_b) -> float:
    """Combined score of level, first-diff and direction similarity."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if a.size != b.size or a.size < 2:
        return 0.0

    # 1) level similarity
    an = normalize(a)
    bn = normalize(b)
    denom = np.linalg.norm(an) * np.linalg.norm(bn)
    level_corr = float(np.dot(an, bn) / denom) if denom != 0 else 0.0

    # 2) first-difference similarity
    da = np.diff(a)
    db = np.diff(b)
    if da.std() == 0 or db.std() == 0:
        diff_corr = 0.0
    else:
        dan = normalize(da)
        dbn = normalize(db)
        denom2 = np.linalg.norm(dan) * np.linalg.norm(dbn)
        diff_corr = float(np.dot(dan, dbn) / denom2) if denom2 != 0 else 0.0

    # 3) directional match
    eps = 1e-6
    sa = np.sign(da)
    sb = np.sign(db)
    sa[np.abs(da) < eps] = 0
    sb[np.abs(db) < eps] = 0
    direction_match = float((sa == sb).mean())

    # clamp negatives to zero
    level_corr = max(0.0, level_corr)
    diff_corr  = max(0.0, diff_corr)

    return (level_corr + diff_corr + direction_match) / 3.0


# ---- LOAD EXISTING POSITIVE PATTERNS ----------------------------------------

def load_patterns():
    query = f"""
        SELECT id, window_start_ts, pattern
        FROM `{POS_PATTERN_TBL}`
        ORDER BY id
    """
    rows = client.query(query).result()

    patterns = []
    for r in rows:
        if r.pattern is None:
            continue
        candles = json.loads(r.pattern)
        closes = np.array([float(c["close_price"]) for c in candles], dtype=float)
        n = len(closes)
        start_ts = r.window_start_ts
        end_ts = start_ts + timedelta(minutes=n - 1)

        patterns.append({
            "id": r.id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "closes": closes,
            "len": n,
        })

    print(f"[INFO] Loaded {len(patterns)} existing positive patterns.")
    return patterns


# ---- LOAD CURRENT 60M WINDOW + FUTURE 5M FROM BTC_OHLC_1M -------------------

def load_current_window_and_future():
    """Return signal_ts, window_start_ts, window_rows, future_rows."""
    # latest timestamp
    query_max = f"""
        SELECT MAX(ts) AS max_ts
        FROM `{OHLC_TABLE}`
        WHERE product_id = @product
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product", "STRING", PRODUCT_ID)
        ]
    )
    rows = list(client.query(query_max, job_config=job_config).result())
    if not rows or rows[0].max_ts is None:
        raise RuntimeError("No candles found for product.")

    max_ts = rows[0].max_ts
    signal_ts = max_ts - timedelta(minutes=FUTURE_MINUTES)
    window_start_ts = signal_ts - timedelta(minutes=WINDOW_MINUTES - 1)

    # fetch window + future
    query_window = f"""
        SELECT ts, open_price, high_price, low_price, close_price, volume
        FROM `{OHLC_TABLE}`
        WHERE product_id = @product
          AND ts BETWEEN @start_ts AND @end_ts
        ORDER BY ts
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product", "STRING", PRODUCT_ID),
            bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", window_start_ts),
            bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", max_ts),
        ]
    )
    rows = list(client.query(query_window, job_config=job_config).result())

    if len(rows) < WINDOW_MINUTES + FUTURE_MINUTES:
        raise RuntimeError("Not enough candles for window + future.")

    window_rows = [r for r in rows if r.ts <= signal_ts][-WINDOW_MINUTES:]
    future_rows = [r for r in rows if r.ts > signal_ts][:FUTURE_MINUTES]

    if len(window_rows) < WINDOW_MINUTES:
        raise RuntimeError("Incomplete 60-minute window.")
    if len(future_rows) < FUTURE_MINUTES:
        raise RuntimeError("Not enough future candles.")

    return signal_ts, window_start_ts, window_rows, future_rows


def compute_future_delta(window_rows, future_rows) -> float:
    current_close = float(window_rows[-1].close_price)
    max_future_close = max(float(r.close_price) for r in future_rows)
    return max_future_close - current_close


def build_pattern_json(window_rows):
    records = []
    for r in window_rows:
        records.append({
            "timestamp": r.ts.isoformat(),
            "open_price": float(r.open_price),
            "high_price": float(r.high_price),
            "low_price": float(r.low_price),
            "close_price": float(r.close_price),
            "volume": float(r.volume),
        })
    return json.dumps(records)


# ---- MAIN PATTERN CHECK + INSERT --------------------------------------------

def check_current_window_and_insert():
    patterns = load_patterns()
    if not patterns:
        print("[WARN] No existing patterns to compare against. Exiting.")
        return

    signal_ts, window_start_ts, window_rows, future_rows = load_current_window_and_future()
    print(f"[INFO] signal_ts={signal_ts}, window_start_ts={window_start_ts}")

    delta = compute_future_delta(window_rows, future_rows)
    print(f"[INFO] Future delta over next {FUTURE_MINUTES} minutes: {delta:.2f} USD")

    # compare shape vs existing positive patterns
    current_closes = np.array(
        [float(r.close_price) for r in window_rows],
        dtype=float
    )

    best_score = 0.0
    best_pattern_id = None
    for p in patterns:
        if p["len"] != len(current_closes):
            continue
        score = shape_similarity(current_closes, p["closes"])
        if score > best_score:
            best_score = score
            best_pattern_id = p["id"]

    print(f"[INFO] Best similarity score: {best_score:.4f} (pattern_id={best_pattern_id})")

    # if it doesn't look like any known pattern, do nothing
    if best_score < SIM_THRESHOLD:
        print(f"[INFO] Best score < SIM_THRESHOLD={SIM_THRESHOLD}. No insert.")
        return

    pattern_json = build_pattern_json(window_rows)
    pattern_hash = hash_seq(current_closes)

    if delta >= MOVE_THRESHOLD:
        # POSITIVE: reached +500 USD -> insert into btc_pattern_1h
        new_id = get_next_id(POS_PATTERN_TBL)
        rows_to_insert = [{
            "id": new_id,
            "signal_ts": signal_ts,
            "window_start_ts": window_start_ts,
            "delta": float(delta),
            "pattern": pattern_json,
        }]
        errors = client.insert_rows_json(POS_PATTERN_TBL, rows_to_insert)
        if errors:
            print("[ERROR] Insert errors (positive table):", errors)
        else:
            print(
                f"[OK] POSITIVE pattern inserted id={new_id}, "
                f"delta={delta:.2f}, score={best_score:.4f}, match_id={best_pattern_id}"
            )
    else:
        # NEGATIVE: pattern shape but DID NOT reach +500 -> btc_pattern_1h_neg_from_pos
        new_id = get_next_id(NEG_PATTERN_TBL)
        rows_to_insert = [{
            "id": new_id,
            "pattern_id": int(best_pattern_id) if best_pattern_id is not None else None,
            "pattern_hash": pattern_hash,
            "signal_ts": signal_ts,
            "window_start_ts": window_start_ts,
            "delta": float(delta),
            "candles": pattern_json,
        }]
        errors = client.insert_rows_json(NEG_PATTERN_TBL, rows_to_insert)
        if errors:
            print("[ERROR] Insert errors (negative table):", errors)
        else:
            print(
                f"[OK] NEGATIVE pattern inserted id={new_id}, "
                f"delta={delta:.2f}, score={best_score:.4f}, "
                f"linked_pattern_id={best_pattern_id}"
            )


if __name__ == "__main__":
    print("[INFO] BTC pattern job started.")
    check_current_window_and_insert()
    print("[INFO] BTC pattern job finished.")
