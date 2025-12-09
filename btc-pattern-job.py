import os
import json
import hashlib
import numpy as np
from datetime import timedelta
from google.cloud import bigquery

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

PROJECT_ID = os.getenv("PROJECT_ID", "bitcoin-480204")
DATASET    = os.getenv("DATASET", "crypto")

PRODUCT_ID      = os.getenv("PRODUCT_ID", "BTC-USD")
SIM_THRESHOLD   = float(os.getenv("SIM_THRESHOLD", "0.75"))
MOVE_THRESHOLD  = float(os.getenv("MOVE_THRESHOLD", "500.0"))  # +USD
WINDOW_MINUTES  = int(os.getenv("WINDOW_MINUTES", "60"))
FUTURE_MINUTES  = int(os.getenv("FUTURE_MINUTES", "5"))

OHLC_TABLE_FQN      = f"{PROJECT_ID}.{DATASET}.btc_ohlc_1m"
POS_PATTERN_FQN     = f"{PROJECT_ID}.{DATASET}.btc_pattern_1h"
NEG_PATTERN_FQN     = f"{PROJECT_ID}.{DATASET}.btc_pattern_1h_neg_from_pos"
SELF_MATCH_FQN      = f"{PROJECT_ID}.{DATASET}.btc_pattern_self_matches"

client = bigquery.Client(project=PROJECT_ID)

# -------------------------------------------------------------------
# GENERIC HELPERS
# -------------------------------------------------------------------

def get_next_id(table_fqn: str) -> int:
    """Return max(id)+1 for a table that uses INTEGER id."""
    query = f"SELECT IFNULL(MAX(id), 0) + 1 AS next_id FROM `{table_fqn}`"
    rows = list(client.query(query).result())
    return rows[0].next_id if rows else 1


def hash_seq(closes):
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
    """Combined similarity: level, first-diff, and direction agreement."""
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

    # 3) directional agreement
    eps = 1e-6
    sa = np.sign(da)
    sb = np.sign(db)
    sa[np.abs(da) < eps] = 0
    sb[np.abs(db) < eps] = 0
    direction_match = float((sa == sb).mean())

    # clamp negatives
    level_corr = max(0.0, level_corr)
    diff_corr  = max(0.0, diff_corr)

    return (level_corr + diff_corr + direction_match) / 3.0


def ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end < b_start or a_start > b_end)

# -------------------------------------------------------------------
# LOAD EXISTING POSITIVE PATTERNS (for similarity / negative-from-pos)
# -------------------------------------------------------------------

def load_existing_patterns():
    """Load all positive patterns for similarity comparison."""
    query = f"""
        SELECT id, window_start_ts, pattern
        FROM `{POS_PATTERN_FQN}`
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

# -------------------------------------------------------------------
# LOAD CURRENT 60M WINDOW + FUTURE 5M FROM btc_ohlc_1m
# -------------------------------------------------------------------

def load_current_window_and_future():
    """Return (signal_ts, window_start_ts, window_rows, future_rows)."""
    # latest timestamp for given product
    query_max = f"""
        SELECT MAX(ts) AS max_ts
        FROM `{OHLC_TABLE_FQN}`
        WHERE product_id = @product
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product", "STRING", PRODUCT_ID)
        ]
    )
    rows = list(client.query(query_max, job_config=job_config).result())
    if not rows or rows[0].max_ts is None:
        raise RuntimeError("No candles found for product in btc_ohlc_1m.")

    max_ts = rows[0].max_ts
    signal_ts = max_ts - timedelta(minutes=FUTURE_MINUTES)
    window_start_ts = signal_ts - timedelta(minutes=WINDOW_MINUTES - 1)

    # window + 5m future
    query_window = f"""
        SELECT ts, open_price, high_price, low_price, close_price, volume
        FROM `{OHLC_TABLE_FQN}`
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
        raise RuntimeError("Not enough candles for 60m window + future.")

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

# -------------------------------------------------------------------
# SELF-MATCH: only for a given (new) pattern_id
# -------------------------------------------------------------------

def run_self_match_for_pattern_id(pattern_id: int):
    """Self-match a single positive pattern against all others."""
    print(f"[SELF] Starting self-match for pattern id={pattern_id}")

    # load base pattern
    base_query = f"""
        SELECT id, window_start_ts, pattern
        FROM `{POS_PATTERN_FQN}`
        WHERE id = @pid
        LIMIT 1
    """
    base_rows = list(
        client.query(
            base_query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("pid", "INT64", pattern_id)
                ]
            ),
        ).result()
    )
    if not base_rows:
        print(f"[SELF] Pattern id={pattern_id} not found, skip self-match.")
        return

    base_row = base_rows[0]
    base_start = base_row.window_start_ts
    base_candles = json.loads(base_row.pattern)
    base_closes = np.array(
        [float(c["close_price"]) for c in base_candles], dtype=float
    )
    base_len = len(base_closes)
    base_end = base_start + timedelta(minutes=base_len - 1)

    # load all other patterns
    others_query = f"""
        SELECT id, window_start_ts, pattern
        FROM `{POS_PATTERN_FQN}`
        WHERE id != @pid
        ORDER BY id
    """
    others = client.query(
        others_query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("pid", "INT64", pattern_id)
            ]
        ),
    ).result()

    matches = []
    total = 0
    for r in others:
        total += 1
        other_id = r.id
        other_start = r.window_start_ts
        ocandles = json.loads(r.pattern)
        ocloses = np.array(
            [float(c["close_price"]) for c in ocandles], dtype=float
        )
        olen = len(ocloses)
        other_end = other_start + timedelta(minutes=olen - 1)

        if olen != base_len:
            continue

        if ranges_overlap(base_start, base_end, other_start, other_end):
            continue

        score = shape_similarity(base_closes, ocloses)
        if score < SIM_THRESHOLD:
            continue

        pid1 = min(pattern_id, other_id)
        pid2 = max(pattern_id, other_id)

        matches.append(
            {
                "pattern_id_1": int(pid1),
                "pattern_id_2": int(pid2),
                "sim_score": float(score),
            }
        )

        print(
            f"[SELF] match base={pattern_id} vs other={other_id} "
            f"(stored {pid1},{pid2}) score={score:.4f}"
        )

    if not matches:
        print(f"[SELF] No self-matches found for pattern id={pattern_id}.")
        return

    errors = client.insert_rows_json(SELF_MATCH_FQN, matches)
    if errors:
        print("[SELF] insert_rows_json errors:", errors)
    else:
        print(
            f"[SELF] Inserted {len(matches)} self-match rows for pattern id={pattern_id}."
        )

# -------------------------------------------------------------------
# MAIN: classify current window and insert + self-match
# -------------------------------------------------------------------

def process_current_window():
    # load historical patterns for similarity / negative-from-pos
    patterns = load_existing_patterns()

    signal_ts, window_start_ts, window_rows, future_rows = (
        load_current_window_and_future()
    )
    print(f"[INFO] signal_ts={signal_ts}, window_start_ts={window_start_ts}")

    delta = compute_future_delta(window_rows, future_rows)
    print(
        f"[INFO] Future delta over next {FUTURE_MINUTES} minutes: {delta:.2f} USD"
    )

    # if no existing patterns, we cannot compute similarity-based label
    if not patterns:
        print("[INFO] No existing patterns to compare with. Skipping.")
        return

    current_closes = np.array(
        [float(r.close_price) for r in window_rows], dtype=float
    )

    # best similarity against existing positive patterns
    best_score = 0.0
    best_pattern_id = None
    for p in patterns:
        if p["len"] != len(current_closes):
            continue
        score = shape_similarity(current_closes, p["closes"])
        if score > best_score:
            best_score = score
            best_pattern_id = p["id"]

    print(
        f"[INFO] Best similarity score: {best_score:.4f} "
        f"(pattern_id={best_pattern_id})"
    )

    if best_score < SIM_THRESHOLD:
        print(
            f"[INFO] Best score < SIM_THRESHOLD={SIM_THRESHOLD}. "
            f"No positive/negative-from-pos insert."
        )
        return

    pattern_json = build_pattern_json(window_rows)
    pattern_hash = hash_seq(current_closes)

    if delta >= MOVE_THRESHOLD:
        # ------------------ POSITIVE PATTERN ------------------
        new_id = get_next_id(POS_PATTERN_FQN)
        rows_to_insert = [
            {
                "id": new_id,
                "signal_ts": signal_ts,
                "window_start_ts": window_start_ts,
                "delta": float(delta),
                "pattern": pattern_json,
            }
        ]
        errors = client.insert_rows_json(POS_PATTERN_FQN, rows_to_insert)
        if errors:
            print("[ERROR] Insert errors (positive table):", errors)
            return

        print(
            f"[OK] POSITIVE pattern inserted id={new_id}, "
            f"delta={delta:.2f}, score={best_score:.4f}, match_id={best_pattern_id}"
        )

        # --------- NEW: run self match for this new positive pattern ---------
        run_self_match_for_pattern_id(new_id)

    else:
        # ------------------ NEGATIVE FROM POSITIVE ------------------
        new_id = get_next_id(NEG_PATTERN_FQN)
        rows_to_insert = [
            {
                "id": new_id,
                "pattern_id": int(best_pattern_id)
                if best_pattern_id is not None
                else None,
                "pattern_hash": pattern_hash,
                "signal_ts": signal_ts,
                "window_start_ts": window_start_ts,
                "delta": float(delta),
                "candles": pattern_json,
            }
        ]
        errors = client.insert_rows_json(NEG_PATTERN_FQN, rows_to_insert)
        if errors:
            print("[ERROR] Insert errors (negative table):", errors)
        else:
            print(
                f"[OK] NEGATIVE pattern inserted id={new_id}, "
                f"delta={delta:.2f}, score={best_score:.4f}, "
                f"linked_pattern_id={best_pattern_id}"
            )


if __name__ == "__main__":
    print("[INFO] BTC pattern + self-match job started.")
    process_current_window()
    print("[INFO] BTC pattern + self-match job finished.")
