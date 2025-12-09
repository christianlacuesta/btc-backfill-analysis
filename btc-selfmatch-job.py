import os
import json
import numpy as np
from datetime import timedelta
from google.cloud import bigquery

# ---------------- CONFIG ----------------

# You can override these via env vars in Cloud Run if needed
PROJECT_ID = os.getenv("PROJECT_ID", "bitcoin-480204")
DATASET    = os.getenv("DATASET", "crypto")

PATTERN_TABLE = f"{PROJECT_ID}.{DATASET}.btc_pattern_1h"
MATCH_TABLE   = f"{PROJECT_ID}.{DATASET}.btc_pattern_self_matches"

# similarity threshold & overlap behavior
SIM_THRESHOLD   = float(os.getenv("SIM_THRESHOLD", "0.75"))
SKIP_OVERLAP_TS = True

client = bigquery.Client(project=PROJECT_ID)


# ---------------- NUMPY HELPERS ----------------

def normalize(series: np.ndarray) -> np.ndarray:
    """Standardize to mean 0, std 1 (or zeros if constant)."""
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return arr
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def shape_similarity(series_a, series_b) -> float:
    """
    Combined shape similarity:
      - Level similarity (correlation of normalized levels)
      - First-difference similarity (correlation of normalized deltas)
      - Directional match (up/down/flat)
    Score is average of the three, clamped to [0,1]-ish.
    """
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

    # clamp negatives so we don't reward anti-correlation
    level_corr = max(0.0, level_corr)
    diff_corr  = max(0.0, diff_corr)

    score = (level_corr + diff_corr + direction_match) / 3.0
    return score


def ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    """Return True if two [start, end] time ranges intersect."""
    return not (a_end < b_start or a_start > b_end)


# ---------------- BIGQUERY LOADERS ----------------

def load_latest_pattern():
    """
    Fetch the latest pattern (MAX id) from btc_pattern_1h.
    Returns:
      {
        "id": int,
        "start_ts": datetime,
        "end_ts": datetime,
        "closes": np.array([...]),
        "len": int,
      }
    or None if table is empty.
    """
    query = f"""
        SELECT id, window_start_ts, pattern
        FROM `{PATTERN_TABLE}`
        ORDER BY id DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        return None

    r = rows[0]
    pid      = r.id
    start_ts = r.window_start_ts
    candles  = json.loads(r.pattern)

    closes = np.array(
        [float(c["close_price"]) for c in candles],
        dtype=float
    )
    n = len(closes)
    end_ts = start_ts + timedelta(minutes=n - 1)

    return {
        "id": pid,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "closes": closes,
        "len": n,
    }


def latest_pattern_already_processed(pattern_id: int) -> bool:
    """
    Check if this pattern already appears in the match table.
    If yes, we treat it as 'already processed' and do nothing.
    """
    query = f"""
        SELECT COUNT(*) AS cnt
        FROM `{MATCH_TABLE}`
        WHERE pattern_id_1 = @pid OR pattern_id_2 = @pid
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("pid", "INT64", pattern_id)
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return rows[0].cnt > 0


def load_all_patterns_except(base_id: int):
    """
    Load all patterns except the one with id = base_id.
    Each item has the same structure as load_latest_pattern() output.
    """
    query = f"""
        SELECT id, window_start_ts, pattern
        FROM `{PATTERN_TABLE}`
        WHERE id != @base_id
        ORDER BY id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("base_id", "INT64", base_id)
        ]
    )
    rows = client.query(query, job_config=job_config).result()

    patterns = []
    for r in rows:
        pid      = r.id
        start_ts = r.window_start_ts
        candles  = json.loads(r.pattern)

        closes = np.array(
            [float(c["close_price"]) for c in candles],
            dtype=float
        )
        n = len(closes)
        end_ts = start_ts + timedelta(minutes=n - 1)

        patterns.append({
            "id": pid,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "closes": closes,
            "len": n,
        })

    return patterns


# ---------------- MAIN: SELF-MATCH LATEST PATTERN ONLY ----------------

def find_self_matches_for_latest():
    # 1) Get latest pattern
    base = load_latest_pattern()
    if base is None:
        print("[INFO] No patterns in btc_pattern_1h. Exiting.")
        return

    base_id = base["id"]
    print(f"[INFO] Latest pattern id={base_id}")

    # 2) If this pattern already appears in match table, skip
    if latest_pattern_already_processed(base_id):
        print(f"[INFO] Pattern {base_id} already processed. Nothing to do.")
        return

    # 3) Load all other patterns
    patterns = load_all_patterns_except(base_id)
    total = len(patterns)
    print(f"[INFO] Loaded {total} other patterns to compare against.")

    base_len   = base["len"]
    base_start = base["start_ts"]
    base_end   = base["end_ts"]

    matches = []
    compared = 0

    for pj in patterns:
        compared += 1
        if compared == 1 or compared % 1000 == 0 or compared == total:
            print(f"   … progress: checked {compared}/{total} rows")

        # length must match
        if pj["len"] != base_len:
            continue

        # skip overlapping time ranges if configured
        if SKIP_OVERLAP_TS and ranges_overlap(
            base_start, base_end,
            pj["start_ts"], pj["end_ts"]
        ):
            continue

        score = shape_similarity(base["closes"], pj["closes"])
        if score < SIM_THRESHOLD:
            continue

        pid1 = min(base_id, pj["id"])
        pid2 = max(base_id, pj["id"])

        matches.append({
            "pattern_id_1": int(pid1),
            "pattern_id_2": int(pid2),
            "sim_score": float(score),
        })

        print(
            f"      ✅ MATCH: base={base_id} vs other={pj['id']} "
            f"(stored as {pid1}, {pid2}) score={score:.4f}"
        )

    print(
        f"[INFO] Finished base pattern {base_id}: "
        f"compared {compared} rows, found {len(matches)} matches."
    )

    # 4) Insert matches into BigQuery
    if not matches:
        print("[INFO] No matches to insert for this new pattern.")
        return

    errors = client.insert_rows_json(MATCH_TABLE, matches)
    if errors:
        print("[ERROR] Insert errors:", errors)
    else:
        print(f"[OK] Inserted {len(matches)} rows into {MATCH_TABLE}.")


if __name__ == "__main__":
    print("[INFO] Self-match job started.")
    find_self_matches_for_latest()
    print("[INFO] Self-match job finished.")
