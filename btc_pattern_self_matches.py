import mysql.connector
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

# --- DB config -------------------------------------------------
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",
    "database": "bitcoin",
    "port": 3306,
}

PATTERN_TABLE = "btc_pattern_1h"
MATCH_TABLE   = "btc_pattern_self_matches"

# similarity settings
SIM_THRESHOLD   = 0.85    # 0..1, raise/lower as you like
SKIP_OVERLAP_TS = True    # skip pairs whose time windows overlap


# ---------------------------------------------------------------
def get_mysql_connection():
    return mysql.connector.connect(**DB_CONFIG)


def load_patterns():
    """
    Load all patterns (id, window_start_ts, pattern JSON)
    and precompute close arrays + length.
    """
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        f"""
        SELECT id, window_start_ts, pattern
        FROM {PATTERN_TABLE}
        ORDER BY id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    patterns = []
    for r in rows:
        pid = r["id"]
        start_ts = r["window_start_ts"]
        candles = json.loads(r["pattern"])
        closes = np.array([float(c["close_price"]) for c in candles],
                          dtype=float)
        n = len(closes)

        # compute the time range for this pattern (1 candle per minute)
        end_ts = start_ts + timedelta(minutes=n - 1)

        patterns.append({
            "id": pid,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "closes": closes,
            "len": n,
        })

    return patterns


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
    """
    Combined shape score (0..1) using:
      - correlation of normalized levels
      - correlation of normalized first differences
      - ratio of matching move directions
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if a.size != b.size or a.size < 2:
        return 0.0

    # 1) level correlation
    an = normalize(a)
    bn = normalize(b)
    denom = np.linalg.norm(an) * np.linalg.norm(bn)
    if denom == 0:
        level_corr = 0.0
    else:
        level_corr = float(np.dot(an, bn) / denom)

    # 2) diff correlation
    da = np.diff(a)
    db = np.diff(b)
    if da.std() == 0 or db.std() == 0:
        diff_corr = 0.0
    else:
        dan = normalize(da)
        dbn = normalize(db)
        denom2 = np.linalg.norm(dan) * np.linalg.norm(dbn)
        if denom2 == 0:
            diff_corr = 0.0
        else:
            diff_corr = float(np.dot(dan, dbn) / denom2)

    # 3) direction match ratio
    eps = 1e-6
    sa = np.sign(da)
    sb = np.sign(db)
    sa[np.abs(da) < eps] = 0
    sb[np.abs(db) < eps] = 0
    direction_match = float((sa == sb).mean())

    # clamp negative correlations to 0
    level_corr = max(0.0, level_corr)
    diff_corr  = max(0.0, diff_corr)

    score = (level_corr + diff_corr + direction_match) / 3.0
    return score


def ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    """Return True if [a_start, a_end] intersects [b_start, b_end]."""
    return not (a_end < b_start or a_start > b_end)


def insert_match(cur, pattern_id_1, pattern_id_2, score):
    cur.execute(
        f"""
        INSERT IGNORE INTO {MATCH_TABLE}
            (pattern_id_1, pattern_id_2, sim_score)
        VALUES (%s, %s, %s)
        """,
        (pattern_id_1, pattern_id_2, float(score)),
    )


# ---------------------------------------------------------------
def find_self_matches():
    patterns = load_patterns()
    total = len(patterns)
    print(f"Loaded {total} patterns")

    # group by length so we only compare same-sized patterns
    groups = {}
    for p in patterns:
        groups.setdefault(p["len"], []).append(p)

    conn = get_mysql_connection()
    cur  = conn.cursor()

    for length, group in groups.items():
        n = len(group)
        if n < 2:
            continue

        print(f"\n--- Length {length} candles: {n} patterns ---")

        # pairwise i < j inside this length group
        num_pairs = n * (n - 1) // 2
        pbar = tqdm(total=num_pairs, desc=f"len={length}")

        for i in range(n):
            pi = group[i]
            for j in range(i + 1, n):
                pj = group[j]
                pbar.update(1)

                # optional: skip overlapping time ranges
                if SKIP_OVERLAP_TS and ranges_overlap(
                        pi["start_ts"], pi["end_ts"],
                        pj["start_ts"], pj["end_ts"]):
                    continue

                score = shape_similarity(pi["closes"], pj["closes"])
                if score < SIM_THRESHOLD:
                    continue

                insert_match(cur, pi["id"], pj["id"], score)
                conn.commit()

                tqdm.write(
                    f"[MATCH] {pi['id']} vs {pj['id']} | "
                    f"len={length} | score={score:.3f}"
                )

        pbar.close()

    cur.close()
    conn.close()


if __name__ == "__main__":
    find_self_matches()
