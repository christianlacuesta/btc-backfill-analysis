import mysql.connector
import json
import numpy as np
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",
    "database": "bitcoin",
    "port": 3306,
}

PATTERN_TABLE = "btc_pattern_1h"
MATCH_TABLE   = "btc_pattern_self_matches"

SIM_THRESHOLD   = 0.75
SKIP_OVERLAP_TS = True


def get_mysql_connection():
    return mysql.connector.connect(**DB_CONFIG)


def load_patterns():
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
        pid      = r["id"]
        start_ts = r["window_start_ts"]
        candles  = json.loads(r["pattern"])

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
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if a.size != b.size or a.size < 2:
        return 0.0

    an = normalize(a)
    bn = normalize(b)
    denom = np.linalg.norm(an) * np.linalg.norm(bn)
    level_corr = float(np.dot(an, bn) / denom) if denom != 0 else 0.0

    da = np.diff(a)
    db = np.diff(b)
    if da.std() == 0 or db.std() == 0:
        diff_corr = 0.0
    else:
        dan = normalize(da)
        dbn = normalize(db)
        denom2 = np.linalg.norm(dan) * np.linalg.norm(dbn)
        diff_corr = float(np.dot(dan, dbn) / denom2) if denom2 != 0 else 0.0

    eps = 1e-6
    sa = np.sign(da)
    sb = np.sign(db)
    sa[np.abs(da) < eps] = 0
    sb[np.abs(db) < eps] = 0
    direction_match = float((sa == sb).mean())

    level_corr = max(0.0, level_corr)
    diff_corr  = max(0.0, diff_corr)

    score = (level_corr + diff_corr + direction_match) / 3.0
    return score


def ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end < b_start or a_start > b_end)


def find_self_matches():
    patterns = load_patterns()
    total = len(patterns)
    print(f"Loaded {total} patterns\n")

    conn = get_mysql_connection()
    cur  = conn.cursor()

    for i in range(total):
        pi = patterns[i]
        print(f"🔵 Base pattern {pi['id']} ({i+1}/{total}) – scanning {total} rows...")

        compared = 0
        matches  = 0

        for j in range(total):
            pj = patterns[j]
            compared += 1

            # progress every 1000 rows (tune this if you want)
            if compared == 1 or compared % 1000 == 0 or compared == total:
                print(f"   … progress: checked {compared}/{total} rows")

            if pi["id"] == pj["id"]:
                continue

            if pi["len"] != pj["len"]:
                continue

            if SKIP_OVERLAP_TS and ranges_overlap(
                pi["start_ts"], pi["end_ts"],
                pj["start_ts"], pj["end_ts"]
            ):
                continue

            score = shape_similarity(pi["closes"], pj["closes"])
            if score < SIM_THRESHOLD:
                continue

            pid1 = min(pi["id"], pj["id"])
            pid2 = max(pi["id"], pj["id"])

            cur.execute(
                f"""
                INSERT IGNORE INTO {MATCH_TABLE}
                    (pattern_id_1, pattern_id_2, sim_score)
                VALUES (%s, %s, %s)
                """,
                (pid1, pid2, float(score)),
            )
            conn.commit()
            matches += 1

            print(
                f"      ✅ MATCH INSERTED: base={pi['id']} vs other={pj['id']} "
                f"(stored as {pid1}, {pid2}) score={score:.4f}"
            )

        print(
            f"🟦 Finished base pattern {pi['id']}: "
            f"compared {compared} rows, found {matches} matches.\n"
        )

    cur.close()
    conn.close()


if __name__ == "__main__":
    find_self_matches()
