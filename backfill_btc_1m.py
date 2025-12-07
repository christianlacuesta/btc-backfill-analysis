import datetime
import time
import requests
import mysql.connector

# ---------- CONFIG ----------

PRODUCT_ID   = "BTC-USD"
GRANULARITY  = 60          # 1 minute
CHUNK_HOURS  = 5           # max window per API call

BASE_URL     = "https://api.exchange.coinbase.com"

START_YEAR   = 2020        # start of history
START_MONTH  = 1           # January 2020

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "btc_user",
    "password": "BtcPass123!",
    "database": "bitcoin",
    "port": 3306,
    # remove ssl_disabled here – not needed for mysql_native_password
}


TABLE_NAME = "btc_ohlc_1m"

# ---------- API FETCH ----------

def fetch_candles_chunk(product_id, start_dt, end_dt, granularity):
    """
    Fetch candles for [start_dt, end_dt) at given granularity.
    Coinbase returns: [ time, low, high, open, close, volume ]
    time is Unix epoch (seconds).
    """
    url = f"{BASE_URL}/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "start": start_dt.replace(microsecond=0).isoformat() + "Z",
        "end":   end_dt.replace(microsecond=0).isoformat() + "Z",
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "btc-ohlc-backfill/1.0"
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()  # [ [time, low, high, open, close, volume], ... ]

    # API returns newest first – sort oldest -> newest
    data.sort(key=lambda x: x[0])
    return data

# ---------- DB INSERT ----------

def save_candles_to_mysql(candles, product_id):
    """
    Insert a batch of candles into MySQL.
    Uses ON DUPLICATE KEY UPDATE to avoid inserting duplicates.
    Assumes ts is stored as UTC (TIMESTAMP or DATETIME).
    """
    if not candles:
        return

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Important: force session timezone to UTC to avoid DST issues
    cursor.execute("SET time_zone = '+00:00'")

    sql = f"""
        INSERT INTO {TABLE_NAME}
        (product_id, ts, open_price, high_price, low_price, close_price, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price  = VALUES(open_price),
            high_price  = VALUES(high_price),
            low_price   = VALUES(low_price),
            close_price = VALUES(close_price),
            volume      = VALUES(volume)
    """

    rows = []
    for c in candles:
        unix_time, low, high, open_, close, volume = c
        # Coinbase time is in seconds since epoch (UTC)
        ts = datetime.datetime.utcfromtimestamp(unix_time)
        rows.append((
            product_id,
            ts,
            float(open_),
            float(high),
            float(low),
            float(close),
            float(volume),
        ))

    cursor.executemany(sql, rows)
    conn.commit()
    cursor.close()
    conn.close()

# ---------- RANGE FETCH HELPERS ----------

def fetch_range_and_store(start_dt, end_dt, product_id=PRODUCT_ID):
    """
    Fetch 1m candles for a given datetime range by stepping in CHUNK_HOURS,
    and store each chunk into MySQL.
    """
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = chunk_start + datetime.timedelta(hours=CHUNK_HOURS)
        if chunk_end > end_dt:
            chunk_end = end_dt

        print(f"  Fetching {chunk_start} -> {chunk_end}")
        candles = fetch_candles_chunk(product_id, chunk_start, chunk_end, GRANULARITY)
        print(f"    Got {len(candles)} candles, saving...")
        save_candles_to_mysql(candles, product_id)

        chunk_start = chunk_end
        time.sleep(0.2)  # be nice to API

def month_iterator(start_year, start_month):
    """
    Yield (month_start, month_end) for each month from start_year/start_month to now.
    month_end is the first day of the next month or 'now', whichever is earlier.
    """
    now = datetime.datetime.utcnow()

    year = start_year
    month = start_month

    while True:
        month_start = datetime.datetime(year, month, 1)

        if month_start >= now:
            break

        # compute first day of next month
        if month == 12:
            next_month = datetime.datetime(year + 1, 1, 1)
        else:
            next_month = datetime.datetime(year, month + 1, 1)

        month_end = min(next_month, now)

        yield month_start, month_end

        # advance to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

# ---------- MAIN ----------

def backfill_from_2020():
    for month_start, month_end in month_iterator(START_YEAR, START_MONTH):
        print(f"=== {month_start.strftime('%Y-%m')} ===")
        fetch_range_and_store(month_start, month_end)

if __name__ == "__main__":
    backfill_from_2020()
    print("Done backfilling BTC 1m candles from 2020.")
