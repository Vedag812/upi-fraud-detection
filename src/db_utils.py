"""
loads the UPI transactions into SQLite and runs 12 analytical queries.
SQL felt like the right choice here since it's how most real data teams would analyze this.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'upi_transactions.db')


def create_database(csv_path, db_path=None):
    if db_path is None:
        db_path = DB_PATH

    print(f"loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("""
        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY,
            timestamp DATETIME,
            sender_upi_id TEXT,
            receiver_upi_id TEXT,
            sender_bank TEXT,
            receiver_bank TEXT,
            amount REAL,
            transaction_type TEXT,
            merchant_category TEXT,
            status TEXT,
            device_os TEXT,
            city TEXT,
            is_fraud INTEGER,
            fraud_type TEXT
        )
    """)

    df.to_sql('transactions', conn, if_exists='replace', index=False)

    # index everything we're going to filter or group by
    cursor.execute("CREATE INDEX idx_timestamp ON transactions(timestamp)")
    cursor.execute("CREATE INDEX idx_sender ON transactions(sender_upi_id)")
    cursor.execute("CREATE INDEX idx_amount ON transactions(amount)")
    cursor.execute("CREATE INDEX idx_fraud ON transactions(is_fraud)")
    cursor.execute("CREATE INDEX idx_city ON transactions(city)")
    cursor.execute("CREATE INDEX idx_bank ON transactions(sender_bank)")

    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"loaded {count:,} rows into {db_path}")
    print(f"db size: {os.path.getsize(db_path) / (1024*1024):.1f} MB")
    conn.close()
    return db_path


def run_query(query, db_path=None):
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


def get_connection(db_path=None):
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(db_path)


# 12 queries covering the main analysis angles
QUERIES = {

    "daily_volume": """
        SELECT DATE(timestamp) as txn_date,
               COUNT(*) as txn_count,
               ROUND(SUM(amount), 2) as total_value,
               ROUND(AVG(amount), 2) as avg_value
        FROM transactions
        WHERE status = 'SUCCESS'
        GROUP BY DATE(timestamp)
        ORDER BY txn_date
    """,

    "top_senders": """
        SELECT sender_upi_id,
               COUNT(*) as txn_count,
               ROUND(SUM(amount), 2) as total_sent,
               ROUND(AVG(amount), 2) as avg_amount,
               MIN(timestamp) as first_txn,
               MAX(timestamp) as last_txn
        FROM transactions
        GROUP BY sender_upi_id
        ORDER BY txn_count DESC
        LIMIT 20
    """,

    "hourly_pattern": """
        SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
               COUNT(*) as txn_count,
               ROUND(AVG(amount), 2) as avg_amount,
               ROUND(SUM(amount), 2) as total_value
        FROM transactions
        GROUP BY hour
        ORDER BY hour
    """,

    "bank_failure_rates": """
        SELECT sender_bank,
               COUNT(*) as total_txns,
               SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failures,
               ROUND(100.0 * SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) / COUNT(*), 2) as fail_rate_pct
        FROM transactions
        GROUP BY sender_bank
        ORDER BY fail_rate_pct DESC
    """,

    # structuring pattern - amounts right under 10k
    "below_threshold": """
        SELECT transaction_id, timestamp, sender_upi_id, amount, city, is_fraud
        FROM transactions
        WHERE amount BETWEEN 9000 AND 9999
          AND status = 'SUCCESS'
        ORDER BY timestamp
        LIMIT 100
    """,

    "fraud_by_type": """
        SELECT fraud_type,
               COUNT(*) as count,
               ROUND(AVG(amount), 2) as avg_amount,
               ROUND(SUM(amount), 2) as total_amount
        FROM transactions
        WHERE is_fraud = 1
        GROUP BY fraud_type
        ORDER BY count DESC
    """,

    "city_fraud_rate": """
        SELECT city,
               COUNT(*) as total_txns,
               SUM(is_fraud) as fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_pct
        FROM transactions
        WHERE city IS NOT NULL
        GROUP BY city
        ORDER BY fraud_rate_pct DESC
    """,

    "weekend_vs_weekday": """
        SELECT 
            CASE 
                WHEN CAST(strftime('%w', timestamp) AS INTEGER) IN (0, 6) THEN 'Weekend'
                ELSE 'Weekday'
            END as day_type,
            COUNT(*) as txn_count,
            ROUND(AVG(amount), 2) as avg_amount,
            ROUND(SUM(amount), 2) as total_value
        FROM transactions
        GROUP BY day_type
    """,

    "monthly_growth": """
        SELECT strftime('%Y-%m', timestamp) as month,
               COUNT(*) as txn_count,
               ROUND(SUM(amount), 2) as total_value,
               ROUND(AVG(amount), 2) as avg_amount
        FROM transactions
        WHERE status = 'SUCCESS'
        GROUP BY month
        ORDER BY month
    """,

    "txn_type_fraud": """
        SELECT transaction_type,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_pct,
               ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2) as avg_fraud_amount,
               ROUND(AVG(CASE WHEN is_fraud = 0 THEN amount END), 2) as avg_normal_amount
        FROM transactions
        GROUP BY transaction_type
        ORDER BY fraud_rate_pct DESC
    """,

    "device_analysis": """
        SELECT device_os,
               COUNT(*) as txn_count,
               SUM(is_fraud) as fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) as fraud_rate_pct,
               ROUND(AVG(amount), 2) as avg_amount
        FROM transactions
        WHERE device_os IS NOT NULL
        GROUP BY device_os
    """,

    "high_value_transactions": """
        SELECT transaction_id, timestamp, sender_upi_id, receiver_upi_id,
               amount, transaction_type, city, is_fraud, fraud_type
        FROM transactions
        WHERE amount > 50000
        ORDER BY amount DESC
        LIMIT 50
    """
}


def run_all_queries(db_path=None):
    results = {}
    for name, query in QUERIES.items():
        try:
            results[name] = run_query(query, db_path)
            print(f"  {name}: {len(results[name])} rows")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    return results


if __name__ == '__main__':
    raw_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'data', 'raw', 'upi_transactions_raw.csv')
    if os.path.exists(raw_csv):
        create_database(raw_csv)
        print("\nrunning queries...")
        results = run_all_queries()
        print(f"\n{len(results)} queries done.")
    else:
        print(f"raw CSV not found at {raw_csv}")
        print("run data_generator.py first.")
