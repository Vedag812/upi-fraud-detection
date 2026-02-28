"""
cleans the raw data and engineers features for the fraud detection models.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath):
    print(f"loading {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    print(f"{len(df):,} rows, {len(df.columns)} columns")
    return df


def check_data_quality(df):
    print("\n--- data quality ---")
    print(f"rows: {len(df):,}")
    print(f"duplicate IDs: {df['transaction_id'].duplicated().sum()}")

    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        pct = missing[col] / len(df) * 100
        print(f"  {col}: {missing[col]:,} missing ({pct:.2f}%)")

    print(f"negative amounts: {(df['amount'] < 0).sum()}")
    print(f"zero amounts: {(df['amount'] == 0).sum()}")
    print(f"range: {df['amount'].min():.2f} to {df['amount'].max():.2f}")


def clean_data(df):
    original_count = len(df)

    df = df.drop_duplicates(subset='transaction_id', keep='first')
    df = df[df['amount'] > 0]

    # fill missing merchant category for non-P2P transactions
    df.loc[(df['merchant_category'].isna()) & (df['transaction_type'] != 'P2P'),
           'merchant_category'] = 'Unknown'

    if df['device_os'].isna().sum() > 0:
        df['device_os'] = df['device_os'].fillna(df['device_os'].mode()[0])

    df['city'] = df['city'].fillna('Unknown')

    # standardize UPI IDs
    df['sender_upi_id'] = df['sender_upi_id'].str.lower().str.strip()
    df['receiver_upi_id'] = df['receiver_upi_id'].str.lower().str.strip()
    df['city'] = df['city'].str.strip()
    df['is_fraud'] = df['is_fraud'].fillna(0).astype(int)

    print(f"cleaned: {len(df):,} rows (dropped {original_count - len(df):,})")
    return df


def add_features(df):
    print("adding features...")

    # basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['date'] = df['timestamp'].dt.date
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(1, 5).astype(int)  # 1-5 AM is the high-fraud window

    # log transform - amounts are heavily right-skewed so this helps with modeling
    df['amount_log'] = np.log1p(df['amount'])

    df = df.sort_values(['sender_upi_id', 'timestamp']).reset_index(drop=True)

    # how long since this user's previous transaction
    df['time_since_last'] = df.groupby('sender_upi_id')['timestamp'].diff().dt.total_seconds()
    df['time_since_last'] = df['time_since_last'].fillna(-1)

    # how many transactions in the same hour (velocity feature)
    df['txn_velocity'] = df.groupby(['sender_upi_id', 'date', 'hour'])['transaction_id'].transform('count')

    # per-user spending stats so we can calculate z-scores later
    user_stats = df.groupby('sender_upi_id')['amount'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['sender_upi_id', 'user_avg_amount', 'user_std_amount']
    user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(1)

    df = df.merge(user_stats, on='sender_upi_id', how='left')
    df['amount_zscore'] = (df['amount'] - df['user_avg_amount']) / df['user_std_amount']
    df['amount_zscore'] = df['amount_zscore'].fillna(0)

    # flag high value txns (top 5%)
    df['is_high_value'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    # first digit for Benford's law analysis later
    df['first_digit'] = df['amount'].astype(str).str[0].astype(int)

    return df


def save_processed(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'upi_transactions_processed.csv')
    df.to_csv(filepath, index=False)
    print(f"saved: {filepath} ({os.path.getsize(filepath) / (1024*1024):.1f} MB)")
    return filepath


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'upi_transactions_raw.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')

    if not os.path.exists(raw_path):
        print("raw data not found, run data_generator.py first.")
    else:
        df = load_raw_data(raw_path)
        check_data_quality(df)
        df = clean_data(df)
        df = add_features(df)
        check_data_quality(df)
        save_processed(df, processed_dir)
        print("\ndone.")
