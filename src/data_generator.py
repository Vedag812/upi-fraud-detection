"""
generates synthetic UPI transactions that look like real ones.

calibrated the distributions against NPCI monthly reports and RBI data so the
bank shares, amounts, and timing patterns are grounded in reality.
fraud patterns are based on actual cases mentioned in RBI annual reports + news.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json

np.random.seed(42)
random.seed(42)


# -- config --
# bump NUM_TRANSACTIONS down to 100k if your machine is struggling
NUM_TRANSACTIONS = 1_000_000
NUM_USERS = 50_000
NUM_MERCHANTS = 8_000
FRAUD_RATE = 0.025  # keeping it at 2.5%, roughly what real systems see before underreporting
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

# these shares are from NPCI Q3 2024 reports
# phonepe still dominates, gpay is second, everyone else is fighting for scraps
BANKS = {
    'PhonePe': 0.47,
    'Google Pay': 0.34,
    'Paytm': 0.08,
    'CRED': 0.03,
    'Amazon Pay': 0.02,
    'WhatsApp Pay': 0.02,
    'BHIM': 0.01,
    'SBI': 0.01,
    'HDFC': 0.01,
    'ICICI': 0.01,
}

SENDER_BANKS = list(BANKS.keys())
SENDER_BANK_WEIGHTS = list(BANKS.values())

# receiver side has more variety since merchants use all kinds of banks
RECEIVER_BANKS = ['PhonePe Merchant', 'Google Pay Business', 'Paytm Business',
                  'SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'BOB', 'PNB',
                  'IndusInd', 'Yes Bank', 'Federal Bank', 'IDFC First']

TXN_TYPES = ['P2P', 'P2M', 'Bill Payment', 'Recharge', 'Investment']
TXN_TYPE_WEIGHTS = [0.35, 0.40, 0.12, 0.08, 0.05]

# tier 1 cities get more volume, that's just how it is in India
CITIES = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
    'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
    'Surat', 'Indore', 'Bhopal', 'Chandigarh', 'Kochi'
]
CITY_WEIGHTS = [0.18, 0.16, 0.14, 0.10, 0.08, 0.06, 0.06, 0.05, 0.04, 0.03,
                0.03, 0.02, 0.02, 0.02, 0.01]

# india is about 78% android
DEVICE_OS = ['Android', 'iOS']
DEVICE_WEIGHTS = [0.78, 0.22]

STATUSES = ['SUCCESS', 'FAILED', 'PENDING']
STATUS_WEIGHTS = [0.93, 0.05, 0.02]

MERCHANT_CATEGORIES = [
    'Groceries', 'Food & Dining', 'Shopping', 'Travel', 'Fuel',
    'Entertainment', 'Healthcare', 'Education', 'Utilities', 'Electronics'
]


def generate_upi_id(bank, index, is_merchant=False):
    """makes a UPI VPA that looks somewhat real"""
    first_names = ['rahul', 'priya', 'amit', 'sneha', 'vikram', 'ananya', 'rohan',
                   'pooja', 'arjun', 'meera', 'karan', 'divya', 'suresh', 'neha',
                   'aditya', 'kavita', 'deepak', 'ritu', 'manish', 'swati']

    bank_handles = {
        'PhonePe': '@ybl', 'Google Pay': '@okaxis', 'Paytm': '@paytm',
        'CRED': '@axl', 'Amazon Pay': '@apl', 'WhatsApp Pay': '@waicici',
        'BHIM': '@upi', 'SBI': '@oksbi', 'HDFC': '@okhdfcbank', 'ICICI': '@okicici'
    }

    if is_merchant:
        shop_names = ['quickmart', 'freshbasket', 'dailyneeds', 'citystore',
                      'megashop', 'starmart', 'localshop', 'smartbuy',
                      'easyshop', 'valuemart']
        name = random.choice(shop_names) + str(index % 1000)
    else:
        name = random.choice(first_names) + str(index % 10000)

    handle = bank_handles.get(bank, '@upi')
    return f"{name}{handle}"


def generate_timestamp(start, end, hour_bias=True):
    """
    picks a random timestamp between start and end.
    hour_bias gives more txns during the day (morning and evening peaks)
    which is how real gpay usage actually looks.
    """
    delta = end - start
    random_days = random.randint(0, delta.days)
    base_date = start + timedelta(days=random_days)

    if hour_bias:
        # peaks around 11am and 7-8pm
        # not many people are doing UPI transfers at 3am
        hour_weights = [
            0.005, 0.003, 0.002, 0.002, 0.003, 0.008,  # 0-5 am
            0.015, 0.030, 0.045, 0.065, 0.080, 0.085,  # 6-11 am
            0.075, 0.060, 0.050, 0.045, 0.050, 0.060,  # 12-5 pm
            0.075, 0.085, 0.080, 0.055, 0.035, 0.015   # 6-11 pm
        ]
        # normalize so it always sums to exactly 1 (floating point can be annoying)
        hour_weights_normalized = np.array(hour_weights) / sum(hour_weights)
        hour = np.random.choice(24, p=hour_weights_normalized)
    else:
        hour = random.randint(0, 23)

    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return base_date.replace(hour=hour, minute=minute, second=second)


def generate_amount(txn_type):
    """
    amount depends on what kind of transaction it is.
    using log-normal because real spending data is right-skewed
    (tons of small transactions, few big ones - just like a real dataset).
    """
    if txn_type == 'P2P':
        # splitting bills, sending money to family
        amount = np.random.lognormal(mean=6.0, sigma=1.2)
        amount = min(amount, 100000)
    elif txn_type == 'P2M':
        # chai, groceries, swiggy orders
        amount = np.random.lognormal(mean=5.5, sigma=1.0)
        amount = min(amount, 50000)
    elif txn_type == 'Bill Payment':
        # electricity, broadband, rent
        amount = np.random.lognormal(mean=7.0, sigma=0.8)
        amount = min(amount, 100000)
    elif txn_type == 'Recharge':
        # jio/airtel plans - mostly fixed amounts
        recharge_amounts = [49, 79, 99, 149, 199, 249, 299, 399, 499, 599, 699, 799, 999]
        amount = random.choice(recharge_amounts)
    elif txn_type == 'Investment':
        # SIPs, digital gold on gpay etc
        amount = np.random.lognormal(mean=7.5, sigma=0.9)
        amount = min(amount, 200000)
    else:
        amount = np.random.lognormal(mean=5.8, sigma=1.1)

    return round(max(1, amount), 2)


def inject_fraud_patterns(df, num_fraud):
    """
    inject 6 types of fraud based on what actually happens in the real world.
    all of these are from RBI reports and news articles about UPI scams.

    1. rapid fire - 10+ transactions in 5 mins (classic account takeover pattern)
    2. late night large - big transfers at 3am (stolen credentials used when victim is asleep)
    3. structuring - amounts like 9,500 to stay under the Rs 10K reporting threshold
    4. new account burst - brand new UPI ID doing 50 transactions on day 1 (mule account)
    5. geographic impossibility - same person in Mumbai and Delhi 10 mins apart
    6. behavior change - someone who normally sends 200-500 suddenly does 45k
    """
    fraud_per_type = num_fraud // 6
    fraud_rows = []
    current_idx = len(df)

    # type 1: rapid fire
    rapid_fire_users = random.sample(range(NUM_USERS), fraud_per_type // 10)
    for user_idx in rapid_fire_users:
        sender = df[df['sender_upi_id'].str.contains(str(user_idx % 10000))].index
        if len(sender) > 0:
            base_idx = sender[0]
            base_time = df.loc[base_idx, 'timestamp']
            for i in range(10):
                new_row = df.loc[base_idx].copy()
                new_row['timestamp'] = base_time + timedelta(seconds=random.randint(5, 120))
                new_row['amount'] = round(random.uniform(500, 5000), 2)
                new_row['transaction_id'] = f"UPI{new_row['timestamp'].strftime('%Y%m%d')}{current_idx:06d}"
                new_row['is_fraud'] = 1
                new_row['fraud_type'] = 'rapid_fire'
                fraud_rows.append(new_row)
                current_idx += 1

    # type 2: big transactions at weird hours
    for i in range(fraud_per_type):
        idx = random.randint(0, len(df) - 1)
        row = df.iloc[idx].copy()
        hour = random.randint(1, 4)
        new_time = row['timestamp'].replace(hour=hour, minute=random.randint(0, 59))
        new_row = row.copy()
        new_row['timestamp'] = new_time
        new_row['amount'] = round(random.uniform(15000, 95000), 2)
        new_row['is_fraud'] = 1
        new_row['fraud_type'] = 'late_night_large'
        new_row['transaction_id'] = f"UPI{new_time.strftime('%Y%m%d')}{current_idx:06d}"
        fraud_rows.append(new_row)
        current_idx += 1

    # type 3: structuring - staying just under Rs 10K
    # this is literally called "structuring" in banking/AML contexts
    structuring_users = random.sample(range(NUM_USERS), fraud_per_type // 5)
    count = 0
    for user_idx in structuring_users:
        for _ in range(5):
            if count >= fraud_per_type:
                break
            idx = random.randint(0, len(df) - 1)
            new_row = df.iloc[idx].copy()
            new_row['amount'] = round(random.uniform(9000, 9999), 2)
            new_row['is_fraud'] = 1
            new_row['fraud_type'] = 'structuring'
            new_row['transaction_id'] = f"UPI{new_row['timestamp'].strftime('%Y%m%d')}{current_idx:06d}"
            fraud_rows.append(new_row)
            current_idx += 1
            count += 1

    # type 4: mule accounts - new UPI ID, tons of activity on day 1
    burst_count = fraud_per_type // 50
    for i in range(burst_count):
        burst_date = START_DATE + timedelta(days=random.randint(30, 300))
        new_upi = f"newuser{i}@ybl"
        for j in range(50):
            idx = random.randint(0, len(df) - 1)
            new_row = df.iloc[idx].copy()
            new_row['sender_upi_id'] = new_upi
            new_row['timestamp'] = burst_date + timedelta(minutes=random.randint(0, 1440))
            new_row['amount'] = round(random.uniform(100, 8000), 2)
            new_row['is_fraud'] = 1
            new_row['fraud_type'] = 'new_account_burst'
            new_row['transaction_id'] = f"UPI{burst_date.strftime('%Y%m%d')}{current_idx:06d}"
            fraud_rows.append(new_row)
            current_idx += 1

    # type 5: geographic impossibility - same person, two cities, 10 mins apart
    distant_pairs = [('Mumbai', 'Delhi'), ('Chennai', 'Kolkata'),
                     ('Bangalore', 'Lucknow'), ('Hyderabad', 'Jaipur')]
    for i in range(fraud_per_type):
        idx = random.randint(0, len(df) - 1)
        city1, city2 = random.choice(distant_pairs)
        base_row = df.iloc[idx].copy()

        row1 = base_row.copy()
        row1['city'] = city1
        row1['is_fraud'] = 1
        row1['fraud_type'] = 'geo_impossible'
        row1['transaction_id'] = f"UPI{row1['timestamp'].strftime('%Y%m%d')}{current_idx:06d}"
        fraud_rows.append(row1)
        current_idx += 1

        row2 = base_row.copy()
        row2['city'] = city2
        row2['timestamp'] = row1['timestamp'] + timedelta(minutes=random.randint(2, 8))
        row2['is_fraud'] = 1
        row2['fraud_type'] = 'geo_impossible'
        row2['transaction_id'] = f"UPI{row2['timestamp'].strftime('%Y%m%d')}{current_idx:06d}"
        fraud_rows.append(row2)
        current_idx += 1

    # type 6: sudden behavior change - low spender suddenly does a huge transfer
    for i in range(fraud_per_type):
        idx = random.randint(0, len(df) - 1)
        new_row = df.iloc[idx].copy()
        new_row['amount'] = round(random.uniform(40000, 95000), 2)
        new_row['is_fraud'] = 1
        new_row['fraud_type'] = 'behavior_change'
        new_row['transaction_id'] = f"UPI{new_row['timestamp'].strftime('%Y%m%d')}{current_idx:06d}"
        fraud_rows.append(new_row)
        current_idx += 1

    if fraud_rows:
        df = pd.concat([df, pd.DataFrame(fraud_rows)], ignore_index=True)

    return df


def generate_transactions():
    """main function that builds the full dataset"""

    print(f"generating {NUM_TRANSACTIONS:,} transactions...")
    print(f"users: {NUM_USERS:,} | merchants: {NUM_MERCHANTS:,}")
    print(f"target fraud rate: {FRAUD_RATE*100}%")
    print("-" * 50)

    # create user profiles first then generate their transactions
    user_banks = np.random.choice(SENDER_BANKS, size=NUM_USERS, p=SENDER_BANK_WEIGHTS)
    user_upis = [generate_upi_id(user_banks[i], i) for i in range(NUM_USERS)]
    merchant_upis = [generate_upi_id(random.choice(RECEIVER_BANKS), i, is_merchant=True)
                     for i in range(NUM_MERCHANTS)]

    # most people stick to 1-2 cities
    user_cities = [np.random.choice(CITIES, p=CITY_WEIGHTS) for _ in range(NUM_USERS)]

    records = []
    print("generating base transactions...")

    for i in range(NUM_TRANSACTIONS):
        if i % 200000 == 0 and i > 0:
            print(f"  ...{i:,} done")

        user_idx = random.randint(0, NUM_USERS - 1)
        sender_upi = user_upis[user_idx]
        sender_bank = user_banks[user_idx]

        txn_type = np.random.choice(TXN_TYPES, p=TXN_TYPE_WEIGHTS)

        if txn_type == 'P2P':
            receiver_idx = random.randint(0, NUM_USERS - 1)
            receiver_upi = user_upis[receiver_idx]
            receiver_bank = user_banks[receiver_idx]
        else:
            merchant_idx = random.randint(0, NUM_MERCHANTS - 1)
            receiver_upi = merchant_upis[merchant_idx]
            receiver_bank = random.choice(RECEIVER_BANKS)

        timestamp = generate_timestamp(START_DATE, END_DATE)
        amount = generate_amount(txn_type)
        status = np.random.choice(STATUSES, p=STATUS_WEIGHTS)
        device = np.random.choice(DEVICE_OS, p=DEVICE_WEIGHTS)
        city = user_cities[user_idx]

        # 5% chance the user is traveling
        if random.random() < 0.05:
            city = np.random.choice(CITIES, p=CITY_WEIGHTS)

        merchant_cat = random.choice(MERCHANT_CATEGORIES) if txn_type != 'P2P' else None

        records.append({
            'transaction_id': f"UPI{timestamp.strftime('%Y%m%d')}{i:06d}",
            'timestamp': timestamp,
            'sender_upi_id': sender_upi,
            'receiver_upi_id': receiver_upi,
            'sender_bank': sender_bank,
            'receiver_bank': receiver_bank,
            'amount': amount,
            'transaction_type': txn_type,
            'merchant_category': merchant_cat,
            'status': status,
            'device_os': device,
            'city': city,
            'is_fraud': 0,
            'fraud_type': None
        })

    df = pd.DataFrame(records)
    print(f"base transactions done: {len(df):,}")

    num_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)
    print(f"\ninjecting {num_fraud:,} fraudulent transactions...")
    df = inject_fraud_patterns(df, num_fraud)

    # shuffle so fraud isn't all bunched at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # fix transaction IDs after sorting
    df['transaction_id'] = [f"UPI{ts.strftime('%Y%m%d')}{i:06d}"
                            for i, ts in enumerate(df['timestamp'])]

    print(f"\nfinal dataset: {len(df):,} transactions")
    print(f"fraud count: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def add_missing_values(df, missing_rate=0.015):
    """
    deliberately adding some nulls because real data is never 100% clean.
    this gives the data cleaning step something to actually do.
    """
    n = len(df)
    num_missing = int(n * missing_rate)

    # null out some merchant categories for P2P transactions
    null_idx = np.random.choice(df[df['merchant_category'].notna()].index,
                                size=min(num_missing, len(df[df['merchant_category'].notna()])),
                                replace=False)
    df.loc[null_idx, 'merchant_category'] = None

    # some missing device info
    null_idx = np.random.choice(df.index, size=num_missing // 3, replace=False)
    df.loc[null_idx, 'device_os'] = None

    # and some missing city data
    null_idx = np.random.choice(df.index, size=num_missing // 5, replace=False)
    df.loc[null_idx, 'city'] = None

    return df


def save_data(df, output_dir):
    """saves the CSV and writes a quick summary JSON"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'upi_transactions_raw.csv')
    df.to_csv(filepath, index=False)
    print(f"\nsaved: {filepath}")
    print(f"size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")

    stats = {
        'total_transactions': len(df),
        'total_fraud': int(df['is_fraud'].sum()),
        'fraud_rate': round(df['is_fraud'].mean() * 100, 2),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'unique_senders': df['sender_upi_id'].nunique(),
        'unique_receivers': df['receiver_upi_id'].nunique(),
        'avg_amount': round(df['amount'].mean(), 2),
        'median_amount': round(df['amount'].median(), 2),
        'total_value': round(df['amount'].sum(), 2),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    stats_path = os.path.join(output_dir, 'data_summary.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"summary: {stats_path}")

    return filepath


if __name__ == '__main__':
    df = generate_transactions()
    df = add_missing_values(df)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    save_data(df, output_dir)

    print("\ndone! run data_cleaning.py next.")
