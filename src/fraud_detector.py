"""
Fraud detection: three methods + Benford's Law + hypothesis testing.

tried a few approaches here - simple z-score, unsupervised ML (isolation forest),
and rule-based logic. then combined them into an ensemble.
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare, mannwhitneyu, chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')


# ---- BENFORD'S LAW ----

def benfords_law_analysis(amounts):
    """
    checks if the first-digit distribution follows Benford's Law.
    P(d) = log10(1 + 1/d), so '1' shows up ~30% and '9' only ~4.6%.
    fraudsters usually break this without realizing it exists.
    """
    first_digits = amounts.astype(str).str.replace('.', '', 1).str.lstrip('0').str[0]
    first_digits = first_digits[first_digits.isin([str(d) for d in range(1, 10)])]
    first_digits = first_digits.astype(int)

    observed_counts = first_digits.value_counts().sort_index()
    for d in range(1, 10):
        if d not in observed_counts.index:
            observed_counts[d] = 0
    observed_counts = observed_counts.sort_index()
    observed_freq = observed_counts / observed_counts.sum()

    expected_freq = pd.Series({d: np.log10(1 + 1/d) for d in range(1, 10)})
    expected_counts = expected_freq * observed_counts.sum()
    stat, p_value = chisquare(observed_counts.values, expected_counts.values)

    return {
        'observed_freq': observed_freq,
        'expected_freq': expected_freq,
        'chi2_statistic': stat,
        'p_value': p_value,
        'conforms': p_value > 0.05,
        'observed_counts': observed_counts,
        'expected_counts': expected_counts
    }


def benfords_by_group(df, group_col='sender_upi_id', min_txns=20):
    """runs benford's per user and flags accounts that deviate significantly"""
    flagged = []
    for name, group in df.groupby(group_col):
        if len(group) < min_txns:
            continue
        result = benfords_law_analysis(group['amount'])
        if not result['conforms']:
            flagged.append({
                group_col: name,
                'txn_count': len(group),
                'chi2': round(result['chi2_statistic'], 2),
                'p_value': round(result['p_value'], 6),
                'avg_amount': round(group['amount'].mean(), 2)
            })
    return pd.DataFrame(flagged)


# ---- METHOD 1: Z-SCORE ----

def zscore_detection(df, threshold=3.0):
    """flags transactions where the amount is 3+ standard deviations from that user's average"""
    if 'amount_zscore' not in df.columns:
        user_stats = df.groupby('sender_upi_id')['amount'].agg(['mean', 'std']).reset_index()
        user_stats.columns = ['sender_upi_id', 'user_mean', 'user_std']
        user_stats['user_std'] = user_stats['user_std'].fillna(1).replace(0, 1)
        df = df.merge(user_stats, on='sender_upi_id', how='left')
        df['amount_zscore'] = (df['amount'] - df['user_mean']) / df['user_std']

    df['zscore_flag'] = (df['amount_zscore'].abs() > threshold).astype(int)
    return df


# ---- METHOD 2: ISOLATION FOREST ----

def isolation_forest_detection(df, contamination=0.03):
    """
    unsupervised ML approach - isolates anomalies through random partitioning.
    works well without labels, which is realistic since most fraud data isn't labeled.
    """
    features = ['amount', 'hour', 'txn_velocity', 'amount_zscore', 'is_night', 'is_weekend']
    available = [f for f in features if f in df.columns]

    if len(available) < 3:
        df['isolation_flag'] = 0
        return df

    X = df[available].fillna(0)
    model = IsolationForest(contamination=contamination, n_estimators=200,
                            random_state=42, n_jobs=-1)
    predictions = model.fit_predict(X)
    df['isolation_flag'] = (predictions == -1).astype(int)
    df['isolation_score'] = model.score_samples(X)
    return df


# ---- METHOD 3: RULE-BASED ----

def rule_based_detection(df):
    """manual rules based on known UPI fraud patterns from RBI reports"""
    rule1 = df['txn_velocity'] > 10 if 'txn_velocity' in df.columns else pd.Series(False, index=df.index)
    rule2 = df['amount'].between(9000, 9999)  # structuring threshold
    rule3 = (df['is_night'] == 1) & (df['amount'] > 5000) if 'is_night' in df.columns else pd.Series(False, index=df.index)
    rule4 = df['amount_zscore'].abs() > 4 if 'amount_zscore' in df.columns else pd.Series(False, index=df.index)
    rule5 = (df['time_since_last'] > 0) & (df['time_since_last'] < 30) if 'time_since_last' in df.columns else pd.Series(False, index=df.index)

    df['rule_flag'] = (rule1 | rule2 | rule3 | rule4 | rule5).astype(int)

    df['rules_triggered'] = ''
    if 'txn_velocity' in df.columns:
        df.loc[rule1, 'rules_triggered'] += 'rapid_fire,'
    df.loc[rule2, 'rules_triggered'] += 'structuring,'
    if 'is_night' in df.columns:
        df.loc[rule3, 'rules_triggered'] += 'odd_hours,'
    if 'amount_zscore' in df.columns:
        df.loc[rule4, 'rules_triggered'] += 'high_zscore,'
    if 'time_since_last' in df.columns:
        df.loc[rule5, 'rules_triggered'] += 'velocity,'

    return df


# ---- ENSEMBLE ----

def ensemble_detection(df):
    """flag if at least 2 out of 3 methods agree - reduces false positives a lot"""
    flag_cols = [c for c in ['zscore_flag', 'isolation_flag', 'rule_flag'] if c in df.columns]
    if not flag_cols:
        df['ensemble_flag'] = 0
        return df
    df['detection_count'] = df[flag_cols].sum(axis=1)
    df['ensemble_flag'] = (df['detection_count'] >= 2).astype(int)
    return df


# ---- EVALUATION ----

def evaluate_method(df, pred_col, actual_col='is_fraud', method_name=''):
    y_true = df[actual_col].values
    y_pred = df[pred_col].values
    cm = confusion_matrix(y_true, y_pred)
    return {
        'method': method_name,
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'true_positives': int(cm[1][1]) if cm.shape == (2, 2) else 0,
        'false_positives': int(cm[0][1]) if cm.shape == (2, 2) else 0,
        'true_negatives': int(cm[0][0]) if cm.shape == (2, 2) else 0,
        'false_negatives': int(cm[1][0]) if cm.shape == (2, 2) else 0,
        'confusion_matrix': cm
    }


def compare_methods(df):
    methods = {
        'Z-Score': 'zscore_flag',
        'Isolation Forest': 'isolation_flag',
        'Rule-Based': 'rule_flag',
        'Ensemble (2/3)': 'ensemble_flag'
    }
    results = []
    for name, col in methods.items():
        if col in df.columns:
            result = evaluate_method(df, col, method_name=name)
            results.append(result)
            print(f"\n{name}:")
            print(f"  precision: {result['precision']:.2%}")
            print(f"  recall:    {result['recall']:.2%}")
            print(f"  f1:        {result['f1_score']:.2%}")
            print(f"  TP={result['true_positives']} FP={result['false_positives']} "
                  f"FN={result['false_negatives']} TN={result['true_negatives']}")
    return pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in results])


# ---- HYPOTHESIS TESTS ----

def hypothesis_tests(df):
    """three statistical tests to validate the fraud patterns we injected"""
    results = []

    # test 1 - are fraud amounts actually higher on average?
    fraud_amts = df[df['is_fraud'] == 1]['amount']
    normal_amts = df[df['is_fraud'] == 0]['amount']
    stat, p = mannwhitneyu(fraud_amts, normal_amts, alternative='greater')
    results.append({
        'test': 'Fraud amounts > Normal amounts',
        'method': 'Mann-Whitney U',
        'statistic': round(stat, 2),
        'p_value': round(p, 8),
        'significant': p < 0.05,
        'interpretation': f"Fraud avg: {fraud_amts.mean():.0f} vs Normal avg: {normal_amts.mean():.0f}"
    })

    # test 2 - is night time actually higher fraud?
    if 'is_night' in df.columns:
        ct = pd.crosstab(df['is_night'], df['is_fraud'])
        chi2, p, dof, exp = chi2_contingency(ct)
        results.append({
            'test': 'Fraud rate higher at night (1-5 AM)',
            'method': 'Chi-squared',
            'statistic': round(chi2, 2),
            'p_value': round(p, 8),
            'significant': p < 0.05,
            'interpretation': 'Night fraud rate vs Day fraud rate'
        })

    # test 3 - does fraud vary across transaction types?
    ct2 = pd.crosstab(df['transaction_type'], df['is_fraud'])
    chi2, p, dof, exp = chi2_contingency(ct2)
    results.append({
        'test': 'Fraud rate varies by transaction type',
        'method': 'Chi-squared',
        'statistic': round(chi2, 2),
        'p_value': round(p, 8),
        'significant': p < 0.05,
        'interpretation': f"Across {df['transaction_type'].nunique()} types"
    })

    return pd.DataFrame(results)


# ---- RUN EVERYTHING ----

def run_full_pipeline(df):
    print("=" * 50)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 50)

    print("\n[1/6] z-score...")
    df = zscore_detection(df)
    print("[2/6] isolation forest...")
    df = isolation_forest_detection(df)
    print("[3/6] rule-based...")
    df = rule_based_detection(df)
    print("[4/6] ensemble...")
    df = ensemble_detection(df)

    print("\n[5/6] comparing methods...")
    comparison = compare_methods(df)

    print("\n[6/6] hypothesis tests...")
    hyp_results = hypothesis_tests(df)
    for _, row in hyp_results.iterrows():
        sig = "YES" if row['significant'] else "no"
        print(f"  {row['test']}: p={row['p_value']:.6f} (significant: {sig})")

    print("\n" + "=" * 50)
    print("done")
    return df, comparison, hyp_results


if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'upi_transactions_processed.csv')

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=['timestamp'])
        df, comparison, hyp_results = run_full_pipeline(df)

        output_dir = os.path.join(base_dir, 'data', 'processed')
        comparison.to_csv(os.path.join(output_dir, 'method_comparison.csv'), index=False)
        hyp_results.to_csv(os.path.join(output_dir, 'hypothesis_tests.csv'), index=False)
        df.to_csv(os.path.join(output_dir, 'upi_transactions_flagged.csv'), index=False)
        print(f"\nensemble flagged {df['ensemble_flag'].sum():,} transactions.")
    else:
        print("processed data not found. run data_cleaning.py first.")
