import pandas as pd
import numpy as np
import os

def run_comparison():
    file_slow = "live_dashboard_feed.csv"
    file_fast = "live_dashboard_feed_fast.csv"

    print("=== OPTIONS ENGINE A/B TEST REPORT ===\n")

    if not os.path.exists(file_slow):
        print(f"❌ ERROR: Missing {file_slow}. Run live_engine.py first.")
        return
    if not os.path.exists(file_fast):
        print(f"❌ ERROR: Missing {file_fast}. Run live_engine_fast.py first.")
        return

    df_slow = pd.read_csv(file_slow)
    df_fast = pd.read_csv(file_fast)

    print("--- 1. SHAPE & COLUMN CHECK ---")
    if len(df_slow) == len(df_fast):
        print(f"✅ Row Count Matches: Both have {len(df_slow)} tickers.")
    else:
        print(f"❌ Row Count Mismatch: Slow={len(df_slow)}, Fast={len(df_fast)}")

    missing_cols = set(df_slow.columns) ^ set(df_fast.columns)
    if not missing_cols:
        print("✅ Column Architecture Matches: 100% identical structures.\n")
    else:
        print(f"❌ Column Mismatch: {missing_cols}\n")

    df_slow.set_index("Ticker", inplace=True)
    df_fast.set_index("Ticker", inplace=True)
    
    common_tickers = df_slow.index.intersection(df_fast.index)
    df_s = df_slow.loc[common_tickers].copy()
    df_f = df_fast.loc[common_tickers].copy()

    print("--- 2. CATEGORICAL / FLAG CHECK ---")
    cat_cols = ['Status', 'Flag']
    for col in cat_cols:
        if col in df_s.columns:
            matches = (df_s[col] == df_f[col]).sum()
            match_pct = (matches / len(common_tickers)) * 100
            if match_pct == 100:
                print(f"✅ {col}: 100% Exact Match")
            else:
                print(f"⚠️ {col}: {match_pct:.1f}% Match (Differences expected if outliers shifted between runs)")
    print()

    print("--- 3. NUMERIC DRIFT ANALYSIS ---")
    print("Note: Minor drift is expected due to market movement between the two script runs.\n")
    
    numeric_cols = df_s.select_dtypes(include=[np.number]).columns
    
    drift_report = []
    for col in numeric_cols:
        s_vals = df_s[col].fillna(0)
        f_vals = df_f[col].fillna(0)
        
        abs_diff = (s_vals - f_vals).abs()
        mean_drift = abs_diff.mean()
        max_drift = abs_diff.max()
        
        drift_report.append({
            "Metric": col,
            "Mean Drift": mean_drift,
            "Max Drift": max_drift
        })

    drift_df = pd.DataFrame(drift_report)
    drift_df['Mean Drift'] = drift_df['Mean Drift'].map('{:.4f}'.format)
    drift_df['Max Drift'] = drift_df['Max Drift'].map('{:.4f}'.format)
    
    print(drift_df.to_string(index=False))
    
    print("\n=== TEST COMPLETE ===")
    print("If Mean Drift is very small (e.g., < 0.50 for Spot, < 0.02 for Vol), both engines are calculating identically.")

if __name__ == "__main__":
    run_comparison()