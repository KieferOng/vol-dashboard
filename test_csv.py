import os
import pandas as pd

HIST_DIR = "massive_historical_options_data"
ETF_UNIVERSE = {
    "INDICES": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", "EWZ", "EEM", "EWW"],
    "BONDS": ["IEF", "TLT", "LQD", "HYG"],
    "COMMODITIES": ["USO", "XOM", "GLD", "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL"],
    "CYCLICALS": ["XLE", "XOP", "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB"],
    "TECH_INNOVATION": ["XLK", "SMH", "ARKK", "IBB", "ARKG"], # Removed ARKX, HACK, ROBO
    "DEFENSIVES": ["XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]
}
ALL_TICKERS = [ticker for sublist in ETF_UNIVERSE.values() for ticker in sublist]

def get_max_streak(series):
    if series.empty: return 0
    return (series.groupby((series != series.shift()).cumsum()).transform('size')).max()

def check_health():
    print(f"--- 📊 CSV Health Diagnostic ({HIST_DIR}) ---")
    results = []
    
    for ticker in ALL_TICKERS:
        file_path = f"{HIST_DIR}/{ticker.lower()}.csv"
        if not os.path.exists(file_path): 
            results.append({
                "Ticker": ticker, "Status": "🔴 MISSING", "IV (Lo/Avg/Hi)": "N/A", 
                "Skew (Lo/Avg/Hi)": "N/A", "Flag": "No CSV found"
            })
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                results.append({
                    "Ticker": ticker, "Status": "🔴 EMPTY", "IV (Lo/Avg/Hi)": "N/A", 
                    "Skew (Lo/Avg/Hi)": "N/A", "Flag": "File is empty"
                })
                continue
            
            window = df.tail(22).copy()
            rows_in_window = len(window)
            
            if rows_in_window < 22:
                results.append({
                    "Ticker": ticker, 
                    "Status": "🔴 INCOMPLETE", 
                    "IV (Lo/Avg/Hi)": "N/A", 
                    "Skew (Lo/Avg/Hi)": "N/A", 
                    "Flag": f"Only {rows_in_window}/22 days available"
                })
                continue

            nan_count = window[['IV30', 'Skew_1M_25D']].isna().sum().sum()
            if nan_count > 0:
                results.append({
                    "Ticker": ticker, 
                    "Status": "🔴 CRITICAL", 
                    "IV (Lo/Avg/Hi)": "N/A", 
                    "Skew (Lo/Avg/Hi)": "N/A", 
                    "Flag": f"Contains {nan_count} NaN values in the 22-day window"
                })
                continue
                
            iv_min, iv_mean, iv_max = window['IV30'].min(), window['IV30'].mean(), window['IV30'].max()
            sk_min, sk_mean, sk_max = window['Skew_1M_25D'].min(), window['Skew_1M_25D'].mean(), window['Skew_1M_25D'].max()
            
            iv_streak = get_max_streak(window['IV30'])
            skew_streak = get_max_streak(window['Skew_1M_25D'])
            worst_streak = max(iv_streak, skew_streak)
            
            iv_unq = window['IV30'].nunique() / rows_in_window
            skew_unq = window['Skew_1M_25D'].nunique() / rows_in_window
            worst_unq = min(iv_unq, skew_unq)

            issues = []
            if worst_streak >= 7 or worst_unq < 0.3:
                status = "🔴 CRITICAL"
                if worst_streak >= 7: issues.append(f"Dead Block ({worst_streak}d)")
                if worst_unq < 0.3: issues.append(f"Low Real Data ({worst_unq*100:.0f}%)")
            elif worst_streak >= 4 or worst_unq < 0.6:
                status = "🟡 WARNING"
                if worst_streak >= 4: issues.append(f"Streak ({worst_streak}d)")
                if worst_unq < 0.6: issues.append(f"Medium Real Data ({worst_unq*100:.0f}%)")
            else:
                status = "🟢 HEALTHY"
                issues.append("Clean 22-Day History")
                
            results.append({
                "Ticker": ticker,
                "Status": status,
                "IV (Lo/Avg/Hi)": f"{iv_min:.1f} / {iv_mean:.1f} / {iv_max:.1f}",
                "Skew (Lo/Avg/Hi)": f"{sk_min:.1f} / {sk_mean:.1f} / {sk_max:.1f}",
                "Flag": " | ".join(issues)
            })
            
        except Exception as e:
             results.append({"Ticker": ticker, "Status": "🔴 ERROR", "IV (Lo/Avg/Hi)": "ERR", "Skew (Lo/Avg/Hi)": "ERR", "Flag": str(e)})

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    print("\n--- Summary Statistics ---")
    print(f"Total Evaluated: {len(df_results)}")
    print(f"🟢 Healthy: {len(df_results[df_results['Status'] == '🟢 HEALTHY'])}")
    print(f"🟡 Warning: {len(df_results[df_results['Status'] == '🟡 WARNING'])}")
    print(f"🔴 Critical: {len(df_results[df_results['Status'] == '🔴 CRITICAL'])}")
    print(f"🔴 Incomplete/Missing: {len(df_results[df_results['Status'].isin(['🔴 INCOMPLETE', '🔴 MISSING'])])}")

if __name__ == "__main__":
    check_health()