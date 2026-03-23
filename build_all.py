import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from options_infra import OptionsClient

OUTPUT_DIR = "real_historical_options_data"
ETF_UNIVERSE = {
    "INDICES": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", "EWZ", "EEM", "EWW"],
    "BONDS": ["IEF", "TLT", "LQD", "HYG"],
    "COMMODITIES": ["USO", "XOM", "GLD", "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL"],
    "CYCLICALS": ["XLE", "XOP", "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB"],
    "TECH_INNOVATION": ["XLK", "SMH", "ARKK", "IBB", "ARKG"],
    "DEFENSIVES": ["XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]
}
ALL_TICKERS = [ticker for sublist in ETF_UNIVERSE.values() for ticker in sublist]

def fetch_risk_free_rate() -> float:
    try:
        return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100.0
    except Exception: 
        return 0.045

def pull_and_format_leg(ticker, client, start, end, target_delta, c_type, rfr, col_name):
    try:
        df = client.get_daily_rolling_by_delta(ticker, start, end, target_delta=target_delta, target_dte=30, contract_type=c_type, r=rfr)
        if df is not None and not df.empty:
            df = df[['date', 'implied_vol']].copy()
            df.rename(columns={'implied_vol': col_name}, inplace=True)
            print(f"      -> Found {col_name} ({len(df)} rows)")
            return df
    except Exception as e:
        print(f"      -> ❌ Error pulling {col_name}: {e}")
        
    print(f"      -> ⚠️ FAILED to find {col_name}.")
    return pd.DataFrame(columns=['date', col_name])

def run_master_build():
    print("Initializing Master History Builder (Resume-Capable)...")
    client = OptionsClient()
    rfr = fetch_risk_free_rate()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    today = datetime.today()
    start_date = (today - timedelta(days=50)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    for i, ticker in enumerate(ALL_TICKERS, 1):
        filename = f"{OUTPUT_DIR}/{ticker.lower()}.csv"
        
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                if len(existing_df) >= 30: 
                    print(f"[{i:02}/{len(ALL_TICKERS)}] ✅ {ticker} already exists with {len(existing_df)} rows. Skipping.")
                    continue
            except Exception:
                pass
        
        print(f"\n[{i:02}/{len(ALL_TICKERS)}] 🔄 Building {ticker}...")

        try:
            df_atm = pull_and_format_leg(ticker, client, start_date, end_date, 0.50, 'call', rfr, 'IV_ATM')
            if df_atm.empty:
                print(f"  ❌ {ticker} totally dead. No ATM options found.")
                continue

            df_put = pull_and_format_leg(ticker, client, start_date, end_date, -0.25, 'put', rfr, 'IV_PUT_WING')
            df_call = pull_and_format_leg(ticker, client, start_date, end_date, 0.25, 'call', rfr, 'IV_CALL_WING')

            merged = pd.merge(df_atm, df_put, on='date', how='outer')
            merged = pd.merge(merged, df_call, on='date', how='outer')
            
            merged['date'] = pd.to_datetime(merged['date'])
            merged = merged.sort_values('date').reset_index(drop=True)

            target_cols = ['IV_ATM', 'IV_PUT_WING', 'IV_CALL_WING']
            for col in target_cols:
                if col not in merged.columns:
                    merged[col] = pd.NA

            merged[target_cols] = merged[target_cols].ffill().bfill()
            
            merged['IV30'] = (merged['IV_ATM'] * 100).round(2)
            merged['Skew_1M_25D'] = ((merged['IV_PUT_WING'] - merged['IV_CALL_WING']) * 100).round(2)

            final_df = merged[['date', 'IV30', 'Skew_1M_25D']].copy()
            final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
            final_df.insert(1, 'Ticker', ticker)
            
            final_df = final_df.tail(45) 
            
            final_df.to_csv(filename, index=False)
            print(f"  💾 Saved {ticker} to disk with {len(final_df)} rows.")
            
            time.sleep(1)

        except Exception as e:
            print(f"  🛑 FATAL ERROR on {ticker}: {e}. Moving to next ticker.")
            continue

    print("\n--- MASTER BUILD COMPLETE ---")

if __name__ == "__main__":
    run_master_build()