import os
import gc
import boto3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from botocore.config import Config
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv
import warnings

warnings.simplefilter(action='ignore')

load_dotenv()
S3_ID = os.getenv("MASSIVE_S3_ID")
S3_KEY = os.getenv("MASSIVE_S3_KEY")
S3_ENDPOINT = os.getenv("MASSIVE_S3_ENDPOINT")
S3_BUCKET = os.getenv("MASSIVE_S3_BUCKET")

OUTPUT_DIR = "massive_historical_options_data"
TEMP_FILE = "temp_opra_day.csv.gz"

ETF_UNIVERSE = {
    "INDICES": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", "EWZ", "EEM", "EWW"],
    "BONDS": ["IEF", "TLT", "LQD", "HYG"],
    "COMMODITIES": ["USO", "XOM", "GLD", "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL"],
    "CYCLICALS": ["XLE", "XOP", "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB"],
    "TECH_INNOVATION": ["XLK", "SMH", "ARKK", "IBB", "ARKG"],
    "DEFENSIVES": ["XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]
}
ALL_TICKERS = [ticker for sublist in ETF_UNIVERSE.values() for ticker in sublist]

class OptionsMath:
    @staticmethod
    def iv_solver(price, S, K, T, r, opt_type='call'):
        intrinsic = max(0.0, S - K) if opt_type == 'call' else max(0.0, K - S)
        if price <= intrinsic or T <= 0: return None 
        def diff(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if opt_type == 'call':
                v = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                v = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return v - price
        try:
            return brentq(diff, 1e-4, 5.0)
        except Exception:
            return None

    @staticmethod
    def black_scholes_delta(S, K, T, r, sigma, opt_type='call'):
        if T <= 0.0001 or sigma <= 0.0001 or S <= 0: return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1.0

def fetch_risk_free_rate() -> float:
    try:
        return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100.0
    except Exception: return 0.045

def run_flatfile_builder():
    print("Initializing OPRA Flatfile ETL Pipeline...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    session = boto3.Session(aws_access_key_id=S3_ID, aws_secret_access_key=S3_KEY)
    s3 = session.client('s3', endpoint_url=S3_ENDPOINT, config=Config(signature_version='s3v4'))
    
    today = datetime.today()
    start_date = today - timedelta(days=35)
    
    print("Fetching historical Spot Prices...")
    spot_df = yf.download(ALL_TICKERS, start=(start_date - timedelta(days=5)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))['Close']
    
    if isinstance(spot_df.index, pd.DatetimeIndex):
        spot_df.index = spot_df.index.tz_localize(None).normalize()
        
    rfr = fetch_risk_free_rate()

    dates_to_pull = pd.date_range(start=start_date, end=today, freq='B') 
    valid_days_processed = 0

    for current_date in dates_to_pull:
        date_str = current_date.strftime('%Y-%m-%d')
        target_ts = pd.Timestamp(current_date).normalize()
        
        object_key = f"us_options_opra/day_aggs_v1/{current_date.year}/{current_date.month:02d}/{date_str}.csv.gz"
        
        try:
            s3.download_file(S3_BUCKET, object_key, TEMP_FILE)
            print(f"\n[{date_str}] Downloaded. Processing...")
        except Exception as e:
            if '404' not in str(e):
                print(f"[{date_str}] AWS Error: {e}")
            continue

        filtered_chunks = []
        try:
            for chunk in pd.read_csv(TEMP_FILE, chunksize=250000, compression='gzip'):
                chunk['underlying'] = chunk['ticker'].str.replace('O:', '', regex=False).str[:-15].str.strip()
                
                valid_universe = chunk[chunk['underlying'].isin(ALL_TICKERS)].copy()
                
                if not valid_universe.empty:
                    s = valid_universe['ticker'].str.replace('O:', '', regex=False)
                    valid_universe['expiry'] = pd.to_datetime(s.str[-15:-9], format='%y%m%d')
                    valid_universe['type'] = s.str[-9].str.lower().map({'c': 'call', 'p': 'put'})
                    valid_universe['strike'] = s.str[-8:].astype(float) / 1000.0
                    filtered_chunks.append(valid_universe)
                    
        except Exception as e:
            print(f"[{date_str}] CSV Processing Error: {e}")
            continue

        if not filtered_chunks:
            print(f"[{date_str}] No relevant ETF data found in file.")
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            continue

        daily_df = pd.concat(filtered_chunks, ignore_index=True)
        daily_df['dte'] = (daily_df['expiry'] - target_ts).dt.days
        daily_df = daily_df[(daily_df['dte'] >= 15) & (daily_df['dte'] <= 45)]

        saved_count = 0
        
        for ticker in ALL_TICKERS:
            try:
                if target_ts in spot_df.index:
                    spot = spot_df.loc[target_ts, ticker]
                else:
                    spot = spot_df[ticker].asof(target_ts)
                
                if pd.isna(spot) or spot <= 0:
                    if ticker == 'SPY': print(f"      -> DEBUG SPY: Gate 1 Failed (Missing Spot Price for {date_str})")
                    continue
            except Exception as e: 
                if ticker == 'SPY': print(f"      -> DEBUG SPY: Gate 1 Exception: {e}")
                continue 
            
            t_df = daily_df[daily_df['underlying'] == ticker].copy()
            if t_df.empty:
                if ticker == 'SPY': print(f"      -> DEBUG SPY: Gate 2 Failed (No strings matched 'SPY' after stripping spaces)")
                continue
            
            t_df = t_df[t_df['close'] > 0]
            vols = []
            
            for _, row in t_df.iterrows():
                iv = OptionsMath.iv_solver(row['close'], spot, row['strike'], row['dte']/365.0, rfr, row['type'])
                if iv and 0.01 < iv < 1.20:
                    delta = OptionsMath.black_scholes_delta(spot, row['strike'], row['dte']/365.0, rfr, iv, row['type'])
                    vols.append({'type': row['type'], 'delta': delta, 'iv': iv})
            
            if not vols:
                if ticker == 'SPY': print(f"      -> DEBUG SPY: Gate 3 Failed (Found contracts, but IV Solver returned None for all. Spot={spot}, Example Strike={t_df['strike'].iloc[0]})")
                continue
                
            v_df = pd.DataFrame(vols)
            
            def get_closest_iv(target_delta, opt_type):
                f = v_df[v_df['type'] == opt_type]
                if f.empty: return None
                return f.iloc[(f['delta'] - target_delta).abs().argmin()]['iv']

            atm_iv = get_closest_iv(0.50, 'call')
            put_iv = get_closest_iv(-0.25, 'put')
            call_iv = get_closest_iv(0.25, 'call')
            
            if not atm_iv:
                if ticker == 'SPY': print(f"      -> DEBUG SPY: Gate 4 Failed (Solved IVs, but couldn't find a Call near 0.50 Delta)")
                continue
            
            skew = round((put_iv - call_iv) * 100, 2) if (put_iv and call_iv) else pd.NA
            new_row = pd.DataFrame([{
                "date": date_str,
                "Ticker": ticker,
                "IV30": round(atm_iv * 100, 2),
                "Skew_1M_25D": skew
            }])
            
            filename = f"{OUTPUT_DIR}/{ticker.lower()}.csv"
            existing_df = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()
            combined = pd.concat([existing_df, new_row], ignore_index=True).drop_duplicates(subset=['date'], keep='last').sort_values('date').tail(45)
            combined.to_csv(filename, index=False)
            saved_count += 1

        print(f"[{date_str}] ✅ Done. Saved {saved_count} tickers.")
        
        if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
        gc.collect() 
        valid_days_processed += 1

    print(f"\n--- ETL PIPELINE COMPLETE. Built {valid_days_processed} trading days. ---")

if __name__ == "__main__":
    run_flatfile_builder()