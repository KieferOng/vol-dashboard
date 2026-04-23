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
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.simplefilter(action='ignore')

S3_ID = os.getenv("MASSIVE_S3_ID")
S3_KEY = os.getenv("MASSIVE_S3_KEY")
S3_ENDPOINT = os.getenv("MASSIVE_S3_ENDPOINT")
S3_BUCKET = os.getenv("MASSIVE_S3_BUCKET")

OUTPUT_DIR = "massive_historical_1y_data"
TEMP_FILE = "temp_opra_daily_upgrade.csv.gz"

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
            if opt_type == 'call': v = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else: v = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return v - price
        try: return brentq(diff, 1e-4, 5.0)
        except Exception: return None

    @staticmethod
    def black_scholes_delta(S, K, T, r, sigma, opt_type='call'):
        if T <= 0.0001 or sigma <= 0.0001 or S <= 0: return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1.0

def fetch_risk_free_rate(target_date) -> float:
    try: 
        rfr_df = yf.Ticker("^IRX").history(start=(target_date - timedelta(days=5)), end=(target_date + timedelta(days=1)))
        return rfr_df['Close'].iloc[-1] / 100.0
    except Exception: 
        return 0.045

def run_daily_updater():
    print("Initializing Daily OPRA Updater...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    session = boto3.Session(aws_access_key_id=S3_ID, aws_secret_access_key=S3_KEY)
    s3 = session.client('s3', endpoint_url=S3_ENDPOINT, config=Config(signature_version='s3v4'))
    
    today = pd.Timestamp.today().normalize()
    target_date = today - pd.tseries.offsets.BDay(1)
    date_str = target_date.strftime('%Y-%m-%d')
    print(f"Targeting end-of-day data for: {date_str}")
    
    start_d = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
    end_d = (target_date + timedelta(days=2)).strftime('%Y-%m-%d')
    spot_df = yf.download(ALL_TICKERS, start=start_d, end=end_d)['Close']
    
    if isinstance(spot_df.columns, pd.MultiIndex):
        try: spot_df.columns = spot_df.columns.get_level_values(1)
        except: pass
        
    if isinstance(spot_df.index, pd.DatetimeIndex):
        spot_df.index = pd.to_datetime(spot_df.index).tz_localize(None).normalize()
    
    rfr = fetch_risk_free_rate(target_date)

    object_key = f"us_options_opra/day_aggs_v1/{target_date.year}/{target_date.month:02d}/{date_str}.csv.gz"
    
    try:
        print("Downloading S3 File...")
        s3.download_file(S3_BUCKET, object_key, TEMP_FILE)
    except Exception as e:
        print(f"Failed to download data for {date_str}. It may not be uploaded yet. ({e})")
        return

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
        print(f"Error parsing chunks: {e}")
        return

    if not filtered_chunks:
        if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
        print("No valid ticker data found in file.")
        return

    daily_df = pd.concat(filtered_chunks, ignore_index=True)
    daily_df['dte'] = (daily_df['expiry'] - target_date).dt.days
    
    daily_df = daily_df[((daily_df['dte'] >= 15) & (daily_df['dte'] <= 45)) | ((daily_df['dte'] >= 75) & (daily_df['dte'] <= 105))].copy()
    daily_df = daily_df[daily_df['close'] > 0]

    saved_count = 0
    for ticker in ALL_TICKERS:
        try:
            if target_date in spot_df.index: raw_spot = spot_df.loc[target_date, ticker]
            else: raw_spot = spot_df[ticker].asof(target_date)
            
            if isinstance(raw_spot, pd.Series): spot = float(raw_spot.iloc[-1])
            else: spot = float(raw_spot)
                
            if pd.isna(spot) or spot <= 0: continue
        except Exception: 
            continue 
        
        t_df = daily_df[daily_df['underlying'] == ticker]
        if t_df.empty: continue
        
        vols_30, vols_90 = [], []
        for _, row in t_df.iterrows():
            iv = OptionsMath.iv_solver(row['close'], spot, row['strike'], row['dte']/365.0, rfr, row['type'])
            if iv and 0.01 < iv < 1.20:
                delta = OptionsMath.black_scholes_delta(spot, row['strike'], row['dte']/365.0, rfr, iv, row['type'])
                dp = {'type': row['type'], 'delta': delta, 'iv': iv}
                if 15 <= row['dte'] <= 45: vols_30.append(dp)
                else: vols_90.append(dp)
        
        v30_df = pd.DataFrame(vols_30)
        v90_df = pd.DataFrame(vols_90) if vols_90 else pd.DataFrame()
        
        def get_closest_iv(df_sub, target_delta, opt_type):
            if df_sub.empty: return None
            f = df_sub[df_sub['type'] == opt_type]
            if f.empty: return None
            return f.iloc[(f['delta'] - target_delta).abs().argmin()]['iv']

        atm_iv_30 = get_closest_iv(v30_df, 0.50, 'call')
        put_iv_30 = get_closest_iv(v30_df, -0.25, 'put')
        call_iv_30 = get_closest_iv(v30_df, 0.25, 'call')
        
        if atm_iv_30 is None: continue
        
        atm_iv_90 = get_closest_iv(v90_df, 0.50, 'call')
        put_iv_90 = get_closest_iv(v90_df, -0.25, 'put')
        call_iv_90 = get_closest_iv(v90_df, 0.25, 'call')
        
        p_c_ratio_30 = round(put_iv_30 / call_iv_30, 4) if (put_iv_30 and call_iv_30) else np.nan
        c_atm_ratio_30 = round(call_iv_30 / atm_iv_30, 4) if (call_iv_30 and atm_iv_30) else np.nan
        p_atm_ratio_30 = round(put_iv_30 / atm_iv_30, 4) if (put_iv_30 and atm_iv_30) else np.nan

        p_c_ratio_90 = round(put_iv_90 / call_iv_90, 4) if (put_iv_90 and call_iv_90) else np.nan
        c_atm_ratio_90 = round(call_iv_90 / atm_iv_90, 4) if (call_iv_90 and atm_iv_90) else np.nan
        p_atm_ratio_90 = round(put_iv_90 / atm_iv_90, 4) if (put_iv_90 and atm_iv_90) else np.nan

        new_row = pd.DataFrame([{
            "date": date_str, 
            "Ticker": ticker, 
            "Spot": round(spot, 2),
            "RFR": round(rfr, 4),
            "IV30": round(atm_iv_30 * 100, 2), 
            "Put_IV_30": round(put_iv_30 * 100, 2) if put_iv_30 else np.nan,
            "Call_IV_30": round(call_iv_30 * 100, 2) if call_iv_30 else np.nan,
            "1M_25dP/25dC": p_c_ratio_30,
            "1M_25dC/ATM": c_atm_ratio_30,
            "1M_25dP/ATM": p_atm_ratio_30,
            "IV90": round(atm_iv_90 * 100, 2) if atm_iv_90 else np.nan,
            "Put_IV_90": round(put_iv_90 * 100, 2) if put_iv_90 else np.nan,
            "Call_IV_90": round(call_iv_90 * 100, 2) if call_iv_90 else np.nan,
            "3M_25dP/25dC": p_c_ratio_90,
            "3M_25dC/ATM": c_atm_ratio_90,
            "3M_25dP/ATM": p_atm_ratio_90
        }])
        
        filename = f"{OUTPUT_DIR}/{ticker.lower()}.csv"
        existing_df = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()
        combined = pd.concat([existing_df, new_row], ignore_index=True).drop_duplicates(subset=['date'], keep='last').sort_values('date').tail(252)
        combined.to_csv(filename, index=False)
        saved_count += 1

    print(f"✅ Daily Update Complete. Appended {saved_count} tickers to the 1Y folder for {date_str}.")
    if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)

if __name__ == "__main__":
    run_daily_updater()