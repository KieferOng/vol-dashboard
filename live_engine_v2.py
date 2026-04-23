import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv
import warnings
import streamlit as st

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()
MASSIVE_API_KEY = st.secrets["MASSIVE_API_KEY"]

ETF_UNIVERSE = {
    "INDICES": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", "EWZ", "EEM", "EWW"],
    "BONDS": ["IEF", "TLT", "LQD", "HYG"],
    "COMMODITIES": ["USO", "XOM", "GLD", "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL"],
    "CYCLICALS": ["XLE", "XOP", "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB"],
    "TECH_INNOVATION": ["XLK", "SMH", "ARKK", "IBB", "ARKG"],
    "DEFENSIVES": ["XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]
}
ALL_TICKERS = [ticker for sublist in ETF_UNIVERSE.values() for ticker in sublist]

def get_global_rf_rate() -> float:
    try:
        return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100.0
    except Exception:
        print("Failed to fetch ^IRX. Defaulting to 4.5%")
        return 0.045

class OptionsMath:
    @staticmethod
    def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
        if T <= 0.0001 or sigma <= 0.0001 or S <= 0: return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1.0

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

class DashboardEngine:
    def __init__(self, ticker: str, rfr: float, hist_dir="massive_historical_options_data"):
        self.ticker = ticker.upper()
        self.headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
        self.rfr = rfr
        self.hist_path = f"{hist_dir}/{ticker.lower()}.csv"

    def calc_equity_metrics(self) -> dict:
        today = datetime.today()
        start_date = (today - timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        url = f"https://api.massive.com/v2/aggs/ticker/{self.ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000"
        
        try:
            res = requests.get(url, headers=self.headers)
            res.raise_for_status()
            data = res.json()
            results = data.get("results", [])
            
            if not results: return {}
            
            df = pd.DataFrame(results).rename(columns={'c': 'Close'})
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            
            df['Return'] = df['Close'].pct_change()
            df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            
            spot = df['Close'].iloc[-1]
            last_date = df.index[-1]
            
            price_1w = df['Close'].asof(last_date - timedelta(days=7))
            price_1m = df['Close'].asof(last_date - timedelta(days=30))
            price_3m = df['Close'].asof(last_date - timedelta(days=90))
            
            perf_1d = df['Return'].iloc[-1]
            perf_1w = (spot / price_1w) - 1 if pd.notna(price_1w) else 0
            perf_1m = (spot / price_1m) - 1 if pd.notna(price_1m) else 0
            perf_3m = (spot / price_3m) - 1 if pd.notna(price_3m) else 0
            
            std_1d = df['Return'].tail(22).std()
            sigma_1d = perf_1d / std_1d if std_1d != 0 else 0
            
            std_1w = df['Return'].rolling(5).sum().tail(22).std()
            sigma_1w = df['Return'].tail(5).sum() / std_1w if std_1w != 0 else 0
            
            hv10 = df['LogReturn'].tail(10).std() * np.sqrt(252)
            hv30 = df['LogReturn'].tail(30).std() * np.sqrt(252)
            
            return {
                "Spot": round(spot, 2),
                "Perf_1D": round(perf_1d * 100, 2),
                "Perf_1W": round(perf_1w * 100, 2),
                "Perf_1M": round(perf_1m * 100, 2),
                "Perf_3M": round(perf_3m * 100, 2),
                "Sigma_1D": round(sigma_1d, 2),
                "Sigma_1W": round(sigma_1w, 2),
                "HV10": round(hv10 * 100, 2),
                "HV30": round(hv30 * 100, 2)
            }
        except Exception: 
            return {}

    def calc_live_options(self, spot: float) -> dict:
        today = datetime.today()
        strike_min, strike_max = spot * 0.4, spot * 2.0
        d15, d45 = (today + timedelta(days=15)).strftime('%Y-%m-%d'), (today + timedelta(days=45)).strftime('%Y-%m-%d')
        d60, d120 = (today + timedelta(days=60)).strftime('%Y-%m-%d'), (today + timedelta(days=120)).strftime('%Y-%m-%d')
        base_url = f"https://api.massive.com/v3/snapshot/options/{self.ticker}"
        
        def fetch_targeted_window(min_date, max_date):
            url = f"{base_url}?expiration_date.gte={min_date}&expiration_date.lte={max_date}&strike_price.gte={strike_min}&strike_price.lte={strike_max}&limit=250"
            results, next_url = [], url
            while next_url:
                res = requests.get(next_url, headers=self.headers)
                if res.status_code != 200: break
                data = res.json()
                results.extend(data.get("results", []))
                next_url = data.get("next_url")
            return results

        try:
            all_results = fetch_targeted_window(d15, d45) + fetch_targeted_window(d60, d120)
            if not all_results: return {}
            
            all_vols = []
            for c in all_results:
                det = c.get("details", {})
                quote = c.get("last_quote", {})
                
                bid = quote.get("bid", 0)
                ask = quote.get("ask", 0)
                close_price = c.get("day", {}).get("close", 0)
                
                if bid > 0 and ask > bid:
                    mkt = (bid + ask) / 2.0
                    spread_pct = (ask - bid) / mkt
                    
                    if spread_pct > 0.50 and close_price > 0:
                        mkt = close_price 
                    elif spread_pct > 1.0:
                        continue 
                else:
                    mkt = close_price
                
                if mkt <= 0 or not det.get("strike_price"): continue
                
                dte = (datetime.strptime(det["expiration_date"], "%Y-%m-%d") - today).days
                if dte <= 5: continue
                
                iv = OptionsMath.iv_solver(mkt, spot, det["strike_price"], dte/365, self.rfr, det["contract_type"].lower())
                if iv and iv > 0.01:
                    delta = OptionsMath.black_scholes_delta(spot, det["strike_price"], dte/365, self.rfr, iv, det["contract_type"].lower())
                    all_vols.append({"Type": det["contract_type"].upper(), "DTE": dte, "IV": iv, "Delta": delta})
            
            df = pd.DataFrame(all_vols)
            if df.empty: return {}

            def get_target_df(df_subset, target_dte):
                if df_subset.empty: return pd.DataFrame()
                closest_dte = df_subset.iloc[(df_subset['DTE'] - target_dte).abs().argmin()]['DTE']
                return df_subset[df_subset['DTE'] == closest_dte]

            df_30 = get_target_df(df[(df['DTE'] >= 15) & (df['DTE'] <= 45)], 30)
            df_90 = get_target_df(df[(df['DTE'] >= 60) & (df['DTE'] <= 120)], 90)

            def get_iv(subset, target_delta, opt_type):
                if subset.empty: return None
                f = subset[subset['Type'] == opt_type]
                if f.empty: return None
                return f.iloc[(f['Delta'] - target_delta).abs().argmin()]['IV']

            iv30_raw = get_iv(df_30, 0.50, 'CALL')
            iv90_raw = get_iv(df_90, 0.50, 'CALL')
            
            if not iv30_raw: return {}
            if iv30_raw > 1.20: return {}

            p25_raw = get_iv(df_30, -0.25, 'PUT')
            if p25_raw and p25_raw < (iv30_raw * 0.2): p25_raw = None
            
            c25_raw = get_iv(df_30, 0.25, 'CALL')
            if c25_raw and c25_raw < (iv30_raw * 0.2): c25_raw = None

            iv30 = round(iv30_raw * 100, 2)
            iv90 = round(iv90_raw * 100, 2) if iv90_raw else None
            ts = round((iv30 / iv90), 2) if (iv30 and iv90) else None
            
            skew_1m = round((p25_raw - c25_raw) * 100, 2) if (p25_raw and c25_raw) else None

            return {"IV30": iv30, "IV90": iv90, "Term_Structure": ts, "Skew_1M_25D": skew_1m}
            
        except Exception: 
            return {}

    def calc_historical_metrics(self, live_iv, live_skew) -> dict:
        if not os.path.exists(self.hist_path): return {"Status": "🔴 MISSING", "Flag": "No CSV found."}
        try:
            df = pd.read_csv(self.hist_path)
            if len(df) < 20: return {"Status": "🔴 INCOMPLETE", "Flag": f"Only {len(df)} rows."}
            window = df.tail(22) 
            
            iv_s = (window['IV30'].groupby((window['IV30'] != window['IV30'].shift()).cumsum()).transform('size')).max()
            sk_s = (window['Skew_1M_25D'].groupby((window['Skew_1M_25D'] != window['Skew_1M_25D'].shift()).cumsum()).transform('size')).max()
            worst_streak = max(iv_s, sk_s)
            worst_unq = min(window['IV30'].nunique(), window['Skew_1M_25D'].nunique()) / len(window)

            if worst_streak >= 6 or worst_unq < 0.4:
                status, flag = "🔴 CRITICAL", f"Dead Block ({worst_streak}d)"
            elif worst_streak >= 4 or worst_unq < 0.6:
                status, flag = "🟡 WARNING", f"Streak ({worst_streak}d)"
            else:
                status, flag = "🟢 HEALTHY", "Clean"

            def calc_stats(live, series, metric_name):
                if live is None or pd.isna(live) or series.empty: 
                    return None, None, None, None, ""
                
                mu, sigma = series.mean(), series.std()
                sigma_floor = max(sigma, 1.0) 
                z = (live - mu) / sigma_floor
                
                anomaly_flag = ""
                if z > 4.0:
                    z = 4.0
                    anomaly_flag = f" | ⚠️ High {metric_name} Outlier"
                elif z < -4.0:
                    z = -4.0
                    anomaly_flag = f" | ⚠️ Low {metric_name} Outlier"

                pct = (series < live).sum() / len(series) * 100
                d1 = live - series.iloc[-1]
                d5 = live - series.iloc[-5] if len(series) >= 5 else 0.0
                
                return round(z, 2), round(pct, 1), round(d1, 2), round(d5, 2), anomaly_flag

            iv_z, iv_p, iv_1d, iv_5d, iv_anomaly = calc_stats(live_iv, window['IV30'], "IV")
            sk_z, sk_p, sk_1d, sk_5d, sk_anomaly = calc_stats(live_skew, window['Skew_1M_25D'], "Skew")
            
            final_flag = flag + iv_anomaly + sk_anomaly
            if ("Outlier" in final_flag) and (status == "🟢 HEALTHY"):
                status = "🟡 WARNING"

            return {
                "Status": status, "Flag": final_flag,
                "IV_Z": iv_z, "IV_Pct": iv_p, "IV_1D": iv_1d, "IV_5D": iv_5d, 
                "Skew_Z": sk_z, "Skew_Pct": sk_p, "Skew_1D": sk_1d, "Skew_5D": sk_5d
            }
        except Exception as e: 
            return {"Status": "🔴 ERROR", "Flag": str(e)}

    def process(self) -> dict:
        row = {"Ticker": self.ticker}
        
        e = self.calc_equity_metrics()
        spot = e.get("Spot", 0.0)
        row.update(e)
        
        o = self.calc_live_options(spot)
        iv30 = o.get("IV30")
        skew = o.get("Skew_1M_25D")
        
        hv10 = e.get("HV10", 0.0)
        carry = round(((iv30 - hv10) / iv30 * 100), 2) if (iv30 is not None and iv30 != 0 and hv10) else None
        
        row.update(o)
        row.update({"Carry": carry})
        
        row.update(self.calc_historical_metrics(iv30, skew))
        
        return row

def process_single_ticker(ticker: str, rfr: float) -> dict:
    try:
        engine = DashboardEngine(ticker, rfr)
        return engine.process()
    except Exception as e:
        print(f"[{ticker}] Fatal error during processing: {e}")
        return {"Ticker": ticker, "Status": "🔴 ERROR", "Flag": "Thread Exception"}

def build_all_tickers():
    print("Initiating Fast Live Market Pull for 50 Tickers...")
    start_time = time.time()
    
    rfr = get_global_rf_rate()
    print(f"Global Risk-Free Rate locked at: {rfr*100:.2f}%")
    
    master_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(process_single_ticker, ticker, rfr): ticker 
            for ticker in ALL_TICKERS
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                master_data.append(data)
                print(f"[{len(master_data)}/50] Processed {ticker}")
            except Exception as exc:
                print(f"[{ticker}] generated an exception: {exc}")

    df = pd.DataFrame(master_data)
    
    ticker_order_map = {t: i for i, t in enumerate(ALL_TICKERS)}
    df['Sort_Index'] = df['Ticker'].map(ticker_order_map)
    df = df.sort_values('Sort_Index').drop(columns=['Sort_Index'])
    
    df.to_csv("live_dashboard_feed_fast.csv", index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Success! Master feed saved to live_dashboard_feed_fast.csv in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    build_all_tickers()