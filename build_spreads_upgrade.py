import os
import time
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

warnings.simplefilter(action='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dotenv()

MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")

ETF_UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", 
                "EWZ", "EEM", "EWW", "IEF", "TLT", "LQD", "HYG", "USO", "XOM", "GLD", 
                "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL", "XLE", "XOP", 
                "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB", "XLK", 
                "SMH", "ARKK", "IBB", "ARKG", "XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]

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
    def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
        if T <= 0.0001 or sigma <= 0.0001 or S <= 0: return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1.0
        
def get_global_rf_rate() -> float:
    try: return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100.0
    except Exception: return 0.045

def calculate_penalized_execution(bid, ask, mkt, is_buy):
    if bid > 0 and ask > bid and ((ask - bid) / mkt) < 0.50:
        spread = ask - bid
        return (mkt + 0.25 * spread) if is_buy else (mkt - 0.25 * spread)
    else:
        # Fallback penalty for illiquid/missing quotes
        return (mkt * 1.05) if is_buy else (mkt * 0.95)

def process_spreads(ticker: str, spot: float, rfr: float) -> list:
    if pd.isna(spot) or spot <= 0: return []
        
    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    strike_min, strike_max = spot * 0.5, spot * 1.5
    d_min = (today + timedelta(days=15)).strftime('%Y-%m-%d')
    d_max = (today + timedelta(days=45)).strftime('%Y-%m-%d')
    
    url = f"https://api.massive.com/v3/snapshot/options/{ticker}?expiration_date.gte={d_min}&expiration_date.lte={d_max}&strike_price.gte={strike_min}&strike_price.lte={strike_max}&limit=250"
    
    results, next_url = [], url
    while next_url:
        try:
            res = requests.get(next_url, headers=headers, timeout=10)
            if res.status_code != 200: break
            data = res.json()
            results.extend(data.get("results", []))
            next_url = data.get("next_url")
        except Exception:
            break

    if not results: return []

    chain = []
    for c in results:
        det = c.get("details", {})
        quote = c.get("last_quote", {})
        bid, ask = quote.get("bid", 0), quote.get("ask", 0)
        close_price = c.get("day", {}).get("close", 0)
        
        if not det.get("strike_price"): continue
        
        mkt = 0.0
        if bid > 0 and ask > bid:
            mkt = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mkt
            if spread_pct > 0.50 and close_price > 0: 
                mkt = close_price 
        elif close_price > 0:
            mkt = close_price
            
        if mkt <= 0.01: continue
            
        exp_date = datetime.strptime(det["expiration_date"], "%Y-%m-%d")
        dte = (exp_date - today).days
        if dte <= 0: continue
        
        opt_type = det["contract_type"].lower()
        iv = OptionsMath.iv_solver(mkt, spot, det["strike_price"], dte/365.0, rfr, opt_type)
        
        if iv and iv > 0.01:
            delta = OptionsMath.black_scholes_delta(spot, det["strike_price"], dte/365.0, rfr, iv, opt_type)
            chain.append({
                "Expiration": det["expiration_date"], 
                "Type": opt_type.upper(), 
                "Strike": det["strike_price"], 
                "Bid": bid, "Ask": ask, "Mkt": mkt, "Delta": delta
            })
            
    df = pd.DataFrame(chain)
    if df.empty: return []
    
    spreads = []
    for exp, group in df.groupby("Expiration"):
        puts = group[group["Type"] == "PUT"].sort_values('Strike')
        calls = group[group["Type"] == "CALL"].sort_values('Strike')
        
        if not puts.empty and len(puts) >= 2:
            buy_idx = (puts['Delta'] - (-0.25)).abs().argmin()
            sell_idx = (puts['Delta'] - (-0.10)).abs().argmin()
            
            if buy_idx != sell_idx:
                buy_leg = puts.iloc[buy_idx]
                sell_leg = puts.iloc[sell_idx]
                
                if abs(buy_leg['Delta'] - (-0.25)) < 0.10 and abs(sell_leg['Delta'] - (-0.10)) < 0.06:
                    if buy_leg['Strike'] > sell_leg['Strike']:
                        exec_buy = calculate_penalized_execution(buy_leg['Bid'], buy_leg['Ask'], buy_leg['Mkt'], is_buy=True)
                        exec_sell = calculate_penalized_execution(sell_leg['Bid'], sell_leg['Ask'], sell_leg['Mkt'], is_buy=False)
                        cost = exec_buy - exec_sell
                        
                        if cost > 0.01: 
                            raw_payout = ((buy_leg['Strike'] - sell_leg['Strike']) - cost) / cost
                            flag = "⚠️ Micro-Premium" if (raw_payout > 40 or cost < 0.05) else "✅ Viable"
                            payout_ratio = min(raw_payout, 99.9)
                            
                            spreads.append({
                                "Ticker": ticker, "Spread_Type": "PUT", "Expiration": exp, 
                                "Strike 1": buy_leg['Strike'], "Strike 2": sell_leg['Strike'], 
                                "S1 %spot": (buy_leg['Strike']/spot)*100, "S2 %spot": (sell_leg['Strike']/spot)*100, 
                                "Cost": cost, "Cost % Spot": (cost/spot)*100, "Payout Ratio": payout_ratio,
                                "Flag": flag
                            })

        if not calls.empty and len(calls) >= 2:
            buy_idx = (calls['Delta'] - 0.25).abs().argmin()
            sell_idx = (calls['Delta'] - 0.10).abs().argmin()
            
            if buy_idx != sell_idx:
                buy_leg = calls.iloc[buy_idx]
                sell_leg = calls.iloc[sell_idx]
                
                if abs(buy_leg['Delta'] - 0.25) < 0.10 and abs(sell_leg['Delta'] - 0.10) < 0.06:
                    if buy_leg['Strike'] < sell_leg['Strike']:
                        exec_buy = calculate_penalized_execution(buy_leg['Bid'], buy_leg['Ask'], buy_leg['Mkt'], is_buy=True)
                        exec_sell = calculate_penalized_execution(sell_leg['Bid'], sell_leg['Ask'], sell_leg['Mkt'], is_buy=False)
                        cost = exec_buy - exec_sell
                        
                        if cost > 0.01:
                            raw_payout = ((sell_leg['Strike'] - buy_leg['Strike']) - cost) / cost
                            flag = "⚠️ Micro-Premium" if (raw_payout > 40 or cost < 0.05) else "✅ Viable"
                            payout_ratio = min(raw_payout, 99.9)
                            
                            spreads.append({
                                "Ticker": ticker, "Spread_Type": "CALL", "Expiration": exp, 
                                "Strike 1": buy_leg['Strike'], "Strike 2": sell_leg['Strike'], 
                                "S1 %spot": (buy_leg['Strike']/spot)*100, "S2 %spot": (sell_leg['Strike']/spot)*100, 
                                "Cost": cost, "Cost % Spot": (cost/spot)*100, "Payout Ratio": payout_ratio,
                                "Flag": flag
                            })

    return spreads

def build_all_spreads():
    print("Pre-fetching Market Spots to avoid Rate Limits...")
    start_time = time.time()
    
    spot_df = yf.download(ETF_UNIVERSE, period="2d")['Close']
    if isinstance(spot_df.columns, pd.MultiIndex):
        spot_df.columns = spot_df.columns.get_level_values(1)
        
    spots_dict = {}
    for ticker in ETF_UNIVERSE:
        try: spots_dict[ticker] = float(spot_df[ticker].dropna().iloc[-1])
        except: spots_dict[ticker] = 0.0

    rfr = get_global_rf_rate()
    master_data = []
    
    print("Scanning live options chain for institutional 25d/10d debit spreads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(process_spreads, ticker, spots_dict[ticker], rfr): ticker 
            for ticker in ETF_UNIVERSE
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try: 
                data = future.result()
                if data:
                    master_data.extend(data)
                    print(f"[{ticker}] Found {len(data)} actionable spread structures.")
                else:
                    print(f"[{ticker}] No viable spreads found.")
            except Exception as e: 
                print(f"[{ticker}] Thread failed: {e}")

    if master_data:
        df = pd.DataFrame(master_data)
        df.to_csv("live_spread_execution.csv", index=False)
        print(f"\n✅ Success! Saved {len(df)} total setups to live_spread_execution.csv in {time.time() - start_time:.1f} seconds.")
    else:
        print("\n❌ Failed to save CSV. No viable spreads generated across the entire universe.")

if __name__ == "__main__":
    build_all_spreads()