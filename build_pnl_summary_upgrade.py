import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

ETF_UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", 
                "EWZ", "EEM", "EWW", "IEF", "TLT", "LQD", "HYG", "USO", "XOM", "GLD", 
                "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL", "XLE", "XOP", 
                "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB", "XLK", 
                "SMH", "ARKK", "IBB", "ARKG", "XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]

class VectorizedOptions:
    @staticmethod
    def bs_price(S, K, T, r, sigma, opt_type='call'):
        """Vectorized Black-Scholes Pricing Array"""
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return np.where(opt_type == 'call', call_price, put_price)

    @staticmethod
    def get_strike_from_delta(S, T, r, sigma, delta, opt_type='call'):
        """Reverse-engineers the exact Option Strike from a given Delta"""
        d1 = norm.ppf(delta) if opt_type == 'call' else norm.ppf(delta + 1.0)
        K = S * np.exp((r + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T))
        return K

def calculate_cumulative(series, periods):
    if len(series) < periods: return None
    return (1 + series.tail(periods)).prod() - 1

def calculate_sharpe(series):
    if series.empty or series.std() == 0: return 0.0
    return np.sqrt(252) * (series.mean() / series.std())

def build_pnl_summary():
    print("Constructing Vectorized Black-Scholes PnL Summary from 1Y Flatfiles...")
    master_results = []
    
    strategies = [
        "Selling Daily ATM Straddle", "Selling Daily ATM Call", "Selling Daily ATM Put", 
        "Selling Daily Strangle", "Selling Daily 25d Call", "Selling Daily 25d Put", 
        "Sell 25d Put, Buy 25d Call", "Stock"
    ]
    
    for ticker in ETF_UNIVERSE:
        file_path = f"massive_historical_1y_data/{ticker.lower()}.csv"
        if not os.path.exists(file_path): continue
            
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date').dropna(subset=['IV30'])
        
        if len(df) < 50: continue  # Skip if not enough history
            
        # Extract T (Day 0) arrays
        S0 = df['Spot'].values
        r0 = df['RFR'].values
        iv_atm = df['IV30'].values / 100.0
        
        # Handle potential NaNs in skew gracefully by falling back to ATM vol
        iv_put25 = df['Put_IV_30'].fillna(df['IV30']).values / 100.0
        iv_call25 = df['Call_IV_30'].fillna(df['IV30']).values / 100.0
        
        # Calculate Option Strikes set on Day 0
        T_entry = 30.0 / 365.0
        K_atm = S0
        K_put25 = VectorizedOptions.get_strike_from_delta(S0, T_entry, r0, iv_put25, -0.25, 'put')
        K_call25 = VectorizedOptions.get_strike_from_delta(S0, T_entry, r0, iv_call25, 0.25, 'call')
        
        # Calculate Prices on Day 0
        P0_atm_c = VectorizedOptions.bs_price(S0, K_atm, T_entry, r0, iv_atm, 'call')
        P0_atm_p = VectorizedOptions.bs_price(S0, K_atm, T_entry, r0, iv_atm, 'put')
        P0_25_c = VectorizedOptions.bs_price(S0, K_call25, T_entry, r0, iv_call25, 'call')
        P0_25_p = VectorizedOptions.bs_price(S0, K_put25, T_entry, r0, iv_put25, 'put')

        # Shift arrays to represent Day 1 (T+1)
        S1 = np.roll(S0, -1)
        r1 = np.roll(r0, -1)
        iv_atm_1 = np.roll(iv_atm, -1)
        iv_put25_1 = np.roll(iv_put25, -1)
        iv_call25_1 = np.roll(iv_call25, -1)
        
        # Calendar days elapsed
        days_diff = pd.Series(df.index).diff().shift(-1).dt.days.values
        T_exit = (30.0 - np.nan_to_num(days_diff, nan=1.0)) / 365.0
        T_exit = np.clip(T_exit, 1/365.0, 30/365.0)  # Prevent negative DTE
        
        # Calculate Prices on Day 1 (Repricing the exact same strikes)
        P1_atm_c = VectorizedOptions.bs_price(S1, K_atm, T_exit, r1, iv_atm_1, 'call')
        P1_atm_p = VectorizedOptions.bs_price(S1, K_atm, T_exit, r1, iv_atm_1, 'put')
        P1_25_c = VectorizedOptions.bs_price(S1, K_call25, T_exit, r1, iv_call25_1, 'call')
        P1_25_p = VectorizedOptions.bs_price(S1, K_put25, T_exit, r1, iv_put25_1, 'put')
        
        # Calculate Normalized Daily PnL (Short position: P0 - P1) divided by Spot
        strat_returns = pd.DataFrame(index=df.index)
        strat_returns["Selling Daily ATM Straddle"] = ((P0_atm_c - P1_atm_c) + (P0_atm_p - P1_atm_p)) / S0
        strat_returns["Selling Daily ATM Call"] = (P0_atm_c - P1_atm_c) / S0
        strat_returns["Selling Daily ATM Put"] = (P0_atm_p - P1_atm_p) / S0
        strat_returns["Selling Daily Strangle"] = ((P0_25_c - P1_25_c) + (P0_25_p - P1_25_p)) / S0
        strat_returns["Selling Daily 25d Call"] = (P0_25_c - P1_25_c) / S0
        strat_returns["Selling Daily 25d Put"] = (P0_25_p - P1_25_p) / S0
        strat_returns["Sell 25d Put, Buy 25d Call"] = strat_returns["Selling Daily 25d Put"] - strat_returns["Selling Daily 25d Call"]
        
        # Fix the stock return shift alignment
        strat_returns["Stock"] = df['Spot'].pct_change().shift(-1)
        
        # Drop the last row because T+1 is unknown
        strat_returns = strat_returns.iloc[:-1].fillna(0)
        
        # Formatting metrics
        ytd_start = pd.to_datetime(f"{datetime.today().year}-01-01")
        
        for strat in strategies:
            series = strat_returns[strat]
            master_results.append({
                "Ticker": ticker, "Strategy": strat,
                "1d": calculate_cumulative(series, 1), 
                "10d": calculate_cumulative(series, 10),
                "20d": calculate_cumulative(series, 20), 
                "60d": calculate_cumulative(series, 60),
                "ytd": (1 + series[series.index >= ytd_start]).prod() - 1 if not series[series.index >= ytd_start].empty else None,
                "1y": calculate_cumulative(series, 252),
                "Sharpe_10d": calculate_sharpe(series.tail(10)),
                "Sharpe_20d": calculate_sharpe(series.tail(20)),
                "Sharpe_60d": calculate_sharpe(series.tail(60)),
                "Sharpe_ytd": calculate_sharpe(series[series.index >= ytd_start]),
                "Sharpe_1y": calculate_sharpe(series)
            })
            
    final_df = pd.DataFrame(master_results)
    final_df.to_csv("pnl_backtest_results.csv", index=False)
    print(f"✅ Success! Vectorized Backtest saved to pnl_backtest_results.csv")

if __name__ == "__main__":
    build_pnl_summary()