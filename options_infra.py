"""
Options Infrastructure for NUSSIF
Version: Pilot (v1.1)
Developer: Kiefer

Purpose:
    - Historical options backtesting software development kit.
    - Connects to Massive.com to retrieve:
        1. Intraday Minute Options Data (OHLC Price, Volume, IV, Greeks)
        2. Daily Options Data (Buy-and-hold, tracks the same option contract over time)
        3. Daily Options Data (Rolling exposure, tracks constant maturity and moneyness)
        4. Daily Options Data (Rolling exposure, tracks constant Delta)

Usage:
    - Refer to the options_infra_manual.ipynb for instructions on how to pull options data from Massive.com:
"""

import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, time

# ==========================================
# 1. Setup & Verification
# ==========================================
def setup():
    """
    Run this once to configure your API key.
    Usage: import options_infra; options_infra.setup()
    """
    print("\n--- Massive.com API Key Setup ---")
    
    # 1. Input API Key
    api_key = input("Enter your Massive.com API Key: ").strip()
    if len(api_key) < 20: 
        print("Error: Key looks too short. Please copy it exactly.")
        return

    # 2. Test API Key
    print("Verifying key with Massive.com...")
    try:
        base_url = "https://api.massive.com"
        url = f"{base_url}/v3/reference/tickers/SPY"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code == 403:
            print("Error 403: Forbidden. Your key is invalid or inactive.")
            return
        elif r.status_code != 200:
            print(f"Connection failed (Status {r.status_code}): {r.text}")
            return
        print("Success. Key is valid.")
        
    except Exception as e:
        print(f"Network error: {e}")
        return

    # 3. Save to .env
    with open(".env", "w") as f:
        f.write(f"POLYGON_API_KEY={api_key}\n")
    print("Saved key to .env")

    # 4. Add to .gitignore
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            if ".env" not in f.read():
                with open(gitignore_path, "a") as f2:
                    f2.write("\n.env\n")
                print("Added .env to .gitignore")
    else:
        with open(gitignore_path, "w") as f:
            f.write(".env\n")
        print("Created .gitignore")

    print("\nSetup complete.")

# ==========================================
# 2. Math Helper Functions
# ==========================================
class OptionsMath:
    @staticmethod
    def parse_contract_details(symbol):
        """
        Parses symbol 'O:SPY231115C00450000' -> (Strike, Expiry, Type)
        """
        s = symbol.replace("O:", "")
        try:
            strike_str = s[-8:]
            type_char = s[-9]       # 'C' or 'P'
            date_str = s[-15:-9]    # YYMMDD
            
            strike = float(strike_str) / 1000.0
            expiry = pd.to_datetime(date_str, format='%y%m%d') + pd.Timedelta(hours=16)
            option_type = 'call' if type_char.upper() == 'C' else 'put'
            
            return strike, expiry, option_type
        except:
            return None, None, None

    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type):
        """
        Calculates Delta, Gamma, Theta, Vega.
        """
        if T <= 0.00001 or sigma <= 0.0001 or S <= 0:
            return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        N_prime = norm.pdf(d1)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * sigma * N_prime) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else: # put
            delta = norm.cdf(d1) - 1
            theta = (- (S * sigma * N_prime) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
        gamma = N_prime / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * N_prime / 100 

        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

    @staticmethod
    def implied_vol_solver(price, S, K, T, r, option_type):
        """
        Back-solves for IV.
        """
        if price < 0.01 or T <= 0: return np.nan

        def price_diff(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                val = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                val = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return val - price

        try:
            return brentq(price_diff, 0.001, 5.0)
        except:
            return np.nan

# ==========================================
# 3. Options Infrastructure
# ==========================================
class OptionsClient:
    def __init__(self, api_key=None):
        load_dotenv()
        self.key = api_key or os.getenv("POLYGON_API_KEY")
        self.BASE_URL = "https://api.massive.com"
        
        if not self.key:
            print("Warning: No API Key found. Run 'setup()' or set POLYGON_API_KEY.")
        
        self.HEADERS = {"Authorization": f"Bearer {self.key}"}
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _get_json(self, url, params=None):
        try:
            r = self.session.get(url, params=params, timeout=30)
            r.raise_for_status() 
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None

    def _check_data_quality(self, df):
        if df.empty: return
        
        nan_count = df['implied_vol'].isna().sum()
        total_rows = len(df)
        
        if nan_count > 0:
            pct = (nan_count / total_rows) * 100
            print(f"Data Quality Warning: {nan_count} rows ({pct:.1f}%) have missing IV/Greeks.")
        
        print(f"Data processed: {total_rows} rows.")
    
    # Public Function 1: get_option_chain fetches active contracts for a specific ticker on a specific past date
    def get_option_chain(self, ticker, as_of_date, verbose=True, save_csv=None, contract_type=None, min_expiry=None, max_expiry=None):
        if verbose:
            filter_msg = f" (Type={contract_type})" if contract_type else ""
            print(f"Fetching option chain for {ticker} on {as_of_date}{filter_msg}...")
            
        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "as_of": as_of_date,
            "expired": "false",
            "limit": 1000
        }
        
        if contract_type: params["contract_type"] = contract_type.lower()
        
        if min_expiry: params["expiration_date.gte"] = min_expiry
        if max_expiry: params["expiration_date.lte"] = max_expiry

        all_contracts = []
        try:
            data = self._get_json(url, params)
            while data and 'results' in data:
                all_contracts.extend(data['results'])
                if 'next_url' in data:
                    data = self._get_json(data['next_url'])
                else:
                    break
        except Exception as e:
            print(f"Error fetching chain: {e}")
            return pd.DataFrame()

        if not all_contracts:
            if verbose: print(f"No contracts found.")
            return pd.DataFrame()

        df = pd.DataFrame(all_contracts)
        cols = ['ticker', 'contract_type', 'strike_price', 'expiration_date']
        df = df[[c for c in cols if c in df.columns]]
        df['expiration_date'] = pd.to_datetime(df['expiration_date'])
        df['days_to_expiry'] = (df['expiration_date'] - pd.to_datetime(as_of_date)).dt.days
        df = df.sort_values(by=['expiration_date', 'strike_price'])
        
        if save_csv: df.to_csv(save_csv, index=False)
        return df
    
    def _get_aggs(self, ticker, from_date, to_date, multiplier, timespan):
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        
        try:
            data = self._get_json(url, params)
            if data and 'results' in data:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching aggregates for {ticker}: {e}")
        return pd.DataFrame()

    def _select_contract(self, ticker, as_of_date, moneyness, target_dte, contract_type, known_stock_price=None):
        if known_stock_price is not None:
            stock_price = known_stock_price
        else:
            daily_df = self._get_aggs(ticker, as_of_date, as_of_date, 1, 'day')
            if daily_df.empty: return None
            stock_price = daily_df.iloc[0]['close']
            
        target_strike = stock_price * moneyness
        
        curr_date = pd.to_datetime(as_of_date)
        target_expiry = curr_date + pd.Timedelta(days=target_dte)
        
        min_date = (target_expiry - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        max_date = (target_expiry + pd.Timedelta(days=15)).strftime('%Y-%m-%d')

        chain = self.get_option_chain(
            ticker, as_of_date, verbose=False, contract_type=contract_type,
            min_expiry=min_date, max_expiry=max_date
        )
        
        if chain.empty: return None

        chain['dte_diff'] = abs(chain['days_to_expiry'] - target_dte)
        candidates = chain[chain['dte_diff'] == chain['dte_diff'].min()].copy()
        
        candidates['strike_diff'] = abs(candidates['strike_price'] - target_strike)
        return candidates.sort_values('strike_diff').iloc[0]['ticker']

    def _process_merged_data(self, merged_df, contract_symbol, user_contract_type, r):
        results = []
        K, expiry, actual_type = OptionsMath.parse_contract_details(contract_symbol)
        
        if not K: return pd.DataFrame()

        calc_type = actual_type if actual_type else user_contract_type

        for _, row in merged_df.iterrows():
            if 'days_to_expiry' in row:
                T = row['days_to_expiry'] / 365.0 
            else:
                T = (expiry - row['timestamp']).total_seconds() / (365 * 24 * 3600)
            
            if T <= 0: continue

            iv = OptionsMath.implied_vol_solver(
                row['close_opt'], row['close_und'], K, T, r, calc_type
            )
            
            greeks = OptionsMath.black_scholes_greeks(
                row['close_und'], K, T, r, iv if not np.isnan(iv) else 0, calc_type
            )
            
            item = row.to_dict()
            item.update(greeks)
            item['implied_vol'] = iv
            results.append(item)
            
        return pd.DataFrame(results)

    # Public Function 2: get_minute_history fetches minute-level data for a specific contract on a specific date
    def get_minute_history(self, date, ticker, moneyness, target_dte, contract_type='call', r=0, save_csv=None):
        contract_symbol = self._select_contract(ticker, date, moneyness, target_dte, contract_type)
        if not contract_symbol: 
            print(f"No {contract_type} found for {ticker} (Target DTE: {target_dte})")
            return pd.DataFrame()
        
        print(f"Analysing Intraday ({contract_type.upper()}): {contract_symbol} | Risk-Free Rate: {r}")

        stock_df = self._get_aggs(ticker, date, date, 1, 'minute')
        opt_df = self._get_aggs(contract_symbol, date, date, 1, 'minute')
        
        if stock_df.empty or opt_df.empty: 
            print("Missing minute data.")
            return pd.DataFrame()

        stock_df = stock_df.sort_values('timestamp')
        opt_df = opt_df.sort_values('timestamp')

        merged = pd.merge_asof(
            stock_df,
            opt_df,
            on='timestamp', 
            suffixes=('_und', '_opt'), 
            direction='backward', 
            tolerance=pd.Timedelta('24hours')
        )

        merged = merged.dropna(subset=['close_opt'])

        df = self._process_merged_data(merged, contract_symbol, contract_type, r)
        self._check_data_quality(df)
        df['ticker'] = contract_symbol
        cols = ['ticker', 'timestamp'] + [c for c in df.columns if c not in ['ticker', 'timestamp']]
        df = df[cols]

        if save_csv:
            df.to_csv(save_csv, index=False)
            print(f"Saved minute history to: {save_csv}")
        return df

    # Public Function 3: Selects ONE option contract and tracks it daily to expiry (Buy-and-hold)
    def get_daily_tracking(self, ticker, start_date, end_date, moneyness, target_dte, contract_type='call', r=0, save_csv=None):
        contract_symbol = self._select_contract(ticker, start_date, moneyness, target_dte, contract_type)
        if not contract_symbol: return pd.DataFrame()
        
        print(f"Tracking Fixed {contract_type.upper()}: {contract_symbol} | Risk-Free Rate: {r}")

        stock_df = self._get_aggs(ticker, start_date, end_date, 1, 'day')
        opt_df = self._get_aggs(contract_symbol, start_date, end_date, 1, 'day')
        
        if stock_df.empty or opt_df.empty: return pd.DataFrame()

        merged = pd.merge(
            opt_df.rename(columns={'close': 'close_opt', 'open': 'open_opt', 'volume': 'volume_opt'}), 
            stock_df.rename(columns={'close': 'close_und', 'open': 'open_und'}), 
            on='timestamp', how='inner'
        )
        
        _, expiry, _ = OptionsMath.parse_contract_details(contract_symbol)
        merged['days_to_expiry'] = (expiry - merged['timestamp']).dt.days

        df = self._process_merged_data(merged, contract_symbol, contract_type, r)
        self._check_data_quality(df)

        df['date'] = df['timestamp'].dt.date
        df = df.drop(columns=['timestamp'])
        
        cols = ['date'] + [c for c in df.columns if c != 'date']
        df = df[cols]

        if save_csv:
            df.to_csv(save_csv, index=False)
            print(f"Saved daily tracking to: {save_csv}")
        return df

    # Public Function 4: Selects a NEW contract every day based on moneyness and days to expiry (Rolling Exposure)
    def get_daily_rolling(self, ticker, start_date, end_date, moneyness, target_dte, contract_type='call', r=0, save_csv=None):
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        all_results = []
        
        print(f"Rolling {contract_type.upper()} Exposure ({len(dates)} days) | Risk-Free Rate: {r}")
        
        for d in dates:
            d_str = d.strftime('%Y-%m-%d')
            
            daily_contract = self._select_contract(ticker, d_str, moneyness, target_dte, contract_type)
            if not daily_contract: continue
            
            stock_df = self._get_aggs(ticker, d_str, d_str, 1, 'day')
            opt_df = self._get_aggs(daily_contract, d_str, d_str, 1, 'day')
            
            if stock_df.empty or opt_df.empty: continue
            
            merged = pd.merge(
                opt_df.rename(columns={'close': 'close_opt', 'open': 'open_opt', 'volume': 'volume_opt'}), 
                stock_df.rename(columns={'close': 'close_und', 'open': 'open_und'}), 
                on='timestamp', how='inner'
            )
            
            _, expiry, _ = OptionsMath.parse_contract_details(daily_contract)
            merged['days_to_expiry'] = (expiry - merged['timestamp']).dt.days
            
            daily_res = self._process_merged_data(merged, daily_contract, contract_type, r)
            daily_res['active_contract'] = daily_contract
            all_results.append(daily_res)
            
        df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        self._check_data_quality(df)
        
        if not df.empty:
            df['date'] = df['timestamp'].dt.date
            df = df.drop(columns=['timestamp'])
            
            cols = ['date'] + [c for c in df.columns if c != 'date']
            df = df[cols]
        
        if save_csv and not df.empty:
            df.to_csv(save_csv, index=False)
            print(f"Saved rolling history to: {save_csv}")
        return df

    def _select_contract_by_delta(self, ticker, as_of_date, target_delta, target_dte, contract_type, r=0):
        """
        Hunts for the specific contract closest to the target Delta.
        Optimised to only pull OTM strikes to save API rate limits, due to demand for 25D put-call skew in Vol Dashboard.
        """
        daily_df = self._get_aggs(ticker, as_of_date, as_of_date, 1, 'day')
        if daily_df.empty: return None
        spot = daily_df.iloc[0]['close']
        
        curr_date = pd.to_datetime(as_of_date)
        target_expiry = curr_date + pd.Timedelta(days=target_dte)
        min_date = (target_expiry - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        max_date = (target_expiry + pd.Timedelta(days=15)).strftime('%Y-%m-%d')

        chain = self.get_option_chain(
            ticker, as_of_date, verbose=False, contract_type=contract_type,
            min_expiry=min_date, max_expiry=max_date
        )
        
        if chain.empty: return None

        chain['dte_diff'] = abs(chain['days_to_expiry'] - target_dte)
        closest_dte = chain['dte_diff'].min()
        candidates = chain[chain['dte_diff'] == closest_dte].copy()
        
        if contract_type.lower() == 'call':
            # Look at ATM and walk UP (OTM Calls have strikes HIGHER than spot)
            # We start slightly below spot (0.98) just to ensure we catch ATM, then grab 40 strikes up.
            candidates = candidates[candidates['strike_price'] >= spot * 0.98].sort_values('strike_price').head(40)
        else:
            # Look at ATM and walk DOWN (OTM Puts have strikes LOWER than spot)
            # We start slightly above spot (1.02) to catch ATM, then grab 40 strikes down.
            candidates = candidates[candidates['strike_price'] <= spot * 1.02].sort_values('strike_price', ascending=False).head(40)

        best_contract = None
        min_delta_diff = float('inf')

        for _, c in candidates.iterrows():
            opt_df = self._get_aggs(c['ticker'], as_of_date, as_of_date, 1, 'day')
            if opt_df.empty: continue
            
            opt_price = opt_df.iloc[0]['close']
            T = c['days_to_expiry'] / 365.0
            if T <= 0: continue

            iv = OptionsMath.implied_vol_solver(opt_price, spot, c['strike_price'], T, r, contract_type.lower())
            if pd.isna(iv) or iv <= 0.01: continue

            delta = OptionsMath.black_scholes_greeks(spot, c['strike_price'], T, r, iv, contract_type.lower())['delta']
            if pd.isna(delta): continue
            
            delta_diff = abs(abs(delta) - abs(target_delta)) 
            
            if delta_diff < min_delta_diff:
                min_delta_diff = delta_diff
                best_contract = c['ticker']

        return best_contract

    # Public Function 5: Selects a NEW contract every day based on DELTA and days to expiry
    def get_daily_rolling_by_delta(self, ticker, start_date, end_date, target_delta, target_dte, contract_type='call', r=0, save_csv=None):
        """
        Constant-Delta Rolling Exposure.
        Calculates IV and Greeks on the fly to find the truest constant-risk contract.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        all_results = []
        
        print(f"Rolling {target_delta}-Delta {contract_type.upper()} Exposure ({len(dates)} days) | Risk-Free Rate: {r}")
        
        for d in dates:
            d_str = d.strftime('%Y-%m-%d')
            
            daily_contract = self._select_contract_by_delta(ticker, d_str, target_delta, target_dte, contract_type, r)
            if not daily_contract: continue
            
            stock_df = self._get_aggs(ticker, d_str, d_str, 1, 'day')
            opt_df = self._get_aggs(daily_contract, d_str, d_str, 1, 'day')
            
            if stock_df.empty or opt_df.empty: continue
            
            merged = pd.merge(
                opt_df.rename(columns={'close': 'close_opt', 'open': 'open_opt', 'volume': 'volume_opt'}), 
                stock_df.rename(columns={'close': 'close_und', 'open': 'open_und'}), 
                on='timestamp', how='inner'
            )
            
            _, expiry, _ = OptionsMath.parse_contract_details(daily_contract)
            merged['days_to_expiry'] = (expiry - merged['timestamp']).dt.days
            
            daily_res = self._process_merged_data(merged, daily_contract, contract_type, r)
            daily_res['active_contract'] = daily_contract
            all_results.append(daily_res)
            
        df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        self._check_data_quality(df)
        
        if not df.empty:
            df['date'] = df['timestamp'].dt.date
            df = df.drop(columns=['timestamp'])
            
            cols = ['date'] + [c for c in df.columns if c != 'date']
            df = df[cols]
        
        if save_csv and not df.empty:
            df.to_csv(save_csv, index=False)
            print(f"Saved delta-rolling history to: {save_csv}")
            
        return df