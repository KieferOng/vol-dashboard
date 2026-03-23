import pandas as pd
import time
from live_engine_v1 import DashboardEngine

def run_test():
    tickers_to_test = [
    "SPY", "QQQ", "IWM", "VGK", "FXI", "EWY", 
    "TLT", "HYG", 
    "GLD", "SLV", "USO", "URA",
    "XLF", "XLE", "XRT", "KRE",
    "XLK", "SMH", "ARKK",
    "XLV"
]
    
    print("--- 🔬 Initializing Live Engine Diagnostic ---")
    
    for ticker in tickers_to_test:
        start_time = time.time()
        print(f"\n📡 Pulling Equity, Live Options, and Historical Context for {ticker}...")
        
        engine = DashboardEngine(ticker)
        dashboard_row = engine.process()
        
        if not dashboard_row or len(dashboard_row) <= 1:
            print(f"❌ Engine failed to return data for {ticker}. Check API or CSV baseline.")
            continue

        df = pd.DataFrame([dashboard_row])
        print("\n" + "="*30)
        print(df.T.to_string(header=False))
        print("="*30)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Processed in {elapsed:.2f} seconds.")
        
        time.sleep(0.5)

if __name__ == "__main__":
    run_test()