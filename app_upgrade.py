import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import from your existing, untouched live_engine_v2
from live_engine_v2 import build_all_tickers

st.set_page_config(page_title="Vol Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- ORIGINAL V2 CSS ---
st.markdown("""
    <style>
    .stApp {background-color: #04040A; color: white;}
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;} 
    header {background-color: transparent !important;}
            
    div[data-testid="stSelectbox"] label {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        text-transform: uppercase; 
        letter-spacing: 0.5px;
    }
    
    .stDataFrame {background-color: #0B0F19;}
    h1, h2, h3, h4 {color: white; font-family: 'Arial', sans-serif;}
    .metric-box {padding: 6px; text-align: center; font-weight: bold; font-size: 15px; border-radius: 4px;}
    .neon-green {background-color: #00E676; color: black;}
    .neon-red {background-color: #FF3D00; color: white;}
    .neutral {background-color: #333333; color: white;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #0B0F19; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px;}
    .stTabs [aria-selected="true"] {background-color: #1A233A; border-bottom: 2px solid #00E676;}
    </style>
""", unsafe_allow_html=True)

# --- ORIGINAL V2 UNIVERSES ---
MACRO_UNIVERSE = {
    "INDICES": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VGK", "INDA", "FXI", "EWJ", "EWY", "EWZ", "EEM", "EWW"],
    "BONDS": ["IEF", "TLT", "LQD", "HYG"],
    "COMMODITIES": ["USO", "XOM", "GLD", "SLV", "FCX", "URA", "UNG", "ADM", "MOS", "GDX", "SIL"]
}

SECTOR_UNIVERSE = {
    "CYCLICALS": ["XLE", "XOP", "ITB", "XLF", "KRE", "XLI", "XLB", "XME", "XRT", "XLY", "XHB"],
    "TECH_INNOVATION": ["XLK", "SMH", "ARKK", "IBB", "ARKG"],
    "DEFENSIVES": ["XLU", "XLV", "XLP", "VNQ", "XLRE", "ITA"]
}

MACRO_TICKERS = [t for sub in MACRO_UNIVERSE.values() for t in sub]
SECTOR_TICKERS = [t for sub in SECTOR_UNIVERSE.values() for t in sub]

# --- DATA LOADERS ---
@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists("live_dashboard_feed_fast.csv"): 
        return None
    df = pd.read_csv("live_dashboard_feed_fast.csv")
    
    ticker_to_cat = {}
    for cat, tkrs in MACRO_UNIVERSE.items():
        for t in tkrs: ticker_to_cat[t] = cat
    for cat, tkrs in SECTOR_UNIVERSE.items():
        for t in tkrs: ticker_to_cat[t] = cat
        
    df['Category'] = df['Ticker'].map(ticker_to_cat)
    
    color_map = {
        'INDICES': '#00E676', 'CYCLICALS': '#00E676',       
        'BONDS': '#00BFFF', 'TECH_INNOVATION': '#00BFFF',   
        'COMMODITIES': '#FF3D00', 'DEFENSIVES': '#FF3D00'   
    }
    df['Chart_Color'] = df['Category'].map(color_map).fillna('#FFFFFF')
    df['Warnings'] = df['Flag'].astype(str).str.replace('Clean', '').str.replace('|', '').str.strip()
    df['Warnings'] = df['Warnings'].replace('', 'None')
    
    def check_divergence(row):
        if pd.isna(row['Perf_1W']) or pd.isna(row['Skew_5D']): return "❌"
        if row['Perf_1W'] > 0.01 and row['Skew_5D'] > 2.0: return "✅"
        if row['Perf_1W'] < -0.01 and row['Skew_5D'] < -2.0: return "✅"
        return "❌"
        
    df['Divergence'] = df.apply(check_divergence, axis=1)
    return df

@st.cache_data(ttl=300)
def load_spreads():
    if not os.path.exists("live_spread_execution.csv"): return pd.DataFrame()
    return pd.read_csv("live_spread_execution.csv")

@st.cache_data(ttl=3600)
def load_history(ticker):
    path = f"massive_historical_1y_data/{ticker.lower()}.csv"
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    if 'Date' not in df.columns and 'date' in df.columns: 
        df.rename(columns={'date':'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date').sort_index()

def refresh_data():
    with st.spinner("Pulling Live Massive API Data..."):
        build_all_tickers()
        st.cache_data.clear()
        st.rerun()

df = load_data()

# --- SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.header("Dashboard Menu")
    
    page_selection = st.radio("Select View:", ["Macro Vol Dashboard", "Strategy & Execution"])
    st.divider()
    
    if st.button("Click to Refresh Data"):
        refresh_data()
    st.divider()

    # Dynamic Sidebar Context based on Page Selection
    if page_selection == "Macro Vol Dashboard":
        with st.expander("Volatility Metrics"):
            st.markdown("""
            **Performance & Sigma**
            * **SPOT:** Current market price.
            * **1D/1W/1M/3M %:** Price returns over specific windows.
            * **σ MOVE:** Performance divided by historical std dev (22 days of rolling daily/weekly returns). Measures how 'extreme' the move is.
            
            **Volatility**
            * **HV10 / HV30:** 10 and 30-day Realised Volatility.
            * **IV30 / IV90:** 30 and 90-day Implied Volatility.
            * **T/S (Term Structure):** Ratio of IV30 to IV90. <1.0 is Contango, >1.0 is Inverted.
            * **IV %-ILE:** Current IV30 rank relative to its 22-day history.
            
            **Skew & Strategy**
            * **1M 25D SKEW:** Relative cost of 25-delta puts vs calls. Positive means puts are more expensive.
            * **CARRY:** Volatility Risk Premium (IV30 - HV10). Short vol when high, long vol when low.
            * **DIVERGENCE:** Triggered when Spot and Skew move in opposite directions (e.g. Spot up, Puts getting more expensive).
            """)

        with st.expander("Heatmap Colour Key"):
            st.markdown("""
            **1. Performance & Direction (PERF %, σ MOVE)**
            * 🟢 **Green:** Positive returns / Bullish upside moves.
            * 🔴 **Red:** Negative returns / Bearish downside moves.

            **2. Volatility (HV, IV)**
            * 🟢 **Green:** Low absolute historical/implied volatility.
            * 🔴 **Red:** High absolute historical/implied volatility.

            **3. Relative Value & Skew (CARRY, 1M 25D SKEW, 5D Δ SKEW)**
            * 🟢 **Green:** Cheap Puts / Low Volatility Risk Premium.
            * 🔴 **Red:** Expensive Puts / High Volatility Risk Premium.

            **4. Term Structure (T/S)**
            * 🟢 **Green:** Contango (< 1.0). Normal market state.
            * 🔴 **Red:** Inverted (> 1.0). Signal of panic/crash.
            """)

        with st.expander("Chart Axis Guide"):
            st.markdown("""
            **1. IV vs Skew Z-Score**
            * **X:** Put-Call Skew Z-Score
            * **Y:** Implied Vol Z-Score
            
            **2. Strategy Compass**
            * **X:** 1-Week Sigma Move
            * **Y:** Implied Vol Z-Score
            
            **3. Contango / Backwardation**
            * **X:** Implied Vol Z-Score
            * **Y:** Term Structure
            
            **4. Carry vs Realised Vol**
            * **X:** 10-Day Realized Volatility
            * **Y:** Carry
            
            **5. Breakout / Reversal Risk**
            * **X:** 1-Week Sigma Move
            * **Y:** 5-Day Change in Skew
            """)
            
    elif page_selection == "Strategy & Execution":
        with st.expander("Strategy Guide & Definitions"):
            st.markdown("""
            **1. 1m Call / Put Monitor**
            * **X-Axis:** A blended gauge of current market panic. It averages the 1M IV Percentile with the 1M IV/RV Ratio Percentile.
            * **Y-Axis:** 1M Call (or Put) Skew Percentile.
            * **Highlighted Zone (Top):** Tickers above the 80th percentile in Skew, indicating historically expensive tail protection/leverage.

            **2. Selected Top Spreads 25d 10d**
            * Scans the live options chain for executable 1-month debit spreads (Buy 25-delta, Sell 10-delta).
            * **Cost:** Estimated net debit assuming execution at ~25% worse than the mid-price (institutional slippage).
            * **Payout Ratio:** Max Potential Profit divided by Execution Cost. Ratios are strictly filtered between 1.0x and 40.0x for realism.

            **3. Daily Options PnL Summary**
            * A vectorised Black-Scholes backtest engine that simulates daily rolling of short option strategies.
            * **Cumulative PnL:** Total return assuming the strategy was executed daily and held to maturity.
            * **Sharpe Ratio:** Risk-adjusted return (annualised). *Note: 10-day and 20-day Sharpes may appear artificially inflated or deflated due to low short-term variance.*

            **4. Historical Percentiles: Skew / Volatility**
            * **Top Chart:** Displays the raw 25d Put / 25d Call ratio.
            * **Bottom Chart:** Displays the raw 25d Call / ATM ratio.
            * Overlays 1-Month (solid lines) and 3-Month (dotted lines) tenors to visualise term structure shifts and historical norms.
            """)

    st.divider()
    st.caption("⚠️ **DISCLAIMER:** Data is powered by Massive's **Real-time** Fair Market Value (FMV) feed. While calculated to be highly accurate, these prices are proprietary aggregates and may differ slightly from specific exchange quotes.")

# ---------------------------------------------------------------------------
# PAGE 1: MACRO VOL DASHBOARD (EXACT UNABRIDGED REPLICA OF APP_V2.PY)
# ---------------------------------------------------------------------------
if page_selection == "Macro Vol Dashboard":
    st.markdown("<h1 style='color:#00E676;'>NUSSIF <span style='color:white;'>VOL DASHBOARD</span></h1>", unsafe_allow_html=True)
    st.caption("👈 **Click the >> icon in the top left to open the dashboard menu to refresh data and view metric definitions.**")
    st.divider()

    if df is None:
        st.error("No data found! Run `python live_engine_fast.py` first, or click 'Click to Refresh Data' in the sidebar.")
        st.stop()

    def build_ribbon(category_name, tickers, source_df, col, header_color="#00E676"):
        with col:
            st.markdown(f"<span style='color:{header_color}; font-weight:bold;'>{category_name}</span>", unsafe_allow_html=True)
            html = "<div style='display:flex; flex-wrap:wrap; gap: 4px;'>"
            for t in tickers:
                row = source_df[source_df['Ticker'] == t]
                if not row.empty:
                    val = row['Perf_1W'].values[0]
                    color_class = "neon-green" if val > 0 else "neon-red" if val < 0 else "neutral"
                    html += f"<div class='metric-box {color_class}' style='flex: 1 1 auto;'>{t}<br>{val:.2f}%</div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    def create_text_scatter(plot_df, x_col, y_col, title, x_label, y_label, x_range=None, y_range=None, highlight_ticker="None"):
        clean_df = plot_df.dropna(subset=[x_col, y_col, 'Perf_1W']).copy()
        
        if highlight_ticker != "None" and highlight_ticker in clean_df['Ticker'].values:
            text_colors = ["#FFFF00" if t == highlight_ticker else "rgba(255,255,255,0.15)" for t in clean_df['Ticker']]
            text_sizes = [24 if t == highlight_ticker else 10 for t in clean_df['Ticker']]
        else:
            text_colors = clean_df['Chart_Color'].tolist()
            text_sizes = [13] * len(clean_df)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=clean_df[x_col], y=clean_df[y_col], text=clean_df['Ticker'], mode="text",
            textfont=dict(color=text_colors, size=text_sizes, family="Arial Black")
        ))
        
        layout_args = dict(
            title=dict(text=title, font=dict(color='white', size=16)),
            plot_bgcolor='#0B0F19', paper_bgcolor='#04040A', 
            xaxis=dict(title=x_label, color='white', showgrid=True, gridcolor='#1A233A', zeroline=True, zerolinecolor='#4A5568'),
            yaxis=dict(title=y_label, color='white', showgrid=True, gridcolor='#1A233A', zeroline=True, zerolinecolor='#4A5568'),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        if x_range: layout_args['xaxis']['range'] = x_range
        if y_range: layout_args['yaxis']['range'] = y_range
            
        fig.update_layout(**layout_args)
        return fig

    def build_dashboard_view(view_df, tab_name, show_ribbon=True):
        if show_ribbon:
            st.markdown("#### 1-WEEK PERFORMANCE SUMMARY")
            is_macro = view_df['Ticker'].isin(MACRO_TICKERS).any()
            
            rc1, rc2, rc3 = st.columns(3)
            if is_macro:
                build_ribbon("INDICES", MACRO_UNIVERSE["INDICES"], view_df, rc1, header_color="#00E676")
                build_ribbon("BONDS", MACRO_UNIVERSE["BONDS"], view_df, rc2, header_color="#00BFFF")
                build_ribbon("COMMODITIES", MACRO_UNIVERSE["COMMODITIES"], view_df, rc3, header_color="#FF3D00")
            else:
                build_ribbon("CYCLICALS", SECTOR_UNIVERSE["CYCLICALS"], view_df, rc1, header_color="#00E676")
                build_ribbon("TECH & INNOVATION", SECTOR_UNIVERSE["TECH_INNOVATION"], view_df, rc2, header_color="#00BFFF")
                build_ribbon("DEFENSIVES", SECTOR_UNIVERSE["DEFENSIVES"], view_df, rc3, header_color="#FF3D00")
            st.write("")

        missing_data = view_df[view_df['IV30'].isna() | view_df['Spot'].isna()]['Ticker'].tolist()
        if missing_data:
            st.warning(f"⚠️ **WARNING:** The following tickers returned 'None' or incomplete data: **{', '.join(missing_data)}**")
        
        anomalies = view_df[view_df['Warnings'] != 'None']
        if not anomalies.empty:
            st.warning(f"🚨 **WARNING:** {len(anomalies)} ticker(s) in this view are showing extreme statistical deviations. Please check the 'WARNINGS' column before taking action.")

        st.markdown("#### VOLATILITY METRICS")
        
        display_df = view_df.copy()
        columns_to_drop = ['Category', 'Chart_Color', 'IV_Z', 'IV_1D', 'IV_5D', 'Skew_Z', 'Skew_1D', 'Status']
        display_df = display_df.drop(columns=[c for c in columns_to_drop if c in display_df.columns])
        
        table_cols = {
            'Ticker': 'TICKER', 'Spot': 'SPOT', 'Perf_1D': '1D PERF %', 'Perf_1W': '1W PERF %', 
            'Perf_1M': '1M PERF %', 'Perf_3M': '3M PERF %', 'Sigma_1D': '1D σ MOVE', 'Sigma_1W': '1W σ MOVE',
            'HV10': 'HV10', 'HV30': 'HV30', 'IV30': 'IV30', 'IV90': 'IV90', 
            'Term_Structure': 'T/S', 'Carry': 'CARRY', 'IV_Pct': 'IV %-ILE',
            'Skew_1M_25D': '1M 25D SKEW', 'Skew_Pct': 'SKEW %-ILE', 'Skew_5D': '5D Δ SKEW',
            'Divergence': 'DIVERGENCE?', 'Warnings': 'WARNINGS'
        }
        display_df = display_df.rename(columns=table_cols)
        
        ordered_cols = [
            'TICKER', 'SPOT', '1D PERF %', '1W PERF %', '1M PERF %', '3M PERF %',
            '1D σ MOVE', '1W σ MOVE', 'HV10', 'HV30', 'IV30', 'IV90', 'T/S',
            '1M 25D SKEW', 'CARRY', 'IV %-ILE', 'SKEW %-ILE', '5D Δ SKEW', 
            'DIVERGENCE?', 'WARNINGS'
        ]
        display_df = display_df[[c for c in ordered_cols if c in display_df.columns]]
        
        format_dict = {
            'SPOT': '{:.1f}', '1D PERF %': '{:.1f}%', '1W PERF %': '{:.1f}%', '1M PERF %': '{:.1f}%', '3M PERF %': '{:.1f}%',
            '1D σ MOVE': '{:.1f}', '1W σ MOVE': '{:.1f}', 'HV10': '{:.1f}', 'HV30': '{:.1f}', 
            'IV30': '{:.1f}', 'IV90': '{:.1f}', 'T/S': '{:.2f}', 'CARRY': '{:.1f}%', 'IV %-ILE': '{:.1f}%',
            '1M 25D SKEW': '{:.1f}', 'SKEW %-ILE': '{:.1f}%', '5D Δ SKEW': '{:.1f}'
        }

        def style_table(styler):
            def color_warnings(val):
                if '⚠️' in str(val) or 'Dead' in str(val) or 'Outlier' in str(val): 
                    return 'color: #FF3D00; font-weight: bold;'
                return ''
                
            def color_divergence(val):
                if val == '✅': return 'color: #00E676; font-weight: bold;' 
                return ''

            styler.format(format_dict, na_rep="N/A")
            
            perf_cols = ['1D PERF %', '1W PERF %', '1M PERF %', '3M PERF %', '1D σ MOVE', '1W σ MOVE']
            for col in perf_cols:
                if col in display_df.columns:
                    m = display_df[col].abs().quantile(0.85) or 1.0
                    styler.background_gradient(cmap='RdYlGn', subset=[col], vmin=-m, vmax=m)

            vol_cols = ['1M 25D SKEW', '5D Δ SKEW', 'CARRY', 'HV10', 'HV30', 'IV30', 'IV90', 'IV %-ILE', 'SKEW %-ILE']
            for col in vol_cols:
                if col in display_df.columns:
                    if col in ['CARRY', '5D Δ SKEW', '1M 25D SKEW']:
                        v_min = display_df[col].quantile(0.05)
                        v_max = display_df[col].quantile(0.95)
                        
                        if pd.isna(v_min) or v_min == v_max: 
                            v_min, v_max = display_df[col].min(), display_df[col].max()
                            
                        styler.background_gradient(cmap='RdYlGn_r', subset=[col], vmin=v_min, vmax=v_max)
                    else:
                        m = display_df[col].quantile(0.90) or 1.0
                        styler.background_gradient(cmap='RdYlGn_r', subset=[col], vmin=0, vmax=m)

            if 'T/S' in display_df.columns:
                m = (display_df['T/S'] - 1.0).abs().quantile(0.90)
                if pd.isna(m) or m == 0: m = 0.1
                styler.background_gradient(cmap='RdYlGn_r', subset=['T/S'], vmin=1.0 - m, vmax=1.0 + m)

            styler.map(color_warnings, subset=['WARNINGS'])
            styler.map(color_divergence, subset=['DIVERGENCE?'])

            text_cols = [c for c in ['WARNINGS'] if c in display_df.columns]
            if text_cols:
                styler.set_properties(subset=text_cols, **{
                    'white-space': 'normal',
                    'min-width': '250px'
                })
            
            styler.set_table_styles([{'selector': 'th', 'props': [('background-color', '#1A233A'), ('color', 'white'), ('border', '1px solid #0B0F19')]}])
            return styler

        st.dataframe(
            display_df.style.pipe(style_table), 
            width="stretch", 
            height=500, 
            hide_index=True,
            column_config={
                "WARNINGS": st.column_config.TextColumn(width="large")
            }
        )
        st.divider()

        st.markdown("#### VOLATILITY CHARTS")
        
        highlight_ticker = st.selectbox(
            "🔎 Search & Highlight Ticker in Volatility Charts", 
            ["None"] + sorted(view_df['Ticker'].tolist()),
            key=f"search_{tab_name}"
        )
        st.write("") 

        label_box_style = dict(
            bgcolor="#000000",
            bordercolor="#3A4A68",
            borderwidth=1.5,
            borderpad=5,
            opacity=0.9,
            standoff=10
        )

        c1, c2 = st.columns(2)
        with c1:
            fig1 = create_text_scatter(view_df, 'Skew_Z', 'IV_Z', "IMPLIED VOL vs SKEW Z-SCORE", "1M 25 DELTA SKEW (PUT-CALL) - Z-SCORE", "IMPLIED VOL - Z-SCORE", [-5.5, 5.5], [-5.5, 5.5], highlight_ticker)
            st.plotly_chart(fig1, width="stretch")
        with c2:
            fig2 = create_text_scatter(view_df, 'Sigma_1W', 'IV_Z', "STRATEGY COMPASS", "SPOT (1W SIGMA MOVE)", "IMPLIED VOL - Z-SCORE", [-5.5, 5.5], [-5.5, 5.5], highlight_ticker)
            for radius in [1.5, 3.0, 4.5]: 
                fig2.add_shape(type="circle", xref="x", yref="y", x0=-radius, y0=-radius, x1=radius, y1=radius, line_color="#3A4A68", line_width=1)
                
            fig2.add_annotation(x=4, y=4, text="<b>SELL CALL SPREAD</b>", showarrow=False, font=dict(color="#FF3D00", size=14), **label_box_style)
            fig2.add_annotation(x=4, y=-4, text="<b>BUY PUT SPREAD</b>", showarrow=False, font=dict(color="#FF3D00", size=14), **label_box_style)
            
            fig2.add_annotation(x=-4, y=4, text="<b>SELL PUT SPREAD</b>", showarrow=False, font=dict(color="#00E676", size=14), **label_box_style)
            fig2.add_annotation(x=-4, y=-4, text="<b>BUY CALL SPREAD</b>", showarrow=False, font=dict(color="#00E676", size=14), **label_box_style)
            
            st.plotly_chart(fig2, width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            fig3 = create_text_scatter(view_df, 'IV_Z', 'Term_Structure', "CONTANGO / BACKWARDATION", "IMPLIED VOL - Z-SCORE", "TERM STRUCTURE (IV30/IV90)", [-5.5, 5.5], [0.7, 1.5], highlight_ticker)
            fig3.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=1.0, y1=1.5, fillcolor="rgba(255, 61, 0, 0.05)", layer="below", line_width=0)
            fig3.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0.7, y1=1.0, fillcolor="rgba(0, 230, 118, 0.05)", layer="below", line_width=0)
            
            fig3.add_annotation(x=4, y=1.45, text="<b>INVERTED</b>", showarrow=False, font=dict(color="#FF3D00", size=16, family="Arial Black"), **label_box_style)
            fig3.add_annotation(x=-4, y=0.75, text="<b>CONTANGO</b>", showarrow=False, font=dict(color="#00E676", size=16, family="Arial Black"), **label_box_style)
            
            st.plotly_chart(fig3, width="stretch")
        with c4:
            fig4 = create_text_scatter(view_df, 'HV10', 'Carry', "CARRY vs 10 DAY REALIZED VOL", "10 DAY REALIZED VOL", "CARRY", [0, 120], [-60, 70], highlight_ticker)
            fig4.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=70, fillcolor="rgba(0, 230, 118, 0.05)", layer="below", line_width=0)
            fig4.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-60, y1=0, fillcolor="rgba(255, 61, 0, 0.05)", layer="below", line_width=0)
            
            st.plotly_chart(fig4, width="stretch")

        c5, c6 = st.columns(2)
        with c5:
            fig5 = create_text_scatter(view_df, 'Sigma_1W', 'Skew_5D', "BREAKOUT / REVERSAL RISK", "ONE WEEK SIGMA MOVE", "5-DAY CHANGE IN SKEW", [-5.5, 5.5], [-40, 40], highlight_ticker)
            
            fig5.add_annotation(x=-4, y=35, text="<b>BREAKOUT RISK</b>", showarrow=False, font=dict(color="#00E676", size=14), **label_box_style)
            fig5.add_annotation(x=4, y=-35, text="<b>BREAKOUT RISK</b>", showarrow=False, font=dict(color="#00E676", size=14), **label_box_style)
            
            fig5.add_annotation(x=4, y=35, text="<b>REVERSAL RISK</b>", showarrow=False, font=dict(color="#FF3D00", size=14), **label_box_style)
            fig5.add_annotation(x=-4, y=-35, text="<b>REVERSAL RISK</b>", showarrow=False, font=dict(color="#FF3D00", size=14), **label_box_style)
            
            st.plotly_chart(fig5, width="stretch")

    tab1, tab2, tab3 = st.tabs(["⭐ MACRO ⭐", "🏭 SECTORS 🏭", "🌎 ALL 50 TICKERS 🌎"])

    with tab1:
        macro_df = df[df['Ticker'].isin(MACRO_TICKERS)]
        build_dashboard_view(macro_df, tab_name="macro", show_ribbon=True)

    with tab2:
        sector_df = df[df['Ticker'].isin(SECTOR_TICKERS)]
        build_dashboard_view(sector_df, tab_name="sectors", show_ribbon=True)

    with tab3:
        st.info("Displaying the full universe of all 50 tickers.")
        build_dashboard_view(df, tab_name="all", show_ribbon=False)

# ---------------------------------------------------------------------------
# PAGE 2: STRATEGY & EXECUTION (NEW ENHANCEMENTS)
# ---------------------------------------------------------------------------
elif page_selection == "Strategy & Execution":
    st.markdown("<h1 style='color:#00BFFF;'>NUSSIF <span style='color:white;'>STRATEGY & EXECUTION</span></h1>", unsafe_allow_html=True)
    st.divider()

    # Create mapping dictionary from the loaded main dataframe to match scatter colors
    ticker_colors = dict(zip(df['Ticker'], df['Chart_Color'])) if df is not None else {}

    # Sort tickers to correctly identify default indexes
    sorted_tickers = sorted(MACRO_TICKERS + SECTOR_TICKERS)
    default_spy_idx = sorted_tickers.index("SPY") if "SPY" in sorted_tickers else 0

    # --- IMAGE 1 (Top): 1M Call/Put Monitor ---
    st.markdown("#### 1m Call / Put Monitor")
    
    scatter_data = []
    
    for tkr in sorted_tickers:
        h_df = load_history(tkr)
        
        required_cols = ['Spot', 'IV30', '1M_25dC/ATM', '1M_25dP/ATM']
        if not h_df.empty and all(c in h_df.columns for c in required_cols) and len(h_df) > 20:
            
            h_df['Return'] = h_df['Spot'].pct_change()
            h_df['RV20'] = h_df['Return'].rolling(20).std() * np.sqrt(252) * 100
            
            h_df['IV_RV_Ratio'] = h_df['IV30'] / h_df['RV20'].replace(0, np.nan)
            
            clean_df = h_df.dropna(subset=['IV30', 'IV_RV_Ratio', '1M_25dC/ATM', '1M_25dP/ATM'])
            
            if not clean_df.empty:
                latest = clean_df.iloc[-1]
                
                iv_pct = (clean_df['IV30'] <= latest['IV30']).mean() * 100
                iv_rv_pct = (clean_df['IV_RV_Ratio'] <= latest['IV_RV_Ratio']).mean() * 100
                
                blended_x_axis = (iv_pct + iv_rv_pct) / 2.0
                
                call_skew_pct = (clean_df['1M_25dC/ATM'] <= latest['1M_25dC/ATM']).mean() * 100
                put_skew_pct = (clean_df['1M_25dP/ATM'] <= latest['1M_25dP/ATM']).mean() * 100
                
                base_color = ticker_colors.get(tkr, '#00E676')

                scatter_data.append({
                    "Ticker": tkr, 
                    "Blended_X": blended_x_axis, 
                    "Put_Skew_Pct": put_skew_pct, 
                    "Call_Skew_Pct": call_skew_pct,
                    "Base_Color": base_color
                })
            
    if scatter_data:
        plot_df = pd.DataFrame(scatter_data)
        c1, c2 = st.columns(2)
        
        with c1:
            fig_c = go.Figure()
            
            # FIXED: Single horizontal band highlighting ONLY the Y >= 80 zone across the whole X-axis
            fig_c.add_shape(type="rect", xref="x", yref="y", x0=-5, x1=105, y0=80, y1=110, fillcolor="rgba(0, 230, 118, 0.15)", layer="below", line_width=0)
            
            # FIXED: Forces all dots to retain their Base_Color, no overrides.
            fig_c.add_trace(go.Scatter(
                x=plot_df['Blended_X'], y=plot_df['Call_Skew_Pct'], text=plot_df['Ticker'], 
                mode='markers+text', textposition="top center", 
                marker=dict(color=plot_df['Base_Color'], size=9, line=dict(color='black', width=1))
            ))
            fig_c.update_layout(
                title="1m Call Monitor", 
                xaxis_title="avg(1m iv/rv %tile, 1m atm %tile)", 
                yaxis_title="Call Skew %tile", 
                plot_bgcolor='white', font=dict(color='black')
            )
            # Extends Y-axis past 100
            fig_c.update_xaxes(range=[-5, 105], showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF')
            fig_c.update_yaxes(range=[-5, 110], showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF')
            st.plotly_chart(fig_c, width='stretch')
            
        with c2:
            fig_p = go.Figure()
            
            # FIXED: Single horizontal band highlighting ONLY the Y >= 80 zone across the whole X-axis
            fig_p.add_shape(type="rect", xref="x", yref="y", x0=-5, x1=105, y0=80, y1=110, fillcolor="rgba(255, 61, 0, 0.15)", layer="below", line_width=0)
            
            # FIXED: Forces all dots to retain their Base_Color, no overrides.
            fig_p.add_trace(go.Scatter(
                x=plot_df['Blended_X'], y=plot_df['Put_Skew_Pct'], text=plot_df['Ticker'], 
                mode='markers+text', textposition="top center", 
                marker=dict(color=plot_df['Base_Color'], size=9, line=dict(color='black', width=1))
            ))
            fig_p.update_layout(
                title="1m Put Monitor", 
                xaxis_title="avg(1m iv/rv %tile, 1m atm %tile)", 
                yaxis_title="Put Skew %tile", 
                plot_bgcolor='white', font=dict(color='black')
            )
            # Extends Y-axis past 100
            fig_p.update_xaxes(range=[-5, 105], showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF')
            fig_p.update_yaxes(range=[-5, 110], showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF')
            st.plotly_chart(fig_p, width='stretch')
    else:
        st.warning("Run `flatfiles_builder_upgrade.py` to calculate exact Skew percentiles.")

    st.divider()

    # --- IMAGE 2: Selected Top Spreads ---
    st.markdown("#### Selected Top Spreads 25d 10d")
    spread_df = load_spreads()
    
    if not spread_df.empty:
        # Search/Filter Function
        spread_search_ticker = st.selectbox(
            "🔎 Filter Spreads by Ticker", 
            ["All"] + sorted(spread_df['Ticker'].unique().tolist()),
            key="spread_ticker_filter"
        )
        st.write("")

        filtered_spreads = spread_df.copy()
        if spread_search_ticker != "All":
            filtered_spreads = filtered_spreads[filtered_spreads['Ticker'] == spread_search_ticker]

        c3, c4 = st.columns(2)
        
        def format_spread_table(df_subset):
            # Mathematically block out unrealistic payout ratios (>40) AND those lower than 1.0
            df_subset = df_subset[(df_subset['Payout Ratio'] <= 40.0) & (df_subset['Payout Ratio'] >= 1.0)]

            display_cols = ['Ticker', 'Expiration', 'Strike 1', 'Strike 2', 'S1 %spot', 'S2 %spot', 'Cost', 'Cost % Spot', 'Payout Ratio']
            df_display = df_subset[display_cols].copy()
            # Sort and display all eligible results
            df_display = df_display.sort_values(by="Payout Ratio", ascending=False)
            
            format_dict = {
                'Strike 1': '{:.1f}', 'Strike 2': '{:.1f}',
                'S1 %spot': '{:.1f}', 'S2 %spot': '{:.1f}', 
                'Cost': '{:.2f}', 'Cost % Spot': '{:.1f}', 'Payout Ratio': '{:.1f}'
            }
            return df_display.style.format(format_dict).background_gradient(subset=['Payout Ratio'], cmap='Blues', vmin=4.0, vmax=11.5).set_table_styles([{'selector': 'th', 'props': [('background-color', 'white'), ('color', 'black')]}])

        with c3:
            st.markdown("**Selected Top Put Spreads 25d 10d**")
            put_display = filtered_spreads[filtered_spreads['Spread_Type'] == 'PUT']
            if put_display.empty:
                st.info("No viable Put spreads found for this selection.")
            else:
                st.dataframe(format_spread_table(put_display), hide_index=True, width='stretch')
        with c4:
            st.markdown("**Selected Top Call Spreads 25d 10d**")
            call_display = filtered_spreads[filtered_spreads['Spread_Type'] == 'CALL']
            if call_display.empty:
                st.info("No viable Call spreads found for this selection.")
            else:
                st.dataframe(format_spread_table(call_display), hide_index=True, width='stretch')
    else:
        st.info("Run `python build_spreads_upgrade.py` to populate execution tables.")

    st.divider()

    # --- IMAGE 3: SPX Daily Options PnL Summary ---
    st.markdown("#### Daily Options PnL Summary")
    
    if os.path.exists("pnl_backtest_results.csv"):
        pnl_df = pd.read_csv("pnl_backtest_results.csv")
        
        pnl_ticker = st.selectbox("Select Ticker for PnL Summary", sorted_tickers, index=default_spy_idx, key="pnl_ticker_select")
        
        selected_pnl = pnl_df[pnl_df['Ticker'] == pnl_ticker].drop(columns=['Ticker'])
        
        cum_cols = ['Strategy', '1d', '10d', '20d', '60d', 'ytd', '1y']
        sharpe_cols = ['Strategy', 'Sharpe_10d', 'Sharpe_20d', 'Sharpe_60d', 'Sharpe_ytd', 'Sharpe_1y']
        
        cum_df = selected_pnl[cum_cols]
        sharpe_df = selected_pnl[sharpe_cols].rename(columns=lambda x: x.replace('Sharpe_', '') if 'Sharpe_' in x else x)

        def format_pct(val):
            if pd.isna(val): return ""
            return f"{val:.1%}"
            
        def format_float(val):
            if pd.isna(val): return ""
            return f"{val:.1f}"

        def color_pnl(val):
            if pd.isna(val) or type(val) == str: return ''
            if val > 0: return 'background-color: #c6efce; color: #006100'
            elif val < 0: return 'background-color: #ffc7ce; color: #9c0006'
            return ''

        st.markdown("**Cumulative PnL**")
        st.dataframe(
            cum_df.style.format({c: format_pct for c in cum_df.columns if c != 'Strategy'})
            .map(color_pnl, subset=[c for c in cum_df.columns if c != 'Strategy'])
            .set_table_styles([{'selector': 'th', 'props': [('background-color', '#9c0006'), ('color', 'white')]}]),
            hide_index=True,
            width='stretch'
        )
        
        st.write("")
        st.markdown("**Sharpe Ratio**")
        st.dataframe(
            sharpe_df.style.format({c: format_float for c in sharpe_df.columns if c != 'Strategy'})
            .map(color_pnl, subset=[c for c in sharpe_df.columns if c != 'Strategy'])
            .set_table_styles([{'selector': 'th', 'props': [('background-color', '#9c0006'), ('color', 'white')]}]),
            hide_index=True,
            width='stretch'
        )
    else:
        st.info("Run `python build_pnl_summary_upgrade.py` to populate this table.")

    st.divider()

    # --- IMAGE 4: Historical Percentiles ---
    st.markdown("#### Historical Percentiles: Skew / Volatility")
    
    sel_ticker = st.selectbox("Select Ticker for Historical Skew", sorted_tickers, index=default_spy_idx, key="history_ticker_select")
    
    hist_df = load_history(sel_ticker)
    
    if not hist_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        
        has_1m = '1M_25dP/25dC' in hist_df.columns and not hist_df['1M_25dP/25dC'].dropna().empty
        has_3m = '3M_25dP/25dC' in hist_df.columns and not hist_df['3M_25dP/25dC'].dropna().empty
        
        if has_1m:
            plot_df_1m = hist_df.dropna(subset=['1M_25dP/25dC', '1M_25dC/ATM'])
            fig.add_trace(
                go.Scatter(x=plot_df_1m.index, y=plot_df_1m['1M_25dP/25dC'], line=dict(color='lightgray'), name=f"{sel_ticker} 1M Skew - 25dP/25dC"), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_df_1m.index, y=plot_df_1m['1M_25dC/ATM'], line=dict(color='pink'), name=f"{sel_ticker} 1M Call Skew - 25dC/ATM"), 
                row=2, col=1
            )
        
        if has_3m:
            plot_df_3m = hist_df.dropna(subset=['3M_25dP/25dC', '3M_25dC/ATM'])
            fig.add_trace(
                go.Scatter(x=plot_df_3m.index, y=plot_df_3m['3M_25dP/25dC'], line=dict(color='gray', dash='dot'), name=f"{sel_ticker} 3M Skew - 25dP/25dC"), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_df_3m.index, y=plot_df_3m['3M_25dC/ATM'], line=dict(color='#8B0000', dash='dot'), name=f"{sel_ticker} 3M Call Skew - 25dC/ATM"), 
                row=2, col=1
            )
        else:
            st.info(f"ℹ️ **Note:** 3-Month options data is unavailable or insufficient for {sel_ticker} due to low chain liquidity. Displaying 1-Month data only.")
            
        fig.update_layout(height=500, plot_bgcolor='white', font=dict(color='black'), showlegend=True)
        
        # Enhanced Gridline Granularity
        fig.update_xaxes(showgrid=True, gridcolor='#E5E7EB', tickformat="%b %Y", dtick="M1", zeroline=True, zerolinecolor='#9CA3AF')
        fig.update_yaxes(showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF')
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning(f"No valid historical data found for {sel_ticker} in the 1Y folder. Ensure `flatfiles_builder_upgrade.py` has finished running.")