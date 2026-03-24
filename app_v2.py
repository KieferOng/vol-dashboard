import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from live_engine_v2 import build_all_tickers

st.set_page_config(page_title="Vol Dashboard", layout="wide", initial_sidebar_state="expanded")

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
        text-transform: uppercase; /* Optional: makes it look more like a header */
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

def refresh_data():
    with st.spinner("Pulling Live Massive API Data..."):
        build_all_tickers()
        st.cache_data.clear()
        st.rerun()

df = load_data()

with st.sidebar:
    st.header("Dashboard Menu")
    if st.button("Click to Refresh Data"):
        refresh_data()
    st.divider()

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

    st.divider()
    st.caption("⚠️ **DISCLAIMER:** Data is powered by Massive's **Real-time** Fair Market Value (FMV) feed. While calculated to be highly accurate, these prices are proprietary aggregates and may differ slightly from specific exchange quotes.")

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