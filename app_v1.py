import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from live_engine_v1 import build_all_tickers

st.set_page_config(page_title="Options Insight | Vol Summary", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp {background-color: #04040A; color: white;}
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;} 
    header {background-color: transparent !important;}
    
    [data-testid="collapsedControl"] {
        color: white !important;
        background-color: #1A233A !important;
        border-radius: 4px;
        margin: 10px;
        border: 1px solid #00E676;
    }
    [data-testid="collapsedControl"] svg {
        fill: white !important;
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
    if not os.path.exists("live_dashboard_feed.csv"): 
        return None
    df = pd.read_csv("live_dashboard_feed.csv")
    
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
    st.header("⚙️ Engine Control")
    if st.button("🔄 Force Refresh Data"):
        refresh_data()
    st.divider()
    st.markdown("""
    **Chart Legend:**
    * **IV_Z:** Relative Fear (vs 22d)
    * **Skew_Z:** Put Premium (vs 22d)
    * **Carry:** Overpriced/Underpriced Vol
    """)
    st.divider()
    st.caption("⚠️ **DISCLAIMER:** Live options data is delayed by at least 15 minutes. This dashboard is for educational and quantitative analysis purposes only and does not constitute financial advice.")

st.markdown("<h1 style='color:#00E676;'>Options <span style='color:white;'>insight</span></h1>", unsafe_allow_html=True)
st.markdown("### WEEKLY VOL SUMMARY")
st.divider()

if df is None:
    st.error("No data found! Run `python live_engine.py` first, or click 'Force Refresh Data' in the sidebar.")
    st.stop()

def build_ribbon(category_name, tickers, source_df, col):
    with col:
        st.markdown(f"<span style='color:#00E676; font-weight:bold;'>{category_name}</span>", unsafe_allow_html=True)
        html = "<div style='display:flex; flex-wrap:wrap; gap: 4px;'>"
        for t in tickers:
            row = source_df[source_df['Ticker'] == t]
            if not row.empty:
                val = row['Perf_1W'].values[0]
                color_class = "neon-green" if val > 0 else "neon-red" if val < 0 else "neutral"
                html += f"<div class='metric-box {color_class}' style='flex: 1 1 auto;'>{t}<br>{val:.2f}%</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

def create_text_scatter(plot_df, x_col, y_col, title, x_label, y_label, x_range=None, y_range=None):
    clean_df = plot_df.dropna(subset=[x_col, y_col, 'Perf_1W']).copy()
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=clean_df[x_col], y=clean_df[y_col], text=clean_df['Ticker'], mode="text",
        textfont=dict(color=clean_df['Chart_Color'], size=13, family="Arial Black")
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

def build_dashboard_view(view_df, show_ribbon=True):
    if show_ribbon:
        st.markdown("#### 1-WEEK PERFORMANCE SUMMARY")
        is_macro = view_df['Ticker'].isin(MACRO_TICKERS).any()
        
        rc1, rc2, rc3 = st.columns(3)
        if is_macro:
            build_ribbon("INDICES", MACRO_UNIVERSE["INDICES"], view_df, rc1)
            build_ribbon("BONDS", MACRO_UNIVERSE["BONDS"], view_df, rc2)
            build_ribbon("COMMODITIES", MACRO_UNIVERSE["COMMODITIES"], view_df, rc3)
        else:
            build_ribbon("CYCLICALS", SECTOR_UNIVERSE["CYCLICALS"], view_df, rc1)
            build_ribbon("TECH & INNOVATION", SECTOR_UNIVERSE["TECH_INNOVATION"], view_df, rc2)
            build_ribbon("DEFENSIVES", SECTOR_UNIVERSE["DEFENSIVES"], view_df, rc3)
        st.write("") 

    missing_data = view_df[view_df['IV30'].isna() | view_df['Spot'].isna()]['Ticker'].tolist()
    if missing_data:
        st.warning(f"⚠️ **MISSING DATA:** The following tickers returned 'None' or incomplete data: **{', '.join(missing_data)}**")
    
    anomalies = view_df[view_df['Warnings'] != 'None']
    if not anomalies.empty:
        st.warning(f"🚨 **ANOMALY ALERT:** {len(anomalies)} ticker(s) in this view are showing extreme statistical deviations. Please check the 'WARNINGS' column.")

    st.markdown("#### DATA TAPE")
    
    display_df = view_df.copy()
    columns_to_drop = ['Category', 'Chart_Color', 'IV_Z', 'IV_1D', 'IV_5D', 'Skew_Z', 'Skew_1D', 'Status']
    display_df = display_df.drop(columns=[c for c in columns_to_drop if c in display_df.columns])
    
    table_cols = {
        'Ticker': 'TICKER', 'Spot': 'SPOT', 'Perf_1D': '1D PERF %', 'Perf_1W': '1W PERF %', 
        'Perf_1M': '1M PERF %', 'Perf_3M': '3M PERF %', 'Sigma_1D': '1D σ MOVE', 'Sigma_1W': '1W σ MOVE',
        'HV10': 'HV10', 'HV30': 'HV30', 'IV30': 'IV30', 'IV90': 'IV90', 
        'Term_Structure': 'T/S', 'Carry': 'CARRY', 'IV_Pct': 'IV %-ILE',
        'Skew_1M_25D': '1M 25D SKEW', 'Skew_Pct': 'SKEW %-ILE', 'Skew_5D': '5D Δ SKEW',
        'Divergence': 'DIVERGENCE?', 'Warnings': 'WARNINGS', 'Flag': 'FLAG'
    }
    display_df = display_df.rename(columns=table_cols)
    
    ordered_cols = [
        'TICKER', 'SPOT', '1D PERF %', '1W PERF %', '1M PERF %', '3M PERF %',
        '1D σ MOVE', '1W σ MOVE', 'HV10', 'HV30', 'IV30', 'IV90', 'T/S',
        '1M 25D SKEW', 'CARRY', 'IV %-ILE', 'SKEW %-ILE', '5D Δ SKEW', 
        'DIVERGENCE?', 'WARNINGS', 'FLAG'
    ]
    display_df = display_df[[c for c in ordered_cols if c in display_df.columns]]
    
    format_dict = {
        'SPOT': '{:.2f}', '1D PERF %': '{:.2f}%', '1W PERF %': '{:.2f}%', '1M PERF %': '{:.2f}%', '3M PERF %': '{:.2f}%',
        '1D σ MOVE': '{:.2f}', '1W σ MOVE': '{:.2f}', 'HV10': '{:.2f}', 'HV30': '{:.2f}', 
        'IV30': '{:.2f}', 'IV90': '{:.2f}', 'T/S': '{:.2f}', 'CARRY': '{:.2f}%', 'IV %-ILE': '{:.2f}%',
        '1M 25D SKEW': '{:.2f}', 'SKEW %-ILE': '{:.2f}%', '5D Δ SKEW': '{:.2f}'
    }
    
    numeric_cols = [
        '1D PERF %', '1W PERF %', '1M PERF %', '3M PERF %', '1D σ MOVE', '1W σ MOVE',
        'HV10', 'HV30', 'IV30', 'IV90', 'T/S', 'CARRY', 'IV %-ILE', 
        '1M 25D SKEW', 'SKEW %-ILE', '5D Δ SKEW'
    ]
    numeric_cols = [c for c in numeric_cols if c in display_df.columns]
    
    def style_table(styler):
        def custom_heatmap(val):
            if pd.isna(val) or not isinstance(val, (int, float)): return ''
            if val <= -5.0: return 'background-color: #8B0000; color: white;'
            if val <= -2.0: return 'background-color: #B22222; color: white;'
            if val < 0: return 'background-color: #FF5252; color: white;'
            if val >= 5.0: return 'background-color: #006400; color: white;'
            if val >= 2.0: return 'background-color: #228B22; color: white;'
            if val > 0: return 'background-color: #00E676; color: black;'
            return ''
            
        def color_warnings(val):
            if '⚠️' in str(val) or 'Dead' in str(val) or 'Outlier' in str(val): 
                return 'color: #FF3D00; font-weight: bold;'
            return ''
            
        def color_divergence(val):
            if val == '✅': return 'color: #00E676; font-weight: bold;' 
            return ''

        styler.format(format_dict, na_rep="N/A")
        styler.map(custom_heatmap, subset=numeric_cols)
        styler.map(color_warnings, subset=['WARNINGS'])
        styler.map(color_divergence, subset=['DIVERGENCE?'])
        styler.set_table_styles([{'selector': 'th', 'props': [('background-color', '#1A233A'), ('color', 'white'), ('border', '1px solid #0B0F19')]}])
        return styler

    st.dataframe(display_df.style.pipe(style_table), width="stretch", height=500, hide_index=True)
    st.divider()

    st.markdown("#### VOLATILITY SURFACE CHARTS")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = create_text_scatter(view_df, 'Skew_Z', 'IV_Z', "IMPLIED VOL vs SKEW Z-SCORE", "1M 25 DELTA SKEW (PUT-CALL) - Z-SCORE", "IMPLIED VOL - Z-SCORE", [-5.5, 5.5], [-5.5, 5.5])
        st.plotly_chart(fig1, width="stretch")
    with c2:
        fig2 = create_text_scatter(view_df, 'Sigma_1W', 'IV_Z', "STRATEGY COMPASS", "SPOT (1W SIGMA MOVE)", "IMPLIED VOL - Z-SCORE", [-5.5, 5.5], [-5.5, 5.5])
        for radius in [1.5, 3.0, 4.5]: 
            fig2.add_shape(type="circle", xref="x", yref="y", x0=-radius, y0=-radius, x1=radius, y1=radius, line_color="#3A4A68", line_width=1)
        fig2.add_annotation(x=4, y=4, text="SELL CALL SPREAD", showarrow=False, font=dict(color="#829AB1", size=14))
        fig2.add_annotation(x=-4, y=4, text="SELL PUT SPREAD", showarrow=False, font=dict(color="#829AB1", size=14))
        fig2.add_annotation(x=4, y=-4, text="BUY PUT SPREAD", showarrow=False, font=dict(color="#829AB1", size=14))
        fig2.add_annotation(x=-4, y=-4, text="BUY CALL SPREAD", showarrow=False, font=dict(color="#829AB1", size=14))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig3 = create_text_scatter(view_df, 'IV_Z', 'Term_Structure', "CONTANGO / BACKWARDATION", "IMPLIED VOL - Z-SCORE", "TERM STRUCTURE (IV30/IV90)", [-5.5, 5.5], [0.7, 1.5])
        fig3.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=1.0, y1=1.5, fillcolor="rgba(255, 61, 0, 0.05)", layer="below", line_width=0)
        fig3.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0.7, y1=1.0, fillcolor="rgba(0, 230, 118, 0.05)", layer="below", line_width=0)
        fig3.add_annotation(x=4, y=1.45, text="INVERTED", showarrow=False, font=dict(color="#FF3D00", size=16, family="Arial Black"))
        fig3.add_annotation(x=-4, y=0.75, text="CONTANGO", showarrow=False, font=dict(color="#00E676", size=16, family="Arial Black"))
        st.plotly_chart(fig3, width="stretch")
    with c4:
        fig4 = create_text_scatter(view_df, 'HV10', 'Carry', "CARRY vs 10 DAY REALIZED VOL", "10 DAY REALIZED VOL", "CARRY", [0, 120], [-60, 70])
        fig4.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=70, fillcolor="rgba(0, 230, 118, 0.05)", layer="below", line_width=0)
        fig4.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-60, y1=0, fillcolor="rgba(255, 61, 0, 0.05)", layer="below", line_width=0)
        st.plotly_chart(fig4, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        fig5 = create_text_scatter(view_df, 'Sigma_1W', 'Skew_5D', "BREAKOUT / REVERSAL RISK", "ONE WEEK SIGMA MOVE", "5-DAY CHANGE IN SKEW", [-5.5, 5.5], [-40, 40])
        fig5.add_annotation(x=-4, y=35, text="BREAKOUT RISK", showarrow=False, font=dict(color="#FF3D00", size=14))
        fig5.add_annotation(x=4, y=-35, text="BREAKOUT RISK", showarrow=False, font=dict(color="#FF3D00", size=14))
        fig5.add_annotation(x=4, y=35, text="REVERSAL RISK", showarrow=False, font=dict(color="#FF3D00", size=14))
        fig5.add_annotation(x=-4, y=-35, text="REVERSAL RISK", showarrow=False, font=dict(color="#FF3D00", size=14))
        st.plotly_chart(fig5, width="stretch")

tab1, tab2, tab3 = st.tabs(["🌎 MACRO & ASSET CLASSES", "🏭 SECTORS & THEMES", "📋 ALL 50 TICKERS"])

with tab1:
    macro_df = df[df['Ticker'].isin(MACRO_TICKERS)]
    build_dashboard_view(macro_df, show_ribbon=True)

with tab2:
    sector_df = df[df['Ticker'].isin(SECTOR_TICKERS)]
    build_dashboard_view(sector_df, show_ribbon=True)

with tab3:
    st.info("Displaying the full universe of all 50 tickers.")
    build_dashboard_view(df, show_ribbon=False)