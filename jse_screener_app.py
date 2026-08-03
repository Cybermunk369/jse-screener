"""
JSE Screener v1.3 - color-coded ratios and scores, pinned columns.
Green shades for positive (darker = stronger), red shades for negative
(darker = stronger). No new signals added - display only.

Run locally:
    pip3 install streamlit yfinance pandas numpy
    python3 -m streamlit run jse_screener_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

RISK_FREE_RATE = 0.08
TRADING_DAYS = 252

JSE_TICKERS = {
    "NPN.JO": "Naspers",
    "SOL.JO": "Sasol",
    "CPI.JO": "Capitec",
    "AGL.JO": "AngloGold Ashanti",
    "SHP.JO": "Shoprite",
    "FSR.JO": "FirstRand",
    "MTN.JO": "MTN Group",
    "SBK.JO": "Standard Bank",
    "ABG.JO": "Absa Group",
    "NED.JO": "Nedbank",
    "CLS.JO": "Clicks Group",
    "WHL.JO": "Woolworths Holdings",
    "PRX.JO": "Prosus",
    "BID.JO": "Bid Corporation",
    "REM.JO": "Remgro",
    "VOD.JO": "Vodacom",
    "IMP.JO": "Impala Platinum",
    "GFI.JO": "Gold Fields",
}

FX_EXPOSURE = {
    "NPN.JO": "Rand Hedge",
    "SOL.JO": "Mixed",
    "CPI.JO": "Domestic",
    "AGL.JO": "Rand Hedge",
    "SHP.JO": "Domestic",
    "FSR.JO": "Domestic",
    "MTN.JO": "Mixed",
    "SBK.JO": "Mixed",
    "ABG.JO": "Domestic",
    "NED.JO": "Domestic",
    "CLS.JO": "Domestic",
    "WHL.JO": "Mixed",
    "PRX.JO": "Rand Hedge",
    "BID.JO": "Rand Hedge",
    "REM.JO": "Mixed",
    "VOD.JO": "Mixed",
    "IMP.JO": "Rand Hedge",
    "GFI.JO": "Rand Hedge",
}


@st.cache_data(ttl=3600)
def get_jse_data(tickers: dict) -> pd.DataFrame:
    rows = []
    for ticker, name in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="6mo")
            if hist.empty:
                continue

            info = t.info

            close_rand = hist["Close"] / 100
            latest_close_rand = close_rand.iloc[-1]
            price_6mo_ago_rand = close_rand.iloc[0]
            momentum_pct = (
                (latest_close_rand - price_6mo_ago_rand) / price_6mo_ago_rand * 100
            )

            daily_returns = close_rand.pct_change().dropna()
            if len(daily_returns) >= 2 and daily_returns.std() != 0:
                annual_return = daily_returns.mean() * TRADING_DAYS
                annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
                sharpe = (annual_return - RISK_FREE_RATE) / annual_vol
            else:
                sharpe = None

            rows.append({
                "Ticker": ticker.replace(".JO", ""),
                "Name": name,
                "Price (R)": round(latest_close_rand, 2),
                "P/E": info.get("trailingPE"),
                "Market Cap (R bn)": (
                    round(info.get("marketCap", 0) / 100 / 1e9, 2)
                    if info.get("marketCap") else None
                ),
                "Sector": info.get("sector", "N/A"),
                "FX Exposure": FX_EXPOSURE.get(ticker, "Unclassified"),
                "6mo Momentum %": round(momentum_pct, 1),
                "Sharpe Ratio": round(sharpe, 2) if sharpe is not None else None,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def add_valuation_score(df):
    valid = df["P/E"].dropna()
    if len(valid) == 0:
        df["Valuation Score"] = 50
        return df
    df["Valuation Score"] = df["P/E"].apply(
        lambda pe: 50 if pd.isna(pe) else max(0, min(100, 100 - (pe / valid.max() * 100)))
    ).round(0)
    return df


def add_momentum_score(df):
    mom = df["6mo Momentum %"]
    if mom.max() == mom.min():
        df["Momentum Score"] = 50
        return df
    df["Momentum Score"] = (
        (mom - mom.min()) / (mom.max() - mom.min()) * 100
    ).round(0)
    return df


def add_sharpe_score(df):
    valid = df["Sharpe Ratio"].dropna()
    if len(valid) == 0 or valid.max() == valid.min():
        df["Sharpe Score"] = 50
        return df
    df["Sharpe Score"] = df["Sharpe Ratio"].apply(
        lambda s: 50 if pd.isna(s) else round(
            (s - valid.min()) / (valid.max() - valid.min()) * 100
        )
    )
    return df


def add_combined_score(df):
    df["Combined Score"] = (
        (df["Valuation Score"] + df["Momentum Score"]) / 2
    ).round(0)
    return df


def _shade(value, max_abs, positive_rgb_light, positive_rgb_dark,
           negative_rgb_light, negative_rgb_dark):
    if pd.isna(value) or max_abs == 0:
        return ""
    intensity = min(abs(value) / max_abs, 1.0)
    light = positive_rgb_light if value >= 0 else negative_rgb_light
    dark = positive_rgb_dark if value >= 0 else negative_rgb_dark
    r = int(light[0] + intensity * (dark[0] - light[0]))
    g = int(light[1] + intensity * (dark[1] - light[1]))
    b = int(light[2] + intensity * (dark[2] - light[2]))
    text_color = "white" if intensity > 0.55 else "black"
    return f"background-color: rgb({r},{g},{b}); color: {text_color}"


def color_ratio(value, max_abs):
    return _shade(
        value, max_abs,
        positive_rgb_light=(220, 237, 200), positive_rgb_dark=(27, 94, 32),
        negative_rgb_light=(255, 224, 224), negative_rgb_dark=(127, 29, 29),
    )


def color_score(value):
    if pd.isna(value):
        return ""
    return color_ratio(value - 50, 50)


def style_table(display_df, momentum_max_abs, sharpe_max_abs):
    styler = display_df.style

    styler = styler.map(
        lambda v: color_ratio(v, momentum_max_abs), subset=["6mo Momentum %"]
    )
    styler = styler.map(
        lambda v: color_ratio(v, sharpe_max_abs), subset=["Sharpe Ratio"]
    )
    for col in ["Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score"]:
        styler = styler.map(color_score, subset=[col])

    return styler


st.set_page_config(page_title="JSE Screener", layout="wide")
st.title("JSE Screener — v1.3")
st.caption(
    "Valuation + momentum ranking, Sharpe ratio, Rand hedge/domestic "
    "classification. Green = positive/strong, red = negative/weak, "
    "darker = more extreme. Free data, updated hourly."
)

with st.spinner("Fetching JSE data..."):
    df = get_jse_data(JSE_TICKERS)

if df.empty:
    st.error("No data returned. Check your internet connection or try again shortly.")
    st.stop()

df = add_valuation_score(df)
df = add_momentum_score(df)
df = add_sharpe_score(df)
df = add_combined_score(df)

momentum_max_abs = df["6mo Momentum %"].abs().max() or 1
sharpe_max_abs = df["Sharpe Ratio"].abs().max() or 1

col1, col2, col3, col4 = st.columns(4)
with col1:
    sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
with col2:
    fx_options = ["All"] + sorted(df["FX Exposure"].unique().tolist())
    fx_filter = st.selectbox("FX Exposure", fx_options)
with col3:
    sort_by = st.selectbox(
        "Sort by",
        ["Combined Score", "Sharpe Ratio", "Valuation Score", "Momentum Score", "6mo Momentum %", "P/E"],
    )
with col4:
    min_score = st.slider("Minimum Combined Score", 0, 100, 0)

filtered = df.copy()
if sector_filter != "All":
    filtered = filtered[filtered["Sector"] == sector_filter]
if fx_filter != "All":
    filtered = filtered[filtered["FX Exposure"] == fx_filter]
filtered = filtered[filtered["Combined Score"] >= min_score]
filtered = filtered.sort_values(sort_by, ascending=False)

display_cols = [
    "Ticker", "Name", "Price (R)", "P/E", "Market Cap (R bn)",
    "Sector", "FX Exposure", "6mo Momentum %", "Sharpe Ratio",
    "Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score",
]
display_df = filtered[display_cols].reset_index(drop=True)

styled = style_table(display_df, momentum_max_abs, sharpe_max_abs)
st.dataframe(
    styled,
    use_container_width=False,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.Column(pinned=True, width="small"),
        "Name": st.column_config.Column(pinned=True, width="medium"),
        "Price (R)": st.column_config.NumberColumn(pinned=True, width="small"),
    },
)

st.caption(
    f"{len(filtered)} of {len(df)} stocks shown. "
    "FX Exposure is a manual classification based on business geography, "
    "not derived from financial filings - treat as approximate. "
    "Sharpe Ratio: risk-adjusted return (annualized, 6mo history, "
    f"{RISK_FREE_RATE*100:.0f}% risk-free rate assumed). "
    "Scores: green above 50, red below 50, darker = further from 50. "
    "Combined Score reflects Valuation + Momentum only. "
    "Not investment advice."
)
