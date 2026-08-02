"""
JSE Screener v1 - the boring version.
Valuation + momentum ranking only. No insider signal yet (that's v2).

Run locally:
    pip3 install streamlit yfinance pandas
    python3 -m streamlit run jse_screener_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd

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

            latest_close_rand = hist["Close"].iloc[-1] / 100
            price_6mo_ago_rand = hist["Close"].iloc[0] / 100
            momentum_pct = (
                (latest_close_rand - price_6mo_ago_rand) / price_6mo_ago_rand * 100
            )

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
                "6mo Momentum %": round(momentum_pct, 1),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def add_valuation_score(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["P/E"].dropna()
    if len(valid) == 0:
        df["Valuation Score"] = 50
        return df
    df["Valuation Score"] = df["P/E"].apply(
        lambda pe: 50 if pd.isna(pe) else max(0, min(100, 100 - (pe / valid.max() * 100)))
    ).round(0)
    return df


def add_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    mom = df["6mo Momentum %"]
    if mom.max() == mom.min():
        df["Momentum Score"] = 50
        return df
    df["Momentum Score"] = (
        (mom - mom.min()) / (mom.max() - mom.min()) * 100
    ).round(0)
    return df


def add_combined_score(df: pd.DataFrame) -> pd.DataFrame:
    df["Combined Score"] = (
        (df["Valuation Score"] + df["Momentum Score"]) / 2
    ).round(0)
    return df


st.set_page_config(page_title="JSE Screener", layout="wide")
st.title("JSE Screener — v1")
st.caption("Valuation + momentum ranking across JSE large-caps. Free data, updated hourly.")

with st.spinner("Fetching JSE data..."):
    df = get_jse_data(JSE_TICKERS)

if df.empty:
    st.error("No data returned. Check your internet connection or try again shortly.")
    st.stop()

df = add_valuation_score(df)
df = add_momentum_score(df)
df = add_combined_score(df)

col1, col2, col3 = st.columns(3)
with col1:
    sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
with col2:
    sort_by = st.selectbox(
        "Sort by",
        ["Combined Score", "Valuation Score", "Momentum Score", "6mo Momentum %", "P/E"],
    )
with col3:
    min_score = st.slider("Minimum Combined Score", 0, 100, 0)

filtered = df.copy()
if sector_filter != "All":
    filtered = filtered[filtered["Sector"] == sector_filter]
filtered = filtered[filtered["Combined Score"] >= min_score]
filtered = filtered.sort_values(sort_by, ascending=False)

st.dataframe(
    filtered[[
        "Ticker", "Name", "Price (R)", "P/E", "Market Cap (R bn)",
        "Sector", "6mo Momentum %", "Valuation Score",
        "Momentum Score", "Combined Score",
    ]],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"{len(filtered)} of {len(df)} stocks shown. "
    "Valuation Score: lower P/E scores higher. "
    "Momentum Score: stronger 6mo price gain scores higher. "
    "Not investment advice."
)