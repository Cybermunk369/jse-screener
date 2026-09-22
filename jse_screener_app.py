"""
JSE Screener v1.4 - correctness pass.

Changes from v1.3:
  - Market cap is no longer divided by 100. yfinance returns marketCap in ZAR
    while price history is in ZAc (cents) - mixed units. v1.3 showed every
    market cap 100x too small.
  - AGL.JO corrected to Anglo American plc. AngloGold Ashanti is ANG.JO.
  - Negative or zero P/E is excluded from the Valuation Score. v1.3 clipped it
    to 100, so loss-making companies ranked as the cheapest stocks.
  - Missing inputs render as a dash instead of a silent neutral 50.
  - Percentile ranking replaces min-max scaling: outlier-robust, and stable as
    the universe grows.
  - Display price uses unadjusted Close; returns use Adj Close.
  - Tickers that fail to load are reported instead of silently dropped.
  - Colour ramp brightens toward extremes so it stays readable in dark mode.

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
MIN_HISTORY_DAYS = 100  # ~6 months of JSE trading days, with slack

JSE_TICKERS = {
    "NPN.JO": "Naspers",
    "SOL.JO": "Sasol",
    "CPI.JO": "Capitec",
    "AGL.JO": "Anglo American",
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
def get_jse_data(tickers: dict):
    """Returns (DataFrame, failures) - failures is a list of (ticker, reason)."""
    rows = []
    failures = []

    for ticker, name in tickers.items():
        try:
            t = yf.Ticker(ticker)
            # auto_adjust=False so Close stays the actual traded price for
            # display, while Adj Close drives return calculations.
            hist = t.history(period="6mo", auto_adjust=False)

            if hist.empty:
                failures.append((ticker, "no price history"))
                continue

            if len(hist) < MIN_HISTORY_DAYS:
                # Short history would be ranked head-to-head against full
                # 6-month windows - not comparable.
                failures.append((ticker, f"only {len(hist)} bars"))
                continue

            info = t.info

            # JSE prices come back in ZAc (cents).
            close_rand = hist["Close"] / 100
            adj_series = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
            adj_rand = adj_series / 100

            latest_close_rand = close_rand.iloc[-1]
            momentum_pct = (adj_rand.iloc[-1] - adj_rand.iloc[0]) / adj_rand.iloc[0] * 100

            daily_returns = adj_rand.pct_change().dropna()
            if len(daily_returns) >= 2 and daily_returns.std() != 0:
                annual_return = daily_returns.mean() * TRADING_DAYS
                annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
                sharpe = (annual_return - RISK_FREE_RATE) / annual_vol
            else:
                sharpe = None

            market_cap = info.get("marketCap")

            rows.append({
                "Ticker": ticker.replace(".JO", ""),
                "Name": name,
                "Price (R)": round(latest_close_rand, 2),
                "P/E": info.get("trailingPE"),
                # marketCap is already in ZAR - do NOT divide by 100.
                "Market Cap (R bn)": round(market_cap / 1e9, 2) if market_cap else None,
                "Sector": info.get("sector", "N/A"),
                "FX Exposure": FX_EXPOSURE.get(ticker, "Unclassified"),
                "6mo Momentum %": round(momentum_pct, 1),
                "Sharpe Ratio": round(sharpe, 2) if sharpe is not None else None,
            })
        except Exception as e:
            failures.append((ticker, type(e).__name__))
            continue

    return pd.DataFrame(rows), failures

def add_valuation_score(df):
    """Percentile rank on P/E - cheaper is better. Negative or zero P/E is
    excluded (a loss-making company is not 'cheap')."""
    pe = df["P/E"].where(df["P/E"] > 0)
    df["Valuation Score"] = pe.rank(pct=True, ascending=False).mul(100).round(0)
    return df

def add_momentum_score(df):
    df["Momentum Score"] = (
        df["6mo Momentum %"].rank(pct=True, ascending=True).mul(100).round(0)
    )
    return df

def add_sharpe_score(df):
    df["Sharpe Score"] = (
        df["Sharpe Ratio"].rank(pct=True, ascending=True).mul(100).round(0)
    )
    return df

def add_combined_score(df):
    """Mean of Valuation and Momentum. Where one input is missing the other
    carries it - the missing input shows as a dash in the table so the user
    can see the score is based on partial data."""
    df["Combined Score"] = (
        df[["Valuation Score", "Momentum Score"]].mean(axis=1, skipna=True).round(0)
    )
    return df

def safe_max_abs(series):
    """Guard against an all-NaN column: NaN is truthy, so `x or 1` is not safe."""
    m = series.abs().max()
    return float(m) if pd.notna(m) and m > 0 else 1.0

def _shade(value, max_abs, positive_rgb_weak, positive_rgb_strong,
           negative_rgb_weak, negative_rgb_strong):
    if pd.isna(value) or max_abs == 0:
        return ""
    intensity = min(abs(value) / max_abs, 1.0)
    weak = positive_rgb_weak if value >= 0 else negative_rgb_weak
    strong = positive_rgb_strong if value >= 0 else negative_rgb_strong
    r = int(weak[0] + intensity * (strong[0] - weak[0]))
    g = int(weak[1] + intensity * (strong[1] - weak[1]))
    b = int(weak[2] + intensity * (strong[2] - weak[2]))
    return f"color: rgb({r},{g},{b}); font-weight: bold; background-color: transparent"

def color_ratio(value, max_abs):
    # Ramps toward a more saturated mid-tone rather than toward black, so the
    # strongest signals stay legible on both light and dark backgrounds.
    return _shade(
        value, max_abs,
        positive_rgb_weak=(110, 150, 115), positive_rgb_strong=(22, 163, 74),
        negative_rgb_weak=(160, 115, 115), negative_rgb_strong=(220, 38, 38),
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

    styler = styler.format({
        "Price (R)": "{:.2f}",
        "P/E": "{:.2f}",
        "Market Cap (R bn)": "{:,.1f}",
        "6mo Momentum %": "{:.1f}",
        "Sharpe Ratio": "{:.2f}",
        "Valuation Score": "{:.0f}",
        "Momentum Score": "{:.0f}",
        "Sharpe Score": "{:.0f}",
        "Combined Score": "{:.0f}",
    }, na_rep="—")

    return styler

st.set_page_config(page_title="JSE Screener", layout="wide")
st.title("JSE Screener — v1.4")
st.caption(
    "Valuation + momentum ranking, Sharpe ratio, Rand hedge/domestic "
    "classification. Scores are percentile ranks across the universe (0-100). "
    "Green = strong, red = weak, stronger colour = further from the middle. "
    "Free data, updated hourly."
)
st.caption(
    "📊 Momentum signal backtested across ~54 monthly periods "
    "(IC ≈ 0.10, ~63% win rate) — a modest but genuine edge, not a strong one."
)

with st.spinner("Fetching JSE data..."):
    df, failures = get_jse_data(JSE_TICKERS)

if df.empty:
    st.error("No data returned. Check your internet connection or try again shortly.")
    st.stop()

if failures:
    st.warning(
        f"{len(failures)} of {len(JSE_TICKERS)} tickers could not be loaded and are "
        "missing from the table below: "
        + ", ".join(f"{t.replace('.JO', '')} ({why})" for t, why in failures)
    )

df = add_valuation_score(df)
df = add_momentum_score(df)
df = add_sharpe_score(df)
df = add_combined_score(df)

momentum_max_abs = safe_max_abs(df["6mo Momentum %"])
sharpe_max_abs = safe_max_abs(df["Sharpe Ratio"])

with st.sidebar:
    st.header("Filters")
    sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
    fx_options = ["All"] + sorted(df["FX Exposure"].unique().tolist())
    fx_filter = st.selectbox("FX Exposure", fx_options)
    sort_by = st.selectbox(
        "Sort by",
        ["Combined Score", "Sharpe Ratio", "Valuation Score", "Momentum Score", "6mo Momentum %", "P/E"],
    )
    min_score = st.slider("Minimum Combined Score", 0, 100, 0)

filtered = df.copy()
if sector_filter != "All":
    filtered = filtered[filtered["Sector"] == sector_filter]
if fx_filter != "All":
    filtered = filtered[filtered["FX Exposure"] == fx_filter]
if min_score > 0:
    # Only applied above zero, so rows with an incomplete Combined Score are
    # not silently dropped at the default setting.
    filtered = filtered[filtered["Combined Score"] >= min_score]
filtered = filtered.sort_values(sort_by, ascending=False)

display_cols = [
    "Ticker", "Name", "Price (R)", "Market Cap (R bn)",
    "Sector", "FX Exposure", "P/E", "6mo Momentum %", "Sharpe Ratio",
    "Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score",
]

display_df = filtered[display_cols].reset_index(drop=True)

styled = style_table(display_df, momentum_max_abs, sharpe_max_abs)
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.Column(pinned=True, width="small"),
        "Name": st.column_config.Column(pinned=True, width="medium"),
        "Price (R)": st.column_config.NumberColumn(pinned=True, width="small"),
    },
)

st.caption(
    f"{len(filtered)} of {len(df)} stocks shown. "
    "Scores are percentile ranks within the loaded universe, so they shift as "
    "the universe changes. Where P/E is missing or negative the Valuation Score "
    "shows — and the Combined Score reflects momentum only. "
    "FX Exposure is a manual classification based on business geography, "
    "not derived from financial filings - treat as approximate. "
    "Sharpe Ratio: risk-adjusted return (annualized, 6mo daily history, "
    f"{RISK_FREE_RATE*100:.0f}% risk-free rate assumed) - a 6-month window is a "
    "noisy estimate. "
    "Combined Score reflects Valuation + Momentum only. "
    "Not investment advice."
)
