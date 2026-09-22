"""
JSE Screener v1.6 - expanded universe with a liquidity screen.

Data is refreshed once a day by fetch_data.py, run from
.github/workflows/refresh.yml after the JSE close. This app just reads
data/screener.csv, so page loads no longer hit Yahoo at all - which is what
removed the rate limiting and the cold-start delay.

If the data file is missing (e.g. before the first scheduled run) the app falls
back to fetching live, so it is never dependent on the pipeline having run.

Run locally:
    pip3 install -r requirements.txt
    python3 -m streamlit run jse_screener_app.py
"""

import json
import os

import pandas as pd
import streamlit as st

import fetch_data
from fetch_data import DATA_FILE, META_FILE, JSE_TICKERS, RISK_FREE_RATE

DISPLAY_COLS = [
    "Ticker", "Name", "Price (R)", "Market Cap (R bn)",
    "Sector", "FX Exposure", "P/E", "6mo Momentum %", "Sharpe Ratio",
    "ADV (R m)",
    "Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score",
]

COLUMN_FORMATS = {
    "Price (R)": "{:.2f}",
    "P/E": "{:.2f}",
    "Market Cap (R bn)": "{:,.1f}",
    "6mo Momentum %": "{:.1f}",
    "Sharpe Ratio": "{:.2f}",
    "ADV (R m)": "{:,.1f}",
    "Valuation Score": "{:.0f}",
    "Momentum Score": "{:.0f}",
    "Sharpe Score": "{:.0f}",
    "Combined Score": "{:.0f}",
}


# ---------------------------------------------------------------- loading

@st.cache_data(ttl=900)
def load_precomputed():
    """Read the daily data file. Returns (df, meta) or (None, None)."""
    if not os.path.exists(DATA_FILE):
        return None, None

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        return None, None

    meta = {}
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE) as fh:
                meta = json.load(fh)
        except (ValueError, OSError):
            meta = {}

    return df, meta


@st.cache_data(ttl=3600)
def load_live():
    """Fallback: fetch directly. Only used when the data file is absent."""
    return fetch_data.build_dataset()


# ---------------------------------------------------------------- styling

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
    """Streamlit's Arrow serialisation ignores Styler.na_rep and prints 'None'
    for missing numbers, which looks like a bug in a financial table. So values
    are pre-formatted to strings here and colours applied via a parallel style
    matrix of the same shape."""
    styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)

    styles["6mo Momentum %"] = display_df["6mo Momentum %"].map(
        lambda v: color_ratio(v, momentum_max_abs)
    )
    styles["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(
        lambda v: color_ratio(v, sharpe_max_abs)
    )
    for col in ["Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score"]:
        styles[col] = display_df[col].map(color_score)

    text = display_df.copy()
    for col, spec in COLUMN_FORMATS.items():
        text[col] = display_df[col].map(
            lambda v, s=spec: "—" if pd.isna(v) else s.format(v)
        )

    return text.style.apply(lambda _: styles, axis=None)


# ---------------------------------------------------------------- app

st.set_page_config(page_title="JSE Screener", layout="wide")
st.title("JSE Screener — v1.6")
st.caption(
    "Valuation + momentum ranking, Sharpe ratio, Rand hedge/domestic "
    "classification. Scores are percentile ranks across the universe (0-100). "
    "Green = strong, red = weak, stronger colour = further from the middle."
)
st.caption(
    "📊 Momentum signal backtested across ~54 monthly periods "
    "(IC ≈ 0.10, ~63% win rate) — a modest but genuine edge, not a strong one."
)

df, meta = load_precomputed()

if df is not None:
    failures = [(f["ticker"], f["reason"]) for f in meta.get("failures", [])]
    stamp = meta.get("generated_sast")
    excluded = meta.get("excluded", [])
    adv_floor = meta.get("min_adv_rand", 5_000_000)
    note = (
        f"🕒 Data as of **{stamp} SAST**, refreshed daily after the JSE close."
        if stamp else "🕒 Precomputed data."
    )
    if excluded:
        note += (
            f"  ·  {len(excluded)} name(s) screened out for illiquidity "
            f"(under R{adv_floor/1e6:.0f}m average daily value traded) or stale prices."
        )
    st.caption(note)
else:
    st.info(
        "No precomputed data file yet — fetching live this once. The daily "
        "refresh job will populate it after the next JSE close."
    )
    with st.spinner("Fetching JSE data..."):
        df, failures = load_live()

if df is None or df.empty:
    st.error("No data available. Try again shortly.")
    st.stop()

if failures:
    st.warning(
        f"{len(failures)} of {len(JSE_TICKERS)} tickers are missing from this "
        "refresh: "
        + ", ".join(f"{t.replace('.JO', '')}" for t, _ in failures)
    )

momentum_max_abs = safe_max_abs(df["6mo Momentum %"])
sharpe_max_abs = safe_max_abs(df["Sharpe Ratio"])

with st.sidebar:
    st.header("Filters")
    sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
    fx_options = ["All"] + sorted(df["FX Exposure"].dropna().unique().tolist())
    fx_filter = st.selectbox("FX Exposure", fx_options)
    sort_by = st.selectbox(
        "Sort by",
        ["Combined Score", "Sharpe Ratio", "Valuation Score", "Momentum Score",
         "6mo Momentum %", "P/E", "Market Cap (R bn)", "ADV (R m)"],
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

display_df = filtered[DISPLAY_COLS].reset_index(drop=True)

styled = style_table(display_df, momentum_max_abs, sharpe_max_abs)
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.Column(pinned=True, width="small"),
        "Name": st.column_config.Column(pinned=True, width="medium"),
        # Values are pre-formatted strings, so these are plain Columns rather
        # than NumberColumns.
        "Price (R)": st.column_config.Column(pinned=True, width="small"),
    },
)

st.caption(
    f"{len(filtered)} of {len(df)} stocks shown. "
    "Scores are percentile ranks within the loaded universe, so they shift as "
    "the universe changes. Where P/E is missing or negative the Valuation Score "
    "shows — and the Combined Score reflects momentum only. "
    "ADV is 20-day average daily value traded; names below the liquidity "
    "floor are dropped before scoring, because momentum and Sharpe computed on "
    "barely-traded prices are not weak signals but false ones. "
    "FX Exposure is a manual classification based on business geography, "
    "not derived from financial filings - treat as approximate. "
    "Sharpe Ratio: risk-adjusted return (annualized, 6mo daily history, "
    f"{RISK_FREE_RATE*100:.0f}% risk-free rate assumed) - a 6-month window is a "
    "noisy estimate. "
    "Combined Score reflects Valuation + Momentum only. "
    "Not investment advice."
)
