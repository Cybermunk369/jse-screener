"""
JSE Screener v1.14 - click a row for the stock report.

Clicking any row opens a pop-up report for that stock (price chart, key
figures, score bars, score history, watchlist button), replacing the v1.13
"Stock detail" picker under the table. The report's address is kept in the page
URL (?stock=KAP), so a link opens straight to it. Row clicks need st.dataframe,
which can't also have a clickable checkbox column, so the ★ column now only
shows what's starred; starring moved into the report and the sidebar search.

v1.13 - stock detail panel, price trend and score history.

Pick any stock under the table to see its six-month price chart, its scores as
bars, and a star button. The table gains a "6m trend" sparkline and a "Δ 1w"
column (change in Combined Score over a week). Both read two new files written
by the daily refresh: data/prices.csv (closes) and data/history.csv (each
day's scores). History starts on the day this version's pipeline first ran and
is not back-filled - older data used a different stock list and scoring, so
"a week ago" figures from it would not be like for like. Δ 1w stays hidden
until a week of history exists.

v1.12 - investor-facing layout.

Title and a summary strip first, then tabs: Screener, Watchlist, How it works.
The methodology notes moved out of the way into "How it works". The table shows
the essential columns by default (all columns on request), missing values show
as a dash via the table's placeholder setting (the numbers stay numeric, so
sorting is unaffected - the v1.7 note below calling "None" unavoidable was
wrong; that setting had been missed), and one-tap quick views replace hunting
through filters.
The sidebar "watchlist only" toggle became the Watchlist tab.

v1.11 - watchlist search box and "watchlist only" view.

A sidebar box adds stocks by name or ticker, and a toggle shows only the
watchlist. Both stay in step with the table stars; all three edit the same
list, still kept in the page address.

v1.10 - watchlist.

Click the ★ in a row to star a stock; starred stocks stay at the top of the
table (then the chosen sort applies). The watchlist is kept in the page
address (?watch=SHP,NPN) because there are no accounts yet - bookmarking the
page saves it. The table is an st.data_editor with only the ★ column editable.

v1.9 - rule-based universe.

The stock list is no longer hand-picked: build_universe.py takes every JSE
ordinary share with a market cap of at least R5bn, rebuilt quarterly by
.github/workflows/universe.yml. The daily liquidity screen still applies on
top. v1.8 folded Sharpe into the Combined Score (40/40/20).

v1.7 - numeric columns sort correctly.

v1.6 pre-formatted every numeric column to a display string (so missing
values could show "-" instead of Streamlit's "None"). That silently broke
the table's built-in column-header sort: Streamlit picks a text or numeric
comparator based on the column's declared type, and a string column sorts
lexicographically ("100.75" comes before "9.50"). Price and Market Cap were
sorting wrong as a result. Columns are numeric again here, formatted via
column_config.NumberColumn instead of pre-formatted strings, so header-click
sort compares numbers. The trade-off: missing values now show Streamlit's
native "None" rather than an em-dash, which is a real Streamlit limitation
(confirmed empirically - na_rep, nullable dtypes, and TextColumn overrides
on numeric data all still show "None", and only a Text-typed column can
show custom missing-value text - but a Text column sorts as a string).

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

import functools
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

import fetch_data
from fetch_data import DATA_FILE, META_FILE, JSE_TICKERS, RISK_FREE_RATE

# Resolved here rather than imported: Streamlit Cloud can re-run this script
# after a deploy while still holding an older fetch_data module in memory, and
# a name that module doesn't have yet takes the whole app down on import.
UNIVERSE_META_FILE = getattr(
    fetch_data, "UNIVERSE_META_FILE",
    os.path.join(os.path.dirname(DATA_FILE), "universe_meta.json"),
)

_DATA_DIR = os.path.dirname(DATA_FILE)
PRICES_FILE = getattr(fetch_data, "PRICES_FILE", os.path.join(_DATA_DIR, "prices.csv"))
HISTORY_FILE = getattr(fetch_data, "HISTORY_FILE", os.path.join(_DATA_DIR, "history.csv"))

SCORE_COLS = ["Combined Score", "Valuation Score", "Momentum Score", "Sharpe Score"]
# Weekly change needs a record from 7-10 calendar days back: 7 so it is a
# full week, up to 10 so a public holiday or a missed refresh still finds one.
WEEK_MIN_DAYS, WEEK_MAX_DAYS = 7, 10
# Single-series chart colour (reference data-viz palette, slot 1).
CHART_BLUE = "#2a78d6"

DISPLAY_COLS = [
    "Ticker", "Name", "Price (R)", "6m trend", "Δ 1w", "Market Cap (R bn)",
    "Sector", "FX Exposure", "P/E", "6mo Momentum %", "Sharpe Ratio",
    "ADV (R m)",
    "Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score",
]

# printf-style specs for st.column_config.NumberColumn (not Python .format() -
# these need to be sprintf syntax so Streamlit renders the number itself while
# keeping the underlying dtype numeric, which is what keeps header-click sort
# comparing numbers instead of strings).
COLUMN_FORMATS = {
    "Price (R)": "%.2f",
    "P/E": "%.2f",
    "Market Cap (R bn)": "%,.1f",
    "6mo Momentum %": "%.1f",
    "Sharpe Ratio": "%.2f",
    "ADV (R m)": "%,.1f",
    "Valuation Score": "%.0f",
    "Momentum Score": "%.0f",
    "Sharpe Score": "%.0f",
    "Combined Score": "%.0f",
    "Δ 1w": "%+.0f",
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


@st.cache_data(ttl=900)
def load_universe_meta():
    """How the stock list was built. Empty dict if it has not been built yet."""
    try:
        with open(UNIVERSE_META_FILE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


@st.cache_data(ttl=900)
def load_prices():
    """Daily closes in rand: one row per trading date, one column per ticker.
    Empty until the refresh has written the file."""
    try:
        return pd.read_csv(PRICES_FILE, index_col="date", parse_dates=["date"])
    except (OSError, ValueError):
        return pd.DataFrame()


@st.cache_data(ttl=900)
def load_history():
    """One row per stock per trading day with that day's scores."""
    try:
        hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    except (OSError, ValueError):
        return pd.DataFrame(columns=["date", "Ticker", "Price (R)"] + SCORE_COLS)
    return hist


def weekly_change(history):
    """Combined Score now minus a week ago, per ticker (NaN where unknown).

    "Now" is the latest recorded day; "a week ago" is the latest record 7-10
    calendar days before it. Scores are ranks against the other stocks, so a
    change can come from the stock itself or from the others moving.
    """
    if history.empty:
        return pd.Series(dtype=float)
    latest = history["date"].max()
    window = history[
        (history["date"] <= latest - pd.Timedelta(days=WEEK_MIN_DAYS))
        & (history["date"] >= latest - pd.Timedelta(days=WEEK_MAX_DAYS))
    ]
    if window.empty:
        return pd.Series(dtype=float)
    then = window[window["date"] == window["date"].max()].set_index("Ticker")["Combined Score"]
    now = history[history["date"] == latest].set_index("Ticker")["Combined Score"]
    return (now - then).dropna()


def trend_points(prices):
    """Weekly closes (every 5th trading day, ending today) for the sparkline."""
    if prices.empty:
        return {}
    weekly = prices.iloc[::-5].iloc[::-1]
    return {t: weekly[t].dropna().round(2).tolist() for t in weekly.columns}


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
    """Colours are applied via a parallel style matrix of the same shape, on
    top of the numeric display_df (not a stringified copy) - see the module
    docstring for why values stay numeric instead of being pre-formatted."""
    styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)

    styles["6mo Momentum %"] = display_df["6mo Momentum %"].map(
        lambda v: color_ratio(v, momentum_max_abs)
    )
    styles["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(
        lambda v: color_ratio(v, sharpe_max_abs)
    )
    for col in ["Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score"]:
        styles[col] = display_df[col].map(color_score)
    # A 10-point move in a week is big for a percentile score; cap the shade there.
    styles["Δ 1w"] = display_df["Δ 1w"].map(lambda v: color_ratio(v, 10))

    return display_df.style.apply(lambda _: styles, axis=None)


# ---------------------------------------------------------------- app

APP_VERSION = "1.14"

# Columns shown by default - enough to act on, narrow enough for a phone.
DEFAULT_COLS = [
    "Ticker", "Name", "Price (R)", "6m trend", "Combined Score", "Δ 1w",
    "6mo Momentum %", "P/E", "Sector",
]

# One-tap views. Each is a plain rule on data already in the table, stated
# next to the view so users can see exactly what they're looking at. They
# filter only; the sidebar "Sort by" still decides the order.
QUICK_VIEWS = {
    "All stocks": (None, "Every ranked stock."),
    "Top 20": (
        lambda d: d[d["Combined Score"].rank(ascending=False, method="first") <= 20],
        "The 20 highest Combined Scores.",
    ),
    "Value": (
        lambda d: d[d["Valuation Score"] >= 70],
        "Valuation Score 70+: the cheapest 30% on P/E (loss-makers excluded).",
    ),
    "Momentum": (
        lambda d: d[d["Momentum Score"] >= 80],
        "Momentum Score 80+: the strongest 20% over six months.",
    ),
    "Steady": (
        lambda d: d[d["Sharpe Score"] >= 80],
        "Sharpe Score 80+: the best 20% on return per unit of risk.",
    ),
    "Rand hedges": (
        lambda d: d[d["FX Exposure"] == "Rand Hedge"],
        "Earn mainly outside South Africa, so they tend to hold up when the rand weakens.",
    ),
}

st.set_page_config(page_title="JSE Screener", page_icon="📈", layout="wide")

df, meta = load_precomputed()
live_fallback = df is None
if live_fallback:
    with st.spinner("Fetching JSE data..."):
        df, failures = load_live()
    meta = {}
else:
    failures = [(f["ticker"], f["reason"]) for f in meta.get("failures", [])]

if df is None or df.empty:
    st.error("No data available. Try again shortly.")
    st.stop()

stamp = meta.get("generated_sast")
excluded = meta.get("excluded", [])
adv_floor = meta.get("min_adv_rand", 5_000_000)
universe_meta = load_universe_meta()
rule = universe_meta.get("rule", {})
cap_floor_bn = (rule.get("min_market_cap_rand") or 5e9) / 1e9

prices = load_prices()
history = load_history()
week_change = weekly_change(history)
has_week_change = not week_change.empty
df = df.copy()
df["Δ 1w"] = df["Ticker"].map(week_change)
trends = trend_points(prices)
df["6m trend"] = df["Ticker"].map(lambda t: trends.get(t))
# Columns with nothing in them yet (before the first refresh on this version,
# or before a week of history) are hidden rather than shown as all dashes.
hidden_cols = set()
if not has_week_change:
    hidden_cols.add("Δ 1w")
if not trends:
    hidden_cols.add("6m trend")

momentum_max_abs = safe_max_abs(df["6mo Momentum %"])
sharpe_max_abs = safe_max_abs(df["Sharpe Ratio"])

# ---------------------------------------------------------------- watchlist
# There are no user accounts yet, so the watchlist is kept in the page address
# (?watch=SHP,NPN): a bookmark saves it, a link shares it. Session state is the
# working copy for this visit.
if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
        t for t in st.query_params.get("watch", "").upper().split(",")
        if t.strip().isalnum()
    ]
if "table_version" not in st.session_state:
    st.session_state.table_version = 0


def save_watchlist(watchlist):
    """Single place that updates the watchlist (session + tables).

    The page address is written only by sync_watch_param() in the main script
    run, so there is exactly one writer, and hand-typed addresses (e.g.
    lowercase ?watch=agl) get normalised on load.
    """
    st.session_state.watchlist = watchlist


def sync_watch_param():
    """Make ?watch= match the watchlist. Runs every script run."""
    wanted = ",".join(st.session_state.watchlist)
    if st.query_params.get("watch", "") != wanted:
        if wanted:
            st.query_params["watch"] = wanted
        else:
            del st.query_params["watch"]


def on_watch_pick():
    """The sidebar watchlist box changed - adopt its value, keeping order."""
    picked = st.session_state.watch_pick
    kept = [t for t in st.session_state.watchlist if t in picked]
    save_watchlist(kept + [t for t in picked if t not in kept])


def on_row_click(row_tickers, table_key):
    """A cell in a table was clicked - open that stock's report.

    Selections are reported by row position in the data the table was given
    (they stay correct if the user re-sorts by a column header), so that
    render's ticker order is passed in. The table is then recreated under a new
    key so the selection clears: otherwise clicking the same row again after
    closing the report would just deselect it.
    """
    cells = st.session_state[table_key]["selection"]["cells"]
    if cells:
        st.session_state.open_stock = row_tickers[int(cells[0][0])]
    st.session_state.table_version += 1


def close_report():
    st.session_state.open_stock = None
    # A fresh table after the report closes: if a star changed the row order
    # meanwhile, the old table ignored the next click (seen in testing).
    st.session_state.table_version += 1


names = dict(zip(df["Ticker"], df["Name"]))
sync_watch_param()

# The open report is kept in the page address (?stock=KAP) like the watchlist,
# so a shared or bookmarked link opens straight to that stock.
if "open_stock" not in st.session_state:
    wanted = st.query_params.get("stock", "").upper()
    st.session_state.open_stock = wanted if wanted in names else None


def sync_stock_param():
    """Make ?stock= match the open report. Runs every script run."""
    stock = st.session_state.open_stock
    if stock:
        if st.query_params.get("stock") != stock:
            st.query_params["stock"] = stock
    elif "stock" in st.query_params:
        del st.query_params["stock"]


sync_stock_param()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("★ Watchlist")
    # Starred stocks that dropped out of today's data (screened out, delisted)
    # stay selectable, so they aren't silently removed from the watchlist.
    watch_options = sorted(names) + [
        t for t in st.session_state.watchlist if t not in names
    ]
    # Mirror the current watchlist into the box before it renders, so stars
    # clicked in a table show up here too.
    st.session_state.watch_pick = list(st.session_state.watchlist)
    st.multiselect(
        "Search and add stocks",
        watch_options,
        key="watch_pick",
        format_func=lambda t: (
            f"{t} · {names[t]}" if t in names else f"{t} · not in today's data"
        ),
        placeholder="Type a name or ticker...",
        on_change=on_watch_pick,
    )

    st.header("Filters")
    sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
    fx_options = ["All"] + sorted(df["FX Exposure"].dropna().unique().tolist())
    fx_filter = st.selectbox("FX Exposure", fx_options)
    min_score = st.slider("Minimum Combined Score", 0, 100, 0)
    sort_by = st.selectbox(
        "Sort by",
        ["Combined Score"] + (["Δ 1w"] if has_week_change else [])
        + ["Sharpe Ratio", "Valuation Score", "Momentum Score",
           "6mo Momentum %", "P/E", "Market Cap (R bn)", "ADV (R m)"],
    )
    all_columns = st.toggle("Show all columns", key="all_columns")


# ---------------------------------------------------------------- helpers
def with_stars(data):
    """Add the ★ column and put starred stocks first, then the chosen sort."""
    data = data.copy()
    data["★"] = data["Ticker"].isin(set(st.session_state.watchlist))
    return data.sort_values(["★", sort_by], ascending=[False, False], kind="stable")


def render_table(data, name):
    """Styled, sortable table; clicking any cell opens that stock's report."""
    display_df = data[["★"] + DISPLAY_COLS].reset_index(drop=True)
    column_config = {
        "★": st.column_config.CheckboxColumn(
            "★", pinned=True, width=40,
            help="On your watchlist. Open a stock to add or remove it.",
        ),
        "Ticker": st.column_config.Column(pinned=True, width="small"),
        "Name": st.column_config.Column(pinned=True, width="medium"),
        "6m trend": st.column_config.LineChartColumn(
            "6m trend", width="small", color="auto",
            help="Share price over six months, weekly points. Green if up, red if down.",
        ),
    }
    # Every numeric column gets a NumberColumn (numeric dtype + numeric sort
    # comparator) with its display format, instead of pre-formatted strings.
    for col, spec in COLUMN_FORMATS.items():
        column_config[col] = st.column_config.NumberColumn(format=spec)
    column_config["Δ 1w"] = st.column_config.NumberColumn(
        "Δ 1w", format="%+.0f",
        help="Change in Combined Score over the last week. Scores are ranks, so "
             "a stock can move because others did.",
    )
    shown_cols = [c for c in (DISPLAY_COLS if all_columns else DEFAULT_COLS)
                  if c not in hidden_cols]
    key = f"{name}_{st.session_state.table_version}"
    # Cell selection rather than row selection: a click anywhere on the row
    # opens the report, not only on the narrow row-marker column. placeholder
    # shows a dash for missing values while columns stay numeric, so sorting
    # keeps working.
    st.dataframe(
        style_table(display_df, momentum_max_abs, sharpe_max_abs),
        key=key,
        use_container_width=True,
        hide_index=True,
        column_order=["★"] + shown_cols,
        column_config=column_config,
        placeholder="—",
        # st.dataframe takes no callback args; bind this render's tickers.
        on_select=functools.partial(on_row_click, display_df["Ticker"].tolist(), key),
        selection_mode="single-cell",
    )
    return display_df


def download_button(display_df, name):
    # The table's own toolbar download uses the browser's "Save as" (File
    # System Access) API, which some embedded browsers expose but can't write
    # through, leaving an empty file. This is built server-side instead and
    # exports exactly the view shown. utf-8-sig so Excel reads it right.
    data_date = (stamp or "")[:10] or "latest"
    export_df = display_df.drop(columns=["★", "6m trend"])
    if "Δ 1w" in hidden_cols:
        export_df = export_df.drop(columns="Δ 1w")
    export_df.insert(0, "Watchlist", display_df["★"].map({True: "yes", False: ""}))
    for col in ["Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score"]:
        export_df[col] = export_df[col].round().astype("Int64")   # 98, not 98.0
    if "Δ 1w" in export_df:
        export_df["Δ 1w"] = export_df["Δ 1w"].round().astype("Int64")
    export_df["P/E"] = export_df["P/E"].round(2)
    st.download_button(
        "⬇ Download CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"jse_screener_{name}_{data_date}.csv",
        mime="text/csv",
        key=f"download_{name}",
        help="Exports this table as shown.",
    )


def toggle_star(ticker):
    watchlist = list(st.session_state.watchlist)
    if ticker in watchlist:
        watchlist.remove(ticker)
    else:
        watchlist.append(ticker)
    save_watchlist(watchlist)


def price_chart(ticker):
    series = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
    if len(series) < 2:
        st.caption("Price chart appears after the next daily refresh.")
        return
    data = series.rename("Price (R)").rename_axis("Date").reset_index()
    # Whole rand on the axis, except for cheap shares where that would repeat.
    y_format = ",.2f" if series.max() < 20 else ",.0f"
    base = alt.Chart(data).encode(
        x=alt.X("Date:T", title=None,
                  axis=alt.Axis(format="%b", tickCount="month", grid=False)),
        y=alt.Y("Price (R):Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(format=y_format, tickCount=4)),
    )
    # Invisible wide points under a nearest-x selection give a hover tooltip
    # anywhere along the line, not only on the 2px stroke itself.
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    chart = alt.layer(
        base.mark_line(color=CHART_BLUE, strokeWidth=2),
        base.mark_point(opacity=0, size=200).add_params(hover).encode(
            tooltip=[alt.Tooltip("Date:T", format="%-d %b %Y"),
                     alt.Tooltip("Price (R):Q", format=",.2f")]
        ),
        base.mark_point(color=CHART_BLUE, size=70, filled=True)
            .transform_filter(hover),
    ).properties(height=240)
    st.altair_chart(chart, use_container_width=True)


def score_history_chart(ticker):
    past = history[history["Ticker"] == ticker].sort_values("date")
    if past["date"].nunique() < 2:
        first = history["date"].min()
        st.caption(
            "Score history is recorded after every daily refresh"
            + (f" (since {first:%-d %b %Y})" if pd.notna(first) else "")
            + ". The chart appears once there are two days to compare."
        )
        return
    data = past.rename(columns={"date": "Date"})
    base = alt.Chart(data).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%-d %b", grid=False)),
        y=alt.Y("Combined Score:Q", title=None, scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 50, 100])),
    )
    tooltip = [alt.Tooltip("Date:T", format="%-d %b %Y")] + [
        alt.Tooltip(f"{c}:Q", format=".0f") for c in SCORE_COLS
    ]
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    chart = alt.layer(
        base.mark_line(color=CHART_BLUE, strokeWidth=2),
        base.mark_point(color=CHART_BLUE, size=64, filled=True),
        base.mark_point(opacity=0, size=300).add_params(hover).encode(tooltip=tooltip),
    ).properties(height=160)
    st.altair_chart(chart, use_container_width=True)


SCORE_NOTES = {
    "Combined Score": "40% value, 40% momentum, 20% Sharpe",
    "Valuation Score": "cheapness on P/E",
    "Momentum Score": "six-month return",
    "Sharpe Score": "return per unit of risk",
}


def fmt(value, spec, prefix="", suffix=""):
    """Format a number for a tile, or a dash when it's missing."""
    return f"{prefix}{value:{spec}}{suffix}" if pd.notna(value) else "—"


def report_body(ticker):
    """The stock report shown in the pop-up."""
    row = df[df["Ticker"] == ticker].iloc[0]
    starred = ticker in st.session_state.watchlist

    facts = [row["Sector"], row["FX Exposure"]]
    st.caption(" · ".join(str(f) for f in facts if pd.notna(f)))
    # Inside a dialog this reruns only the report, so the label flips at once;
    # the table's ★ column catches up when the report closes (close_report
    # reruns the page).
    st.button(
        "★ Remove from watchlist" if starred else "☆ Add to watchlist",
        key=f"star_{ticker}", on_click=toggle_star, args=(ticker,),
    )

    pe = row["P/E"] if pd.notna(row["P/E"]) and row["P/E"] > 0 else None
    tiles = st.container(horizontal=True, wrap=True, gap="small")
    tiles.metric("Price", fmt(row["Price (R)"], ",.2f", "R"), border=True, width="content")
    tiles.metric(
        "6m return", fmt(row["6mo Momentum %"], "+.1f", suffix="%"),
        border=True, width="content",
        help="Including dividends. The chart shows the traded price, so it can "
             "differ slightly.",
    )
    tiles.metric(
        "P/E", fmt(pe, ".1f"), border=True, width="content",
        help="Trailing price-to-earnings. A dash means no profit to measure.",
    )
    tiles.metric(
        "Sharpe (6m)", fmt(row["Sharpe Ratio"], ".2f"), border=True, width="content",
        help="Six-month return per unit of risk. A short window, so noisy.",
    )
    tiles.metric(
        "Market cap", fmt(row["Market Cap (R bn)"], ",.0f", "R", "bn"),
        border=True, width="content",
    )
    tiles.metric(
        "Traded daily", fmt(row["ADV (R m)"], ",.0f", "R", "m"),
        border=True, width="content",
        help="Average value traded per day over the last 20 trading days.",
    )

    # Side by side when there's room, stacked in a narrow window.
    panes = st.container(horizontal=True, wrap=True, gap="large")
    left = panes.container(width=500)
    right = panes.container(width=380)
    with left:
        st.markdown("**Share price, last six months (R)**")
        price_chart(ticker)
    with right:
        st.markdown("**Scores** (0-100, against the other stocks)")
        change = row.get("Δ 1w")
        for col in SCORE_COLS:
            value = row[col]
            label = col.replace(" Score", "")
            if pd.isna(value):
                st.progress(0, text=f"{label}: no score")
                continue
            extra = (f" · {change:+.0f} in a week"
                     if col == "Combined Score" and pd.notna(change) else "")
            st.progress(int(value) / 100,
                        text=f"{label}: **{value:.0f}**{extra} · {SCORE_NOTES[col]}")
        st.markdown("**Combined Score history**")
        score_history_chart(ticker)
    st.caption(f"Prices as of {pretty_stamp(stamp)} SAST. Not investment advice.")


def open_report(ticker):
    """Show the report pop-up. Called on every run while a stock is open, so
    it stays open across reruns until the user closes it."""
    title = f"{names.get(ticker, ticker)} ({ticker})"
    st.dialog(title, width="large", on_dismiss=close_report)(report_body)(ticker)


def pretty_stamp(s):
    try:
        return pd.Timestamp(s).strftime("%-d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return s or "unknown"


# ---------------------------------------------------------------- header
st.title("📈 JSE Screener")
st.caption(
    f"Every liquid JSE share worth R{cap_floor_bn:.0f}bn or more, ranked daily on "
    "value, momentum and risk-adjusted return. "
    + (f"Data as of {pretty_stamp(stamp)} SAST." if stamp else "")
)
if live_fallback:
    st.info("Today's precomputed data isn't available yet, so this is a live fetch.")

top = df.loc[df["Combined Score"].idxmax()]
mover = df.loc[df["6mo Momentum %"].idxmax()]
# A wrapping row rather than fixed columns: tiles keep a readable minimum
# width and move to a second line in a narrow window instead of cutting off.
tiles = st.container(horizontal=True, wrap=True, gap="small")
m1 = m2 = m3 = m4 = tiles
# Short labels, the number as the value, the ticker as a grey sub-line - so
# the tiles still read in a narrow window (long values get cut off with "...").
m1.metric(
    "Ranked", len(df), delta=f"{len(excluded)} screened out",
    delta_color="off", delta_arrow="off", border=True, width=170,
    help="Stocks scored today. The rest were left out as illiquid, stale or "
         "too new - see How it works.",
)
m2.metric(
    "Top score", f"{top['Combined Score']:.0f}", delta=top["Ticker"],
    delta_color="off", delta_arrow="off", border=True, width=170,
    help=f"Highest Combined Score: {top['Name']}",
)
m3.metric(
    "Best 6m run", f"{mover['6mo Momentum %']:+.0f}%", delta=mover["Ticker"],
    delta_color="off", delta_arrow="off", border=True, width=170,
    help=f"Strongest six-month price move: {mover['Name']}",
)
m4.metric(
    "Watchlist", len(st.session_state.watchlist), delta="starred",
    delta_color="off", delta_arrow="off", border=True, width=170,
    help="Open a stock and tap ☆, or search in the sidebar.",
)

tab_screen, tab_watch, tab_how = st.tabs(
    ["📊 Screener", "★ Watchlist", "ℹ️ How it works"]
)

# ---------------------------------------------------------------- screener
with tab_screen:
    view = st.segmented_control(
        "Quick view", list(QUICK_VIEWS), default="All stocks",
        key="quick_view", label_visibility="collapsed",
    ) or "All stocks"
    view_rule, view_note = QUICK_VIEWS[view]

    filtered = df
    if sector_filter != "All":
        filtered = filtered[filtered["Sector"] == sector_filter]
    if fx_filter != "All":
        filtered = filtered[filtered["FX Exposure"] == fx_filter]
    if min_score > 0:
        # Only applied above zero, so rows with an incomplete Combined Score
        # are not silently dropped at the default setting.
        filtered = filtered[filtered["Combined Score"] >= min_score]
    if view_rule is not None:
        filtered = view_rule(filtered)

    st.caption(f"**{view}** · {view_note} {len(filtered)} of {len(df)} shown. "
               "**Click any row for the full stock report.** Starred stocks stay at the top.")
    if filtered.empty:
        st.info("No stocks match this view with the current sidebar filters.")
    else:
        shown = render_table(with_stars(filtered), "screener")
        download_button(shown, "screener")
    st.caption(
        "Scores rank each stock against the others from 0 to 100 - green is "
        "strong, red is weak. Not investment advice. Details in *How it works*."
    )

# ---------------------------------------------------------------- watchlist
with tab_watch:
    watchlist = st.session_state.watchlist
    missing = [t for t in watchlist if t not in names]
    if not watchlist:
        st.info(
            "Your watchlist is empty. Click any stock in the Screener and tap "
            "☆ Add to watchlist, or search for one in the sidebar."
        )
    else:
        starred = df[df["Ticker"].isin(watchlist)]
        if not starred.empty:
            shown = render_table(with_stars(starred), "watchlist")
            download_button(shown, "watchlist")
        st.caption(
            f"★ {len(watchlist)} on your watchlist, shown regardless of the "
            "screener filters. Click a row for its report. "
            + (f"Not in today's data: {', '.join(missing)}. " if missing else "")
            + "It lives in this page's web address - bookmark the page to keep it."
        )

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown(
        f"""
#### The scores
Each stock gets three scores from 0 to 100. They are **percentile ranks**: a
Momentum Score of 80 means stronger momentum than 80% of the stocks here.

- **Valuation** - how cheap the share is on P/E. Loss-makers (no or negative
  P/E) get no Valuation Score rather than looking "cheap".
- **Momentum** - price change over the last six months.
- **Sharpe** - six-month return per unit of risk (annualised, daily prices,
  {RISK_FREE_RATE*100:.0f}% risk-free rate). Six months is a short window, so treat it as noisy.

The **Combined Score** blends them 40% Valuation, 40% Momentum, 20% Sharpe. When
a score is missing, its weight shifts to the others.

**Δ 1w** is the change in Combined Score over the last week, from the scores
recorded after each daily refresh{(" since " + f"{history['date'].min():%-d %b %Y}") if not history.empty else ""}.
Because scores are ranks, a stock can move because the others did.

#### How much to trust them
The momentum signal was backtested over about 54 monthly periods: an
information coefficient of about 0.10 and a ~63% win rate - a modest but
genuine edge, not a strong one. Valuation and Sharpe are included on sound
principles but have not been backtested here.

#### Which stocks are included
Every JSE ordinary share with a market cap of R{cap_floor_bn:.0f}bn or more
({universe_meta.get('count', '?')} names), rebuilt every quarter
{('- last on ' + universe_meta['generated_sast'][:10]) if universe_meta.get('generated_sast') else ''}.
Each day, shares trading under R{adv_floor/1e6:.0f}m a day on average are left out,
because momentum and Sharpe on barely-traded prices are false signals, as are
shares with broken or too-short price history.

**FX exposure** (Rand hedge / Mixed / Domestic) is a manual judgement of where
each business earns its money - approximate, not taken from filings.

*Not investment advice.*
"""
    )
    with st.expander(f"Screened out today ({len(excluded)})"):
        if excluded:
            st.dataframe(
                pd.DataFrame(
                    [{"Ticker": e["ticker"].replace(".JO", ""), "Reason": e["reason"]}
                     for e in excluded]
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("None.")
    if failures:
        st.caption(
            f"Data errors in this refresh ({len(failures)} of "
            f"{meta.get('tickers_requested', len(JSE_TICKERS))}): "
            + ", ".join(t.replace(".JO", "") for t, _ in failures)
            + ". These usually clear on the next daily refresh."
        )
    st.caption(
        f"Data as of {pretty_stamp(stamp)} SAST · refreshed after each JSE close · "
        f"version {APP_VERSION}"
    )

# ---------------------------------------------------------------- stock report
# Last, so the page behind the pop-up is fully drawn.
if st.session_state.open_stock in names:
    open_report(st.session_state.open_stock)
