"""
JSE Screener v1.12 - investor-facing layout.

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

import json
import os

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

DISPLAY_COLS = [
    "Ticker", "Name", "Price (R)", "Market Cap (R bn)",
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

    return display_df.style.apply(lambda _: styles, axis=None)


# ---------------------------------------------------------------- app

APP_VERSION = "1.12"

# Columns shown by default - enough to act on, narrow enough for a phone.
DEFAULT_COLS = [
    "Ticker", "Name", "Price (R)", "Combined Score", "6mo Momentum %", "P/E",
    "Sector",
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
    # The table editors keep positional edits; new keys discard them once the
    # rows re-order (see on_star_edit).
    st.session_state.table_version += 1


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


def on_star_edit(row_tickers, editor_key):
    """Apply ★ clicks in either table to the watchlist.

    The editor reports changes by row position in the table it was given, so
    the ticker order of that render is passed in. Starred rows then move, which
    would make the editor's stored edits point at the wrong rows - so every
    editor is recreated under a new key after each change.
    """
    edits = st.session_state[editor_key]["edited_rows"]
    watchlist = list(st.session_state.watchlist)
    for row, change in edits.items():
        if "★" not in change:
            continue
        ticker = row_tickers[int(row)]
        if change["★"] and ticker not in watchlist:
            watchlist.append(ticker)
        elif not change["★"] and ticker in watchlist:
            watchlist.remove(ticker)
    save_watchlist(watchlist)


names = dict(zip(df["Ticker"], df["Name"]))
sync_watch_param()

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
        ["Combined Score", "Sharpe Ratio", "Valuation Score", "Momentum Score",
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
    """Styled, sortable table whose only editable column is the ★."""
    display_df = data[["★"] + DISPLAY_COLS].reset_index(drop=True)
    column_config = {
        "★": st.column_config.CheckboxColumn(
            "★", pinned=True, width=40,
            help="Star a stock to add it to your watchlist.",
        ),
        "Ticker": st.column_config.Column(pinned=True, width="small"),
        "Name": st.column_config.Column(pinned=True, width="medium"),
    }
    # Every numeric column gets a NumberColumn (numeric dtype + numeric sort
    # comparator) with its display format, instead of pre-formatted strings.
    for col, spec in COLUMN_FORMATS.items():
        column_config[col] = st.column_config.NumberColumn(format=spec)
    key = f"{name}_{st.session_state.table_version}"
    # st.data_editor rather than st.dataframe so the ★ column is clickable.
    # Every other column is locked, and Styler colours still apply to locked
    # columns. placeholder shows a dash for missing values while the column
    # stays numeric, so sorting keeps working.
    st.data_editor(
        style_table(display_df, momentum_max_abs, sharpe_max_abs),
        key=key,
        use_container_width=True,
        hide_index=True,
        column_order=["★"] + (DISPLAY_COLS if all_columns else DEFAULT_COLS),
        column_config=column_config,
        disabled=DISPLAY_COLS,
        num_rows="fixed",
        placeholder="—",
        on_change=on_star_edit,
        args=(display_df["Ticker"].tolist(), key),
    )
    return display_df


def download_button(display_df, name):
    # The table's own toolbar download uses the browser's "Save as" (File
    # System Access) API, which some embedded browsers expose but can't write
    # through, leaving an empty file. This is built server-side instead and
    # exports exactly the view shown. utf-8-sig so Excel reads it right.
    data_date = (stamp or "")[:10] or "latest"
    export_df = display_df.drop(columns="★")
    export_df.insert(0, "Watchlist", display_df["★"].map({True: "yes", False: ""}))
    for col in ["Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score"]:
        export_df[col] = export_df[col].round().astype("Int64")   # 98, not 98.0
    export_df["P/E"] = export_df["P/E"].round(2)
    st.download_button(
        "⬇ Download CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"jse_screener_{name}_{data_date}.csv",
        mime="text/csv",
        key=f"download_{name}",
        help="Exports this table as shown.",
    )


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
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Stocks ranked", len(df), border=True,
    help=f"{len(excluded)} more screened out today (illiquid, stale or too new). "
         "See How it works.",
)
m2.metric(
    "Top Combined Score", f"{top['Ticker']} · {top['Combined Score']:.0f}",
    border=True, help=top["Name"],
)
m3.metric(
    "Strongest 6-month run", f"{mover['Ticker']} · {mover['6mo Momentum %']:+.0f}%",
    border=True, help=mover["Name"],
)
m4.metric(
    "Your watchlist", len(st.session_state.watchlist), border=True,
    help="Star stocks in the table, or search in the sidebar.",
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
               "Starred stocks stay at the top.")
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
            "Your watchlist is empty. Tick the ★ next to any stock in the "
            "Screener, or search for one in the sidebar."
        )
    else:
        starred = df[df["Ticker"].isin(watchlist)]
        if not starred.empty:
            shown = render_table(with_stars(starred), "watchlist")
            download_button(shown, "watchlist")
        st.caption(
            f"★ {len(watchlist)} on your watchlist, shown regardless of the "
            "screener filters. "
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
