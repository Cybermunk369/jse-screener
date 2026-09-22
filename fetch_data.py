"""
fetch_data.py - offline data refresh for the JSE Screener.

Run on a schedule by .github/workflows/refresh.yml after the JSE close. It
fetches prices and fundamentals, applies liquidity screens, computes scores,
and writes data/screener.csv. The Streamlit app then just reads that file.

Why: fetching at page load meant every cold start hammered Yahoo from
Streamlit Cloud's shared IPs, which is what caused the rate limiting. Doing the
work once a day on a GitHub runner removes that entirely, makes the app load
instantly, and lets the universe grow without making the problem worse.

Also importable - the app falls back to build_dataset() if the data file is
missing, so the app never depends on the pipeline having run yet.

Usage:
    python fetch_data.py              # refresh data/screener.csv
    python fetch_data.py --validate   # test the universe, write nothing
"""

import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

RISK_FREE_RATE = 0.08
TRADING_DAYS = 252
MIN_HISTORY_DAYS = 100        # ~6 months of JSE trading days, with slack
FETCH_RETRIES = 3
FETCH_PAUSE_SECONDS = 0.5     # be polite to Yahoo as the universe grows

# Liquidity screen. The JSE's small-cap tail is genuinely untradeable - shares
# that print the same close for days because nothing traded. Momentum and
# Sharpe computed on stale prices are not weak signals, they are fake ones, so
# these names are dropped BEFORE scoring. Leaving them in would also corrupt
# every percentile rank in the table.
MIN_ADV_RAND = 5_000_000      # 20-day average daily value traded
ADV_WINDOW = 20
STALE_DAYS = 5                # identical closes for this many days = not trading

# Refresh must not silently shrink the table. The gate is on the share of the
# universe that hit a data error (bad response, no history) - not on the row
# count vs the last run - because the universe is now rule-based and rebuilt
# quarterly, so it can legitimately shrink. Names screened out on purpose
# (illiquid, stale, too new) are not data errors and do not count.
MAX_FAILURE_RATE = 0.10

# Anchored to this file's directory, not the working directory. Streamlit
# Cloud and GitHub Actions do not guarantee the same cwd, and a relative path
# that silently misses just falls back to the built-in universe - which is
# exactly the kind of quiet wrong answer this app must not give.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "screener.csv")
META_FILE = os.path.join(DATA_DIR, "metadata.json")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.csv")
UNIVERSE_META_FILE = os.path.join(DATA_DIR, "universe_meta.json")

SAST = timezone(timedelta(hours=2))

# Fallback universe, used only if data/universe.csv is missing.
FALLBACK_UNIVERSE = {
    "NPN.JO": ("Naspers", "Rand Hedge"),
    "SOL.JO": ("Sasol", "Mixed"),
    "CPI.JO": ("Capitec", "Domestic"),
    "AGL.JO": ("Anglo American", "Rand Hedge"),
    "SHP.JO": ("Shoprite", "Domestic"),
    "FSR.JO": ("FirstRand", "Domestic"),
    "MTN.JO": ("MTN Group", "Mixed"),
    "SBK.JO": ("Standard Bank", "Mixed"),
    "ABG.JO": ("Absa Group", "Domestic"),
    "NED.JO": ("Nedbank", "Domestic"),
    "CLS.JO": ("Clicks Group", "Domestic"),
    "WHL.JO": ("Woolworths Holdings", "Mixed"),
    "PRX.JO": ("Prosus", "Rand Hedge"),
    "BID.JO": ("Bid Corporation", "Rand Hedge"),
    "REM.JO": ("Remgro", "Mixed"),
    "VOD.JO": ("Vodacom", "Mixed"),
    "IMP.JO": ("Impala Platinum", "Rand Hedge"),
    "GFI.JO": ("Gold Fields", "Rand Hedge"),
}

COLUMNS = [
    "Ticker", "Name", "Price (R)", "Market Cap (R bn)", "Sector", "FX Exposure",
    "P/E", "6mo Momentum %", "Sharpe Ratio", "ADV (R m)",
    "Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score",
]


# ---------------------------------------------------------------- universe

def load_universe(path=UNIVERSE_FILE):
    """Read data/universe.csv -> {ticker: (name, fx_exposure)}."""
    if not os.path.exists(path):
        return dict(FALLBACK_UNIVERSE)

    universe = {}
    # utf-8-sig tolerates the byte-order mark Excel adds when saving a CSV.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip()
            if not ticker or ticker.startswith("#"):
                continue
            universe[ticker] = (
                (row.get("name") or ticker).strip(),
                (row.get("fx_exposure") or "Unclassified").strip() or "Unclassified",
            )
    return universe or dict(FALLBACK_UNIVERSE)


# Kept for backwards compatibility with anything importing these names.
JSE_TICKERS = {t: n for t, (n, _) in load_universe().items()}
FX_EXPOSURE = {t: fx for t, (_, fx) in load_universe().items()}


# ---------------------------------------------------------------- fetching

class Excluded(Exception):
    """Ticker fetched fine but failed a screen - not a data error."""


def fetch_one(ticker, name, fx_exposure):
    """Fetch and screen a single ticker. Returns a row dict, or raises."""
    t = yf.Ticker(ticker)

    # auto_adjust=False so Close stays the actual traded price for display,
    # while Adj Close drives return calculations.
    hist = t.history(period="6mo", auto_adjust=False)
    if hist.empty:
        raise ValueError("no price history")
    if len(hist) < MIN_HISTORY_DAYS:
        # A short window would be ranked head-to-head against full 6-month
        # windows, which is not a like-for-like comparison. With a rule-based
        # universe this is usually a recent listing, so it is a screen rather
        # than a data error.
        raise Excluded(f"too new ({len(hist)} trading days, need {MIN_HISTORY_DAYS})")

    # JSE prices come back in ZAc (cents).
    close_rand = hist["Close"] / 100

    if close_rand.tail(STALE_DAYS).nunique() == 1:
        raise Excluded(f"stale price ({STALE_DAYS}d unchanged)")

    adv_rand = float((close_rand * hist["Volume"]).tail(ADV_WINDOW).mean())
    if not np.isfinite(adv_rand) or adv_rand < MIN_ADV_RAND:
        raise Excluded(f"illiquid (ADV R{adv_rand/1e6:.1f}m)")

    adj_series = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
    adj_rand = adj_series / 100

    momentum_pct = (adj_rand.iloc[-1] - adj_rand.iloc[0]) / adj_rand.iloc[0] * 100

    daily_returns = adj_rand.pct_change().dropna()
    if len(daily_returns) >= 2 and daily_returns.std() != 0:
        annual_return = daily_returns.mean() * TRADING_DAYS
        annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (annual_return - RISK_FREE_RATE) / annual_vol
    else:
        sharpe = None

    info = t.info
    market_cap = info.get("marketCap")

    return {
        "Ticker": ticker.replace(".JO", ""),
        "Name": name,
        "Price (R)": round(float(close_rand.iloc[-1]), 2),
        "P/E": info.get("trailingPE"),
        # marketCap is already in ZAR - do NOT divide by 100.
        "Market Cap (R bn)": round(market_cap / 1e9, 2) if market_cap else None,
        "Sector": info.get("sector", "N/A"),
        "FX Exposure": fx_exposure,
        "6mo Momentum %": round(float(momentum_pct), 1),
        "Sharpe Ratio": round(float(sharpe), 2) if sharpe is not None else None,
        "ADV (R m)": round(adv_rand / 1e6, 1),
    }


def fetch_all(universe, verbose=True):
    """Fetch every ticker with retries.

    Returns (rows, failures, excluded). `failures` are data errors worth
    worrying about; `excluded` are names that were screened out on purpose.
    """
    rows, failures, excluded = [], [], []
    total = len(universe)

    for i, (ticker, (name, fx)) in enumerate(universe.items(), start=1):
        last_error = None
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                rows.append(fetch_one(ticker, name, fx))
                last_error = None
                break
            except Excluded as e:
                excluded.append((ticker, str(e)))
                last_error = None
                if verbose:
                    print(f"  [{i}/{total}] {ticker} excluded - {e}")
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < FETCH_RETRIES:
                    time.sleep(attempt * 2)

        if last_error:
            failures.append((ticker, last_error))
            if verbose:
                print(f"  [{i}/{total}] {ticker} FAILED - {last_error}")
        elif verbose and not (excluded and excluded[-1][0] == ticker):
            print(f"  [{i}/{total}] {ticker} ok")

        time.sleep(FETCH_PAUSE_SECONDS)

    return rows, failures, excluded


# ---------------------------------------------------------------- scoring

def add_valuation_score(df):
    """Percentile rank on P/E - cheaper is better. Negative or zero P/E is
    excluded, because a loss-making company is not 'cheap'."""
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


# Valuation and Momentum carry equal, dominant weight; Sharpe is a smaller
# risk-adjustment overlay. Momentum is the only one of the three with a
# backtest behind it (see the app's momentum caption) - Valuation and Sharpe
# are included on theoretical grounds, not a verified edge, which is why
# Sharpe gets the smallest slice rather than an equal third.
COMBINED_SCORE_WEIGHTS = {
    "Valuation Score": 0.4,
    "Momentum Score": 0.4,
    "Sharpe Score": 0.2,
}


def add_combined_score(df, weights=None):
    """Weighted mean of Valuation, Momentum, and Sharpe Score. Weights are
    renormalized across whichever inputs are present for a given row, so a
    missing Valuation Score (bad or negative P/E) doesn't get silently
    treated as 0 - Momentum and Sharpe pick up its share proportionally
    instead of the row just being penalised twice."""
    weights = weights or COMBINED_SCORE_WEIGHTS
    cols = list(weights.keys())
    w = pd.Series(weights)
    scores = df[cols]
    present = scores.notna()
    weighted_sum = scores.fillna(0).mul(w, axis=1).sum(axis=1)
    weight_total = present.mul(w, axis=1).sum(axis=1)
    combined = (weighted_sum / weight_total).round(0)
    combined[weight_total == 0] = np.nan
    df["Combined Score"] = combined
    return df


def score(df):
    df = add_valuation_score(df)
    df = add_momentum_score(df)
    df = add_sharpe_score(df)
    df = add_combined_score(df)
    return df


def build_dataset(universe=None, verbose=False):
    """Fetch and score in one step. Returns (DataFrame, failures).

    The app uses this as a fallback when the precomputed file is absent.
    """
    universe = universe or load_universe()
    rows, failures, _ = fetch_all(universe, verbose=verbose)
    if not rows:
        return pd.DataFrame(columns=COLUMNS), failures
    return score(pd.DataFrame(rows)), failures


# ---------------------------------------------------------------- output

def write_outputs(df, failures, excluded, requested):
    os.makedirs(DATA_DIR, exist_ok=True)

    df = df.reindex(columns=COLUMNS)
    df.to_csv(DATA_FILE, index=False)

    now_utc = datetime.now(timezone.utc)
    meta = {
        "generated_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_sast": now_utc.astimezone(SAST).strftime("%Y-%m-%d %H:%M:%S"),
        "tickers_requested": requested,
        "tickers_loaded": int(len(df)),
        "min_adv_rand": MIN_ADV_RAND,
        "failures": [{"ticker": t, "reason": r} for t, r in failures],
        "excluded": [{"ticker": t, "reason": r} for t, r in excluded],
    }
    with open(META_FILE, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    return meta


def validate():
    """Test every ticker in the universe and report. Writes nothing.

    Use this after editing data/universe.csv, before relying on a refresh.
    """
    universe = load_universe()
    print(f"Validating {len(universe)} tickers from {UNIVERSE_FILE}...\n")

    rows, failures, excluded = fetch_all(universe)

    print(f"\n{'='*58}")
    print(f"  tradeable : {len(rows)}")
    print(f"  excluded  : {len(excluded)}  (screened out on purpose)")
    print(f"  failed    : {len(failures)}  (bad ticker or data error)")
    print(f"{'='*58}")

    if excluded:
        print("\nScreened out (illiquid, stale, or too new):")
        for ticker, reason in excluded:
            print(f"  {ticker:<10} {reason}")

    if failures:
        print("\nFailed - check these ticker codes:")
        for ticker, reason in failures:
            print(f"  {ticker:<10} {reason}")

    return 0 if rows else 1


def main():
    # Never refresh from the built-in fallback list: that would quietly
    # replace a full table with 18 names.
    if not os.path.exists(UNIVERSE_FILE):
        print(f"FAILED: {UNIVERSE_FILE} not found. Keeping the existing data file.",
              file=sys.stderr)
        return 1

    universe = load_universe()
    requested = len(universe)
    print(f"Fetching {requested} JSE tickers from {UNIVERSE_FILE}...")

    rows, failures, excluded = fetch_all(universe)
    loaded = len(rows)

    print(f"\nTradeable {loaded}/{requested} "
          f"({len(excluded)} screened out, {len(failures)} failed).")

    # Quality gate. A bad fetch (e.g. Yahoo rate-limiting the runner) must not
    # overwrite a good table with a thin one.
    if loaded == 0:
        print("FAILED: no tickers loaded. Keeping the existing data file.",
              file=sys.stderr)
        return 1

    if len(failures) > requested * MAX_FAILURE_RATE:
        print(
            f"FAILED: {len(failures)} of {requested} tickers hit a data error "
            f"(limit {MAX_FAILURE_RATE:.0%}). Keeping the existing data file.",
            file=sys.stderr,
        )
        for ticker, reason in failures:
            print(f"  {ticker}: {reason}", file=sys.stderr)
        return 1

    meta = write_outputs(score(pd.DataFrame(rows)), failures, excluded, requested)
    print(f"Wrote {DATA_FILE} and {META_FILE} at {meta['generated_sast']} SAST.")

    if excluded:
        print(f"\n{len(excluded)} screened out:")
        for ticker, reason in excluded:
            print(f"  {ticker}: {reason}")
    if failures:
        print(f"\n{len(failures)} failed:")
        for ticker, reason in failures:
            print(f"  {ticker}: {reason}")

    return 0


if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(validate())
    sys.exit(main())
