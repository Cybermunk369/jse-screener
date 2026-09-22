"""
fetch_data.py - offline data refresh for the JSE Screener.

Run on a schedule by .github/workflows/refresh.yml after the JSE close. It
fetches prices and fundamentals, computes scores, and writes data/screener.csv.
The Streamlit app then just reads that file.

Why: fetching at page load meant every cold start hammered Yahoo from
Streamlit Cloud's shared IPs, which is what caused the rate limiting. Doing the
work once a day on a GitHub runner removes that entirely, makes the app load
instantly, and lets the universe grow without making the problem worse.

Also importable - the app falls back to build_dataset() if the data file is
missing, so the app never depends on the pipeline having run yet.

Usage:
    python fetch_data.py
"""

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
FETCH_PAUSE_SECONDS = 0.5     # be polite to Yahoo when the universe grows
MIN_SUCCESS_RATE = 0.80       # below this the refresh fails and keeps old data

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "screener.csv")
META_FILE = os.path.join(DATA_DIR, "metadata.json")

SAST = timezone(timedelta(hours=2))

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

COLUMNS = [
    "Ticker", "Name", "Price (R)", "Market Cap (R bn)", "Sector", "FX Exposure",
    "P/E", "6mo Momentum %", "Sharpe Ratio",
    "Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score",
]


# ---------------------------------------------------------------- fetching

def fetch_one(ticker, name):
    """Fetch a single ticker. Returns a row dict, or raises."""
    t = yf.Ticker(ticker)

    # auto_adjust=False so Close stays the actual traded price for display,
    # while Adj Close drives return calculations.
    hist = t.history(period="6mo", auto_adjust=False)
    if hist.empty:
        raise ValueError("no price history")
    if len(hist) < MIN_HISTORY_DAYS:
        # A short window would be ranked head-to-head against full 6-month
        # windows, which is not a like-for-like comparison.
        raise ValueError(f"only {len(hist)} bars")

    info = t.info

    # JSE prices come back in ZAc (cents).
    close_rand = hist["Close"] / 100
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

    market_cap = info.get("marketCap")

    return {
        "Ticker": ticker.replace(".JO", ""),
        "Name": name,
        "Price (R)": round(float(close_rand.iloc[-1]), 2),
        "P/E": info.get("trailingPE"),
        # marketCap is already in ZAR - do NOT divide by 100.
        "Market Cap (R bn)": round(market_cap / 1e9, 2) if market_cap else None,
        "Sector": info.get("sector", "N/A"),
        "FX Exposure": FX_EXPOSURE.get(ticker, "Unclassified"),
        "6mo Momentum %": round(float(momentum_pct), 1),
        "Sharpe Ratio": round(float(sharpe), 2) if sharpe is not None else None,
    }


def fetch_all(tickers, verbose=True):
    """Fetch every ticker with retries. Returns (rows, failures)."""
    rows, failures = [], []

    for i, (ticker, name) in enumerate(tickers.items(), start=1):
        last_error = None
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                rows.append(fetch_one(ticker, name))
                last_error = None
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < FETCH_RETRIES:
                    time.sleep(attempt * 2)

        if last_error:
            failures.append((ticker, last_error))
            if verbose:
                print(f"  [{i}/{len(tickers)}] {ticker} FAILED - {last_error}")
        elif verbose:
            print(f"  [{i}/{len(tickers)}] {ticker} ok")

        time.sleep(FETCH_PAUSE_SECONDS)

    return rows, failures


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


def add_combined_score(df):
    """Mean of Valuation and Momentum. Where one input is missing the other
    carries it - the missing input renders as a dash in the app so the user can
    see the score rests on partial data."""
    df["Combined Score"] = (
        df[["Valuation Score", "Momentum Score"]].mean(axis=1, skipna=True).round(0)
    )
    return df


def score(df):
    df = add_valuation_score(df)
    df = add_momentum_score(df)
    df = add_sharpe_score(df)
    df = add_combined_score(df)
    return df


def build_dataset(tickers=None, verbose=False):
    """Fetch and score in one step. Returns (DataFrame, failures).

    The app uses this as a fallback when the precomputed file is absent.
    """
    tickers = tickers or JSE_TICKERS
    rows, failures = fetch_all(tickers, verbose=verbose)
    if not rows:
        return pd.DataFrame(columns=COLUMNS), failures
    return score(pd.DataFrame(rows)), failures


# ---------------------------------------------------------------- output

def write_outputs(df, failures, requested):
    os.makedirs(DATA_DIR, exist_ok=True)

    df = df.reindex(columns=COLUMNS)
    df.to_csv(DATA_FILE, index=False)

    now_utc = datetime.now(timezone.utc)
    meta = {
        "generated_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_sast": now_utc.astimezone(SAST).strftime("%Y-%m-%d %H:%M:%S"),
        "tickers_requested": requested,
        "tickers_loaded": int(len(df)),
        "failures": [{"ticker": t, "reason": r} for t, r in failures],
    }
    with open(META_FILE, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    return meta


def main():
    requested = len(JSE_TICKERS)
    print(f"Fetching {requested} JSE tickers...")

    rows, failures = fetch_all(JSE_TICKERS)
    loaded = len(rows)
    rate = loaded / requested if requested else 0

    print(f"\nLoaded {loaded}/{requested} ({rate:.0%}).")

    # Quality gate: a bad fetch must not overwrite good data with a thin file.
    # Failing here leaves the previous day's committed data in place.
    if rate < MIN_SUCCESS_RATE:
        print(
            f"FAILED: success rate {rate:.0%} is below the {MIN_SUCCESS_RATE:.0%} "
            "threshold. Keeping the existing data file.",
            file=sys.stderr,
        )
        for ticker, reason in failures:
            print(f"  {ticker}: {reason}", file=sys.stderr)
        return 1

    meta = write_outputs(score(pd.DataFrame(rows)), failures, requested)
    print(f"Wrote {DATA_FILE} and {META_FILE} at {meta['generated_sast']} SAST.")

    if failures:
        print(f"{len(failures)} ticker(s) missing from this refresh:")
        for ticker, reason in failures:
            print(f"  {ticker}: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
