"""
tools/company_coverage.py - how complete is Yahoo's company data for our JSE
universe? Report only: prints to the log, writes nothing.

Run by .github/workflows/company-coverage.yml (manual). Used to decide which
company fields are reliable enough to show in the app's stock report, and to
check units (JSE prices come back in cents; dividend yield has been reported
as a fraction in some yfinance versions and a percentage in others).
"""

import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_data import load_universe  # noqa: E402

FIELDS = [
    "longBusinessSummary", "industry", "website", "fullTimeEmployees",
    "dividendYield", "trailingAnnualDividendYield", "payoutRatio",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "regularMarketPrice", "currency",
    "returnOnEquity", "profitMargins", "debtToEquity", "revenueGrowth",
    "earningsGrowth", "priceToBook", "beta",
]
UNIT_CHECK = ["NPN.JO", "SHP.JO", "SBK.JO", "FSR.JO", "MTN.JO", "KAP.JO", "BTI.JO"]


def main():
    universe = load_universe()
    rows = []
    for i, ticker in enumerate(universe, 1):
        row = {"ticker": ticker}
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            for f in FIELDS:
                row[f] = info.get(f)
            try:
                cal = t.calendar or {}
                dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
                row["nextEarnings"] = str(dates[0]) if dates else None
            except Exception as e:  # calendar is often missing for JSE names
                row["nextEarnings"] = None
            row["error"] = None
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(f"[{i}/{len(universe)}] {ticker}", "ERR " + row["error"] if row["error"] else "ok", flush=True)
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    n = len(df)
    print("\n=== COVERAGE (share of the universe with a value) ===")
    for f in FIELDS + ["nextEarnings"]:
        present = df[f].notna() & (df[f].astype(str).str.strip() != "")
        print(f"COV {f:<30} {present.sum():>4}/{n}  {present.mean():6.1%}")

    s = df["longBusinessSummary"].dropna().astype(str).str.len()
    if len(s):
        print(f"\nSUMMARY LENGTH chars: min {s.min()}  median {int(s.median())}  max {s.max()}")

    print("\n=== UNIT CHECK ===")
    show = ["ticker", "regularMarketPrice", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
            "currency", "dividendYield", "trailingAnnualDividendYield", "payoutRatio",
            "returnOnEquity", "profitMargins", "debtToEquity", "revenueGrowth", "nextEarnings"]
    for _, r in df[df["ticker"].isin(UNIT_CHECK)][show].iterrows():
        print("UNIT " + " | ".join(f"{k}={r[k]}" for k in show))

    print("\n=== MISSING SUMMARY ===")
    print("MISS " + ", ".join(df.loc[df["longBusinessSummary"].isna(), "ticker"]))
    print("\n=== SAMPLE SUMMARY (SHP) ===")
    shp = df.loc[df["ticker"] == "SHP.JO", "longBusinessSummary"]
    print("SAMPLE " + (str(shp.iloc[0])[:600] if len(shp) else "n/a"))
    errs = df["error"].notna().sum()
    print(f"\nERRORS {errs}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
