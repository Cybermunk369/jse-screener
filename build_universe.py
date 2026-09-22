"""
build_universe.py - rule-based JSE universe.

Replaces the hand-picked ticker list with a rule anyone can check:

    every ordinary share listed on the JSE with a market cap of at least R5bn

Liquidity is deliberately NOT part of this rule. The daily refresh in
fetch_data.py already drops anything trading under R5m a day, and applying it
there (daily) rather than here (quarterly) means a stock that becomes liquid
mid-quarter is picked up straight away.

Run quarterly by .github/workflows/universe.yml. Writes:
    data/universe.csv        - the universe fetch_data.py reads
    data/universe_meta.json  - the rule, the date, and what changed

Manual input lives in data/universe_overrides.csv (ticker, name,
fx_exposure, exclude). It supplies display names and the Rand hedge /
domestic classification, which is a judgement call no data feed provides,
and lets a specific ticker be excluded (e.g. a duplicate share class).
New names that are not in the overrides file come through as "Unclassified"
and are listed in the run log so they can be classified.

Usage:
    python build_universe.py             # rebuild data/universe.csv
    python build_universe.py --dry-run   # show the result, write nothing
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import yfinance as yf
from yfinance import EquityQuery

from fetch_data import DATA_DIR, SAST, UNIVERSE_FILE, UNIVERSE_META_FILE

OVERRIDES_FILE = os.path.join(DATA_DIR, "universe_overrides.csv")

MIN_MARKET_CAP_RAND = 5_000_000_000
EXCHANGE = "JNB"
PAGE_SIZE = 250                 # Yahoo's maximum per request
MAX_RESULTS = 2_000             # hard stop on pagination
FETCH_RETRIES = 3

# Quality gates. A broken or partial screener response must not overwrite a
# good universe, so the run fails (and keeps the existing file) if any of
# these trip.
MIN_UNIVERSE = 40               # the Top 40 alone clears the floor
MAX_SHRINK = 0.70               # vs the current universe
# Large, unambiguous JSE names. All must be present (proves the response is
# complete) and their market caps must look like rands, not cents - mixing
# the two is the exact bug that once made every market cap 100x too small.
ANCHORS = ["NPN.JO", "FSR.JO", "SBK.JO", "SHP.JO", "MTN.JO"]
ANCHOR_CAP_RANGE = (1e10, 1e13)  # R10bn to R10tn

# Things Yahoo sometimes tags as EQUITY that are not ordinary shares.
NON_ORDINARY = re.compile(
    r"\bpref|\bpreference|\bdebenture|\bwarrant|\bETF\b|\bETN\b|\bnotes?\b"
    r"|\bbonds?\b|\btracker\b|\bindex\b|satrix|newgold|new gold|1nvest"
    r"|sygnia itrix|coreshares|etfsa|absa capital",
    re.IGNORECASE,
)

NAME_SUFFIXES = re.compile(
    r"[\s,]+(limited|ltd\.?|plc|p\.l\.c\.|n\.v\.|s\.a\.|se|ag|inc\.?)$",
    re.IGNORECASE,
)


class GateFailed(Exception):
    """The screener response failed a sanity check - keep the old universe."""


# ---------------------------------------------------------------- inputs

def load_overrides(path=OVERRIDES_FILE):
    """{ticker: {"name", "fx_exposure", "exclude"}} from the manual file."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip()
            if not ticker or ticker.startswith("#"):
                continue
            out[ticker] = {
                "name": (row.get("name") or "").strip(),
                "fx_exposure": (row.get("fx_exposure") or "").strip(),
                "exclude": (row.get("exclude") or "").strip().lower()
                in ("1", "yes", "y", "true", "x"),
            }
    return out


def load_current_universe(path=UNIVERSE_FILE):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [
            (row.get("ticker") or "").strip()
            for row in csv.DictReader(fh)
            if (row.get("ticker") or "").strip()
        ]


# ---------------------------------------------------------------- fetch

def _screen_page(offset):
    query = EquityQuery("eq", ["exchange", EXCHANGE])
    last_error = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            return yf.screen(
                query,
                offset=offset,
                size=PAGE_SIZE,
                sortField="intradaymarketcap",
                sortAsc=False,
            )
        except Exception as e:  # network / rate limit
            last_error = e
            time.sleep(attempt * 5)
    raise GateFailed(f"screener request failed: {type(last_error).__name__}: {last_error}")


def fetch_jse_equities():
    """Every JSE equity Yahoo knows about. Returns (quotes, reported_total)."""
    quotes, offset, total = [], 0, None
    while offset < MAX_RESULTS:
        result = _screen_page(offset)
        page = result.get("quotes") or []
        if total is None:
            total = result.get("total")
        quotes.extend(page)
        offset += len(page)
        if not page or (total is not None and offset >= total):
            break
        time.sleep(1)
    return quotes, total


# ---------------------------------------------------------------- rules

def clean_name(name):
    name = (name or "").strip()
    previous = None
    while previous != name:
        previous = name
        name = NAME_SUFFIXES.sub("", name).strip()
    return name


def check_anchors(by_symbol):
    missing = [a for a in ANCHORS if a not in by_symbol]
    if missing:
        raise GateFailed(
            f"screener response looks incomplete - missing {', '.join(missing)}"
        )
    lo, hi = ANCHOR_CAP_RANGE
    bad = {
        a: by_symbol[a].get("marketCap")
        for a in ANCHORS
        if not isinstance(by_symbol[a].get("marketCap"), (int, float))
        or not lo <= by_symbol[a]["marketCap"] <= hi
    }
    if bad:
        detail = ", ".join(f"{a}={v}" for a, v in bad.items())
        raise GateFailed(
            "anchor market caps outside the plausible rand range "
            f"(R{lo/1e9:.0f}bn-R{hi/1e12:.0f}tn) - units may have changed: {detail}"
        )


def apply_rules(quotes, overrides):
    """Returns (kept, dropped) where dropped is [(symbol, name, reason)]."""
    by_symbol = {}
    for q in quotes:
        sym = q.get("symbol")
        if sym and sym not in by_symbol:
            by_symbol[sym] = q

    check_anchors(by_symbol)

    kept, dropped = [], []
    for sym, q in by_symbol.items():
        raw_name = q.get("longName") or q.get("shortName") or sym
        cap = q.get("marketCap")

        if not sym.endswith(".JO"):
            reason = "not a .JO listing"
        elif q.get("quoteType") not in (None, "EQUITY"):
            reason = f"quoteType {q.get('quoteType')}"
        elif NON_ORDINARY.search(raw_name) or NON_ORDINARY.search(q.get("shortName") or ""):
            reason = "not an ordinary share"
        elif not isinstance(cap, (int, float)):
            reason = "no market cap"
        elif cap < MIN_MARKET_CAP_RAND:
            continue  # the normal case - not worth listing
        elif overrides.get(sym, {}).get("exclude"):
            reason = "excluded in universe_overrides.csv"
        else:
            reason = None

        if reason:
            dropped.append((sym, raw_name, reason))
            continue

        ov = overrides.get(sym, {})
        kept.append({
            "ticker": sym,
            "name": ov.get("name") or clean_name(raw_name) or sym,
            "fx_exposure": ov.get("fx_exposure") or "Unclassified",
            "market_cap_rbn": round(cap / 1e9, 1),
        })

    kept.sort(key=lambda r: r["market_cap_rbn"], reverse=True)
    return kept, dropped


def check_size(kept, current):
    if len(kept) < MIN_UNIVERSE:
        raise GateFailed(f"only {len(kept)} names passed the rule (minimum {MIN_UNIVERSE})")
    if current and len(kept) < len(current) * MAX_SHRINK:
        raise GateFailed(
            f"universe would shrink from {len(current)} to {len(kept)} "
            f"(below {MAX_SHRINK:.0%}) - looks like a partial response"
        )


# ---------------------------------------------------------------- output

def write_outputs(kept, meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(UNIVERSE_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["ticker", "name", "fx_exposure", "market_cap_rbn"]
        )
        writer.writeheader()
        writer.writerows(kept)
    with open(UNIVERSE_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


def main(dry_run=False):
    overrides = load_overrides()
    current = load_current_universe()

    print(f"Screening all {EXCHANGE} equities on Yahoo...")
    try:
        quotes, total = fetch_jse_equities()
        print(f"  {len(quotes)} quotes returned (Yahoo reports {total} total)")
        kept, dropped = apply_rules(quotes, overrides)
        check_size(kept, current)
    except GateFailed as e:
        print(f"FAILED: {e}. Keeping the existing universe.", file=sys.stderr)
        return 1

    kept_set = {r["ticker"] for r in kept}
    added = sorted(kept_set - set(current))
    removed = sorted(set(current) - kept_set)
    unclassified = [r["ticker"] for r in kept if r["fx_exposure"] == "Unclassified"]

    print(f"\nUniverse: {len(kept)} JSE ordinary shares with market cap >= "
          f"R{MIN_MARKET_CAP_RAND/1e9:.0f}bn\n")
    print(f"  {'ticker':<9} {'mkt cap':>10}  {'fx':<13} name")
    for r in kept:
        print(f"  {r['ticker']:<9} R{r['market_cap_rbn']:>8,.1f}bn  "
              f"{r['fx_exposure']:<13} {r['name']}")

    if dropped:
        print(f"\nAbove the floor but dropped ({len(dropped)}):")
        for sym, name, reason in dropped:
            print(f"  {sym:<9} {reason:<36} {name}")
    print(f"\nAdded vs current universe ({len(added)}): {', '.join(added) or '-'}")
    print(f"Removed vs current universe ({len(removed)}): {', '.join(removed) or '-'}")
    if unclassified:
        print(f"\nNeeds an FX classification in {os.path.basename(OVERRIDES_FILE)} "
              f"({len(unclassified)}): {', '.join(unclassified)}")

    if dry_run:
        print("\nDry run - nothing written.")
        return 0

    now = datetime.now(timezone.utc)
    meta = {
        "generated_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_sast": now.astimezone(SAST).strftime("%Y-%m-%d %H:%M:%S"),
        "rule": {
            "exchange": EXCHANGE,
            "min_market_cap_rand": MIN_MARKET_CAP_RAND,
            "instruments": "ordinary shares (ETFs, prefs, notes excluded)",
            "source": "Yahoo Finance screener via yfinance",
        },
        "count": len(kept),
        "added": added,
        "removed": removed,
        "unclassified": unclassified,
        "dropped": [{"ticker": s, "reason": r} for s, _, r in dropped],
    }
    write_outputs(kept, meta)
    print(f"\nWrote {UNIVERSE_FILE} and {UNIVERSE_META_FILE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
