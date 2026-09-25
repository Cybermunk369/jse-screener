"""
backtest/run_backtest.py - how well have the screener's indicators predicted
returns on the JSE? Research only: writes backtest/results/, never touches the
app's data.

Run by .github/workflows/backtest.yml (manual). Needs Yahoo, so it runs on a
GitHub runner, not locally.

What it tests, month by month (point in time - only information that was
public at each month-end is used):

  * Cross-sectional indicators: at each month-end, rank the liquid stocks by
    the indicator and see how well that ranking lined up with the NEXT month's
    total return (information coefficient, IC), and how the best fifth did
    against the worst fifth.
      - price-based (full history): 6-month momentum, 6-month Sharpe,
        price vs its 50-day average, 20-day average above/below 50-day
      - dividend yield (trailing 12 months of paid dividends / price)
      - from annual accounts, lagged 4 months after year-end so results were
        published: P/E, return on equity, profit margin, debt to equity,
        revenue growth (Yahoo keeps only ~4 years of accounts)
      - the app's Combined Score (40% value, 40% momentum, 20% Sharpe)
  * Top 20 by Combined Score, equal weight, rebalanced monthly, after costs,
    against the Satrix 40 ETF and an equal-weighted average of all stocks.
  * Moving-average crosses (20-day vs 50-day): what happened after each
    "golden cross" (buy signal) and "death cross" (sell signal), compared
    with the average stock over the same days; and a hold-when-above rule
    against simply holding, after trading costs.

Known limits, printed in the results too:
  * Survivorship: the stock list is today's (JSE shares >= R5bn now). Firms
    that shrank or delisted are missing, which flatters results.
  * Yahoo keeps ~4 years of annual accounts, so accounts-based tests are short.
  * P/E for companies reporting in dollars/pounds/euros is converted with the
    exchange rate on each date; EPS figures are Yahoo's and can be restated.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
UNIVERSE_FILE = os.path.join(REPO, "data", "universe.csv")
OUT_DIR = os.path.join(HERE, "results")

START = "2015-06-01"
ACCOUNTS_LAG_MONTHS = 4        # JSE results are due within 3 months of year-end
COST_PER_TRADE = 0.003         # 0.3% of the amount traded, each way
MIN_ADV_RAND = 5_000_000       # same liquidity screen as the app
MIN_HISTORY = 126              # ~6 months of trading days
STALE_DAYS = 5
MAX_DAILY_JUMP = 20
TOP_N = 20
RISK_FREE = 0.08
CASH_RATE = 0.07               # rough SA cash / money-market yield, 2015-2026
BENCHMARK = "STX40.JO"         # Satrix 40 ETF: tracks the FTSE/JSE Top 40
FX_TICKERS = {"USD": "USDZAR=X", "GBP": "GBPZAR=X", "EUR": "EURZAR=X",
              "GBp": "GBPZAR=X"}
MA_FAST, MA_SLOW = 20, 50
EVENT_HORIZONS = (20, 60)      # trading days after a cross
PAUSE = 0.3

# Direction in which each indicator is expected to predict: +1 = higher is
# better, -1 = lower is better. ICs are reported in this direction, so a
# positive IC always means "worked as the app assumes".
INDICATORS = {
    "momentum_6m": ("6-month momentum", +1),
    "sharpe_6m": ("6-month Sharpe", +1),
    "price_vs_ma50": ("Price vs 50-day average", +1),
    "ma20_above_ma50": ("20-day above 50-day average", +1),
    "dividend_yield": ("Dividend yield", +1),
    "pe": ("P/E (cheaper = better)", -1),
    "roe": ("Return on equity", +1),
    "margin": ("Profit margin", +1),
    "debt_equity": ("Debt to equity (less = better)", -1),
    "revenue_growth": ("Revenue growth", +1),
    "combined": ("Combined Score (40/40/20)", +1),
}
COMBINED_WEIGHTS = {"valuation": 0.4, "momentum": 0.4, "sharpe": 0.2}


# ---------------------------------------------------------------- fetching

def load_universe(path=UNIVERSE_FILE):
    u = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(u["ticker"], u["name"]))


def _row(frame, names):
    """First matching row of a Yahoo statement, as a Series by period end."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in frame.index:
            s = pd.to_numeric(frame.loc[name], errors="coerce")
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.dropna()
    return pd.Series(dtype=float)


def fetch_ticker(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(start=START, auto_adjust=False)
    if hist.empty:
        raise ValueError("no history")
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist = hist[~hist.index.duplicated(keep="last")]
    hist = hist.dropna(subset=["Close"])
    px = pd.DataFrame({
        "close": hist["Close"] / 100,                       # ZAc -> rand
        "adj": hist.get("Adj Close", hist["Close"]) / 100,
        "volume": hist["Volume"],
    })
    divs = t.dividends
    if len(divs):
        divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
    info = t.info or {}
    try:
        inc, bal = t.income_stmt, t.balance_sheet
    except Exception:
        inc, bal = None, None
    fin = pd.DataFrame({
        "net_income": _row(inc, ["Net Income", "Net Income Common Stockholders"]),
        "revenue": _row(inc, ["Total Revenue", "Operating Revenue"]),
        "eps": _row(inc, ["Diluted EPS", "Basic EPS"]),
        "equity": _row(bal, ["Stockholders Equity", "Common Stock Equity",
                             "Total Equity Gross Minority Interest"]),
        "debt": _row(bal, ["Total Debt"]),
    }).sort_index()
    return {"px": px, "divs": divs, "fin": fin, "info": info}


def fetch_fx():
    fx = {}
    for cur, sym in FX_TICKERS.items():
        h = yf.Ticker(sym).history(start=START)
        if len(h):
            s = h["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            fx[cur] = s[~s.index.duplicated(keep="last")]
        time.sleep(PAUSE)
    return fx


# ---------------------------------------------------------------- cleaning

def repair_history(px):
    """Yahoo's long JSE histories carry the odd bad print: a single day at
    100x (cents/rand mix-up) or a stretch quoted in the other unit. Drop
    one- to three-day spikes that reverse, and rescale a whole earlier
    stretch when the series steps by ~100x and stays there. Returns the
    repaired frame and the number of fixes."""
    px = px.copy()
    fixes = 0
    for _ in range(3):
        c = px["close"]
        step = c / c.shift(1)
        jumps = np.where((step > MAX_DAILY_JUMP / 4) | (step < 4 / MAX_DAILY_JUMP))[0]
        if not len(jumps):
            break
        drop = set()
        for i in jumps:
            if i - 1 in drop or i in drop:
                continue
            base = c.iloc[i - 1]
            # A spike that comes back within 3 days: drop those days.
            back = None
            for k in range(i + 1, min(i + 4, len(c))):
                if 0.5 < c.iloc[k] / base < 2:
                    back = k
                    break
            if back is not None:
                drop.update(range(i, back))
                continue
            # A lasting ~100x step: rescale everything before it.
            ratio = c.iloc[i] / base
            for factor in (100.0, 0.01):
                if 0.5 < ratio / factor < 2:
                    px.iloc[:i, px.columns.get_indexer(["close", "adj"])] *= factor
                    fixes += 1
                    break
        if drop:
            px = px.drop(px.index[sorted(drop)])
            fixes += len(drop)
    return px, fixes


def bad_history(px):
    """Same guard as the app: a 20x one-day move means mixed rand/cents."""
    for col in ("close", "adj"):
        step = (px[col] / px[col].shift(1)).dropna()
        step = step[step > 0]
        if len(step) and (step.max() > MAX_DAILY_JUMP or step.min() < 1 / MAX_DAILY_JUMP):
            return True
    return False


def dividend_units(divs, px, info):
    """Yahoo JSE dividends are usually in cents like the price, but not
    always. Calibrate against Yahoo's current dividend yield (a percentage);
    returns the factor that converts `divs` to rand, or None if unreliable."""
    if not len(divs) or not len(px):
        return None
    last = px.index[-1]
    ttm = divs[divs.index > last - pd.Timedelta(days=365)].sum()
    if ttm <= 0:
        return 0.01                                  # no recent dividends
    price = px["close"].iloc[-1]
    y_cents = ttm / 100 / price * 100                # % if divs are cents
    ref = info.get("dividendYield")
    if ref and ref > 0:
        for factor, y in ((0.01, y_cents), (1.0, y_cents * 100)):
            if 1 / 3 <= y / ref <= 3:
                return factor
        return None
    return 0.01 if 0 < y_cents < 30 else None


def eps_fx_factor(info):
    cur = info.get("financialCurrency") or "ZAR"
    return cur


def pe_units(features, info):
    """Check our latest P/E against Yahoo's current trailing P/E. EPS is
    sometimes in cents; returns the factor to apply to our P/E, or None when
    the two disagree too much to trust (P/E then left out for that stock)."""
    ref = info.get("trailingPE")
    ours = features["pe"].dropna() if "pe" in features else pd.Series(dtype=float)
    if not ref or ref <= 0 or ours.empty:
        return 1.0
    ratio = ours.iloc[-1] / ref
    if 1 / 3 <= ratio <= 3:
        return 1.0
    if 1 / 300 <= ratio <= 1 / 30:
        return 100.0
    return None


# ---------------------------------------------------------------- features

def month_ends(index):
    s = pd.Series(index, index=index)
    return s.groupby([index.year, index.month]).max().tolist()


def point_in_time_accounts(fin, when, lag_months=ACCOUNTS_LAG_MONTHS):
    """Latest annual accounts published by `when` (period end + lag), and the
    year before (for growth). Returns (latest_row, prior_row) or (None, None)."""
    if fin is None or fin.empty:
        return None, None
    available = fin[fin.index + pd.DateOffset(months=lag_months) <= when]
    if available.empty:
        return None, None
    latest = available.iloc[-1]
    prior = available.iloc[-2] if len(available) > 1 else None
    return latest, prior


def stock_features(data, dates, fx, div_factor):
    """Indicators for one stock at each month-end in `dates` (DataFrame)."""
    px = data["px"]
    close, adj, vol = px["close"], px["adj"], px["volume"]
    ma_f = close.rolling(MA_FAST).mean()
    ma_s = close.rolling(MA_SLOW).mean()
    adv = (close * vol).rolling(20).mean()
    rets = adj.pct_change()
    cur = eps_fx_factor(data["info"])
    fx_series = fx.get(cur) if cur not in ("ZAR", "ZAc") else None
    divs = data["divs"]
    rows = []
    for d in dates:
        pos = px.index.searchsorted(d, side="right") - 1
        if pos < MIN_HISTORY or px.index[pos] < d - pd.Timedelta(days=7):
            continue
        day = px.index[pos]
        window = close.iloc[pos - STALE_DAYS + 1: pos + 1]
        r = {"date": d, "price": close.iloc[pos], "adj": adj.iloc[pos],
             "adv": adv.iloc[pos],
             "stale": window.nunique() == 1}
        r["momentum_6m"] = adj.iloc[pos] / adj.iloc[pos - MIN_HISTORY] - 1
        dr = rets.iloc[pos - MIN_HISTORY + 1: pos + 1].dropna()
        sd = dr.std()
        r["sharpe_6m"] = ((dr.mean() * 252 - RISK_FREE) / (sd * np.sqrt(252))
                          if sd and np.isfinite(sd) else np.nan)
        r["price_vs_ma50"] = close.iloc[pos] / ma_s.iloc[pos] - 1 if ma_s.iloc[pos] else np.nan
        r["ma20_above_ma50"] = float(ma_f.iloc[pos] > ma_s.iloc[pos]) if np.isfinite(ma_s.iloc[pos]) else np.nan
        if div_factor is not None and len(divs):
            ttm = divs[(divs.index > day - pd.Timedelta(days=365)) & (divs.index <= day)].sum()
            r["dividend_yield"] = ttm * div_factor / close.iloc[pos]
        else:
            r["dividend_yield"] = np.nan
        latest, prior = point_in_time_accounts(data["fin"], day)
        for k in ("pe", "roe", "margin", "debt_equity", "revenue_growth"):
            r[k] = np.nan
        if latest is not None:
            eps = latest.get("eps")
            if pd.notna(eps) and eps > 0:
                rate = 1.0
                if fx_series is not None:
                    fpos = fx_series.index.searchsorted(day, side="right") - 1
                    rate = fx_series.iloc[fpos] if fpos >= 0 else np.nan
                    if cur == "GBp":
                        rate = rate / 100
                if np.isfinite(rate):
                    r["pe"] = close.iloc[pos] / (eps * rate)
            eq = latest.get("equity")
            ni = latest.get("net_income")
            rev = latest.get("revenue")
            if pd.notna(eq) and eq > 0 and pd.notna(ni):
                r["roe"] = ni / eq
            if pd.notna(rev) and rev > 0 and pd.notna(ni):
                r["margin"] = ni / rev
            if pd.notna(eq) and eq > 0 and pd.notna(latest.get("debt")):
                r["debt_equity"] = latest.get("debt") / eq
            if prior is not None and pd.notna(rev) and pd.notna(prior.get("revenue")) \
                    and prior.get("revenue") > 0:
                r["revenue_growth"] = rev / prior.get("revenue") - 1
        rows.append(r)
    return pd.DataFrame(rows)


def add_forward_returns(panel, adj_prices, dates):
    """Next month-end total return for each (date, ticker)."""
    nxt = dict(zip(dates[:-1], dates[1:]))
    out = []
    for (d, tk), row in panel.iterrows():
        if d not in nxt or tk not in adj_prices:
            out.append(np.nan)
            continue
        s = adj_prices[tk]
        p1 = s.index.searchsorted(nxt[d], side="right") - 1
        p0 = s.index.searchsorted(d, side="right") - 1
        if p1 <= p0 or s.index[p1] < nxt[d] - pd.Timedelta(days=7):
            out.append(np.nan)
        else:
            out.append(s.iloc[p1] / s.iloc[p0] - 1)
    panel["fwd_1m"] = out
    return panel


def pct_rank(s, ascending=True):
    return s.rank(pct=True, ascending=ascending) * 100


def add_combined(panel):
    """The app's Combined Score, computed within each month."""
    def per_month(g):
        val = pct_rank(g["pe"].where(g["pe"] > 0), ascending=False)
        mom = pct_rank(g["momentum_6m"])
        shp = pct_rank(g["sharpe_6m"])
        scores = pd.DataFrame({"valuation": val, "momentum": mom, "sharpe": shp})
        w = pd.Series(COMBINED_WEIGHTS)
        present = scores.notna()
        total = present.mul(w, axis=1).sum(axis=1)
        g = g.copy()
        g["combined"] = (scores.fillna(0).mul(w, axis=1).sum(axis=1) / total).where(total > 0)
        return g
    return panel.groupby(level=0, group_keys=False).apply(per_month)


# ---------------------------------------------------------------- evaluation

def evaluate_indicator(panel, col, direction):
    months, ics, spreads, q_rets = [], [], [], []
    for d, g in panel.groupby(level=0):
        g = g[[col, "fwd_1m"]].dropna()
        if len(g) < 25 or g[col].nunique() < (2 if col == "ma20_above_ma50" else 3):
            continue
        x = g[col] * direction
        ic = x.rank().corr(g["fwd_1m"].rank())
        if col == "ma20_above_ma50":
            top = g[x > x.min()]["fwd_1m"]
            bot = g[x == x.min()]["fwd_1m"]
            if len(top) < 5 or len(bot) < 5:
                continue
            q = [bot.mean(), np.nan, np.nan, np.nan, top.mean()]
        else:
            buckets = pd.qcut(x.rank(method="first"), 5, labels=False)
            q = [g["fwd_1m"][buckets == i].mean() for i in range(5)]
        months.append(d)
        ics.append(ic)
        q_rets.append(q)
        spreads.append(q[4] - q[0])
    if not months:
        return None
    ics, spreads = np.array(ics), np.array(spreads)
    n = len(ics)
    q = np.array(q_rets, dtype=float)
    return {
        "months": n,
        "first": str(months[0].date()), "last": str(months[-1].date()),
        "avg_stocks": float(panel[col].groupby(level=0).count().loc[months].mean()),
        "mean_ic": float(ics.mean()),
        "ic_t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n))) if n > 2 else None,
        "ic_positive_share": float((ics > 0).mean()),
        "spread_monthly": float(np.nanmean(spreads)),
        "spread_annual": float((1 + np.nanmean(spreads)) ** 12 - 1),
        "spread_positive_share": float((spreads > 0).mean()),
        "quintile_monthly": [None if np.isnan(v) else float(v) for v in np.nanmean(q, axis=0)],
        "ic_series": {str(m.date()): float(v) for m, v in zip(months, ics)},
        "spread_cum": dict(zip([str(m.date()) for m in months],
                               np.cumprod(1 + np.nan_to_num(spreads)).round(4).tolist())),
    }


def top_n_strategy(panel, bench_monthly, col="combined", n=TOP_N):
    """Equal-weight top-n by `col`, rebalanced monthly, with trading costs."""
    rows, prev = [], set()
    for d, g in panel.groupby(level=0):
        g = g[[col, "fwd_1m"]].dropna()
        if len(g) < max(40, n * 2):
            continue
        picks = g.sort_values(col, ascending=False).head(n)
        held = set(picks.index.get_level_values(1))
        turnover = len(held - prev) / n if prev else 1.0
        gross = picks["fwd_1m"].mean()
        rows.append({"date": d, "gross": gross,
                     "net": gross - 2 * COST_PER_TRADE * turnover,
                     "universe": g["fwd_1m"].mean(),
                     "turnover": turnover,
                     "benchmark": bench_monthly.get(d, np.nan)})
        prev = held
    df = pd.DataFrame(rows).set_index("date")
    return df


def perf_stats(r):
    r = pd.Series(r).dropna()
    if r.empty:
        return None
    growth = (1 + r).cumprod()
    years = len(r) / 12
    dd = growth / growth.cummax() - 1
    return {"months": int(len(r)),
            "cagr": float(growth.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
            "vol": float(r.std() * np.sqrt(12)),
            "max_drawdown": float(dd.min()),
            "total": float(growth.iloc[-1] - 1)}


def ma_cross_study(prices, liquid_days):
    """Event study of 20/50-day crosses plus a hold-when-above timing rule."""
    events = {"golden": {h: [] for h in EVENT_HORIZONS},
              "death": {h: [] for h in EVENT_HORIZONS}}
    # Equal-weight daily index of all stocks, for "vs the average stock".
    adj = pd.DataFrame({t: p["adj"] for t, p in prices.items()})
    daily = adj.pct_change(fill_method=None)
    avg = daily.mean(axis=1).fillna(0)
    avg_growth = (1 + avg).cumprod()
    timing, hold = {}, {}
    for t, p in prices.items():
        close, a = p["close"], p["adj"]
        f, s = close.rolling(MA_FAST).mean(), close.rolling(MA_SLOW).mean()
        above = (f > s).where(s.notna())
        cross = above.astype(float).diff()
        liquid = liquid_days.get(t, pd.Series(False, index=close.index))
        for i in np.where(cross.fillna(0) != 0)[0]:
            day = close.index[i]
            if not liquid.get(day, False) or i + 1 >= len(close):
                continue
            kind = "golden" if cross.iloc[i] > 0 else "death"
            entry = i + 1                                   # act next day
            for h in EVENT_HORIZONS:
                if entry + h >= len(a):
                    continue
                stock = a.iloc[entry + h] / a.iloc[entry] - 1
                d0, d1 = a.index[entry], a.index[entry + h]
                mkt = avg_growth.asof(d1) / avg_growth.asof(d0) - 1
                events[kind][h].append({"excess": float(stock - mkt), "raw": float(stock),
                                        "date": str(day.date()), "ticker": t})
        # Timing rule: hold only while 20-day > 50-day (decided at yesterday's
        # close), pay costs on every switch.
        r = a.pct_change(fill_method=None)
        pos = above.shift(1).fillna(0)
        switches = pos.diff().abs().fillna(0)
        cash = (1 + CASH_RATE) ** (1 / 252) - 1        # out of the stock = in cash
        timing[t] = (pos * r + (1 - pos) * cash
                     - switches * COST_PER_TRADE).where(s.shift(1).notna())
        hold[t] = r.where(s.shift(1).notna())
    summary = {}
    for kind in events:
        summary[kind] = {}
        for h in EVENT_HORIZONS:
            ex = np.array([e["excess"] for e in events[kind][h]])
            if len(ex) == 0:
                continue
            good = (ex > 0) if kind == "golden" else (ex < 0)
            summary[kind][str(h)] = {
                "count": int(len(ex)),
                "mean_excess": float(ex.mean()),
                "median_excess": float(np.median(ex)),
                "hit_rate": float(good.mean()),
                "t": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else None,
            }
    # Does a golden cross beat a death cross? Both are measured against the
    # same average stock, so this difference cancels any drift the event
    # sample shares (e.g. volatile stocks crossing more often).
    for h in EVENT_HORIZONS:
        g = np.array([e["excess"] for e in events["golden"][h]])
        d = np.array([e["excess"] for e in events["death"][h]])
        if len(g) > 2 and len(d) > 2:
            diff = g.mean() - d.mean()
            se = np.sqrt(g.var(ddof=1) / len(g) + d.var(ddof=1) / len(d))
            summary.setdefault("golden_minus_death", {})[str(h)] = {
                "diff": float(diff), "t": float(diff / se)}
    by_year = {}
    for kind in ("golden", "death"):
        for e in events[kind][EVENT_HORIZONS[-1]]:
            by_year.setdefault(e["date"][:4], {}).setdefault(kind, []).append(e["excess"])
    summary["by_year"] = {y: {k: {"count": len(v), "mean_excess": float(np.mean(v))}
                              for k, v in ks.items()} for y, ks in sorted(by_year.items())}
    tim = pd.DataFrame(timing).mean(axis=1).dropna()
    hol = pd.DataFrame(hold).mean(axis=1).dropna()
    both = pd.concat([tim.rename("timing"), hol.rename("hold")], axis=1).dropna()
    monthly = (1 + both).groupby([both.index.year, both.index.month]).prod() - 1
    monthly.index = [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y, m in monthly.index]
    summary["timing_rule"] = {"timing": perf_stats(monthly["timing"]),
                              "hold": perf_stats(monthly["hold"]),
                              "growth": {str(d.date()): [round(float(a), 4), round(float(b), 4)]
                                         for d, a, b in zip(monthly.index,
                                                            (1 + monthly["timing"]).cumprod(),
                                                            (1 + monthly["hold"]).cumprod())}}
    return summary


# ---------------------------------------------------------------- main

def run(universe, verbose=True):
    fx = fetch_fx()
    data, problems, repaired = {}, {}, {}
    for i, t in enumerate(universe, 1):
        try:
            d = fetch_ticker(t)
            d["px"], fixed = repair_history(d["px"])
            if fixed:
                repaired[t] = fixed
            if bad_history(d["px"]):
                problems[t] = "bad price history (rand/cents jump)"
            elif len(d["px"]) < MIN_HISTORY + 21:
                problems[t] = "too little history"
            else:
                data[t] = d
        except Exception as e:
            problems[t] = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"[{i}/{len(universe)}] {t} {'ok' if t in data else problems[t]}", flush=True)
        time.sleep(PAUSE)

    bench = yf.Ticker(BENCHMARK).history(start=START, auto_adjust=False)
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    bench_adj = bench.get("Adj Close", bench["Close"]).dropna()

    all_days = sorted(set().union(*[set(d["px"].index) for d in data.values()]))
    dates = [pd.Timestamp(x) for x in month_ends(pd.DatetimeIndex(all_days))]
    # Drop the current, unfinished month.
    if dates and dates[-1].month == pd.Timestamp.now().month and dates[-1].year == pd.Timestamp.now().year:
        dates = dates[:-1]

    frames, div_units, pe_unreliable = [], {}, []
    for t, d in data.items():
        factor = dividend_units(d["divs"], d["px"], d["info"])
        div_units[t] = factor
        f = stock_features(d, dates, fx, factor)
        if len(f):
            pf = pe_units(f, d["info"])
            if pf is None:
                pe_unreliable.append(t)
                f["pe"] = np.nan
            else:
                f["pe"] = f["pe"] * pf
            f.loc[(f["pe"] <= 0) | (f["pe"] > 300), "pe"] = np.nan
            f["ticker"] = t
            frames.append(f)
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    # Same screens the app applies, at each date.
    eligible = (panel["adv"] >= MIN_ADV_RAND) & ~panel["stale"]
    panel = panel[eligible]
    panel = add_forward_returns(panel, {t: d["px"]["adj"] for t, d in data.items()}, dates)
    panel = add_combined(panel)

    results = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
               "settings": {"start": START, "accounts_lag_months": ACCOUNTS_LAG_MONTHS,
                            "cost_per_trade": COST_PER_TRADE, "min_adv_rand": MIN_ADV_RAND,
                            "top_n": TOP_N, "benchmark": BENCHMARK,
                            "ma_fast": MA_FAST, "ma_slow": MA_SLOW, "cash_rate": CASH_RATE},
               "coverage": {"universe": len(universe), "with_prices": len(data),
                            "excluded": problems,
                            "repaired_bad_prints": repaired,
                            "dividend_units_unreliable": [t for t, f in div_units.items() if f is None],
                            "pe_unreliable": pe_unreliable,
                            "stocks_per_month": {str(d.date()): int(n) for d, n in
                                                 panel.groupby(level=0).size().items()}},
               "indicators": {}}
    for col, (label, direction) in INDICATORS.items():
        res = evaluate_indicator(panel, col, direction)
        if res:
            res["label"] = label
            res["direction"] = direction
            results["indicators"][col] = res
        if verbose:
            print(f"{label}: " + (f"IC {res['mean_ic']:+.3f} (t {res['ic_t']:+.1f}), "
                                  f"best-worst fifth {res['spread_annual']:+.1%}/yr over {res['months']} months"
                                  if res else "not enough data"))

    bm = {}
    for i, d in enumerate(dates[:-1]):
        a0 = bench_adj.asof(d)
        a1 = bench_adj.asof(dates[i + 1])
        if pd.notna(a0) and pd.notna(a1) and bench_adj.index[0] <= d:
            bm[d] = a1 / a0 - 1
    strategies = {}
    for col in ("combined", "momentum_6m"):
        df = top_n_strategy(panel, bm, col=col)
        strategies[col] = {
            "net": perf_stats(df["net"]), "gross": perf_stats(df["gross"]),
            "universe": perf_stats(df["universe"]), "benchmark": perf_stats(df["benchmark"]),
            "avg_turnover": float(df["turnover"].iloc[1:].mean()) if len(df) > 1 else None,
            "beat_benchmark_share": float((df["net"] > df["benchmark"]).mean()),
            "growth": {str(d.date()): [round(float(x), 4) for x in v] for d, v in
                       zip(df.index, zip((1 + df["net"]).cumprod(),
                                         (1 + df["universe"]).cumprod(),
                                         (1 + df["benchmark"].fillna(0)).cumprod()))},
        }
    results["top20"] = strategies

    liquid_days = {}
    for t, d in data.items():
        adv = (d["px"]["close"] * d["px"]["volume"]).rolling(20).mean()
        liquid_days[t] = adv >= MIN_ADV_RAND
    results["ma_cross"] = ma_cross_study({t: d["px"] for t, d in data.items()}, liquid_days)
    return results, panel


def main():
    universe = list(load_universe())
    results, panel = run(universe)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    panel.reset_index().to_csv(os.path.join(OUT_DIR, "panel.csv.gz"), index=False,
                               compression="gzip", float_format="%.6g")
    print("Wrote", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
