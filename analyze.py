"""quant-facts: read-only market analysis. Facts and standard indicators only.

Usage:
  python analyze.py SPY AAPL [options]
  python analyze.py --watchlist watchlist.txt [options]

Options:
  --range 5y            history window: 3mo 6mo 1y 2y 5y 10y max
  --news N              show N recent headlines per ticker (factual list)
  --chart               render the indicator PNG per ticker (output/)
  --backtest FAST,SLOW  run the honest SMA-crossover backtest
  --optimize            grid-search fast/slow on train data only
  --split 0.7           chronological train/test split fraction
  --walkforward TR,TE   re-pick the rule every TE bars on the trailing TR
  --anchored            walk-forward trains from the first bar, not a window
  --cost-bps 5          commission per position change, basis points
  --slip-bps 5          slippage per position change, basis points
  --refresh             ignore today's cache, refetch
  --out DIR             chart output directory

This tool never says what to do with money. Ask it a what-should-I-do
question and it points you at a licensed advisor, by design. Every ticker
report and the advisor line are gated through compliance.guard before
printing; the two fixed strings that skip it (the refusal pointer and the
chart-path line) are covered by the boundary tests instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt  # noqa: E402
import compliance  # noqa: E402
import indicators as ind  # noqa: E402
import walkforward as wfa  # noqa: E402
from data import PROVIDER, Bars, get_bars  # noqa: E402
from news import Headline, fetch_news, render_news  # noqa: E402

WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def facts_block(bars: Bars, from_cache: bool) -> str:
    """The indicator table: descriptive statistics of the price history."""
    c = bars.close
    r = bars.return_series()
    lines = [
        f"=== {bars.symbol} ({bars.currency})  {len(bars)} daily bars  "
        f"{bars.dates[0].isoformat()} to {bars.dates[-1].isoformat()}",
        f"    source: {PROVIDER}" + ("  [today's cache]" if from_cache else ""),
        f"  last close        {c[-1]:,.2f}",
    ]
    look = min(len(c), 252)
    lines.append(
        f"  52w high / low    {max(c[-look:]):,.2f} / {min(c[-look:]):,.2f}"
    )
    rets = "   ".join(
        f"{name} {_pct(ind.pct_change(r, days))}"
        for name, days in WINDOWS.items()
    )
    lines.append(f"  trailing returns  {rets}   (adjusted close)")
    for n in (20, 50, 200):
        if len(c) >= n:
            v = ind.sma(c, n)[-1]
            rel = "above" if c[-1] > v else "below"
            label = f"SMA({n})"
            lines.append(
                f"  {label:<18}{v:,.2f}   close is {rel} it"
            )
    if len(c) >= 15:
        lines.append(f"  RSI(14) Wilder    {ind.rsi(c, 14)[-1]:.1f}")
    if len(c) >= 35:
        line, sig, hist = ind.macd(c)
        lines.append(
            f"  MACD(12,26,9)     line {line[-1]:+,.3f}   "
            f"signal {sig[-1]:+,.3f}   histogram {hist[-1]:+,.3f}"
        )
    if len(c) >= 20:
        mid, up, lo = ind.bollinger(c, 20, 2.0)
        width = up[-1] - lo[-1]
        pct_b = (c[-1] - lo[-1]) / width if width else 0.0
        lines.append(
            f"  Bollinger(20,2)   lower {lo[-1]:,.2f}   mid {mid[-1]:,.2f}   "
            f"upper {up[-1]:,.2f}   %B {pct_b:.2f}"
        )
    if len(c) >= 15:
        a = ind.atr(bars.high, bars.low, c, 14)[-1]
        lines.append(
            f"  ATR(14) Wilder    {a:,.2f}   ({a / c[-1]:.1%} of close)"
        )
    vol = ind.annualized_vol(ind.daily_returns(r)[-63:])
    if vol is not None:
        lines.append(f"  ann. vol (3m)     {vol:.1%}")
    lines.append(
        f"  max drawdown      {ind.max_drawdown(r):+.1%}   "
        "(deepest peak-to-trough in the window shown)"
    )
    lines.append(
        "  These are descriptive statistics of past prices. None of them"
        " predicts anything."
    )
    return "\n".join(lines)


def build_ticker_report(bars: Bars, from_cache: bool = False,
                        news_items: list[Headline] | None = None,
                        backtest_result: bt.BacktestResult | None = None,
                        walkforward_result: wfa.WalkForwardResult | None = None,
                        ) -> str:
    """Assemble one ticker's full report and gate it through the money
    boundary. Raises ComplianceError rather than ever printing advice."""
    parts = [facts_block(bars, from_cache)]
    if news_items is not None:
        parts.append(render_news(bars.symbol, news_items))
    if backtest_result is not None:
        parts.append(bt.render_report(backtest_result))
    if walkforward_result is not None:
        parts.append(wfa.render_report(walkforward_result))
    return compliance.guard("\n\n".join(parts))


def read_watchlist(path: Path) -> list[str]:
    """The watchlist file: one ticker per line, # comments allowed."""
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        token = raw.split("#", 1)[0].strip()
        if token:
            out.append(token)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="quant/analyze.py",
        description="Read-only market facts and standard indicators. "
                    "Never advice.",
    )
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--watchlist", type=Path)
    ap.add_argument("--range", dest="range_", default="5y")
    ap.add_argument("--news", type=int, default=0, metavar="N")
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--backtest", metavar="FAST,SLOW")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--walkforward", metavar="TRAIN,TEST")
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slip-bps", type=float, default=5.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    questions = [t for t in args.tickers if compliance.looks_like_question(t)]
    if questions:
        print(compliance.refuse_advice(" ".join(args.tickers)))
        return 2

    tickers = list(args.tickers)
    if args.watchlist:
        tickers += [t for t in read_watchlist(args.watchlist)
                    if t not in tickers]
    if not tickers:
        ap.error("give at least one ticker or --watchlist FILE")

    costs = bt.Costs(cost_bps=args.cost_bps, slippage_bps=args.slip_bps)
    failures = 0
    for symbol in tickers:
        try:
            bars, cached = get_bars(symbol, args.range_, refresh=args.refresh)
        except Exception as exc:
            failures += 1
            print(f"=== {symbol}: FETCH FAILED: {exc}", file=sys.stderr)
            continue
        news_items = None
        if args.news:
            try:
                news_items = fetch_news(symbol, limit=args.news)
            except Exception as exc:
                news_items = []
                print(f"    news fetch failed for {symbol}: {exc}",
                      file=sys.stderr)
        result = None
        if args.optimize:
            result = bt.optimize(bars, split=args.split, costs=costs)
        elif args.backtest:
            fast, slow = (int(x) for x in args.backtest.split(","))
            result = bt.run(bars, fast=fast, slow=slow, split=args.split,
                            costs=costs)
        wf_result = None
        if args.walkforward:
            train_n, test_n = (int(x) for x in args.walkforward.split(","))
            wf_result = wfa.run(bars, train_bars=train_n, test_bars=test_n,
                                costs=costs, anchored=args.anchored)
        print(build_ticker_report(bars, cached, news_items, result,
                                  wf_result))
        if args.chart:
            from charts import render_indicator_chart
            out = args.out / f"{bars.symbol}_indicators.png" if args.out else None
            print(f"  chart -> {render_indicator_chart(bars, out)}")
        print()
    print(compliance.guard(compliance.ADVISOR_LINE))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
