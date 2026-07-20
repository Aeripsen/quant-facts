"""Honest backtest harness for a user-supplied SMA-crossover rule.

What "honest" means here, mechanically (citations in RESEARCH.md):

- No look-ahead. The signal computed from bar t's close earns returns
  only from bar t+1 onward (position = yesterday's signal). A rule that
  trades the same close it just observed is the classic leakage bug.
- Chronological train/test split. Bars are split by date, never
  shuffled. In-sample and out-of-sample stats are reported side by side
  and labeled; only the out-of-sample line deserves attention, and even
  it is one path of one asset.
- Costs are charged. Every position change pays commission plus
  slippage (in basis points of traded notional). Zero-cost backtests
  systematically flatter high-turnover rules.
- Trials are counted. When the optimizer grid-searches parameters it
  fits on TRAIN ONLY and reports how many configurations were tried,
  because the probability that the best backtest is an overfit artifact
  rises with every trial (Bailey, Borwein, Lopez de Prado, Zhu 2014).
- Survivorship is named. Testing a still-listed ticker already selects
  a survivor; the harness says so in every report.

Every rendered report opens with compliance.BACKTEST_DISCLAIMER. This is
a historical simulation of the past, not a prediction, and not advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from compliance import BACKTEST_DISCLAIMER
from data import Bars
from indicators import sma

TRADING_DAYS = 252


@dataclass
class Costs:
    """Per-side friction in basis points of traded notional."""

    cost_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def per_change(self) -> float:
        return (self.cost_bps + self.slippage_bps) / 1e4


@dataclass
class SegmentStats:
    """Descriptive statistics of one simulated segment. All of these
    describe the replayed past; none of them estimates the future."""

    label: str
    start: str
    end: str
    n_bars: int
    total_return: float
    cagr: float | None
    ann_vol: float | None
    sharpe: float | None
    max_drawdown: float
    trades: int
    exposure: float
    bh_total_return: float
    bh_max_drawdown: float


@dataclass
class BacktestResult:
    symbol: str
    fast: int
    slow: int
    costs: Costs
    split: float
    in_sample: SegmentStats
    out_sample: SegmentStats
    trials: int = 1
    # exposed for tests and inspection; aligned to bars[1:]
    positions: list[int] = field(default_factory=list)
    strat_returns: list[float] = field(default_factory=list)
    market_returns: list[float] = field(default_factory=list)


def sma_cross_signal(closes: list[float], fast: int, slow: int) -> list[int]:
    """1 while SMA(fast) > SMA(slow), else 0 (long-or-flat, no shorting).
    Undefined warmup bars are flat. Signal at index t uses closes up to
    and including t; it must never be traded on bar t."""
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be under slow ({slow})")
    f, s = sma(closes, fast), sma(closes, slow)
    return [
        1 if (f[i] is not None and s[i] is not None and f[i] > s[i]) else 0
        for i in range(len(closes))
    ]


def run(bars: Bars, fast: int = 50, slow: int = 200, split: float = 0.7,
        costs: Costs | None = None) -> BacktestResult:
    """Simulate the crossover rule with next-bar execution and costs."""
    costs = costs or Costs()
    if not 0.5 <= split <= 0.9:
        raise ValueError("split must be between 0.5 and 0.9")
    series = bars.return_series()
    if len(series) <= slow + 10:
        raise ValueError(
            f"{len(series)} bars is too short for slow={slow}; "
            "fetch a longer range"
        )
    signal = sma_cross_signal(series, fast, slow)

    # Bar t earns position * market return, where the position is the
    # signal from bar t-1. This is the no-look-ahead line.
    positions = [0] + [signal[t - 1] for t in range(1, len(series))]
    market = [series[t] / series[t - 1] - 1.0 for t in range(1, len(series))]
    strat: list[float] = []
    for t in range(1, len(series)):
        turn = abs(positions[t] - positions[t - 1])
        strat.append(positions[t] * market[t - 1] - turn * costs.per_change)

    cut = int(len(series) * split)
    result = BacktestResult(
        symbol=bars.symbol, fast=fast, slow=slow, costs=costs, split=split,
        in_sample=_segment("IN-SAMPLE (train)", bars, positions, strat,
                           market, 1, cut),
        out_sample=_segment("OUT-OF-SAMPLE (test)", bars, positions, strat,
                            market, cut, len(series)),
        positions=positions[1:], strat_returns=strat, market_returns=market,
    )
    return result


def _segment(label: str, bars: Bars, positions: list[int],
             strat: list[float], market: list[float],
             lo: int, hi: int) -> SegmentStats:
    """Stats over bar indexes [lo, hi). Return lists are offset by one
    (entry k describes bar k+1)."""
    seg_strat = strat[lo - 1: hi - 1]
    seg_mkt = market[lo - 1: hi - 1]
    seg_pos = positions[lo:hi]
    n = len(seg_strat)
    total = _compound(seg_strat)
    bh_total = _compound(seg_mkt)
    changes = sum(
        1 for a, b in zip([positions[lo - 1]] + seg_pos, seg_pos) if a != b
    )
    return SegmentStats(
        label=label,
        start=bars.dates[lo].isoformat(),
        end=bars.dates[hi - 1].isoformat(),
        n_bars=n,
        total_return=total,
        cagr=_cagr(total, n),
        ann_vol=_ann_vol(seg_strat),
        sharpe=_sharpe(seg_strat),
        max_drawdown=_equity_drawdown(seg_strat),
        trades=changes,
        exposure=sum(seg_pos) / n if n else 0.0,
        bh_total_return=bh_total,
        bh_max_drawdown=_equity_drawdown(seg_mkt),
    )


def optimize(bars: Bars, fast_grid: tuple[int, ...] = (10, 20, 50),
             slow_grid: tuple[int, ...] = (100, 150, 200),
             split: float = 0.7, costs: Costs | None = None) -> BacktestResult:
    """Grid-search fast/slow ON THE TRAIN SEGMENT ONLY, then report the
    winner's untouched test segment. The trials count is carried into the
    report because an unreported trial count is how overfit backtests get
    sold (Bailey et al. 2014)."""
    best: BacktestResult | None = None
    trials = 0
    for f in fast_grid:
        for s in slow_grid:
            if f >= s:
                continue
            trials += 1
            r = run(bars, fast=f, slow=s, split=split, costs=costs)
            key = r.in_sample.sharpe if r.in_sample.sharpe is not None else -9e9
            best_key = (best.in_sample.sharpe
                        if best and best.in_sample.sharpe is not None else -9e9)
            if best is None or key > best_key:
                best = r
    if best is None:
        raise ValueError("empty parameter grid")
    best.trials = trials
    return best


def render_report(r: BacktestResult) -> str:
    """The full backtest block, disclaimer first, facts after."""
    lines = [
        BACKTEST_DISCLAIMER,
        "",
        f"BACKTEST  {r.symbol}  rule: SMA({r.fast}) x SMA({r.slow}) "
        "crossover, long-or-flat, next-bar execution",
        f"costs: {r.costs.cost_bps:.0f} bps commission + "
        f"{r.costs.slippage_bps:.0f} bps slippage per position change "
        "(estimates; real fills differ)",
        f"configurations tried on the train segment: {r.trials}"
        + ("  (parameters were chosen on train data; only the test line"
           " below is out-of-sample)" if r.trials > 1 else ""),
        f"chronological split at {r.split:.0%}: no shuffling, no test data"
        " touched during any parameter choice",
        "",
    ]
    for seg in (r.in_sample, r.out_sample):
        lines += [
            f"--- {seg.label}  {seg.start} to {seg.end}  "
            f"({seg.n_bars} bars) ---",
            f"  rule    total {_pct(seg.total_return):>9}   "
            f"CAGR {_pct(seg.cagr):>8}   vol {_pct(seg.ann_vol):>8}   "
            f"Sharpe {_num(seg.sharpe):>5} (rf=0)",
            f"          max drawdown {_pct(seg.max_drawdown):>8}   "
            f"position changes {seg.trades}   "
            f"time in market {seg.exposure:.0%}",
            f"  hold    total {_pct(seg.bh_total_return):>9}   "
            f"max drawdown {_pct(seg.bh_max_drawdown):>8}   "
            "(same bars, no rule; a fact for scale, not a benchmark to"
            " beat)",
        ]
    lines += [
        "",
        "This simulated one past path of one surviving ticker. It is not"
        " a prediction, not an expected return, and not advice.",
    ]
    return "\n".join(lines)


# --- small stat helpers (facts about a finished return list) ---

def _compound(returns: list[float]) -> float:
    eq = 1.0
    for x in returns:
        eq *= 1.0 + x
    return eq - 1.0


def _cagr(total_return: float, n_bars: int) -> float | None:
    years = n_bars / TRADING_DAYS
    if years <= 0 or total_return <= -1.0:
        return None
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _ann_vol(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    m = sum(returns) / len(returns)
    var = sum((x - m) ** 2 for x in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def _sharpe(returns: list[float]) -> float | None:
    """Annualized mean over stdev at zero risk-free rate. Reported to two
    decimals and taken with salt: on one path it is a description, not a
    quality score (Harvey and Liu 2015 on haircutting these)."""
    vol = _ann_vol(returns)
    if not vol:
        return None
    mean_ann = (sum(returns) / len(returns)) * TRADING_DAYS
    return mean_ann / vol


def _equity_drawdown(returns: list[float]) -> float:
    eq, peak, worst = 1.0, 1.0, 0.0
    for x in returns:
        eq *= 1.0 + x
        peak = max(peak, eq)
        worst = min(worst, eq / peak - 1.0)
    return worst


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"
