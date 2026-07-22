"""Walk-forward evaluation of the SMA-crossover search.

A single chronological split answers one question: how did the one rule
picked on the train segment do on the bars after it. Walk-forward asks the
question the way a rule would actually have been run: pick the best
configuration on a training window, trade that pick on the bars that follow,
roll forward, pick again (Pardo 1992, 2008). Every traded bar is then out of
sample for the pick that traded it, and the sequence of picks becomes data
in its own right: how often the winner changes between windows is parameter
instability that a single split cannot see.

One run produces two paths on identical bars with identical costs:

- REFIT: always trade the latest window's winner, switch costs charged at
  every boundary where the position had to change.
- FROZEN: the first window's winner held to the end, never re-picked. Its
  picking data also precedes every traded bar, so the comparison is fair,
  and what re-fitting did to this path is printed as a fact instead of
  assumed in either direction.

What walk-forward does not fix, and the report says so: each window's pick
is still the luckiest of N trials on its own training window; the window
lengths and the grid are analyst choices, and searching many walk-forward
configurations to quote the best one recreates the selection bias this
harness measures (Bailey, Borwein, Lopez de Prado, Zhu 2014); and the whole
thing is still one path of one surviving ticker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from backtest import (Costs, SegmentStats, _compound, _num, _pct, _segment,
                      _sharpe, sma_cross_signal)
from compliance import BACKTEST_DISCLAIMER
from data import Bars


@dataclass
class WindowRecord:
    """One re-fit: where the pick trained, what it was, how its bars went."""

    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_test_bars: int
    fast: int
    slow: int
    train_sharpe: float | None
    test_return: float
    switched: bool


@dataclass
class WalkForwardResult:
    symbol: str
    train_bars: int
    test_bars: int
    anchored: bool
    costs: Costs
    trials_per_window: int
    windows: list[WindowRecord]
    refit: SegmentStats
    frozen: SegmentStats
    frozen_fast: int
    frozen_slow: int
    n_switches: int
    n_distinct: int
    # exposed for tests and inspection; positions[t] is the position held on
    # bar t (flat before the first test bar), return entry k describes bar k+1
    positions: list[int] = field(default_factory=list)
    refit_returns: list[float] = field(default_factory=list)
    frozen_positions: list[int] = field(default_factory=list)
    frozen_returns: list[float] = field(default_factory=list)


def _returns_for(closes: list[float], positions: list[int],
                 costs: Costs) -> list[float]:
    """Per-bar returns of a finished position path: entry k describes bar
    k+1, matching backtest.run's alignment. Every position change pays
    costs.per_change, which is what charges a re-fit switch the moment the
    incoming rule disagrees with the outgoing one."""
    out = []
    for t in range(1, len(closes)):
        turn = abs(positions[t] - positions[t - 1])
        out.append(positions[t] * (closes[t] / closes[t - 1] - 1.0)
                   - turn * costs.per_change)
    return out


def _standalone_returns(closes: list[float], signal: list[int],
                        costs: Costs) -> list[float]:
    """One config traded alone from bar 1, its own costs only. These series
    are what the train windows are scored on."""
    positions = [0] + [signal[t - 1] for t in range(1, len(closes))]
    return _returns_for(closes, positions, costs)


def _pick(configs: list[tuple[int, int]],
          standalone: dict[tuple[int, int], list[float]],
          k_lo: int, k_hi: int) -> tuple[int, int]:
    """The config with the best train-window Sharpe. Ties fall to the
    earlier grid entry, so a rerun picks the same rule every time."""
    def key(c: tuple[int, int]) -> float:
        s = _sharpe(standalone[c][k_lo:k_hi])
        return -math.inf if s is None else s
    return max(configs, key=key)


def run(bars: Bars, train_bars: int, test_bars: int,
        fast_grid: tuple[int, ...] = (10, 20, 50),
        slow_grid: tuple[int, ...] = (100, 150, 200),
        costs: Costs | None = None, anchored: bool = False
        ) -> WalkForwardResult:
    """Roll the grid search forward: pick on the train window, trade the
    pick on the next test_bars bars, advance, repeat. The last window keeps
    whatever bars remain, and its shorter length is printed rather than
    hidden. anchored=True grows the train window from the first bar instead
    of sliding it."""
    costs = costs or Costs()
    closes = bars.return_series()
    configs = [(f, s) for f in fast_grid for s in slow_grid if f < s]
    if not configs:
        raise ValueError("empty parameter grid")
    slow_max = max(s for _, s in configs)
    if train_bars < slow_max + 10:
        raise ValueError(
            f"train window of {train_bars} bars is too short for "
            f"slow={slow_max}; the slowest average needs room to warm up")
    if test_bars < 2:
        raise ValueError("test step must be at least 2 bars")
    if len(closes) < train_bars + test_bars:
        raise ValueError(
            f"{len(closes)} bars cannot fit a {train_bars}-bar train window "
            f"plus a {test_bars}-bar test step; fetch a longer range")

    signals = {c: sma_cross_signal(closes, *c) for c in configs}
    standalone = {c: _standalone_returns(closes, signals[c], costs)
                  for c in configs}
    market = [closes[t] / closes[t - 1] - 1.0 for t in range(1, len(closes))]

    first = train_bars
    positions = [0] * len(closes)
    picks: list[tuple[int, int]] = []
    spans: list[tuple[int, int]] = []
    lo = first
    while lo < len(closes):
        hi = min(lo + test_bars, len(closes))
        k_lo = 0 if anchored else max(0, lo - train_bars - 1)
        pick = _pick(configs, standalone, k_lo, lo - 1)
        for t in range(lo, hi):
            positions[t] = signals[pick][t - 1]
        picks.append(pick)
        spans.append((lo, hi))
        lo = hi

    strat = _returns_for(closes, positions, costs)

    windows: list[WindowRecord] = []
    for i, ((w_lo, w_hi), pick) in enumerate(zip(spans, picks)):
        k_lo = 0 if anchored else max(0, w_lo - train_bars - 1)
        windows.append(WindowRecord(
            index=i + 1,
            train_start=bars.dates[0 if anchored
                                   else w_lo - train_bars].isoformat(),
            train_end=bars.dates[w_lo - 1].isoformat(),
            test_start=bars.dates[w_lo].isoformat(),
            test_end=bars.dates[w_hi - 1].isoformat(),
            n_test_bars=w_hi - w_lo,
            fast=pick[0], slow=pick[1],
            train_sharpe=_sharpe(standalone[pick][k_lo:w_lo - 1]),
            test_return=_compound(strat[w_lo - 1:w_hi - 1]),
            switched=i > 0 and pick != picks[i - 1],
        ))

    frozen_positions = [0] * len(closes)
    for t in range(first, len(closes)):
        frozen_positions[t] = signals[picks[0]][t - 1]
    frozen_strat = _returns_for(closes, frozen_positions, costs)

    return WalkForwardResult(
        symbol=bars.symbol, train_bars=train_bars, test_bars=test_bars,
        anchored=anchored, costs=costs, trials_per_window=len(configs),
        windows=windows,
        refit=_segment("refit", bars, positions, strat, market,
                       first, len(closes)),
        frozen=_segment("frozen", bars, frozen_positions, frozen_strat,
                        market, first, len(closes)),
        frozen_fast=picks[0][0], frozen_slow=picks[0][1],
        n_switches=sum(1 for w in windows if w.switched),
        n_distinct=len(set(picks)),
        positions=positions, refit_returns=strat,
        frozen_positions=frozen_positions, frozen_returns=frozen_strat,
    )


def render_report(r: WalkForwardResult) -> str:
    """The walk-forward block, disclaimer first, facts after."""
    mode = "anchored" if r.anchored else "rolling"
    lines = [
        BACKTEST_DISCLAIMER,
        "",
        f"WALK-FORWARD  {r.symbol}  rule family: SMA(fast) x SMA(slow) "
        "crossover, long-or-flat, next-bar execution",
        f"costs: {r.costs.cost_bps:.0f} bps commission + "
        f"{r.costs.slippage_bps:.0f} bps slippage per position change,"
        " including the change a re-fit switch forces",
        f"{mode} {r.train_bars}-bar train window, {r.test_bars}-bar test "
        f"step, {r.trials_per_window} configurations re-searched per window",
        "every test bar is out of sample for the pick that traded it: each"
        " window's rule was chosen on bars strictly before that window",
        "",
        "--- PICKS, WINDOW BY WINDOW ---",
        f"  {'w':>2}  {'train span':<24} {'picked':>8} {'train SR':>9}  "
        f"{'test span':<24} {'test ret':>9}",
    ]
    for w in r.windows:
        mark = "  *" if w.switched else ""
        lines.append(
            f"  {w.index:>2}  {w.train_start}..{w.train_end}  "
            f"{w.fast:>3}x{w.slow:<4} {_num(w.train_sharpe):>9}  "
            f"{w.test_start}..{w.test_end}  {_pct(w.test_return):>9}{mark}"
        )
    if r.n_switches:
        lines.append("      * the pick changed entering this window")
    short = r.windows[-1].n_test_bars if r.windows else 0
    if r.windows and short < r.test_bars:
        lines.append(
            f"      the last window ran out of data at {short} bars"
        )
    seg = r.refit
    lines += [
        "",
        f"--- STITCHED OUT-OF-SAMPLE PATH  {seg.start} to {seg.end}  "
        f"({seg.n_bars} bars) ---",
    ]
    for tag, s in (("refit", r.refit), ("frozen", r.frozen)):
        lines += [
            f"  {tag:<7} total {_pct(s.total_return):>9}   "
            f"CAGR {_pct(s.cagr):>8}   vol {_pct(s.ann_vol):>8}   "
            f"Sharpe {_num(s.sharpe):>5} (rf=0)",
            f"          max drawdown {_pct(s.max_drawdown):>8}   "
            f"position changes {s.trades}   "
            f"time in market {s.exposure:.0%}",
        ]
    lines += [
        f"  frozen = the first window's pick ({r.frozen_fast}x"
        f"{r.frozen_slow}) held to the end; its picking data also precedes"
        " every bar above",
        f"  hold    total {_pct(seg.bh_total_return):>9}   "
        f"max drawdown {_pct(seg.bh_max_drawdown):>8}   "
        "(same bars, no rule; a fact for scale, not a benchmark to beat)",
        "",
        "--- WHAT RE-FITTING DID ---",
        f"  the pick changed at {r.n_switches} of "
        f"{max(len(r.windows) - 1, 0)} boundaries; {r.n_distinct} distinct "
        f"rule{'' if r.n_distinct == 1 else 's'} picked across "
        f"{len(r.windows)} windows",
        "  refit minus frozen, total return on identical bars and costs: "
        f"{(r.refit.total_return - r.frozen.total_return) * 100:+.1f} points",
        "  (one path: that difference is a fact about this path, not"
        " evidence re-fitting adds or subtracts anything in general)",
        "",
        "--- WHAT WALK-FORWARD DOES NOT FIX ---",
        f"  each window's pick is still the luckiest of "
        f"{r.trials_per_window} trials on its own train window",
        "  the window lengths and the grid are analyst choices; searching"
        " many walk-forward configurations and quoting the best one"
        " recreates the selection bias this harness measures",
        "  this is still one path of one surviving ticker",
        "",
        "This simulated one past path of one surviving ticker. It is not"
        " a prediction, not an expected return, and not advice.",
    ]
    return "\n".join(lines)
