"""Measure what the parameter search costs, on data with nothing in it.

Generates seeded geometric random walks (no drift edge to find), runs the same
25-configuration grid the optimizer runs, and prints the winner's train Sharpe
next to its haircut Sharpe, its deflated Sharpe, and the PBO of the search. The
naive column is the identical winning rule scored as if it had been the only
thing tried, which is the comparison the whole module exists to make.

Offline and deterministic. Writes artifacts/selection_bias_random_walk.txt.

    python selection_bias_demo.py
"""
from __future__ import annotations

import random
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt  # noqa: E402
from data import Bars  # noqa: E402

SEEDS = (7, 4, 9, 13, 21, 33, 42, 55)
FAST_GRID = (5, 10, 20, 30, 40)
SLOW_GRID = (60, 80, 100, 150, 200)
N_BARS = 1600


def random_walk_bars(seed: int, n: int = N_BARS) -> Bars:
    rng = random.Random(seed)
    price = 100.0
    closes = [price]
    for _ in range(n - 1):
        price *= 1.0 + rng.gauss(0.0003, 0.011)
        closes.append(price)
    d0 = date(2018, 1, 1)
    return Bars(
        symbol=f"WALK{seed}", currency="USD",
        dates=[d0 + timedelta(days=i) for i in range(n)],
        open=list(closes), high=[c * 1.01 for c in closes],
        low=[c * 0.99 for c in closes], close=list(closes),
        adjclose=list(closes), volume=[1000.0] * n,
    )


def main() -> int:
    rows = []
    for seed in SEEDS:
        bars = random_walk_bars(seed)
        searched = bt.optimize(bars, fast_grid=FAST_GRID, slow_grid=SLOW_GRID)
        naive = bt.run(bars, fast=searched.fast, slow=searched.slow)
        s, n = searched.overfit, naive.overfit
        rows.append((
            seed, searched.fast, searched.slow, s.sharpe,
            n.haircuts[0].haircut_sharpe if n.haircuts else None,
            s.haircuts[0].haircut_sharpe if s.haircuts else None,
            n.deflated_sharpe, s.deflated_sharpe, s.pbo,
            searched.out_sample.sharpe,
        ))

    out = [
        "selection bias on data with no edge in it",
        f"generated {date.today().isoformat()} by selection_bias_demo.py",
        "",
        f"{len(SEEDS)} seeded geometric random walks, {N_BARS} bars each, "
        f"{len(FAST_GRID) * len(SLOW_GRID)} SMA-crossover configurations "
        "searched on the first 70% of each.",
        "There is no signal in a random walk, so every train Sharpe below is "
        "selection, and the corrected columns are what says so.",
        "",
        f"{'seed':>5} {'rule':>10} {'train SR':>9} {'bonf 1':>7} "
        f"{'bonf 25':>8} {'DSR 1':>7} {'DSR 25':>7} {'PBO':>6} {'test SR':>8}",
    ]
    for (seed, f, sl, sr, hn, hs, dn, ds, p, oos) in rows:
        out.append(
            f"{seed:>5} {f:>4}x{sl:<5} {sr:>9.2f} {_n(hn):>7} {_n(hs):>8} "
            f"{_n(dn, 3):>7} {_n(ds, 3):>7} {_n(p, 3):>6} {oos:>8.2f}"
        )
    drops = [dn - ds for (*_, dn, ds, _p, _o) in rows
             if dn is not None and ds is not None]
    pbos = [p for (*_, p, _o) in rows if p is not None]
    out += [
        "",
        f"deflated Sharpe drop from 1 trial to 25: min {min(drops):.3f}, "
        f"mean {statistics.mean(drops):.3f}, max {max(drops):.3f}",
        f"PBO across the same searches: min {min(pbos):.3f}, "
        f"mean {statistics.mean(pbos):.3f}, max {max(pbos):.3f}",
        "",
        "bonf 1 / bonf 25 are the Bonferroni haircut Sharpe at one trial and "
        "at 25; DSR is the deflated Sharpe (probability the true Sharpe "
        "clears the luckiest-of-N benchmark).",
        "PBO on independent noise sits near 0.5 with a wide spread across "
        "seeds; a single value is not a verdict, which is why the spread is "
        "printed with it.",
    ]
    text = "\n".join(out) + "\n"
    path = Path(__file__).resolve().parent / "artifacts"
    path.mkdir(exist_ok=True)
    (path / "selection_bias_random_walk.txt").write_text(text,
                                                         encoding="utf-8")
    print(text)
    return 0


def _n(x: float | None, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


if __name__ == "__main__":
    raise SystemExit(main())
