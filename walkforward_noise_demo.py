"""Walk-forward on data with no edge in it.

selection_bias_demo.py shows what an in-sample search manufactures on merit-free
data and what the corrections charge for it. This demo asks the follow-up
question: does walking the same search forward rescue anything? It runs the
exact walk-forward configuration of the committed SPY artifact (rolling 756-bar
train window, 126-bar test step, the 9-configuration SMA grid, default costs)
over the same seeded geometric random walks, where by construction there is
nothing to find.

Two facts come out, and the artifact prints both per seed:

- The stitched out-of-sample path earns nothing systematic. Every traded bar is
  out of sample for the pick that traded it, so on merit-free data the refit
  line has no selection bias left to flatter it; what remains is drift minus
  costs, centered near zero, spread wide across seeds.
- The picks churn. Window winners are luck, and luck does not persist, so the
  re-fits keep replacing one lucky rule with another. The switch counts sit next
  to the SPY run's for scale.

Walk-forward is an honest scorekeeper, not an edge factory: it stops in-sample
luck from reaching the headline, and it cannot conjure a return that was never
in the data. That is the whole point of running it here.

Offline and deterministic. Writes artifacts/walkforward_random_walk.txt.

    python walkforward_noise_demo.py
"""
from __future__ import annotations

import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import walkforward as wf  # noqa: E402
from selection_bias_demo import SEEDS, random_walk_bars  # noqa: E402

N_BARS = 1600
TRAIN, TEST = 756, 126  # the committed SPY artifact's configuration


def main() -> int:
    rows = []
    for seed in SEEDS:
        bars = random_walk_bars(seed, N_BARS)
        r = wf.run(bars, train_bars=TRAIN, test_bars=TEST)
        rows.append((
            seed, len(r.windows), r.n_distinct, r.n_switches,
            r.refit.total_return, r.refit.sharpe,
            r.frozen.total_return, r.refit.bh_total_return,
        ))

    out = [
        "walk-forward on data with no edge in it",
        f"generated {date.today().isoformat()} by walkforward_noise_demo.py",
        "",
        f"{len(SEEDS)} seeded geometric random walks, {N_BARS} bars each, the "
        f"SPY artifact's exact configuration: rolling {TRAIN}-bar train, "
        f"{TEST}-bar step, 9 SMA configurations re-searched per window, "
        "default costs.",
        "Every traded bar is out of sample for the pick that traded it, and "
        "there is nothing to find by construction, so the refit column is what "
        "walk-forward earns on pure luck: drift minus costs, not edge.",
        "",
        f"{'seed':>5} {'win':>4} {'rules':>6} {'switch':>7} {'refit':>9} "
        f"{'refit SR':>9} {'frozen':>9} {'hold':>9}",
    ]
    for (seed, n_win, n_rules, n_sw, ref, sr, fro, hold) in rows:
        out.append(
            f"{seed:>5} {n_win:>4} {n_rules:>6} {n_sw:>7} {ref:>+9.1%} "
            f"{_n(sr):>9} {fro:>+9.1%} {hold:>+9.1%}"
        )
    srs = [r[5] for r in rows if r[5] is not None]
    switches = [r[3] for r in rows]
    out += [
        "",
        f"refit Sharpe across seeds: min {min(srs):+.2f}, "
        f"mean {statistics.mean(srs):+.2f}, max {max(srs):+.2f} "
        "(centered near zero; single seeds swing wide, which is the point of "
        "printing all of them)",
        f"picks changed at {min(switches)} to {max(switches)} of "
        f"{rows[0][1] - 1} boundaries per walk (SPY, same configuration: 3 of "
        "13); winners on noise do not persist, so re-fitting keeps replacing "
        "them",
        "",
        "Read next to artifacts/selection_bias_random_walk.txt: the in-sample "
        "search there looks brilliant on the same walks until the corrections "
        "charge for it; the walk-forward line here never looked brilliant, "
        "because every reported bar was out of sample. Same data, same grid, "
        "honest scoring on both sides.",
    ]
    text = "\n".join(out) + "\n"
    path = Path(__file__).resolve().parent / "artifacts"
    path.mkdir(exist_ok=True)
    (path / "walkforward_random_walk.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _n(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
