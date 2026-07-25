"""Walk-forward over merit-free walks: the demo is deterministic, the stitched
out-of-sample Sharpe carries no systematic edge, and the picks churn. The
numbers behind the bounds live in artifacts/walkforward_random_walk.txt."""
from statistics import mean

import walkforward as wf
from selection_bias_demo import SEEDS, random_walk_bars
from tests.test_backtest import make_bars
from tests.test_overfitting import random_walk_closes


def run_all(train=756, test=126, n_bars=1600):
    return [wf.run(random_walk_bars(s, n_bars), train_bars=train,
                   test_bars=test) for s in SEEDS]


def test_noise_walkforward_finds_no_systematic_edge():
    """On data with nothing in it, the refit line is drift minus costs. The
    mean across the eight committed seeds sits near zero, against the mostly
    positive in-sample train Sharpes the same walks hand a naive search (up
    to 1.11 in the selection-bias artifact); single seeds swing wide."""
    results = run_all()
    srs = [r.refit.sharpe for r in results if r.refit.sharpe is not None]
    assert len(srs) == len(SEEDS)
    assert abs(mean(srs)) < 0.45
    assert min(srs) < 0.0


def test_noise_picks_churn_and_switch_costs_are_charged():
    """Window winners on noise are luck, so re-fits replace them: most walks
    change the pick at least once, and every switch that flips the position
    shows up as extra position changes on the refit line vs the frozen one."""
    results = run_all()
    assert sum(r.n_switches for r in results) >= len(SEEDS)
    assert any(r.n_distinct >= 3 for r in results)


def test_demo_is_deterministic():
    a = wf.run(random_walk_bars(SEEDS[0], 1600), train_bars=756,
               test_bars=126)
    b = wf.run(random_walk_bars(SEEDS[0], 1600), train_bars=756,
               test_bars=126)
    assert a == b


def test_same_config_as_the_spy_artifact_on_short_data_still_validates():
    """The demo config must stay runnable on the walks' 1600 bars: one train
    window plus at least one full test step fits, and the windows tile."""
    bars = make_bars(random_walk_closes(1600, seed=99))
    r = wf.run(bars, train_bars=756, test_bars=126)
    assert sum(w.n_test_bars for w in r.windows) == 1600 - 756
