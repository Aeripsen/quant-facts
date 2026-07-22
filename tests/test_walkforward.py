"""Walk-forward: windows tile the out-of-sample span, picks cannot see
their own or later bars, switch costs are charged, a single-config walk
matches the plain backtest, and the report keeps the money boundary."""
import pytest

import backtest as bt
import walkforward as wf
from compliance import check_output
from tests.test_backtest import make_bars
from tests.test_overfitting import random_walk_closes

GRID = {"fast_grid": (5, 10, 20), "slow_grid": (60, 100)}


def test_windows_tile_the_out_of_sample_span():
    bars = make_bars(random_walk_closes(1600, seed=1))
    r = wf.run(bars, 400, 200, **GRID)
    assert len(r.windows) == 6
    assert r.windows[0].test_start == bars.dates[400].isoformat()
    assert r.windows[-1].test_end == bars.dates[-1].isoformat()
    assert sum(w.n_test_bars for w in r.windows) == 1600 - 400
    for a, b in zip(r.windows, r.windows[1:]):
        assert a.test_end < b.test_start
    assert r.refit.n_bars == 1600 - 400
    assert r.frozen.n_bars == r.refit.n_bars


def test_the_last_window_keeps_the_leftover_bars():
    bars = make_bars(random_walk_closes(1150, seed=2))
    r = wf.run(bars, 400, 300, **GRID)
    assert [w.n_test_bars for w in r.windows] == [300, 300, 150]
    assert "ran out of data at 150 bars" in wf.render_report(r)


def test_future_bars_cannot_change_any_pick():
    """Triple the price over the last stretch of the series; every pick
    trained on bars before the stretch, so every pick must be identical,
    while the stitched path (which trades those bars) must move."""
    closes = random_walk_closes(1200, seed=5)
    plain = wf.run(make_bars(closes), 400, 200, **GRID)
    spiked = closes[:1050] + [c * 3.0 for c in closes[1050:]]
    spiky = wf.run(make_bars(spiked), 400, 200, **GRID)
    assert [(w.fast, w.slow) for w in plain.windows] == \
        [(w.fast, w.slow) for w in spiky.windows]
    assert plain.refit.total_return != spiky.refit.total_return


def test_returns_charge_every_position_change():
    closes = [100.0, 110.0, 99.0, 99.0, 108.9]
    costs = bt.Costs(cost_bps=50.0, slippage_bps=50.0)
    got = wf._returns_for(closes, [0, 1, 1, 0, 1], costs)
    per = costs.per_change
    assert got[0] == pytest.approx(0.10 - per)      # enter on bar 1
    assert got[1] == pytest.approx(-0.10)           # held, no charge
    assert got[2] == pytest.approx(-per)            # exit: flat bar, fee only
    assert got[3] == pytest.approx(0.10 - per)      # re-enter on bar 4


def test_single_config_walk_matches_the_plain_backtest():
    """With one config there is nothing to re-pick: from the first test bar
    on, positions and returns must equal the plain harness run of that rule.
    The only allowed difference is the entry bar, where the walk starts flat
    and may pay a position change the plain run already paid earlier."""
    bars = make_bars(random_walk_closes(1400, seed=3))
    r = wf.run(bars, 400, 250, fast_grid=(20,), slow_grid=(100,))
    plain = bt.run(bars, fast=20, slow=100, split=0.5)
    assert r.trials_per_window == 1
    assert r.n_distinct == 1 and r.n_switches == 0
    for t in range(400, 1400):
        assert r.positions[t] == plain.positions[t - 1]
    for k in range(400, 1399):
        assert r.refit_returns[k] == pytest.approx(plain.strat_returns[k])
    assert r.refit.total_return == pytest.approx(r.frozen.total_return)
    assert r.refit.sharpe == pytest.approx(r.frozen.sharpe)


def test_frozen_holds_the_first_pick():
    bars = make_bars(random_walk_closes(1600, seed=8))
    r = wf.run(bars, 420, 200, **GRID)
    assert (r.frozen_fast, r.frozen_slow) == \
        (r.windows[0].fast, r.windows[0].slow)
    sig = bt.sma_cross_signal(bars.return_series(), r.frozen_fast,
                              r.frozen_slow)
    for t in range(420, 1600):
        assert r.frozen_positions[t] == sig[t - 1]


def test_anchored_trains_from_the_first_bar():
    bars = make_bars(random_walk_closes(1200, seed=4))
    r = wf.run(bars, 400, 200, anchored=True, **GRID)
    assert all(w.train_start == bars.dates[0].isoformat()
               for w in r.windows)
    rolling = wf.run(bars, 400, 200, anchored=False, **GRID)
    assert [w.train_start for w in rolling.windows] == \
        [bars.dates[w_lo - 400].isoformat()
         for w_lo in range(400, 1200, 200)]


def test_impossible_inputs_fail_loudly():
    bars = make_bars(random_walk_closes(900, seed=6))
    with pytest.raises(ValueError, match="too short"):
        wf.run(bars, 100, 200, **GRID)
    with pytest.raises(ValueError, match="at least 2"):
        wf.run(bars, 400, 1, **GRID)
    with pytest.raises(ValueError, match="cannot fit"):
        wf.run(bars, 800, 200, **GRID)
    with pytest.raises(ValueError, match="empty"):
        wf.run(bars, 400, 200, fast_grid=(200,), slow_grid=(100,))


def test_report_prints_the_disclaimer_the_windows_and_the_caveats():
    bars = make_bars(random_walk_closes(1600, seed=7))
    r = wf.run(bars, 400, 200, **GRID)
    text = wf.render_report(r)
    check_output(text)
    assert "HISTORICAL SIMULATION. NOT A PREDICTION. NOT ADVICE." in text
    assert "PICKS, WINDOW BY WINDOW" in text
    assert text.count("..") >= len(r.windows) * 2
    assert "WHAT RE-FITTING DID" in text
    assert "WHAT WALK-FORWARD DOES NOT FIX" in text
    assert "luckiest of 6 trials" in text
    assert "not advice." in text
