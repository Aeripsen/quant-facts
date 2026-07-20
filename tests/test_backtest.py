"""Backtest harness honesty checks: no look-ahead, costs charged,
chronological split, trials counted, disclaimer always rendered."""
from datetime import date, timedelta

import pytest

import backtest as bt
from data import Bars


def make_bars(closes, symbol="FAKE"):
    n = len(closes)
    d0 = date(2024, 1, 1)
    return Bars(
        symbol=symbol, currency="USD",
        dates=[d0 + timedelta(days=i) for i in range(n)],
        open=list(closes), high=[c * 1.01 for c in closes],
        low=[c * 0.99 for c in closes], close=list(closes),
        adjclose=list(closes), volume=[1000.0] * n,
    )


def test_signal_uses_only_past_and_present_closes():
    # flat then a jump: the crossover fires ON the jump bar, so the
    # position (yesterday's signal) must still be flat that bar
    closes = [100.0] * 5 + [110.0, 120.0, 130.0]
    sig = bt.sma_cross_signal(closes, 1, 2)
    assert sig[4] == 0
    assert sig[5] == 1  # fires on the jump bar itself...


def test_no_look_ahead_jump_day_return_not_captured():
    closes = [100.0] * 12 + [150.0, 151.0, 152.0, 153.0, 154.0, 155.0]
    bars = make_bars(closes)
    r = bt.run(bars, fast=1, slow=2, split=0.7,
               costs=bt.Costs(cost_bps=0.0, slippage_bps=0.0))
    # positions[k] belongs to bar k+1; the +50% day is bar index 12
    jump_idx = 12
    assert r.positions[jump_idx - 1] == 0  # ...but is traded one bar later
    assert r.strat_returns[jump_idx - 1] == pytest.approx(0.0)
    assert r.market_returns[jump_idx - 1] == pytest.approx(0.5)
    # the day AFTER the jump the position is on
    assert r.positions[jump_idx] == 1
    assert r.strat_returns[jump_idx] == pytest.approx(
        r.market_returns[jump_idx])


def test_costs_reduce_returns():
    closes = [100.0 + i for i in range(60)]
    bars = make_bars(closes)
    free = bt.run(bars, 2, 5, costs=bt.Costs(0.0, 0.0))
    paid = bt.run(bars, 2, 5, costs=bt.Costs(25.0, 25.0))
    assert free.in_sample.trades >= 1
    assert (paid.in_sample.total_return
            < free.in_sample.total_return)


def test_split_is_chronological_and_disjoint():
    closes = [100.0 * (1.001 ** i) for i in range(200)]
    r = bt.run(make_bars(closes), 5, 20, split=0.7)
    assert r.in_sample.end < r.out_sample.start
    assert r.in_sample.n_bars + r.out_sample.n_bars == len(closes) - 1


def test_too_short_history_raises():
    with pytest.raises(ValueError, match="too short"):
        bt.run(make_bars([100.0] * 150), fast=50, slow=200)


def test_bad_params_raise():
    bars = make_bars([100.0] * 300)
    with pytest.raises(ValueError):
        bt.run(bars, fast=20, slow=10)
    with pytest.raises(ValueError):
        bt.run(bars, fast=5, slow=20, split=0.2)


def test_optimize_fits_on_train_only_and_counts_trials():
    closes = [100.0 * (1.002 ** i) * (1.0 + 0.02 * ((i // 7) % 2))
              for i in range(400)]
    bars = make_bars(closes)
    r = bt.optimize(bars, fast_grid=(2, 5), slow_grid=(10, 20), split=0.7)
    assert r.trials == 4
    assert (r.fast, r.slow) in {(2, 10), (2, 20), (5, 10), (5, 20)}
    text = bt.render_report(r)
    assert "configurations tried on the train segment: 4" in text
    assert "only the test line below is out-of-sample" in text


def test_report_is_labeled_honestly():
    closes = [100.0 * (1.001 ** i) for i in range(300)]
    r = bt.run(make_bars(closes), 5, 20)
    text = bt.render_report(r)
    assert "HISTORICAL SIMULATION. NOT A PREDICTION. NOT ADVICE." in text
    assert "IN-SAMPLE (train)" in text
    assert "OUT-OF-SAMPLE (test)" in text
    assert "survivorship" in text.lower()
    assert "not advice" in text.lower()
    assert "bps" in text
    # buy and hold appears as a fact for scale
    assert "hold" in text.lower()
