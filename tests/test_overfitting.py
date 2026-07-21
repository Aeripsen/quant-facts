"""Selection-bias corrections: haircuts shrink with trials, the deflated
Sharpe of a searched rule falls below the naive one, and PBO separates a
real edge from a grid searched over noise."""
import math
import random
from statistics import mean

import pytest

import backtest as bt
import overfitting as ov
from tests.test_backtest import make_bars


def noise_series(n, mu=0.0, sigma=0.01, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def random_walk_closes(n, seed=0, mu=0.0003, sigma=0.011):
    rng = random.Random(seed)
    price = 100.0
    out = [price]
    for _ in range(n - 1):
        price *= 1.0 + rng.gauss(mu, sigma)
        out.append(price)
    return out


def ann_sharpe(returns):
    m = sum(returns) / len(returns)
    var = sum((x - m) ** 2 for x in returns) / (len(returns) - 1)
    return m / math.sqrt(var) * math.sqrt(ov.TRADING_DAYS)


def test_haircut_grows_with_the_number_of_trials():
    r = noise_series(1000, mu=0.0006, seed=3)
    sr = ann_sharpe(r)
    one = ov.haircut_sharpe(sr, 1000, 1)[0]
    many = ov.haircut_sharpe(sr, 1000, 100)[0]
    assert one.haircut_fraction == pytest.approx(0.0, abs=1e-9)
    assert many.haircut_sharpe < one.haircut_sharpe
    assert many.adjusted_p > one.adjusted_p


def test_bonferroni_multiplies_the_p_value_by_the_trial_count():
    h = ov.haircut_sharpe(1.0, 500, 20)[0]
    single = ov.haircut_sharpe(1.0, 500, 1)[0]
    assert h.adjusted_p == pytest.approx(min(1.0, single.p_value * 20))


def test_haircut_is_empty_for_a_sharpe_at_or_below_zero():
    assert ov.haircut_sharpe(-0.4, 500, 10) == ()
    assert ov.haircut_sharpe(0.0, 500, 10) == ()


def test_expected_max_sharpe_rises_with_trials_and_is_zero_for_one():
    assert ov.expected_max_sharpe(1, 0.01) == 0.0
    assert ov.expected_max_sharpe(2, 0.01) < ov.expected_max_sharpe(200, 0.01)


def test_deflated_sharpe_falls_when_the_same_sharpe_came_from_a_search():
    r = noise_series(1500, mu=0.0007, seed=11)
    sr = ann_sharpe(r)
    skew, kurt = ov.moments(r)
    var = ov.trial_sharpe_variance([sr * f for f in
                                    (1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.3)])
    naive = ov.deflated_sharpe(sr, len(r), skew, kurt, 1, 0.0)
    searched = ov.deflated_sharpe(sr, len(r), skew, kurt, 200, var)
    assert naive > 0.95
    assert searched < naive - 0.1


def test_pbo_is_low_when_one_configuration_has_a_real_edge():
    trials = [noise_series(1000, mu=0.0012, seed=1)]
    trials += [noise_series(1000, mu=0.0, seed=s) for s in range(2, 10)]
    value, logits = ov.pbo(trials, 16)
    assert len(logits) == math.comb(16, 8)
    assert value < 0.2


def test_pbo_separates_an_edge_from_a_field_of_noise():
    """One seed proves nothing: PBO on independent noise sits near 0.5 with a
    wide spread (0.16 to 0.76 over these seeds), so the claim that can be
    pinned is the gap between the two populations, not any single value."""
    edge, noise = [], []
    for s in range(5):
        winners = [noise_series(1000, mu=0.0012, seed=s * 100)]
        winners += [noise_series(1000, seed=s * 100 + j) for j in range(1, 9)]
        edge.append(ov.pbo(winners, 10)[0])
        noise.append(ov.pbo(
            [noise_series(1000, seed=s * 1000 + j) for j in range(8)], 10)[0])
    assert mean(noise) > mean(edge) + 0.15
    assert max(edge) < 0.35


def test_pbo_refuses_impossible_inputs():
    assert ov.pbo([noise_series(500)], 16) == (None, [])
    assert ov.pbo([noise_series(500), noise_series(500, seed=2)], 5)[0] is None
    assert ov.pbo([noise_series(20), noise_series(20, seed=2)], 16)[0] is None


def test_search_over_a_random_walk_lowers_the_deflated_sharpe():
    """The headline claim: same harness, same data, same winning rule, scored
    once as the only thing tried and once as the best of 25. The deflated
    Sharpe falls every time, and the Bonferroni haircut takes the searched
    Sharpe to zero on data that has no edge in it. Numbers behind the
    thresholds are in artifacts/selection_bias_random_walk.txt."""
    grid_fast, grid_slow = (5, 10, 20, 30, 40), (60, 80, 100, 150, 200)
    drops = []
    for seed in (7, 4, 13):
        bars = make_bars(random_walk_closes(1600, seed=seed))
        searched = bt.optimize(bars, fast_grid=grid_fast, slow_grid=grid_slow)
        naive = bt.run(bars, fast=searched.fast, slow=searched.slow)

        assert searched.trials == 25
        assert naive.overfit.n_trials == 1
        # same rule, same bars, same Sharpe: only the trial count differs
        assert searched.in_sample.sharpe == pytest.approx(
            naive.in_sample.sharpe)
        assert (searched.overfit.deflated_sharpe
                < naive.overfit.deflated_sharpe)
        assert naive.overfit.haircuts[0].haircut_fraction < 1e-9
        assert searched.overfit.haircuts[0].haircut_sharpe == 0.0
        assert searched.overfit.pbo is not None
        drops.append(naive.overfit.deflated_sharpe
                     - searched.overfit.deflated_sharpe)
    assert mean(drops) > 0.1


def test_report_prints_the_corrections_and_the_pbo():
    bars = make_bars(random_walk_closes(1600, seed=4))
    r = bt.optimize(bars, fast_grid=(5, 20), slow_grid=(60, 120))
    text = bt.render_report(r)
    assert "SELECTION-BIAS CORRECTIONS" in text
    assert "bonferroni" in text
    assert "deflated Sharpe" in text
    assert "PBO" in text
    assert f"{r.overfit.pbo:.3f}" in text


def test_single_run_reports_one_trial_and_no_pbo():
    r = bt.run(make_bars(random_walk_closes(1200, seed=9)), 20, 50)
    assert r.overfit.n_trials == 1
    assert r.overfit.pbo is None
    text = bt.render_report(r)
    assert "1 trial)" in text
    assert "PBO              n/a" in text
