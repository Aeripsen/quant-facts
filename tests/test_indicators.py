"""Indicator math against hand-computed values. All synthetic data."""
import math

import pytest

import indicators as ind


def test_sma_basic():
    out = ind.sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert out[:2] == [None, None]
    assert out[2:] == [pytest.approx(2.0), pytest.approx(3.0), pytest.approx(4.0)]


def test_sma_warmup_is_none_never_backfilled():
    out = ind.sma(list(map(float, range(300))), 200)
    assert all(v is None for v in out[:199])
    assert all(v is not None for v in out[199:])


def test_ema_seeded_with_sma():
    # n=3 on 1..5: seed at index 2 is 2.0; alpha=0.5 -> 3.0, 4.0
    out = ind.ema([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_rsi_wilder_hand_case():
    # n=2 on closes 1,2,3,2,3 -> changes +1,+1,-1,+1
    # idx2: avg gain 1, loss 0 -> 100
    # idx3: gain (1*1+0)/2=0.5, loss (0*1+1)/2=0.5 -> RS=1 -> 50
    # idx4: gain (0.5*1+1)/2=0.75, loss (0.5*1+0)/2=0.25 -> RS=3 -> 75
    out = ind.rsi([1.0, 2.0, 3.0, 2.0, 3.0], 2)
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(100.0)
    assert out[3] == pytest.approx(50.0)
    assert out[4] == pytest.approx(75.0)


def test_rsi_flat_series_is_neutral():
    out = ind.rsi([5.0] * 10, 3)
    assert out[3] == pytest.approx(50.0)


def test_macd_hand_case():
    # fast=1 (EMA1 = price), slow=2, signal=1 on 1..4:
    # EMA2: seed 1.5 at idx1, then 2.5, 3.5 -> MACD = 0.5 from idx1 on
    line, sig, hist = ind.macd([1.0, 2.0, 3.0, 4.0], fast=1, slow=2, signal=1)
    assert line[0] is None
    assert line[1:] == [pytest.approx(0.5)] * 3
    assert sig[1:] == [pytest.approx(0.5)] * 3
    assert hist[1:] == [pytest.approx(0.0)] * 3


def test_macd_rejects_fast_not_under_slow():
    with pytest.raises(ValueError):
        ind.macd([1.0] * 50, fast=26, slow=12)


def test_bollinger_constant_series_collapses_to_middle():
    mid, up, lo = ind.bollinger([7.0] * 25, 20, 2.0)
    assert mid[-1] == up[-1] == lo[-1] == pytest.approx(7.0)


def test_bollinger_population_std():
    # window 1..5: mean 3, population std sqrt(2)
    mid, up, lo = ind.bollinger([1.0, 2.0, 3.0, 4.0, 5.0], 5, 2.0)
    assert mid[-1] == pytest.approx(3.0)
    assert up[-1] == pytest.approx(3.0 + 2.0 * math.sqrt(2.0))
    assert lo[-1] == pytest.approx(3.0 - 2.0 * math.sqrt(2.0))


def test_true_range_uses_previous_close_gaps():
    # gap up: H-L is 1 but H-prevC is 5 -> TR must be 5
    tr = ind.true_range([10.0, 16.0], [8.0, 15.0], [9.0, 15.5])
    assert tr[0] == pytest.approx(2.0)
    assert tr[1] == pytest.approx(7.0)  # |16 - 9| beats 16-15 and |15-9|


def test_atr_wilder_hand_case():
    # constant TR of 2 -> ATR 2 everywhere it is defined
    high = [10.0, 11.0, 12.0]
    low = [8.0, 9.0, 10.0]
    close = [9.0, 10.0, 11.0]
    out = ind.atr(high, low, close, 2)
    assert out[0] is None
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(2.0)


def test_pct_change_and_drawdown():
    xs = [100.0, 110.0, 99.0, 121.0]
    assert ind.pct_change(xs, 1) == pytest.approx(121.0 / 99.0 - 1.0)
    assert ind.pct_change(xs, 10) is None
    # peak 110 -> trough 99
    assert ind.max_drawdown(xs) == pytest.approx(99.0 / 110.0 - 1.0)


def test_too_short_input_raises_not_guesses():
    with pytest.raises(ValueError):
        ind.sma([1.0, 2.0], 3)
    with pytest.raises(ValueError):
        ind.rsi([1.0, 2.0, 3.0], 14)
