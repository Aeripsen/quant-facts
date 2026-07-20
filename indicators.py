"""Standard technical indicators, computed as published, in pure Python.

Every function takes plain lists and returns a list aligned to the input,
with None where the indicator is not yet defined (the warmup window). No
value is ever backfilled or extrapolated: if the math needs 200 closes,
the first 199 slots are None, visibly.

Provenance (full citations in RESEARCH.md):
- RSI and ATR use Wilder's original smoothing (Wilder 1978, "New Concepts
  in Technical Trading Systems").
- MACD is Appel's 12/26 EMA spread with a 9-period signal EMA.
- Bollinger Bands are a 20-period SMA with bands at 2 population standard
  deviations (Bollinger 2001).
These are descriptive statistics of past prices. None of them predicts
anything; they summarize what already happened.
"""
from __future__ import annotations

import math

Series = list  # list[float | None], aligned to the input bars


def sma(xs: list[float], n: int) -> Series:
    """Simple moving average over the last n values."""
    _need(xs, n)
    out: Series = [None] * len(xs)
    running = 0.0
    for i, x in enumerate(xs):
        running += x
        if i >= n:
            running -= xs[i - n]
        if i >= n - 1:
            out[i] = running / n
    return out


def ema(xs: list[float], n: int) -> Series:
    """Exponential moving average, seeded with the SMA of the first n
    values (the standard convention), alpha = 2 / (n + 1)."""
    _need(xs, n)
    out: Series = [None] * len(xs)
    alpha = 2.0 / (n + 1.0)
    seed = sum(xs[:n]) / n
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(xs)):
        prev = prev + alpha * (xs[i] - prev)
        out[i] = prev
    return out


def rsi(closes: list[float], n: int = 14) -> Series:
    """Relative Strength Index with Wilder smoothing.

    First average gain/loss is the simple mean of the first n changes;
    each later average is (prev * (n - 1) + current) / n. All-gain windows
    read 100, all-loss windows read 0.
    """
    _need(closes, n + 1)
    out: Series = [None] * len(closes)
    gains, losses = 0.0, 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain, avg_loss = gains / n, losses / n
    out[n] = _rsi_value(avg_gain, avg_loss)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (n - 1) + max(d, 0.0)) / n
        avg_loss = (avg_loss * (n - 1) + max(-d, 0.0)) / n
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def macd(closes: list[float], fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[Series, Series, Series]:
    """MACD line (EMA fast minus EMA slow), signal EMA, histogram."""
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be under slow ({slow})")
    _need(closes, slow + signal)
    ef, es = ema(closes, fast), ema(closes, slow)
    line: Series = [None] * len(closes)
    for i in range(len(closes)):
        if ef[i] is not None and es[i] is not None:
            line[i] = ef[i] - es[i]
    first = next(i for i, v in enumerate(line) if v is not None)
    valid = [v for v in line[first:]]
    sig_tail = ema(valid, signal)
    sig: Series = [None] * len(closes)
    for j, v in enumerate(sig_tail):
        if v is not None:
            sig[first + j] = v
    hist: Series = [
        (line[i] - sig[i]) if line[i] is not None and sig[i] is not None else None
        for i in range(len(closes))
    ]
    return line, sig, hist


def bollinger(closes: list[float], n: int = 20,
              k: float = 2.0) -> tuple[Series, Series, Series]:
    """Middle SMA(n) with bands at k population standard deviations,
    the convention Bollinger himself uses. Returns (mid, upper, lower)."""
    _need(closes, n)
    mid = sma(closes, n)
    upper: Series = [None] * len(closes)
    lower: Series = [None] * len(closes)
    for i in range(n - 1, len(closes)):
        window = closes[i - n + 1: i + 1]
        m = mid[i]
        sd = math.sqrt(sum((x - m) ** 2 for x in window) / n)
        upper[i] = m + k * sd
        lower[i] = m - k * sd
    return mid, upper, lower


def true_range(high: list[float], low: list[float],
               close: list[float]) -> list[float]:
    """TR = max(H-L, |H-prevC|, |L-prevC|); first bar is plain H-L."""
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high/low/close lengths differ")
    tr = [high[0] - low[0]]
    for i in range(1, len(close)):
        pc = close[i - 1]
        tr.append(max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc)))
    return tr


def atr(high: list[float], low: list[float], close: list[float],
        n: int = 14) -> Series:
    """Average True Range, Wilder smoothing: first ATR is the mean of the
    first n true ranges, then (prev * (n - 1) + TR) / n."""
    _need(close, n)
    tr = true_range(high, low, close)
    out: Series = [None] * len(close)
    prev = sum(tr[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(close)):
        prev = (prev * (n - 1) + tr[i]) / n
        out[i] = prev
    return out


def pct_change(xs: list[float], periods: int) -> float | None:
    """Simple return over the trailing window, or None if too short."""
    if len(xs) <= periods or xs[-1 - periods] == 0:
        return None
    return xs[-1] / xs[-1 - periods] - 1.0


def daily_returns(xs: list[float]) -> list[float]:
    return [xs[i] / xs[i - 1] - 1.0 for i in range(1, len(xs)) if xs[i - 1] != 0]


def annualized_vol(returns: list[float], trading_days: int = 252) -> float | None:
    """Sample standard deviation of daily returns, scaled by sqrt(252)."""
    if len(returns) < 2:
        return None
    m = sum(returns) / len(returns)
    var = sum((r - m) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(trading_days)


def max_drawdown(xs: list[float]) -> float:
    """Deepest peak-to-trough decline over the series, as a negative
    fraction (a fact about the past path, nothing more)."""
    peak, worst = xs[0], 0.0
    for x in xs:
        peak = max(peak, x)
        if peak > 0:
            worst = min(worst, x / peak - 1.0)
    return worst


def _need(xs: list, n: int) -> None:
    if n < 1:
        raise ValueError("window must be >= 1")
    if len(xs) < n:
        raise ValueError(f"need at least {n} values, got {len(xs)}")
