"""Selection-bias corrections for a Sharpe ratio that came out of a search.

A Sharpe ratio picked as the best of N tried configurations is not the same
number as a Sharpe ratio from a single pre-registered rule, and the gap grows
with N. Three published corrections are implemented here, in pure Python:

- Multiple-testing haircut (Harvey and Liu 2015, "Backtesting", JPM 42(1)).
  Turn the Sharpe into a t-statistic, adjust its p-value for the number of
  tests (Bonferroni, Holm, Benjamini-Hochberg-Yekutieli), map the adjusted
  p-value back to a Sharpe. The haircut is the fraction lost.
- Deflated Sharpe ratio (Bailey and Lopez de Prado 2014, "The Deflated
  Sharpe Ratio", Journal of Portfolio Management 40(5)). The probability
  that the true Sharpe is above zero, after subtracting the Sharpe you would
  expect from the luckiest of N trials and after correcting for skew and fat
  tails in the returns.
- Probability of backtest overfitting via combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado, Zhu 2017, "The
  Probability of Backtest Overfitting", J. Computational Finance 20(4)).
  Split the sample into S blocks, take every half-and-half train/test
  partition, and count how often the configuration that won in-sample landed
  below the median out-of-sample.

All three describe a finished search over past data. None of them says a
strategy will work, and a good score here is not evidence of anything except
that this particular selection bias was measured instead of ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329
PBO_SPLITS = 16
_N = NormalDist()


@dataclass
class Haircut:
    """One multiple-testing adjustment of a searched Sharpe ratio."""

    method: str
    n_trials: int
    t_stat: float
    p_value: float
    adjusted_p: float
    haircut_sharpe: float
    haircut_fraction: float


@dataclass
class OverfitStats:
    """Everything the harness knows about how much the search cost."""

    sharpe: float
    n_obs: int
    n_trials: int
    skew: float | None
    kurtosis: float | None
    trial_sharpe_variance: float | None
    expected_max_sharpe: float | None
    deflated_sharpe: float | None
    haircuts: tuple[Haircut, ...]
    pbo: float | None = None
    pbo_splits: int | None = None
    pbo_obs: int | None = None


def moments(returns: list[float]) -> tuple[float | None, float | None]:
    """Sample skew and non-excess kurtosis of a return list."""
    n = len(returns)
    if n < 4:
        return None, None
    m = sum(returns) / n
    m2 = sum((x - m) ** 2 for x in returns) / n
    if m2 <= 0:
        return None, None
    s = math.sqrt(m2)
    skew = sum((x - m) ** 3 for x in returns) / (n * s ** 3)
    kurt = sum((x - m) ** 4 for x in returns) / (n * s ** 4)
    return skew, kurt


def _per_bar(sharpe_ann: float) -> float:
    return sharpe_ann / math.sqrt(TRADING_DAYS)


def trial_sharpe_variance(sharpes_ann: list[float]) -> float:
    """Variance of the tried Sharpes, in the per-bar units the deflation
    formula expects. One trial means no spread to measure, so zero."""
    vals = [_per_bar(s) for s in sharpes_ann if s is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - 1)


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """The Sharpe (per bar) the luckiest of n_trials independent trials
    reaches on merit-free data, given the variance of the trial Sharpes.

    Bailey and Lopez de Prado 2014, equation for E[max SR]: the Gumbel
    approximation to the expected maximum of n_trials draws. With one trial
    there is no selection and the benchmark is zero.
    """
    if n_trials <= 1 or trial_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_variance) * ((1.0 - EULER_GAMMA) * a
                                        + EULER_GAMMA * b)


def deflated_sharpe(sharpe_ann: float, n_obs: int, skew: float | None,
                    kurtosis: float | None, n_trials: int,
                    trial_variance: float) -> float | None:
    """Probability that the true Sharpe exceeds the selection benchmark.

    This is the probabilistic Sharpe ratio evaluated against E[max SR] over
    n_trials rather than against zero, with the non-normality correction from
    the same paper. Returns a probability in [0, 1]; below 0.95 is the usual
    read of "this could be selection luck".
    """
    if n_obs < 4 or skew is None or kurtosis is None:
        return None
    sr = _per_bar(sharpe_ann)
    sr0 = expected_max_sharpe(n_trials, trial_variance)
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return None
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _N.cdf(z)


def _t_to_sharpe(t: float, sharpe_ann: float, t_orig: float) -> float:
    if t_orig <= 0:
        return 0.0
    return sharpe_ann * max(t, 0.0) / t_orig


def _two_sided_p(t: float) -> float:
    return 2.0 * (1.0 - _N.cdf(abs(t)))


def _p_to_t(p: float) -> float:
    p = min(max(p, 1e-15), 1.0)
    return _N.inv_cdf(1.0 - p / 2.0)


def haircut_sharpe(sharpe_ann: float, n_obs: int, n_trials: int,
                   trial_sharpes: list[float] | None = None
                   ) -> tuple[Haircut, ...]:
    """Multiple-testing haircuts of a searched Sharpe (Harvey and Liu 2015).

    Bonferroni needs only the trial count. Holm and BHY need the p-values of
    every trial, so they are computed only when the trial Sharpes are handed
    in; that is the honest version of the procedure, since inventing the
    unobserved tests would be inventing the answer.
    """
    if n_obs < 2 or sharpe_ann <= 0:
        return ()
    t = _per_bar(sharpe_ann) * math.sqrt(n_obs)
    p = _two_sided_p(t)
    n = max(n_trials, 1)

    def build(method: str, adj_p: float) -> Haircut:
        adj_p = min(adj_p, 1.0)
        hs = _t_to_sharpe(_p_to_t(adj_p), sharpe_ann, t)
        return Haircut(method=method, n_trials=n, t_stat=t, p_value=p,
                       adjusted_p=adj_p, haircut_sharpe=hs,
                       haircut_fraction=1.0 - hs / sharpe_ann)

    out = [build("bonferroni", p * n)]
    if trial_sharpes and len(trial_sharpes) > 1:
        ps = sorted(_two_sided_p(_per_bar(s) * math.sqrt(n_obs))
                    for s in trial_sharpes)
        m = len(ps)
        k = min(sum(1 for q in ps if q < p) + 1, m)
        # Holm step-down: the adjusted p-value is monotone in rank order.
        holm = max((m - j) * ps[j] for j in range(k))
        c = sum(1.0 / i for i in range(1, m + 1))
        bhy = min(m * c * ps[j] / (j + 1) for j in range(k - 1, m))
        out.append(build("holm", holm))
        out.append(build("bhy", bhy))
    return tuple(out)


def _block_moments(returns: list[float], edges: list[tuple[int, int]]
                   ) -> list[tuple[int, float, float]]:
    out = []
    for lo, hi in edges:
        seg = returns[lo:hi]
        out.append((len(seg), sum(seg), sum(x * x for x in seg)))
    return out


def _sharpe_from_blocks(blocks: list[tuple[int, float, float]],
                        picked: tuple[int, ...]) -> float:
    n = sum(blocks[b][0] for b in picked)
    if n < 2:
        return float("-inf")
    s = sum(blocks[b][1] for b in picked)
    ss = sum(blocks[b][2] for b in picked)
    mean = s / n
    var = (ss - n * mean * mean) / (n - 1)
    if var <= 0:
        return float("-inf")
    return mean / math.sqrt(var)


def pbo(trial_returns: list[list[float]], n_splits: int = PBO_SPLITS
        ) -> tuple[float | None, list[float]]:
    """Probability of backtest overfitting by combinatorially symmetric
    cross-validation (Bailey et al. 2017).

    Takes one return series per tried configuration, all the same length.
    Whatever window is handed in is the window this measures; CSCV in the
    paper runs over the whole available sample, so the caller has to say
    which bars it passed rather than let the reader assume.
    Splits the sample into n_splits contiguous blocks, and for every way of
    using half the blocks as train and the rest as test, checks where the
    in-sample winner ranked out-of-sample. PBO is the share of partitions
    where it landed at or below the out-of-sample median, which is the rate
    at which picking the best backtest would have picked a below-median rule.

    Returns (pbo, logits). The logits are ln(w / (1 - w)) of the winner's
    relative out-of-sample rank, one per partition, kept so the distribution
    can be inspected rather than trusted.
    """
    if len(trial_returns) < 2 or n_splits < 4 or n_splits % 2:
        return None, []
    length = min(len(r) for r in trial_returns)
    block = length // n_splits
    if block < 2:
        return None, []
    used = block * n_splits
    edges = [(i * block, (i + 1) * block) for i in range(n_splits)]
    per_strategy = [_block_moments(r[:used], edges) for r in trial_returns]

    n_strat = len(trial_returns)
    all_blocks = range(n_splits)
    logits: list[float] = []
    for train in combinations(all_blocks, n_splits // 2):
        test = tuple(b for b in all_blocks if b not in train)
        is_sharpes = [_sharpe_from_blocks(bl, train) for bl in per_strategy]
        best = max(range(n_strat), key=lambda j: is_sharpes[j])
        oos = [_sharpe_from_blocks(bl, test) for bl in per_strategy]
        rank = sum(1 for v in oos if v < oos[best]) + 1
        w = rank / (n_strat + 1.0)
        logits.append(math.log(w / (1.0 - w)))
    if not logits:
        return None, []
    return sum(1 for x in logits if x <= 0) / len(logits), logits
