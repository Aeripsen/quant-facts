# What professional quant/technical analysis actually uses (cited)

Research pass for the quant tool, 2026-07-15. Purpose: make the tool credible
by building only what practitioners actually compute, and honest by naming
what none of it can do. Sources marked [live] were re-verified by web search
on 2026-07-15; the rest are standard bibliographic records of well-known
published work.

## 1. The standard indicators and where they come from

Professionals compute a small, stable core of descriptive statistics on
price series. The tool implements these exactly as published:

- **Moving averages (SMA/EMA).** Trend smoothing; the 50/200-day pair is the
  common convention behind "golden cross" talk. A moving average is just a
  lagged mean; the tool reports where price sits relative to it as a fact.
- **RSI (Relative Strength Index)** and **ATR (Average True Range)**: both
  introduced by J. Welles Wilder in "New Concepts in Technical Trading
  Systems", Trend Research, 1978, along with ADX and Parabolic SAR. The
  smoothing scheme in the book (first value a simple mean, then
  (prev x (n-1) + current) / n) is what "RSI(14)" means everywhere, and what
  this tool computes. [live]
  https://archive.org/details/newconceptsintec00wild ,
  https://en.wikipedia.org/wiki/J._Welles_Wilder_Jr.
- **MACD**: Gerald Appel's moving average convergence/divergence, the 12/26
  EMA spread with a 9-period signal EMA, developed in the late 1970s.
  https://en.wikipedia.org/wiki/MACD
- **Bollinger Bands**: John Bollinger's 20-period SMA with bands at 2
  standard deviations (population), from his trading work in the 1980s and
  the book "Bollinger on Bollinger Bands" (McGraw-Hill, 2001).
  https://en.wikipedia.org/wiki/Bollinger_Bands

The honest part: these are all functions of past prices. The academic
evidence that they predict returns is weak, mixed, and decays once published;
practitioners use them mostly as risk and regime framing, not as oracles. A
tool that turned RSI 72 into a verdict would be manufacturing certainty that
the underlying math does not contain. This tool prints the number.

## 2. Factor and return basics

The reference frame professionals measure returns against:

- **Fama and French (1993)**, "Common risk factors in the returns on stocks
  and bonds", Journal of Financial Economics 33, 3-56: market, size, and
  value factors explain most diversified-portfolio return variation.
- **Jegadeesh and Titman (1993)**, "Returns to Buying Winners and Selling
  Losers", Journal of Finance 48(1), 65-91: the momentum anomaly.
- **Carhart (1997)**, "On Persistence in Mutual Fund Performance", Journal of
  Finance 52(1), 57-82: the four-factor model; fund alpha mostly disappears
  after factors and costs.
- **Harvey, Liu and Zhu (2016)**, "... and the Cross-Section of Expected
  Returns", Review of Financial Studies 29(1), 5-68: 316+ factors published
  by 2015; under multiple-testing corrections most fail, and the authors
  argue new claims need t-statistics above 3, not 2. [live, via the
  companion Backtesting paper] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489

Takeaway for a personal tool: simple return, volatility, drawdown, and a
factor-aware humility. Single-name returns are mostly market beta plus noise;
nothing in a price chart changes that.

## 3. Honest backtesting methodology

What separates a credible backtest from a sales document, per the people who
studied it:

- **Chronological out-of-sample testing.** Fit or choose anything on a train
  window, evaluate untouched later data, never shuffle time. In-sample fit is
  nearly free to manufacture; out-of-sample performance is the only line with
  information in it, and even that is one path. Harvey and Liu (2015),
  "Backtesting", Journal of Portfolio Management 42(1); SSRN 2345489. [live]
  https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF
- **No look-ahead.** A signal computed from bar t's close cannot be executed
  at bar t's close. Same family of leakage: normalizing with full-sample
  statistics, using restated financials, resampling before the split. Lopez
  de Prado, "Advances in Financial Machine Learning", Wiley 2018, chapters 7
  and 11, treats leakage as the default failure mode of financial ML.
- **Transaction costs and slippage.** Real institutional trading costs,
  measured from live executions rather than assumed, materially shrink paper
  returns, and high-turnover rules suffer most. Frazzini, Israel and
  Moskowitz, "Trading Costs", 2018, SSRN 3229719. A backtest with zero costs
  is a different (and easier) game than markets.
- **Report the number of trials.** The expected maximum Sharpe ratio of N
  random strategies grows with N (on the order of sqrt(2 ln N)), so the best
  of many backtests looks impressive by construction. Bailey, Borwein, Lopez
  de Prado and Zhu, "Pseudo-Mathematics and Financial Charlatanism: The
  Effects of Backtest Overfitting on Out-of-Sample Performance", Notices of
  the AMS 61(5), 2014, 458-471; SSRN 2308659. [live]
  https://www.ams.org/notices/201405/rnoti-p458.pdf ,
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
  Their sharper follow-up metric is the Probability of Backtest Overfitting
  (Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest
  Overfitting", Journal of Computational Finance 20(4), 2017; SSRN 2326253).
- **Haircut reported Sharpe ratios.** Multiple-testing corrections
  (Bonferroni, Holm, BHY) applied to backtested Sharpe ratios; the flat "cut
  it in half" rule is wrong in both directions. Harvey and Liu (2015), above.
  [live]

## 4. The classic traps, named

The failure catalog every credible quant leads with, and this tool prints:

1. **Backtest overfitting / selection bias.** Try enough parameter
   combinations and a random-walk dataset will hand you a beautiful equity
   curve. Unreported trials are how it gets sold (Bailey et al. 2014). The
   harness counts and prints trials, fits on train only.
2. **Survivorship bias.** Datasets of currently-listed tickers exclude the
   ones that died, biasing history upward. Documented for mutual funds by
   Elton, Gruber and Blake (1996), "Survivorship Bias and Mutual Fund
   Performance", Review of Financial Studies 9(4); the same logic applies to
   backtesting any still-listed symbol, which is exactly what a single-ticker
   harness does. The report says so on every run.
3. **Look-ahead leakage.** Any use of information that did not exist at
   decision time. The subtlest versions are procedural (full-sample scalers,
   same-close execution), which is why the harness hard-codes next-bar
   execution rather than trusting the user to remember.
4. **Data snooping across a research community.** Even a single honest
   researcher inherits the bias of everyone who searched the same data
   before them. White (2000), "A Reality Check for Data Snooping",
   Econometrica 68(5), 1097-1126.
5. **Cost blindness.** Paper strategies that trade daily die of fees and
   spread in production (Frazzini et al. 2018). Costs default ON here.

## 5. What this means for the tool

- Compute the published indicator set, exactly as defined, and stop there:
  no composite scores, no signals dressed as insight.
- Backtests: chronological split, next-bar execution, costs on, trials
  printed, out-of-sample labeled, disclaimer boxed at the top. One past path
  of one survivor, and it says so.
- News: headlines are reproduced with source and date, unscored. Deciding
  what news means for money is judgment, and judgment is out of scope by
  design.
- The whole surface is descriptive. The moment output crosses from "what is"
  to "what to do", `compliance.py` kills the run. That is the product: the
  honesty is the credibility.
