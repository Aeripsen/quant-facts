# quant-facts - read-only market analysis (facts, never advice)

A small, dependency-light toolkit that fetches public daily price history for a
watchlist, computes the standard technical indicators, lists news headlines as
published, and runs an honest backtest of a simple rule. It informs a decision;
it never makes one.

Extracted from a private personal automation project; this tool was its
self-contained part and stands alone. Two design commitments carry the whole
thing: the money boundary is enforced by code that fails the run rather than
emit advice, and the backtest harness prints its own caveats (look-ahead,
costs, trials, survivorship) instead of hiding them.

## The money boundary (read this first)

This tool does NOT recommend what to trade, does NOT rate or rank tickers as
good or bad, does NOT connect to any brokerage, does NOT execute anything, and
gives NO personalized financial advice. It reports facts: prices that happened,
indicator arithmetic on those prices, headlines third parties published, and
simulations of the past labeled as simulations of the past.

Any what-should-I-do question is a licensed-advisor question and the tool says
exactly that. This is enforced in code, not just promised here:

- `compliance.py` holds a banned-phrase list; every ticker report and the
  advisor line pass through `compliance.guard()` before printing and the run
  dies with `ComplianceError` rather than emit advisory phrasing in its own
  voice. Two fixed strings skip the guard by construction: the `refuse_advice()`
  refusal (which only ever emits the advisor pointer, and whose output is
  checked by `test_what_should_i_do_is_refused`) and the `chart -> <path>` line
  (a filename). The banned-phrase scan in `tests/test_boundary.py` reads every
  source file, so neither string can drift into advice phrasing unnoticed.
- Question-shaped CLI input (`should`, `what`, a `?`, a lowercase trade verb)
  gets `refuse_advice()`: the advisor pointer, nothing else.
- `tests/test_boundary.py` scans every source file, doc file, and rendered
  report for advice phrasing, end to end.
- Quoted third-party headlines are marked with a `> ` prefix and reported
  verbatim as facts about what was published; everything else is tool voice
  and is scanned.

## Honest backtesting, or nothing

A backtest here replays one rule against one past path of one ticker that is
still listed. That sentence contains three warnings, and the harness prints
them with every run. The caveats it enforces (cited in `RESEARCH.md`):

- **No look-ahead.** A signal computed from bar t trades on bar t+1, never on
  the close it just observed.
- **Chronological train/test split.** In-sample and out-of-sample stats are
  separated and labeled; only out-of-sample deserves attention, and even that
  is a single path, not a distribution.
- **Costs charged.** Commission plus slippage in basis points on every
  position change. Zero-cost backtests flatter high-turnover rules.
- **Trials counted, then charged for.** `--optimize` fits parameters on the
  train segment only and prints how many configurations were tried, because
  the odds that the best backtest is an overfit artifact rise with every
  trial (Bailey, Borwein, Lopez de Prado, Zhu 2014). Every report then prints
  what the search cost: the multiple-testing haircut of the Sharpe
  (Bonferroni, Holm, BHY), the deflated Sharpe, and the probability of
  backtest overfitting from combinatorially symmetric cross-validation. On
  SPY over ten years the winning rule's train Sharpe of 0.88 haircuts to 0.51
  once you account for the nine-configuration search that produced it.
- **Survivorship named.** Testing a ticker that still exists already selects
  a winner; the report says so every time.

Historical, not a prediction, not advice. Every backtest block opens with that
disclaimer in a box you cannot miss.

## Usage

```
# install (Python 3.11+)
pip install -r requirements.txt

# facts + indicators for tickers, or a whole watchlist
python analyze.py SPY AAPL
python analyze.py --watchlist watchlist.txt

# add headlines (factual list) and a chart PNG (written to output/)
python analyze.py SPY --news 5 --chart

# honest backtest of an SMA crossover, with costs
python analyze.py SPY --backtest 50,200 --cost-bps 5 --slip-bps 5

# grid-search fast/slow on the train segment only, report the test segment
python analyze.py SPY --optimize --split 0.7

# what the same search does to data with no edge in it (offline, seeded)
python selection_bias_demo.py
```

Ticker format is Yahoo's: `SPY`, `VFV.TO`, `^GSPC`. History ranges: `3mo` to
`max` via `--range`.

## Data sources (free, no keys)

- Prices: the Yahoo Finance public chart API (the same endpoint the yfinance
  library wraps; called directly to stay dependency-light). Fetches are cached
  as CSV in `cache/` (gitignored) so same-day reruns are offline. Stooq's CSV
  endpoint was evaluated first but sits behind a JavaScript anti-bot check as
  of 2026-07-15.
- News: Yahoo Finance per-ticker RSS, with Google News RSS as fallback. Feed
  content is treated as hostile input: parsed with defusedxml, tags stripped,
  control characters removed, length-capped.

## Layout

| File | What it is |
| --- | --- |
| `analyze.py` | CLI and report assembly; reports gated through compliance |
| `data.py` | fetch, parse, CSV cache (offline-testable) |
| `indicators.py` | SMA, EMA, RSI (Wilder), MACD, Bollinger, ATR, in pure Python |
| `news.py` | RSS headlines as facts, sanitized, quoted, unscored |
| `backtest.py` | the honest harness: split, next-bar execution, costs, trials |
| `overfitting.py` | what the search cost: haircut Sharpe, deflated Sharpe, PBO |
| `selection_bias_demo.py` | runs the grid over seeded random walks; writes `artifacts/` |
| `charts.py` | matplotlib PNG, dataviz-method palette, three panels |
| `compliance.py` | the boundary: banned phrases, guard, refusal |
| `RESEARCH.md` | what professionals actually use, cited, traps included |
| `artifacts/` | committed output of real runs, dated, not regenerated silently |
| `tests/` | 50+ pytest tests, all offline and synthetic |

Run checks from the repo root:

```
python -m pytest tests -q
python -m ruff check .
```

The tests are all offline and synthetic; nothing in them touches the network.

## What the indicators are (and are not)

SMA, EMA, RSI, MACD, Bollinger Bands, and ATR are descriptive statistics of
past prices, computed here exactly as published (Wilder smoothing for RSI and
ATR, Appel's 12/26/9 for MACD, population standard deviation for Bollinger).
Provenance and the evidence on their limits are in `RESEARCH.md`. None of them
predicts anything, and the tool never converts them into a verdict; it prints
the numbers and the numbers are yours.

## License

MIT. See `LICENSE`.
