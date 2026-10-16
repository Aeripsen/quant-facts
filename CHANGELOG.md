# Changelog

Dated entries, newest first, written from the commit history.

## 2026-10-16 (branch improve-2026-10-16)

- README: a 3-minute section on the honesty harness (the one-line look-ahead fix and the test
  that proves it, trials counted and charged with numbers from `artifacts/`), a Mermaid flow of a
  run, an offline quickstart, and limitations in one place.
- `tests/test_docs_numbers.py`: every run number the README quotes must appear in the committed
  artifact it came from, and the "six of seven" count is recounted from the table.
- `tests/test_boundary.py`: the advice-phrasing scan now reads every root `.md` and `.cff` file
  instead of a fixed pair, so this file and `CITATION.cff` are scanned too.
- Checked on 2026-10-16: both seeded demos rewrite their artifacts identically apart from the
  "generated <date>" line.
- Added `CITATION.cff` and this file.

## 2026-07-25

- Walk-forward negative control: the SPY configuration on merit-free random walks earns nothing
  systematic and churns its picks; run committed.

## 2026-07-23

- CI status badge. Ruff pinned to its classic E4/E7/E9/F rule set so a wider default in a new
  ruff release does not fail the lint step.

## 2026-07-22

- Walk-forward: rolling re-pick, stitched with switch costs, re-fit against a frozen first pick;
  wired into the CLI and the ticker report; SPY run committed.

## 2026-07-21

- Overfitting corrections: multiple-testing haircut (Bonferroni, Holm, BHY), deflated Sharpe,
  probability of backtest overfitting; wired into the report, with tests.
- Random-walk selection-bias demo; SPY and demo runs committed; two counts corrected after a check
  against the committed artifacts; SPY artifact regenerated so a rerun matches byte for byte.

## 2026-07-20

- Initial release: read-only market facts, honest backtest harness, money boundary in code. CI.
