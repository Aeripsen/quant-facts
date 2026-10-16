"""Every number the README quotes from a run must appear in the committed
artifact of that run. A doc that drifts from its evidence fails here, the same
way a report that drifts into advice fails test_boundary.py."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

SPY = "spy_10y_2026-07-21.txt"
NOISE = "selection_bias_random_walk.txt"
WF_SPY = "walkforward_spy_10y_2026-07-22.txt"
WF_NOISE = "walkforward_random_walk.txt"

# (text in README.md, artifact file, text that must be in that artifact)
CLAIMS: list[tuple[str, str, str]] = [
    ("train Sharpe of 0.88", SPY, "Sharpe  0.88"),
    ("the haircut Sharpe stays 0.88", SPY,
     "haircut Sharpe 0.88   (0% of it was the search)"),
    ("9-configuration search", SPY,
     "configurations tried on the train segment: 9"),
    ('0.51 ("42% of it was the search")', SPY,
     "haircut Sharpe 0.51   (42% of it was the search)"),
    ("BHY to 0.31", SPY, "haircut Sharpe 0.31"),
    ("(PBO) is 0.404", SPY, "PBO              0.404"),
    ("8 seeded random walks, 25 configurations", NOISE,
     "8 seeded geometric random walks, 1600 bars each, 25 SMA-crossover"),
    ("winning train Sharpe reaches 1.11", NOISE, "1.11"),
    ("deflated Sharpe drops by 0.210 on average", NOISE,
     "min 0.075, mean 0.210"),
    ("3 of 13 boundaries", WF_SPY, "the pick changed at 3 of 13 boundaries"),
    ("26.4 points behind", WF_SPY,
     "refit minus frozen, total return on identical bars and costs: -26.4"),
    ("re-fit Sharpe averages -0.34", WF_NOISE, "mean -0.34"),
]


def _readme() -> str:
    """README text with every whitespace run collapsed, so line wrapping in
    the prose never decides a test."""
    return " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())


@pytest.mark.parametrize("readme_text,artifact,artifact_text", CLAIMS)
def test_readme_number_is_backed_by_its_artifact(readme_text: str,
                                                 artifact: str,
                                                 artifact_text: str) -> None:
    assert readme_text in _readme(), f"README no longer says {readme_text!r}"
    assert artifact_text in (ART / artifact).read_text(encoding="utf-8"), (
        f"{artifact} does not contain {artifact_text!r}")


def test_six_of_seven_positive_noise_winners_haircut_to_zero() -> None:
    """'six of the seven positive winners to 0.00' is a count over the
    table, so count it from the table instead of matching a string."""
    rows = []
    for line in (ART / NOISE).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+\d+\s+\S+\s+(-?\d+\.\d+)\s+(\S+)\s+(\S+)", line)
        if m:
            rows.append((float(m.group(1)), m.group(3)))
    positive = [b25 for sr, b25 in rows if sr > 0]
    assert len(rows) == 8
    assert len(positive) == 7
    assert positive.count("0.00") == 6
    assert "six of the seven positive winners to 0.00" in _readme()


def test_every_artifact_the_readme_names_exists() -> None:
    for name in re.findall(r"artifacts/([\w.-]+\.txt)", _readme()):
        assert (ART / name).is_file(), name
