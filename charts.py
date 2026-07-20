"""Chart rendering (matplotlib PNG): descriptive pictures of public data.

Colors are assigned by entity and never
cycled: price is ink, the fast average is always blue, the slow average
is always amber, the RSI line is blue, MACD line blue / signal amber,
and the MACD histogram uses the diverging teal/orange pair because it is
a polarity around zero. The palette was checked for color-vision-deficiency
separation; the two lower-contrast hues get relief via direct labels and
legends. One y-axis per panel, never dual-axis.

The pictures show what already happened. The footer says so on every
render.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import indicators as ind  # noqa: E402
from data import Bars  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"    # fast / primary indicator line
AMBER = "#eda100"   # slow / signal line
TEAL = "#1baf7a"    # positive half of a diverging pair
ORANGE = "#eb6834"  # negative half of a diverging pair

FOOT = ("Descriptive indicators computed from public daily price history. "
        "These summarize the past; they are not a forecast and not "
        "investment advice.")
DEFAULT_OUT = Path(__file__).resolve().parent / "output"


def _base_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def _plot_defined(ax, dates, series, **kw):
    """Plot a warmup-padded indicator series, skipping the None head."""
    xs = [d for d, v in zip(dates, series) if v is not None]
    ys = [v for v in series if v is not None]
    if xs:
        ax.plot(xs, ys, **kw)
    return xs, ys


def _end_label(ax, xs, ys, text, color):
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, -3),
                    textcoords="offset points", color=color,
                    fontsize=8, weight="bold")


def render_indicator_chart(bars: Bars, out: Path | None = None,
                           sma_fast: int = 50, sma_slow: int = 200) -> Path:
    """One PNG, three stacked panels sharing the date axis:
    price with SMAs and Bollinger band, RSI(14), MACD(12,26,9)."""
    out = Path(out) if out else DEFAULT_OUT / f"{bars.symbol}_indicators.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    dates, closes = bars.dates, bars.close

    fig, (ax_p, ax_r, ax_m) = plt.subplots(
        3, 1, figsize=(11, 8.2), sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.14},
    )
    for ax in (ax_p, ax_r, ax_m):
        _base_axes(ax)

    # --- panel 1: price, moving averages, Bollinger envelope ---
    if len(closes) >= 20:
        mid, up, lo = ind.bollinger(closes, 20, 2.0)
        xs = [d for d, u in zip(dates, up) if u is not None]
        ax_p.fill_between(
            xs,
            [v for v in lo if v is not None],
            [v for v in up if v is not None],
            color=BLUE, alpha=0.08, linewidth=0,
            label="Bollinger(20, 2)",
        )
    ax_p.plot(dates, closes, color=INK, linewidth=1.4, label="close")
    _end_label(ax_p, dates, closes, f"{closes[-1]:,.2f}", INK2)
    if len(closes) >= sma_fast:
        xs, ys = _plot_defined(ax_p, dates, ind.sma(closes, sma_fast),
                               color=BLUE, linewidth=2,
                               label=f"SMA {sma_fast}")
        _end_label(ax_p, xs, ys, f"SMA{sma_fast}", BLUE)
    if len(closes) >= sma_slow:
        xs, ys = _plot_defined(ax_p, dates, ind.sma(closes, sma_slow),
                               color=AMBER, linewidth=2,
                               label=f"SMA {sma_slow}")
        _end_label(ax_p, xs, ys, f"SMA{sma_slow}", AMBER)
    ax_p.set_title(
        f"{bars.symbol}  daily close with standard indicators "
        f"({dates[0].isoformat()} to {dates[-1].isoformat()}, "
        f"{bars.currency})",
        color=INK, fontsize=13, loc="left", pad=12, weight="bold",
    )
    ax_p.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax_p.legend(loc="upper left", frameon=False, fontsize=8.5,
                labelcolor=INK2)

    # --- panel 2: RSI(14), Wilder ---
    if len(closes) >= 15:
        rsi = ind.rsi(closes, 14)
        _plot_defined(ax_r, dates, rsi, color=BLUE, linewidth=1.6)
        last = next((v for v in reversed(rsi) if v is not None), None)
        if last is not None:
            _end_label(ax_r, [dates[-1]], [last], f"{last:.0f}", BLUE)
    for level in (30, 70):
        ax_r.axhline(level, color=BASELINE, linewidth=1, linestyle="--")
    ax_r.set_ylim(0, 100)
    ax_r.set_yticks([30, 50, 70])
    ax_r.text(0.003, 0.86, "RSI(14), Wilder. 70/30 are conventional "
              "reference lines, not signals.",
              transform=ax_r.transAxes, color=INK2, fontsize=8.5)

    # --- panel 3: MACD(12, 26, 9) ---
    if len(closes) >= 35:
        line, sig, hist = ind.macd(closes, 12, 26, 9)
        pos_x = [d for d, h in zip(dates, hist) if h is not None and h >= 0]
        pos_y = [h for h in hist if h is not None and h >= 0]
        neg_x = [d for d, h in zip(dates, hist) if h is not None and h < 0]
        neg_y = [h for h in hist if h is not None and h < 0]
        ax_m.bar(pos_x, pos_y, width=1.0, color=TEAL, alpha=0.55,
                 linewidth=0)
        ax_m.bar(neg_x, neg_y, width=1.0, color=ORANGE, alpha=0.55,
                 linewidth=0)
        _plot_defined(ax_m, dates, line, color=BLUE, linewidth=1.6,
                      label="MACD")
        _plot_defined(ax_m, dates, sig, color=AMBER, linewidth=1.6,
                      label="signal")
        ax_m.axhline(0, color=BASELINE, linewidth=1)
        ax_m.legend(loc="upper left", frameon=False, fontsize=8.5,
                    labelcolor=INK2)
    ax_m.text(0.003, 1.05, "MACD(12, 26, 9). Histogram = MACD minus "
              "signal; teal above zero, orange below.",
              transform=ax_m.transAxes, color=INK2, fontsize=8.5)

    ax_m.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_m.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.text(0.01, 0.008, FOOT, color=MUTED, fontsize=7.5)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
