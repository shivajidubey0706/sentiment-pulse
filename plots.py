"""Small set of report-ready charts (matplotlib)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.6,
    "legend.frameon": False,
})


def price_and_score(f: pd.DataFrame, coin: str):
    """Two stacked panels (shared x, separate y): price on a log scale, then the daily score."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(f.index, f["close"], color=BLUE)
    a1.set_yscale("log"); a1.set_ylabel("Close, USD (log)")
    a1.set_title(f"{coin}: price and daily sentiment score")
    s = f["score"].rolling(7).mean()
    a2.axhline(0, color=INK2, lw=0.8)
    a2.fill_between(f.index, s, 0, where=s >= 0, color=AQUA, alpha=0.5, lw=0)
    a2.fill_between(f.index, s, 0, where=s < 0, color=ORANGE, alpha=0.5, lw=0)
    a2.set_ylabel("Score, z (7-day avg)")
    fig.tight_layout()
    return fig


def lead_lag(f: pd.DataFrame, coin: str, feature="score", max_lag=7):
    """Correlation of today's score with returns k days earlier (negative k) and later (positive k).
    Bars on the right side = sentiment leading price; left side = sentiment following price."""
    lags = range(-max_lag, max_lag + 1)
    corr = [f[feature].corr(f["ret_1"].shift(-k)) for k in lags]
    n = f[[feature, "ret_1"]].dropna().shape[0]
    band = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [ORANGE if k < 0 else BLUE for k in lags]
    ax.bar(list(lags), corr, color=colors, width=0.7, zorder=2)
    ax.axhspan(-band, band, color=GRID, alpha=0.6, lw=0, label="95% noise band", zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("k (days): left = past returns, right = future returns")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{coin}: does sentiment lead or follow returns?")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def decay(reg: pd.DataFrame, coin: str):
    """t-statistic of the score coefficient by forecast horizon."""
    r = reg[reg["feature"] == "score"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(r["h"].astype(str), r["t_stat"], color=BLUE, width=0.55)
    for y in (1.96, -1.96):
        ax.axhline(y, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Horizon (days ahead)"); ax.set_ylabel("t-stat of score")
    ax.set_title(f"{coin}: signal strength by horizon")
    fig.tight_layout()
    return fig


def regimes(rt: pd.DataFrame, coin: str):
    """Score coefficient (bp per 1 z) inside each regime, with 95% intervals."""
    rt = rt.copy()
    rt["label"] = rt["regime"] + ": " + rt["state"].astype(str)
    se = (rt["coef_bp"] / rt["t_stat"]).abs()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rt) + 1))
    y = np.arange(len(rt))[::-1]
    ax.errorbar(rt["coef_bp"], y, xerr=1.96 * se, fmt="o", color=BLUE,
                ecolor=INK2, elinewidth=1, capsize=3, markersize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, rt["label"])
    ax.set_xlabel("Next-day return per +1 z of score (basis points)")
    ax.set_title(f"{coin}: where does sentiment matter?")
    fig.tight_layout()
    return fig


def rolling(rc: pd.Series, coin: str):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rc.index, rc.values, color=BLUE)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("bp per +1 z")
    ax.set_title(f"{coin}: 180-day rolling score coefficient (next-day return)")
    fig.tight_layout()
    return fig


def cumulative_strategy(fc: pd.DataFrame, coin: str,
                        models=("Ridge (controls only)", "Ridge (+ sentiment)")):
    """Growth of 1 unit following the sign of each forecast, next-day horizon, gross of costs."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(fc.index, np.exp(fc["actual"].cumsum()), color=INK2, lw=1.2, label="Buy and hold")
    for m, c in zip(models, (YELLOW, BLUE)):
        ax.plot(fc.index, np.exp((np.sign(fc[m]) * fc["actual"]).cumsum()), color=c, label=m)
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log)")
    ax.set_title(f"{coin}: out-of-sample long/short by forecast sign (no trading costs)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
