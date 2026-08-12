"""Standard backtest charts, in matplotlib.

Every function returns the matplotlib ``Figure`` so you can keep customising or
save it. None of them uses a secondary Y axis: when two different magnitudes
are involved they go into stacked panels sharing the X axis.

Light and dark themes, both with colourblind-validated palettes
(``set_theme("dark")``).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure

# ---------------------------------------------------------------------- theme
THEMES: Dict[str, dict] = {
    "light": {
        "surface": "#FCFCFB",
        "panel": "#FFFFFF",
        "ink": "#1C1B1A",
        "ink_soft": "#57534E",
        "ink_muted": "#8A857F",
        "grid": "#E7E5E1",
        # Fixed categorical order (never cycled, never reassigned by rank)
        "series": ["#3B6FE0", "#C2410C", "#0F8F7A", "#9333EA", "#B45309"],
        "positive": "#0F8F7A",
        "negative": "#C0392B",
        "neutral": "#8A857F",
        "band": "#3B6FE0",
    },
    "dark": {
        "surface": "#1A1A19",
        "panel": "#232322",
        "ink": "#F2F0ED",
        "ink_soft": "#B8B3AD",
        "ink_muted": "#88837D",
        "grid": "#343432",
        "series": ["#4F83E8", "#CF6A21", "#0F9C80", "#9C5FD8", "#B07E1F"],
        "positive": "#0F9C80",
        "negative": "#D45B4E",
        "neutral": "#88837D",
        "band": "#4F83E8",
    },
}

_ACTIVE = "light"


def set_theme(mode: str = "light") -> dict:
    """Activate a theme (``"light"`` or ``"dark"``) and set the rcParams."""
    global _ACTIVE
    if mode not in THEMES:
        raise ValueError(f"Unknown theme: {mode}")
    _ACTIVE = mode
    t = THEMES[mode]
    mpl.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["ink_soft"],
            "axes.titlecolor": t["ink"],
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "text.color": t["ink"],
            "xtick.color": t["ink_muted"],
            "ytick.color": t["ink_muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 110,
            "font.size": 9,
        }
    )
    for spine in ("top", "right"):
        mpl.rcParams[f"axes.spines.{spine}"] = False
    return t


def theme() -> dict:
    """The active theme's colours."""
    return THEMES[_ACTIVE]


def _diverging_cmap() -> LinearSegmentedColormap:
    t = theme()
    return LinearSegmentedColormap.from_list(
        "pl", [t["negative"], t["surface"], t["positive"]], N=256
    )


def _sequential_cmap() -> LinearSegmentedColormap:
    t = theme()
    return LinearSegmentedColormap.from_list(
        "seq", [t["surface"], t["series"][0], "#0B2B6B"], N=256
    )


def _fmt_money(ax, currency: str = "") -> None:
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}{(' ' + currency) if currency else ''}")
    )


def _dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _finish(fig: Figure, title: Optional[str] = None, subtitle: Optional[str] = None) -> Figure:
    t = theme()
    if title:
        fig.suptitle(title, color=t["ink"], fontsize=13, fontweight="600",
                     x=0.01, ha="left", y=0.985)
    if subtitle:
        fig.text(0.01, 0.947, subtitle, color=t["ink_muted"], fontsize=9, ha="left")
    with warnings.catch_warnings():   # colorbars are not tight_layout friendly
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1.0))
    return fig


# ------------------------------------------------------------------- EQUITY
def plot_equity(result, show_balance: bool = True, show_drawdown: bool = True,
                log: bool = False, figsize: Tuple[float, float] = (11, 6),
                title: Optional[str] = None) -> Figure:
    """Equity (and balance) curve with drawdown in a lower panel."""
    t = theme()
    eq = result.equity_curve
    dd = result.drawdown

    if show_drawdown:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax2 = None

    ax.plot(eq.index, eq.values, color=t["series"][0], label="Equity", zorder=3)
    if show_balance and "balance" in result.equity:
        ax.plot(result.equity.index, result.equity["balance"].values,
                color=t["series"][1], linewidth=1.3, alpha=0.9, label="Balance", zorder=2)
    ax.fill_between(eq.index, eq.values, eq.min(), color=t["series"][0], alpha=0.06, zorder=1)

    # Directly label the deepest drawdown
    if len(dd) and dd["dd_abs"].min() < 0:
        worst = dd["dd_abs"].idxmin()
        ax.scatter([worst], [eq.loc[worst]], s=42, color=t["negative"], zorder=4,
                   edgecolor=t["surface"], linewidth=2)
        ax.annotate(
            f"Max DD {result.stats['max_dd_pct']:.1f}%",
            xy=(worst, eq.loc[worst]), xytext=(8, -14), textcoords="offset points",
            color=t["ink_soft"], fontsize=8.5,
        )

    ax.axhline(result.stats["initial_balance"], color=t["ink_muted"], linewidth=1.0,
               linestyle=(0, (4, 4)), alpha=0.7)
    ax.set_ylabel("Equity")
    if log:
        ax.set_yscale("log")
    _fmt_money(ax)
    ax.legend(loc="best", ncols=2)

    if ax2 is not None:
        ax2.fill_between(dd.index, dd["dd_pct"].values, 0, color=t["negative"], alpha=0.28)
        ax2.plot(dd.index, dd["dd_pct"].values, color=t["negative"], linewidth=1.2)
        ax2.set_ylabel("Drawdown %")
        ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        _dates(ax2)
    else:
        _dates(ax)

    subtitle = (
        f"{result.spec.symbol} · net {result.stats['net_profit']:,.2f} "
        f"({result.stats['return_pct']:.1f}%) · PF {result.stats['profit_factor']:.2f} "
        f"· {result.stats['trades']} trades"
    )
    return _finish(fig, title or "Equity curve", subtitle)


def plot_drawdown(result, figsize: Tuple[float, float] = (11, 4)) -> Figure:
    """Underwater chart: how deep and how long below the running peak."""
    t = theme()
    dd = result.drawdown
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dd.index, dd["dd_pct"].values, 0, color=t["negative"], alpha=0.3)
    ax.plot(dd.index, dd["dd_pct"].values, color=t["negative"], linewidth=1.4)
    ax.axhline(0, color=t["ink_muted"], linewidth=1)
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _dates(ax)
    return _finish(
        fig, "Drawdown",
        f"max {result.stats['max_dd_pct']:.2f}% "
        f"({result.stats['max_dd_abs']:,.2f}) · longest {result.stats.get('max_dd_duration')}",
    )


# --------------------------------------------------------------- PRICE/TRADES
def plot_price_trades(result, max_bars: int = 4000, show_sl_tp: bool = False,
                      figsize: Tuple[float, float] = (12, 6),
                      start=None, end=None) -> Figure:
    """Price with entries and exits marked, coloured by outcome."""
    t = theme()
    data = result.data
    trades = result.trades
    if start is not None or end is not None:
        data = data.loc[pd.Timestamp(start or data.index[0]): pd.Timestamp(end or data.index[-1])]
        if not trades.empty:
            trades = trades[(trades["open_time"] >= data.index[0]) &
                            (trades["open_time"] <= data.index[-1])]
    if len(data) > max_bars:
        data = data.iloc[::int(np.ceil(len(data) / max_bars))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data["close"].values, color=t["ink_muted"], linewidth=1.0,
            label="Price (close)", zorder=1)

    if not trades.empty:
        wins = trades[trades["net_profit"] > 0]
        losses = trades[trades["net_profit"] <= 0]
        for frame, color, label in (
            (wins, t["positive"], "Winners"),
            (losses, t["negative"], "Losers"),
        ):
            if frame.empty:
                continue
            for _, tr in frame.iterrows():
                ax.plot([tr["open_time"], tr["close_time"]],
                        [tr["open_price"], tr["close_price"]],
                        color=color, linewidth=1.4, alpha=0.75, zorder=2)
            longs = frame[frame["type"] == "buy"]
            shorts = frame[frame["type"] == "sell"]
            ax.scatter(longs["open_time"], longs["open_price"], marker="^", s=46,
                       color=color, edgecolor=t["surface"], linewidth=1.2, zorder=3,
                       label=f"{label} (long)" if len(longs) else None)
            ax.scatter(shorts["open_time"], shorts["open_price"], marker="v", s=46,
                       color=color, edgecolor=t["surface"], linewidth=1.2, zorder=3,
                       label=f"{label} (short)" if len(shorts) else None)

        if show_sl_tp:
            for _, tr in trades.iterrows():
                for level, style in ((tr.get("sl"), (0, (2, 2))), (tr.get("tp"), (0, (1, 2)))):
                    if level and level == level:
                        ax.plot([tr["open_time"], tr["close_time"]], [level, level],
                                color=t["ink_muted"], linewidth=0.8, linestyle=style, alpha=0.6)

    ax.set_ylabel(f"Price ({result.spec.profit_currency})")
    _dates(ax)
    ax.legend(loc="best", ncols=2, fontsize=8)
    return _finish(fig, f"Trades on {result.spec.symbol}", f"{len(trades)} trades shown")


# --------------------------------------------------------------- DISTRIBUTION
def plot_trade_distribution(result, bins: int = 40,
                            figsize: Tuple[float, float] = (11, 7)) -> Figure:
    """P&L histogram, cumulative P&L, holding time and MAE/MFE."""
    t = theme()
    trades = result.trades
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax1, ax2), (ax3, ax4) = axes

    if trades.empty:
        for ax in (ax1, ax2, ax3, ax4):
            ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                    color=t["ink_muted"], transform=ax.transAxes)
        return _finish(fig, "Trade distribution")

    pnl = trades["net_profit"].astype(float)

    # 1) Outcome histogram, winners and losers apart
    edges = np.histogram_bin_edges(pnl, bins=bins)
    ax1.hist(pnl[pnl > 0], bins=edges, color=t["positive"], alpha=0.85, label="Winners")
    ax1.hist(pnl[pnl <= 0], bins=edges, color=t["negative"], alpha=0.85, label="Losers")
    ax1.axvline(0, color=t["ink_muted"], linewidth=1)
    ax1.axvline(pnl.mean(), color=t["series"][0], linewidth=1.6, linestyle=(0, (4, 3)))
    ax1.annotate(f"mean {pnl.mean():,.1f}", xy=(pnl.mean(), ax1.get_ylim()[1] * 0.9),
                 xytext=(6, 0), textcoords="offset points", color=t["ink_soft"], fontsize=8)
    ax1.set_title("Result per trade")
    ax1.set_xlabel("Net profit")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # 2) Cumulative P&L by trade number
    cum = pnl.cumsum()
    ax2.plot(range(1, len(cum) + 1), cum.values, color=t["series"][0])
    ax2.fill_between(range(1, len(cum) + 1), cum.values, 0, color=t["series"][0], alpha=0.08)
    ax2.axhline(0, color=t["ink_muted"], linewidth=1)
    ax2.set_title("Cumulative P&L by trade")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Cumulative profit")

    # 3) Holding time
    hours = trades["duration_s"].astype(float) / 3600.0
    ax3.hist(hours, bins=min(bins, 30), color=t["series"][2], alpha=0.9)
    ax3.set_title("Holding time")
    ax3.set_xlabel("Hours")
    ax3.set_ylabel("Frequency")

    # 4) MAE vs MFE
    if {"mae", "mfe"}.issubset(trades.columns):
        colors = np.where(pnl > 0, t["positive"], t["negative"])
        ax4.scatter(trades["mae"].abs(), trades["mfe"], c=colors, s=26, alpha=0.75,
                    edgecolor=t["surface"], linewidth=0.6)
        lim = max(trades["mae"].abs().max(), trades["mfe"].max(), 1e-9)
        ax4.plot([0, lim], [0, lim], color=t["ink_muted"], linewidth=1, linestyle=(0, (4, 4)))
        ax4.set_title("MAE against MFE")
        ax4.set_xlabel("Adverse excursion (|MAE|)")
        ax4.set_ylabel("Favourable excursion (MFE)")

    return _finish(fig, "Trade distribution",
                   f"{len(trades)} trades · expectancy {pnl.mean():,.2f} per trade")


# ------------------------------------------------------------------- MONTHLY
def plot_monthly_heatmap(result, figsize: Tuple[float, float] = (11, 4.5)) -> Figure:
    """Year x month heatmap of returns (%), diverging palette."""
    t = theme()
    table = result.monthly_table()
    fig, ax = plt.subplots(figsize=figsize)
    if table.empty:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                color=t["ink_muted"], transform=ax.transAxes)
        return _finish(fig, "Monthly returns")

    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(abs(finite).max(), 1e-6) if finite.size else 1.0
    ax.imshow(values, cmap=_diverging_cmap(),
              norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")

    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                    color=t["ink"] if abs(value) < limit * 0.55 else t["surface"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return _finish(fig, "Monthly returns (%)",
                   "green = positive month · red = negative · last column is the year")


def plot_returns_distribution(result, freq: str = "ME",
                              figsize: Tuple[float, float] = (11, 4)) -> Figure:
    """Bar chart of returns per period (monthly by default)."""
    t = theme()
    rets = result.returns(freq)
    fig, ax = plt.subplots(figsize=figsize)
    if rets.empty:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                color=t["ink_muted"], transform=ax.transAxes)
        return _finish(fig, "Returns per period")
    colors = [t["positive"] if v >= 0 else t["negative"] for v in rets.values]
    width = (rets.index[1] - rets.index[0]).days * 0.7 if len(rets) > 1 else 20
    ax.bar(rets.index, rets.values, color=colors, width=width)
    ax.axhline(0, color=t["ink_muted"], linewidth=1)
    ax.set_ylabel("Return %")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _dates(ax)
    return _finish(fig, "Returns per period",
                   f"{(rets > 0).mean() * 100:.0f}% of periods positive")


def plot_rolling(result, window: int = 500, metric: str = "sharpe",
                 figsize: Tuple[float, float] = (11, 4)) -> Figure:
    """A rolling metric (sharpe, volatility or return) over the equity curve."""
    t = theme()
    series = result.rolling(window, metric)
    fig, ax = plt.subplots(figsize=figsize)
    if series.empty:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                color=t["ink_muted"], transform=ax.transAxes)
        return _finish(fig, f"Rolling {metric}")
    ax.plot(series.index, series.values, color=t["series"][0])
    ax.axhline(0, color=t["ink_muted"], linewidth=1)
    ax.set_ylabel(metric)
    _dates(ax)
    return _finish(fig, f"Rolling {metric} ({window} bars)")


# ----------------------------------------------------------------- DASHBOARD
def plot_dashboard(result, figsize: Tuple[float, float] = (14, 10)) -> Figure:
    """Summary panel: equity, drawdown, key metrics, monthly and distribution."""
    t = theme()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, height_ratios=[2.1, 0.8, 1.5, 1.5], hspace=0.55, wspace=0.28)

    # --- equity
    ax_eq = fig.add_subplot(gs[0, :])
    eq = result.equity_curve
    ax_eq.plot(eq.index, eq.values, color=t["series"][0], label="Equity")
    if "balance" in result.equity:
        ax_eq.plot(result.equity.index, result.equity["balance"].values,
                   color=t["series"][1], linewidth=1.2, alpha=0.9, label="Balance")
    ax_eq.fill_between(eq.index, eq.values, eq.min(), color=t["series"][0], alpha=0.06)
    ax_eq.axhline(result.stats["initial_balance"], color=t["ink_muted"],
                  linewidth=1, linestyle=(0, (4, 4)))
    ax_eq.set_title("Equity curve")
    _fmt_money(ax_eq)
    _dates(ax_eq)
    ax_eq.legend(loc="best", ncols=2)

    # --- drawdown
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_eq)
    dd = result.drawdown
    ax_dd.fill_between(dd.index, dd["dd_pct"].values, 0, color=t["negative"], alpha=0.3)
    ax_dd.plot(dd.index, dd["dd_pct"].values, color=t["negative"], linewidth=1.1)
    ax_dd.set_title("Drawdown %")
    ax_dd.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _dates(ax_dd)

    # --- key metrics (text tiles, not a chart)
    ax_kpi = fig.add_subplot(gs[2, 0])
    ax_kpi.axis("off")
    s = result.stats
    rows = [
        ("Net profit", f"{s['net_profit']:,.2f}", s["net_profit"] >= 0),
        ("Return", f"{s['return_pct']:.2f}%", s["return_pct"] >= 0),
        ("CAGR", f"{s['cagr_pct']:.2f}%", s["cagr_pct"] >= 0),
        ("Max drawdown", f"{s['max_dd_pct']:.2f}%", False),
        ("Profit factor", f"{s['profit_factor']:.2f}", s["profit_factor"] >= 1),
        ("Sharpe", f"{s['sharpe']:.2f}", s["sharpe"] >= 0),
        ("Win rate", f"{s['win_rate']:.1f}%", s["win_rate"] >= 50),
        ("Trades", f"{s['trades']:,}", True),
        ("Expectancy", f"{s['expectancy']:,.2f}", s["expectancy"] >= 0),
        ("Costs", f"{s['total_commission'] - s['total_swap']:,.2f}", False),
    ]
    for k, (label, value, good) in enumerate(rows):
        y = 1.0 - k * 0.105
        ax_kpi.text(0.0, y, label, fontsize=8.5, color=t["ink_muted"], va="top")
        ax_kpi.text(1.0, y, value, fontsize=9.5, fontweight="600", ha="right", va="top",
                    color=t["positive"] if good else t["ink"])
    ax_kpi.set_title("Key metrics", loc="left")

    # --- P&L distribution
    ax_hist = fig.add_subplot(gs[2, 1])
    if not result.trades.empty:
        pnl = result.trades["net_profit"].astype(float)
        edges = np.histogram_bin_edges(pnl, bins=30)
        ax_hist.hist(pnl[pnl > 0], bins=edges, color=t["positive"], alpha=0.85, label="Winners")
        ax_hist.hist(pnl[pnl <= 0], bins=edges, color=t["negative"], alpha=0.85, label="Losers")
        ax_hist.axvline(0, color=t["ink_muted"], linewidth=1)
        ax_hist.legend()
    ax_hist.set_title("Result per trade")

    # --- cumulative P&L
    ax_cum = fig.add_subplot(gs[2, 2])
    if not result.trades.empty:
        cum = result.trades["net_profit"].astype(float).cumsum()
        ax_cum.plot(range(1, len(cum) + 1), cum.values, color=t["series"][0])
        ax_cum.axhline(0, color=t["ink_muted"], linewidth=1)
    ax_cum.set_title("Cumulative P&L")
    ax_cum.set_xlabel("Trade #")

    # --- monthly
    ax_month = fig.add_subplot(gs[3, :])
    table = result.monthly_table()
    if not table.empty:
        values = table.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        limit = max(abs(finite).max(), 1e-6) if finite.size else 1.0
        ax_month.imshow(values, cmap=_diverging_cmap(),
                        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        ax_month.set_xticks(range(len(table.columns)), table.columns)
        ax_month.set_yticks(range(len(table.index)), table.index)
        ax_month.grid(False)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isfinite(value):
                    ax_month.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                                  color=t["ink"] if abs(value) < limit * 0.55 else t["surface"])
    ax_month.set_title("Monthly returns (%)")

    subtitle = (
        f"{result.spec.symbol} · {result.stats['start']} -> {result.stats['end']} · "
        f"broker {result.profile.name} · {type(result.strategy).__name__} {result.params}"
    )
    fig.suptitle("Backtest report", color=t["ink"], fontsize=14, fontweight="600",
                 x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.962, subtitle, color=t["ink_muted"], fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------- MONTE CARLO
def plot_montecarlo(mc_result, figsize: Tuple[float, float] = (12, 8),
                    max_paths: int = 200) -> Figure:
    """Fan of simulated paths plus outcome and drawdown distributions."""
    t = theme()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax1, ax2), (ax3, ax4) = axes

    paths = mc_result.paths
    x = np.arange(paths.shape[1])

    # 1) Percentile fan (single-hue sequential ramp)
    for row in paths[: min(max_paths, paths.shape[0])]:
        ax1.plot(x, row, color=t["series"][0], alpha=0.05, linewidth=0.8)
    for low, high, alpha in [(5, 95, 0.14), (25, 75, 0.24)]:
        ax1.fill_between(x, np.percentile(paths, low, axis=0),
                         np.percentile(paths, high, axis=0),
                         color=t["series"][0], alpha=alpha, linewidth=0,
                         label=f"P{low}-P{high}")
    ax1.plot(x, np.percentile(paths, 50, axis=0), color=t["series"][0], linewidth=2,
             label="Median")
    if mc_result.original_curve is not None:
        ax1.plot(np.arange(len(mc_result.original_curve)), mc_result.original_curve,
                 color=t["series"][1], linewidth=1.8, label="Actual backtest")
    ax1.axhline(mc_result.initial_balance, color=t["ink_muted"], linewidth=1,
                linestyle=(0, (4, 4)))
    ax1.set_title(f"Simulated paths ({mc_result.n_simulations:,} scenarios)")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="best", ncols=2)

    # 2) Final equity distribution
    finals = mc_result.final_equity
    ax2.hist(finals, bins=50, color=t["series"][0], alpha=0.85)
    ax2.axvline(mc_result.initial_balance, color=t["ink_muted"], linewidth=1.4,
                linestyle=(0, (4, 3)))
    ax2.axvline(np.median(finals), color=t["series"][1], linewidth=1.8)
    ax2.annotate(f"median {np.median(finals):,.0f}",
                 xy=(np.median(finals), ax2.get_ylim()[1] * 0.92),
                 xytext=(6, 0), textcoords="offset points", color=t["ink_soft"], fontsize=8)
    ax2.set_title("Final equity")
    ax2.set_xlabel("Equity")
    ax2.set_ylabel("Scenarios")

    # 3) Max drawdown distribution
    ax3.hist(mc_result.max_drawdowns_pct, bins=50, color=t["negative"], alpha=0.85)
    # Drawdowns are negative, so the bad tail sits at the 5th percentile.
    p95 = np.percentile(mc_result.max_drawdowns_pct, 5)
    ax3.axvline(p95, color=t["ink"], linewidth=1.5, linestyle=(0, (4, 3)))
    ax3.annotate(f"P95 {p95:.1f}%", xy=(p95, ax3.get_ylim()[1] * 0.92), xytext=(6, 0),
                 textcoords="offset points", color=t["ink_soft"], fontsize=8)
    ax3.set_title("Max drawdown per scenario")
    ax3.set_xlabel("Drawdown %")
    ax3.set_ylabel("Scenarios")

    # 4) Numeric summary
    ax4.axis("off")
    rows = [
        ("Scenarios", f"{mc_result.n_simulations:,}"),
        ("Method", mc_result.method),
        ("Final equity P05", f"{np.percentile(finals, 5):,.0f}"),
        ("Final equity median", f"{np.median(finals):,.0f}"),
        ("Final equity P95", f"{np.percentile(finals, 95):,.0f}"),
        ("Median drawdown", f"{np.median(mc_result.max_drawdowns_pct):.2f}%"),
        ("Drawdown P95 (bad)", f"{p95:.2f}%"),
        ("Worst drawdown", f"{np.min(mc_result.max_drawdowns_pct):.2f}%"),
        ("Probability of loss", f"{mc_result.probability_of_loss * 100:.1f}%"),
        ("Risk of ruin", f"{mc_result.risk_of_ruin * 100:.1f}%"),
        ("P(DD > threshold)", f"{mc_result.probability_dd_exceeds * 100:.1f}%"),
    ]
    for k, (label, value) in enumerate(rows):
        y = 1.0 - k * 0.095
        ax4.text(0.0, y, label, fontsize=8.5, color=t["ink_muted"], va="top")
        ax4.text(1.0, y, value, fontsize=9.5, fontweight="600", ha="right", va="top",
                 color=t["ink"])
    ax4.set_title("Summary", loc="left")

    return _finish(fig, "Monte Carlo simulation",
                   "reshuffling / resampling of the backtest's trades")


# -------------------------------------------------------------- OPTIMIZATION
def plot_optimization(opt_result, metric: Optional[str] = None,
                      params: Optional[Sequence[str]] = None,
                      figsize: Tuple[float, float] = (11, 5)) -> Figure:
    """Optimization output: bars for one parameter, a heatmap for two."""
    t = theme()
    df = opt_result.results if hasattr(opt_result, "results") else opt_result
    metric = metric or getattr(opt_result, "metric", "net_profit")
    names = list(params or getattr(opt_result, "param_names", []))

    fig, ax = plt.subplots(figsize=figsize)
    if len(names) >= 2:
        table = df.pivot_table(index=names[1], columns=names[0], values=metric, aggfunc="mean")
        values = table.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and finite.min() < 0 < finite.max():
            limit = max(abs(finite).max(), 1e-9)
            norm, cmap = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), _diverging_cmap()
        else:
            norm, cmap = None, _sequential_cmap()
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(table.columns)),
                      [f"{c:g}" if isinstance(c, (int, float)) else c for c in table.columns])
        ax.set_yticks(range(len(table.index)),
                      [f"{r:g}" if isinstance(r, (int, float)) else r for r in table.index])
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.grid(False)
        bar = fig.colorbar(image, ax=ax, pad=0.02)
        bar.set_label(metric)
        bar.outline.set_visible(False)
        # Mark the best point directly
        best = df.loc[df[metric].idxmax()]
        try:
            xi = list(table.columns).index(best[names[0]])
            yi = list(table.index).index(best[names[1]])
            ax.scatter([xi], [yi], s=110, facecolor="none", edgecolor=t["ink"], linewidth=2)
            ax.annotate("best", xy=(xi, yi), xytext=(8, 8), textcoords="offset points",
                        color=t["ink"], fontsize=8.5)
        except (ValueError, KeyError):
            pass
    elif len(names) == 1:
        grouped = df.groupby(names[0])[metric].mean()
        colors = [t["positive"] if v >= 0 else t["negative"] for v in grouped.values]
        ax.bar(range(len(grouped)), grouped.values, color=colors)
        ax.set_xticks(range(len(grouped)),
                      [f"{v:g}" if isinstance(v, (int, float)) else v for v in grouped.index])
        ax.axhline(0, color=t["ink_muted"], linewidth=1)
        ax.set_xlabel(names[0])
        ax.set_ylabel(metric)
    else:
        ax.scatter(range(len(df)), df[metric], color=t["series"][0], s=24)
        ax.set_xlabel("combination")
        ax.set_ylabel(metric)

    return _finish(fig, f"Optimization - {metric}", f"{len(df)} combinations evaluated")


def plot_walk_forward(wf_result, figsize: Tuple[float, float] = (11, 6)) -> Figure:
    """Compare in-sample and out-of-sample performance for each window."""
    t = theme()
    df = wf_result.windows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"hspace": 0.2})
    x = np.arange(len(df))
    width = 0.38
    ax1.bar(x - width / 2, df["is_metric"], width, color=t["series"][0], label="In-sample")
    ax1.bar(x + width / 2, df["oos_metric"], width, color=t["series"][1], label="Out-of-sample")
    ax1.axhline(0, color=t["ink_muted"], linewidth=1)
    ax1.set_ylabel(wf_result.metric)
    ax1.set_title("Performance per window")
    ax1.legend(ncols=2)

    equity = wf_result.oos_equity
    if equity is not None and len(equity):
        ax2.plot(equity.index, equity.values, color=t["series"][2])
        ax2.fill_between(equity.index, equity.values, equity.min(),
                         color=t["series"][2], alpha=0.08)
        ax2.set_ylabel("OOS equity")
        _dates(ax2)
    ax2.set_title("Chained out-of-sample curve")

    return _finish(
        fig, "Walk-forward",
        f"{len(df)} windows · average efficiency {wf_result.efficiency:.2f} (OOS/IS)",
    )


# ---------------------------------------------------------------- save it all
def save_all_charts(result, folder: str, prefix: str = "chart", dpi: int = 130,
                    close: bool = True, extension: str = "png") -> List[str]:
    """Build and save the whole standard chart set to disk."""
    os.makedirs(folder, exist_ok=True)
    builders = {
        "dashboard": lambda: plot_dashboard(result),
        "equity": lambda: plot_equity(result),
        "drawdown": lambda: plot_drawdown(result),
        "trades": lambda: plot_price_trades(result),
        "distribution": lambda: plot_trade_distribution(result),
        "monthly": lambda: plot_monthly_heatmap(result),
        "returns": lambda: plot_returns_distribution(result),
    }
    saved = []
    for name, builder in builders.items():
        try:
            fig = builder()
        except Exception as exc:  # one broken chart must not kill the report
            print(f"  warning: could not build '{name}': {exc}")
            continue
        path = os.path.join(folder, f"{prefix}_{name}.{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
        if close:
            plt.close(fig)
    return saved


set_theme("light")
