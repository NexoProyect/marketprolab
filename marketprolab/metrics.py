"""Performance statistics derived from the equity curve and the trade list."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.DataFrame:
    """Absolute and relative drawdown against the running peak."""
    peak = equity.cummax()
    abs_dd = equity - peak
    pct_dd = np.where(peak > 0, abs_dd / peak * 100.0, 0.0)
    return pd.DataFrame({"peak": peak, "dd_abs": abs_dd, "dd_pct": pct_dd}, index=equity.index)


def max_drawdown(equity: pd.Series) -> Dict[str, float]:
    dd = drawdown_series(equity)
    idx_abs = dd["dd_abs"].idxmin() if len(dd) else None
    idx_pct = dd["dd_pct"].idxmin() if len(dd) else None
    return {
        "max_dd_abs": float(dd["dd_abs"].min()) if len(dd) else 0.0,
        "max_dd_pct": float(dd["dd_pct"].min()) if len(dd) else 0.0,
        "max_dd_time": idx_abs,
        "max_dd_pct_time": idx_pct,
    }


def drawdown_duration(equity: pd.Series) -> Dict[str, object]:
    """Longest and average time spent under water."""
    if equity.empty:
        return {"max_dd_duration": pd.Timedelta(0), "avg_dd_duration": pd.Timedelta(0)}
    peak = equity.cummax()
    under = equity < peak
    durations = []
    start = None
    for ts, flag in under.items():
        if flag and start is None:
            start = ts
        elif not flag and start is not None:
            durations.append(ts - start)
            start = None
    if start is not None:
        durations.append(equity.index[-1] - start)
    if not durations:
        return {"max_dd_duration": pd.Timedelta(0), "avg_dd_duration": pd.Timedelta(0)}
    return {
        "max_dd_duration": max(durations),
        "avg_dd_duration": sum(durations, pd.Timedelta(0)) / len(durations),
    }


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    """Periods per year, from the median spacing of the series."""
    if len(index) < 3:
        return 1.0
    delta = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    if delta <= 0:
        return 1.0
    return (365.0 * 24 * 3600) / delta


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(excess.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    dd_std = downside.std(ddof=1)
    if not dd_std or np.isnan(dd_std):
        return 0.0
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))


def streaks(values: List[bool]) -> Dict[str, int]:
    """Longest winning and losing streaks."""
    best = worst = current = 0
    sign = 0
    for win in values:
        step = 1 if win else -1
        current = current + step if sign == step else step
        sign = step
        best = max(best, current)
        worst = min(worst, current)
    return {"max_win_streak": best, "max_loss_streak": abs(worst)}


def compute_stats(
    trades: pd.DataFrame,
    equity: pd.Series,
    balance: Optional[pd.Series] = None,
    initial_balance: float = 0.0,
    risk_free_rate: float = 0.0,
    exposure: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """The full metric dictionary for a backtest."""
    stats: Dict[str, object] = {}
    equity = equity.astype(float)
    n_bars = len(equity)
    initial = initial_balance or (float(equity.iloc[0]) if n_bars else 0.0)
    final = float(equity.iloc[-1]) if n_bars else initial

    stats["start"] = equity.index[0] if n_bars else None
    stats["end"] = equity.index[-1] if n_bars else None
    stats["duration"] = (equity.index[-1] - equity.index[0]) if n_bars else pd.Timedelta(0)
    stats["bars"] = n_bars
    stats["initial_balance"] = initial
    stats["final_equity"] = final
    stats["net_profit"] = final - initial
    stats["return_pct"] = (final / initial - 1.0) * 100.0 if initial else 0.0
    stats["peak_equity"] = float(equity.max()) if n_bars else initial
    stats["min_equity"] = float(equity.min()) if n_bars else initial

    years = stats["duration"].total_seconds() / (365.25 * 24 * 3600) if n_bars else 0.0
    stats["years"] = years
    if years > 0 and initial > 0 and final > 0:
        stats["cagr_pct"] = ((final / initial) ** (1 / years) - 1.0) * 100.0
    else:
        stats["cagr_pct"] = 0.0

    stats.update(max_drawdown(equity))
    stats.update(drawdown_duration(equity))
    stats["recovery_factor"] = (
        stats["net_profit"] / abs(stats["max_dd_abs"]) if stats["max_dd_abs"] else 0.0
    )
    stats["calmar"] = (
        stats["cagr_pct"] / abs(stats["max_dd_pct"]) if stats["max_dd_pct"] else 0.0
    )

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ppy = _annualization_factor(equity.index) if n_bars > 2 else 1.0
    stats["periods_per_year"] = ppy
    stats["sharpe"] = sharpe_ratio(returns, ppy, risk_free_rate)
    stats["sortino"] = sortino_ratio(returns, ppy, risk_free_rate)
    stats["volatility_annual_pct"] = (
        float(returns.std(ddof=1) * np.sqrt(ppy) * 100.0) if len(returns) > 1 else 0.0
    )
    if exposure is not None and len(exposure):
        stats["exposure_pct"] = float((exposure > 0).mean() * 100.0)
        stats["avg_open_volume"] = float(exposure.mean())
    else:
        stats["exposure_pct"] = np.nan
        stats["avg_open_volume"] = np.nan

    # ---------------------------------------------------------------- trades
    if trades is None or trades.empty:
        stats.update(
            {
                "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "avg_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0, "gross_profit": 0.0,
                "gross_loss": 0.0, "total_commission": 0.0, "total_swap": 0.0,
                "total_costs": 0.0, "max_win_streak": 0, "max_loss_streak": 0,
                "avg_bars_held": 0.0, "avg_duration": pd.Timedelta(0),
                "trades_per_month": 0.0, "sqn": 0.0, "payoff_ratio": 0.0,
                "avg_mae": 0.0, "avg_mfe": 0.0, "avg_slippage_points": 0.0,
                "long_trades": 0, "short_trades": 0, "long_win_rate": 0.0,
                "short_win_rate": 0.0, "long_profit": 0.0, "short_profit": 0.0,
                "kelly_pct": 0.0, "ulcer_index": 0.0,
            }
        )
        return stats

    pnl = trades["net_profit"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    stats["trades"] = int(len(pnl))
    stats["wins"] = int(len(wins))
    stats["losses"] = int(len(losses))
    stats["win_rate"] = float(len(wins) / len(pnl) * 100.0)
    stats["gross_profit"] = float(wins.sum())
    stats["gross_loss"] = float(losses.sum())
    stats["profit_factor"] = (
        float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    )
    stats["expectancy"] = float(pnl.mean())
    stats["avg_trade"] = float(pnl.mean())
    stats["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    stats["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    stats["payoff_ratio"] = (
        abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] else float("inf")
    )
    stats["largest_win"] = float(pnl.max())
    stats["largest_loss"] = float(pnl.min())
    stats["total_commission"] = float(trades["commission"].sum())
    stats["total_swap"] = float(trades["swap"].sum())
    stats["total_costs"] = stats["total_commission"] - stats["total_swap"]
    stats.update(streaks(list(pnl > 0)))
    stats["avg_bars_held"] = float(trades["bars_held"].mean())
    stats["avg_duration"] = pd.to_timedelta(trades["duration_s"].mean(), unit="s")
    months = max(stats["duration"].days / 30.44, 1e-9)
    stats["trades_per_month"] = float(len(pnl) / months)
    std = pnl.std(ddof=1)
    stats["sqn"] = float(np.sqrt(len(pnl)) * pnl.mean() / std) if std else 0.0
    stats["avg_mae"] = float(trades["mae"].mean()) if "mae" in trades else 0.0
    stats["avg_mfe"] = float(trades["mfe"].mean()) if "mfe" in trades else 0.0
    stats["avg_slippage_points"] = (
        float(trades["slippage_points"].mean()) if "slippage_points" in trades else 0.0
    )

    longs = trades[trades["type"] == "buy"]
    shorts = trades[trades["type"] == "sell"]
    stats["long_trades"] = int(len(longs))
    stats["short_trades"] = int(len(shorts))
    stats["long_win_rate"] = (
        float((longs["net_profit"] > 0).mean() * 100.0) if len(longs) else 0.0
    )
    stats["short_win_rate"] = (
        float((shorts["net_profit"] > 0).mean() * 100.0) if len(shorts) else 0.0
    )
    stats["long_profit"] = float(longs["net_profit"].sum()) if len(longs) else 0.0
    stats["short_profit"] = float(shorts["net_profit"].sum()) if len(shorts) else 0.0

    # Kelly fraction (theoretical optimum) and Ulcer index
    w = stats["win_rate"] / 100.0
    payoff = stats["payoff_ratio"]
    stats["kelly_pct"] = (
        float((w - (1 - w) / payoff) * 100.0) if payoff not in (0, float("inf")) else 0.0
    )
    dd_pct = drawdown_series(equity)["dd_pct"]
    stats["ulcer_index"] = float(np.sqrt((dd_pct ** 2).mean()))

    if "reason" in trades:
        stats["exits_by_reason"] = trades["reason"].value_counts().to_dict()
    return stats


def periodic_returns(equity: pd.Series, freq: str = "ME") -> pd.Series:
    """Return per period in % (``ME`` monthly, ``W`` weekly, ``YE`` yearly)."""
    if equity.empty:
        return pd.Series(dtype=float)
    resampled = equity.resample(freq).last().dropna()
    first = equity.iloc[0]
    prev = pd.concat([pd.Series([first], index=[equity.index[0]]), resampled]).shift(1)
    prev = prev.reindex(resampled.index).ffill()
    return ((resampled / prev - 1.0) * 100.0).dropna()


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month table of returns, in %."""
    monthly = periodic_returns(equity, "ME")
    if monthly.empty:
        return pd.DataFrame()
    frame = monthly.to_frame("ret")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    table = frame.pivot_table(index="year", columns="month", values="ret")
    table.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1] for m in table.columns
    ]
    table["Year"] = ((table.fillna(0) / 100.0 + 1.0).prod(axis=1) - 1.0) * 100.0
    return table


def rolling_metric(equity: pd.Series, window: int, metric: str = "sharpe") -> pd.Series:
    """A rolling metric over the equity curve."""
    returns = equity.pct_change().dropna()
    ppy = _annualization_factor(equity.index)
    if metric == "sharpe":
        mean = returns.rolling(window).mean()
        std = returns.rolling(window).std(ddof=1)
        return (mean / std * np.sqrt(ppy)).dropna()
    if metric == "volatility":
        return (returns.rolling(window).std(ddof=1) * np.sqrt(ppy) * 100).dropna()
    if metric == "return":
        return (equity / equity.shift(window) - 1.0).dropna() * 100
    raise ValueError(f"Unknown metric: {metric}")


def format_stats(stats: Dict[str, object]) -> str:
    """A console-friendly text report."""
    def money(x):
        return f"{x:>14,.2f}"

    def pct(x):
        return f"{x:>13,.2f}%"

    lines = [
        "===================== BACKTEST RESULTS =====================",
        f"Period               : {stats['start']}  ->  {stats['end']}",
        f"Duration / bars      : {stats['duration']}  /  {stats['bars']:,}",
        "--------------------------- Account -------------------------",
        f"Initial balance      : {money(stats['initial_balance'])}",
        f"Final equity         : {money(stats['final_equity'])}",
        f"Net profit           : {money(stats['net_profit'])}   ({stats['return_pct']:.2f}%)",
        f"CAGR                 : {pct(stats['cagr_pct'])}",
        f"Max drawdown         : {money(stats['max_dd_abs'])}   ({stats['max_dd_pct']:.2f}%)",
        f"Longest drawdown     : {stats.get('max_dd_duration')}",
        f"Recovery factor      : {stats['recovery_factor']:>13,.2f}",
        "---------------------------- Risk ---------------------------",
        f"Sharpe / Sortino     : {stats['sharpe']:>7,.2f} / {stats['sortino']:,.2f}",
        f"Calmar / SQN         : {stats['calmar']:>7,.2f} / {stats['sqn']:,.2f}",
        f"Annual volatility    : {pct(stats['volatility_annual_pct'])}",
        f"Ulcer index          : {stats['ulcer_index']:>14,.2f}",
        f"Exposure             : {stats['exposure_pct']:>13,.2f}%",
        "---------------------------- Trades -------------------------",
        f"Trades               : {stats['trades']:>14,}"
        f"   ({stats['long_trades']} long / {stats['short_trades']} short)",
        f"Win rate             : {stats['win_rate']:>13,.2f}%"
        f"   ({stats['wins']}/{stats['trades']})",
        f"Profit factor        : {stats['profit_factor']:>14,.2f}",
        f"Expectancy per trade : {money(stats['expectancy'])}",
        f"Avg win / avg loss   : {stats['avg_win']:>10,.2f} / {stats['avg_loss']:,.2f}"
        f"   (payoff {stats['payoff_ratio']:.2f})",
        f"Largest win / loss   : {stats['largest_win']:>10,.2f} / {stats['largest_loss']:,.2f}",
        f"Max win/loss streak  : {stats['max_win_streak']:>10} / {stats['max_loss_streak']}",
        f"Average duration     : {stats.get('avg_duration')}  ({stats['avg_bars_held']:.1f} bars)",
        f"Trades per month     : {stats['trades_per_month']:>14,.1f}",
        "---------------------------- Costs --------------------------",
        f"Commission           : {money(stats['total_commission'])}",
        f"Swap                 : {money(stats['total_swap'])}",
        f"Average slippage     : {stats['avg_slippage_points']:>10,.2f} points",
    ]
    if stats.get("exits_by_reason"):
        lines.append(f"Exits by reason      : {stats['exits_by_reason']}")
    lines.append("=============================================================")
    return "\n".join(lines)
