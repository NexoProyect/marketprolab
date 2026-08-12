"""Monte Carlo simulation on top of backtest results.

A backtest is *one* sample of a possible future. Monte Carlo reshuffles or
resamples the trades to estimate the range of outcomes and, above all, the
drawdown you should be prepared to sit through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """The output of a simulation."""

    paths: np.ndarray                  # (n_simulations, n_trades + 1)
    final_equity: np.ndarray
    max_drawdowns_pct: np.ndarray      # negative values
    max_drawdowns_abs: np.ndarray
    initial_balance: float
    n_simulations: int
    method: str
    probability_of_loss: float
    risk_of_ruin: float
    probability_dd_exceeds: float
    dd_threshold_pct: float
    original_curve: Optional[np.ndarray] = None
    percentiles: Dict[str, Any] = field(default_factory=dict)

    def summary(self, print_it: bool = True) -> str:
        p = self.percentiles
        text = "\n".join(
            [
                "================= MONTE CARLO =================",
                f"Scenarios            : {self.n_simulations:,}   method: {self.method}",
                f"Initial balance      : {self.initial_balance:,.2f}",
                "------------------ Final equity ---------------",
                f"  P05 / P25          : {p['final_p05']:,.2f} / {p['final_p25']:,.2f}",
                f"  Median             : {p['final_p50']:,.2f}",
                f"  P75 / P95          : {p['final_p75']:,.2f} / {p['final_p95']:,.2f}",
                f"  Worst / best       : {p['final_min']:,.2f} / {p['final_max']:,.2f}",
                "------------------ Max drawdown ---------------",
                f"  Median             : {p['dd_p50']:.2f}%",
                f"  P95 (bad)          : {p['dd_p95']:.2f}%",
                f"  P99 (very bad)     : {p['dd_p99']:.2f}%",
                f"  Worst scenario     : {p['dd_max']:.2f}%",
                "------------------ Probabilities --------------",
                f"  Ending in a loss   : {self.probability_of_loss * 100:.2f}%",
                f"  Risk of ruin       : {self.risk_of_ruin * 100:.2f}%",
                f"  DD > {self.dd_threshold_pct:.0f}%           : "
                f"{self.probability_dd_exceeds * 100:.2f}%",
                "===============================================",
            ]
        )
        if print_it:
            print(text)
        return text

    def plot(self, **kwargs):
        from .plotting import plot_montecarlo

        return plot_montecarlo(self, **kwargs)

    def to_html(self, path: str = "montecarlo_report.html", **kwargs) -> str:
        """Write a standalone HTML report of the simulation."""
        from .report import montecarlo_report

        return montecarlo_report(self, path, **kwargs)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "final_equity": self.final_equity,
                "max_drawdown_pct": self.max_drawdowns_pct,
                "max_drawdown_abs": self.max_drawdowns_abs,
            }
        )

    def __repr__(self) -> str:
        return (f"<MonteCarloResult {self.n_simulations:,} scenarios, "
                f"DD P95 {self.percentiles['dd_p95']:.1f}%, "
                f"ruin {self.risk_of_ruin * 100:.1f}%>")


def _equity_paths(pnl_matrix: np.ndarray, initial_balance: float) -> np.ndarray:
    cumulative = np.cumsum(pnl_matrix, axis=1)
    start = np.full((pnl_matrix.shape[0], 1), initial_balance)
    return np.hstack([start, start + cumulative])


def _drawdowns(paths: np.ndarray) -> tuple:
    peaks = np.maximum.accumulate(paths, axis=1)
    abs_dd = paths - peaks
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_dd = np.where(peaks > 0, abs_dd / peaks * 100.0, 0.0)
    return pct_dd.min(axis=1), abs_dd.min(axis=1)


def monte_carlo(
    source,
    n_simulations: int = 1000,
    method: str = "shuffle",
    initial_balance: Optional[float] = None,
    block_size: int = 10,
    skip_probability: float = 0.0,
    pnl_noise_pct: float = 0.0,
    extra_cost_per_trade: float = 0.0,
    n_trades: Optional[int] = None,
    ruin_level_pct: float = 0.0,
    dd_threshold_pct: float = 20.0,
    seed: Optional[int] = 42,
    compound_risk: bool = False,
) -> MonteCarloResult:
    """Simulate thousands of alternative sequences of the same trades.

    Parameters
    ----------
    source
        A ``BacktestResult``, a trades ``DataFrame``, or a sequence of net
        results per trade.
    method
        * ``"shuffle"``    - reorder the trades (same set).
        * ``"bootstrap"``  - resample with replacement (repeats and omissions).
        * ``"block"``      - block bootstrap: preserves streaks.
        * ``"normal"``     - draw P&L from a normal with the real mean and sigma.
    skip_probability
        Chance of skipping each trade (missed signals, disconnections, or simply
        not being at the screen).
    pnl_noise_pct
        Multiplicative noise on each result (e.g. ``10`` = +/-10%), modelling the
        fact that the same trade does not always play out the same way.
    extra_cost_per_trade
        Additional flat cost per trade (worse spread, higher commission).
    ruin_level_pct
        Percentage of the initial balance below which the account counts as ruined.
    compound_risk
        When ``True``, each result is scaled by the current equity (position size
        proportional to capital) instead of a fixed lot.
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------- input
    original_curve = None
    if hasattr(source, "trades") and hasattr(source, "stats"):
        pnl = (source.trades["net_profit"].to_numpy(dtype=float)
               if not source.trades.empty else np.array([]))
        initial_balance = (initial_balance if initial_balance is not None
                           else float(source.stats["initial_balance"]))
        original_curve = (
            initial_balance + np.concatenate([[0.0], np.cumsum(pnl)]) if pnl.size else None
        )
    elif isinstance(source, pd.DataFrame):
        pnl = source["net_profit"].to_numpy(dtype=float)
        initial_balance = float(initial_balance or 10_000.0)
    else:
        pnl = np.asarray(source, dtype=float)
        initial_balance = float(initial_balance or 10_000.0)

    if pnl.size == 0:
        raise ValueError("There are no trades to simulate")

    length = int(n_trades or len(pnl))

    # ------------------------------------------------------------ sampling
    if method == "shuffle":
        if length > len(pnl):
            matrix = np.array(
                [rng.permutation(np.resize(pnl, length)) for _ in range(n_simulations)]
            )
        else:
            matrix = np.array([rng.permutation(pnl)[:length] for _ in range(n_simulations)])
    elif method == "bootstrap":
        matrix = rng.choice(pnl, size=(n_simulations, length), replace=True)
    elif method == "block":
        blocks = max(1, int(np.ceil(length / block_size)))
        starts = rng.integers(0, max(1, len(pnl) - block_size), size=(n_simulations, blocks))
        matrix = np.empty((n_simulations, blocks * block_size))
        for i in range(n_simulations):
            matrix[i] = np.concatenate([pnl[s: s + block_size] if s + block_size <= len(pnl)
                                        else np.resize(pnl[s:], block_size)
                                        for s in starts[i]])
        matrix = matrix[:, :length]
    elif method == "normal":
        matrix = rng.normal(pnl.mean(), pnl.std(ddof=1) or 1e-9, size=(n_simulations, length))
    else:
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------- perturbations
    if pnl_noise_pct:
        matrix = matrix * (1.0 + rng.normal(0.0, pnl_noise_pct / 100.0, size=matrix.shape))
    if extra_cost_per_trade:
        matrix = matrix - abs(extra_cost_per_trade)
    if skip_probability:
        matrix = matrix * (rng.random(matrix.shape) >= skip_probability)

    # -------------------------------------------------------------- curves
    if compound_risk:
        paths = np.empty((matrix.shape[0], matrix.shape[1] + 1))
        paths[:, 0] = initial_balance
        for j in range(matrix.shape[1]):
            scale = np.maximum(paths[:, j], 0.0) / initial_balance
            paths[:, j + 1] = paths[:, j] + matrix[:, j] * scale
    else:
        paths = _equity_paths(matrix, initial_balance)

    final_equity = paths[:, -1]
    dd_pct, dd_abs = _drawdowns(paths)

    ruin_level = initial_balance * ruin_level_pct / 100.0
    ruined = (paths.min(axis=1) <= ruin_level).mean()

    percentiles = {
        "final_p05": float(np.percentile(final_equity, 5)),
        "final_p25": float(np.percentile(final_equity, 25)),
        "final_p50": float(np.percentile(final_equity, 50)),
        "final_p75": float(np.percentile(final_equity, 75)),
        "final_p95": float(np.percentile(final_equity, 95)),
        "final_min": float(final_equity.min()),
        "final_max": float(final_equity.max()),
        # Drawdowns are negative: the bad tail lives in the low percentiles.
        "dd_p50": float(np.percentile(dd_pct, 50)),
        "dd_p95": float(np.percentile(dd_pct, 5)),
        "dd_p99": float(np.percentile(dd_pct, 1)),
        "dd_max": float(dd_pct.min()),
        "dd_best": float(dd_pct.max()),
    }

    return MonteCarloResult(
        paths=paths,
        final_equity=final_equity,
        max_drawdowns_pct=dd_pct,
        max_drawdowns_abs=dd_abs,
        initial_balance=initial_balance,
        n_simulations=int(n_simulations),
        method=method,
        probability_of_loss=float((final_equity < initial_balance).mean()),
        risk_of_ruin=float(ruined),
        probability_dd_exceeds=float((dd_pct <= -abs(dd_threshold_pct)).mean()),
        dd_threshold_pct=float(abs(dd_threshold_pct)),
        original_curve=original_curve,
        percentiles=percentiles,
    )


def monte_carlo_bars(
    backtest,
    n_simulations: int = 50,
    method: str = "bootstrap_returns",
    block_size: int = 24,
    seed: Optional[int] = 42,
    progress: bool = True,
) -> pd.DataFrame:
    """Monte Carlo on the **price data**, not on the trades.

    Rebuilds synthetic price series by resampling returns (optionally in
    blocks) and re-runs the whole strategy on each one. Much slower than
    :func:`monte_carlo`, and much more honest: the strategy faces price paths
    it has never seen.
    """
    from .engine import Backtest

    rng = np.random.default_rng(seed)
    data = backtest.data
    close = data["close"].to_numpy()
    log_returns = np.diff(np.log(close))
    n = len(log_returns)

    rows = []
    for k in range(n_simulations):
        if method == "bootstrap_returns":
            sampled = rng.choice(log_returns, size=n, replace=True)
        elif method == "block":
            blocks = int(np.ceil(n / block_size))
            starts = rng.integers(0, max(1, n - block_size), size=blocks)
            sampled = np.concatenate([log_returns[s: s + block_size] for s in starts])[:n]
        elif method == "shuffle":
            sampled = rng.permutation(log_returns)
        else:
            raise ValueError(f"Unknown method: {method}")

        synth_close = close[0] * np.exp(np.concatenate([[0.0], np.cumsum(sampled)]))
        scale = synth_close / close
        synth = data.copy()
        for column in ("open", "high", "low", "close"):
            synth[column] = data[column].to_numpy() * scale
        synth["high"] = synth[["open", "high", "low", "close"]].max(axis=1)
        synth["low"] = synth[["open", "high", "low", "close"]].min(axis=1)

        result = Backtest(synth, backtest.strategy_class, backtest.spec, backtest.profile,
                          backtest.config, strategy_params=backtest.strategy_params,
                          warmup_bars=backtest.warmup_bars).run()
        rows.append(
            {
                "simulation": k + 1,
                "net_profit": result.stats["net_profit"],
                "return_pct": result.stats["return_pct"],
                "max_dd_pct": result.stats["max_dd_pct"],
                "profit_factor": result.stats["profit_factor"],
                "sharpe": result.stats["sharpe"],
                "trades": result.stats["trades"],
            }
        )
        if progress:
            print(f"  {k + 1}/{n_simulations} synthetic paths", end="\r", flush=True)
    if progress:
        print(" " * 60, end="\r")
    return pd.DataFrame(rows)


def confidence_intervals(mc: MonteCarloResult,
                         levels: Sequence[float] = (5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Percentile bands of the simulated equity curve, trade by trade."""
    data = {f"P{int(level):02d}": np.percentile(mc.paths, level, axis=0) for level in levels}
    return pd.DataFrame(data)


def required_capital(mc: MonteCarloResult, confidence: float = 95.0,
                     safety_factor: float = 1.5) -> Dict[str, float]:
    """Suggested minimum capital to survive the expected drawdown.

    Takes the drawdown at the given confidence level and applies a safety factor.
    """
    dd_pct = abs(float(np.percentile(mc.max_drawdowns_pct, 100 - confidence)))
    dd_abs = abs(float(np.percentile(mc.max_drawdowns_abs, 100 - confidence)))
    return {
        "confidence": confidence,
        "expected_dd_pct": dd_pct,
        "expected_dd_abs": dd_abs,
        "suggested_capital": dd_abs * safety_factor,
        "safety_factor": safety_factor,
    }
