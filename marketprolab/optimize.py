"""Strategy optimization: grid, random and walk-forward.

Windows note: with ``n_jobs > 1`` the strategy must live in an importable
module (not a notebook) and the script must be guarded with
``if __name__ == "__main__":``, because parallelism uses processes.
"""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

Objective = Union[str, Callable[[Any], float]]

# Metrics where lower is better
_MINIMIZE = {"max_dd_pct", "max_dd_abs", "ulcer_index", "volatility_annual_pct"}


def _score(result, objective: Objective) -> float:
    if callable(objective):
        return float(objective(result))
    value = result.stats.get(objective)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return -np.inf
    value = float(value)
    return -value if objective in _MINIMIZE else value


@dataclass
class OptimizationResult:
    """The table of every combination that was tried."""

    results: pd.DataFrame
    param_names: List[str]
    metric: str
    best_params: Dict[str, Any]
    best_result: Any = None
    elapsed: float = 0.0

    @property
    def best(self) -> pd.Series:
        return self.results.iloc[0]

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.results.head(n)

    def stable_top(self, n: int = 10, neighbours: int = 1) -> pd.DataFrame:
        """Rank by the average of the parameter neighbourhood, not by the peak.

        An isolated peak is usually overfitting; a genuinely good setting is
        surrounded by neighbours that are also good. This scores every
        combination by the mean of its neighbourhood in the grid.
        """
        df = self.results.copy()
        if not self.param_names:
            return df.head(n)
        grids = {p: sorted(df[p].unique()) for p in self.param_names}
        index = {p: {v: k for k, v in enumerate(grids[p])} for p in self.param_names}
        lookup = df.set_index(self.param_names)[self.metric].to_dict()

        scores = []
        for _, row in df.iterrows():
            coords = [index[p][row[p]] for p in self.param_names]
            values = []
            for offsets in itertools.product(range(-neighbours, neighbours + 1),
                                             repeat=len(self.param_names)):
                key = []
                ok = True
                for p, c, off in zip(self.param_names, coords, offsets):
                    j = c + off
                    if not 0 <= j < len(grids[p]):
                        ok = False
                        break
                    key.append(grids[p][j])
                if not ok:
                    continue
                value = lookup.get(tuple(key) if len(key) > 1 else key[0])
                if value is not None and math.isfinite(value):
                    values.append(value)
            scores.append(np.mean(values) if values else -np.inf)
        df["neighbourhood_score"] = scores
        return df.sort_values("neighbourhood_score", ascending=False).head(n)

    def plot(self, **kwargs):
        from .plotting import plot_optimization

        return plot_optimization(self, **kwargs)

    def to_html(self, path: str = "optimization_report.html", **kwargs) -> str:
        """Write a standalone HTML report of the optimization."""
        from .report import optimization_report

        return optimization_report(self, path, **kwargs)

    def __repr__(self) -> str:
        return (f"<OptimizationResult {len(self.results)} combinations, "
                f"best {self.metric}={self.results.iloc[0][self.metric]:.4f} "
                f"at {self.best_params}>")


# --------------------------------------------------------------- parallel run
_WORKER_BT = None


def _init_worker(backtest):
    global _WORKER_BT
    _WORKER_BT = backtest


def _run_worker(params: Dict[str, Any]) -> Dict[str, Any]:
    return _row(params, _WORKER_BT.run(**params))


def _row(params: Dict[str, Any], result) -> Dict[str, Any]:
    stats = result.stats
    row = dict(params)
    row.update(
        {
            "net_profit": stats["net_profit"],
            "return_pct": stats["return_pct"],
            "cagr_pct": stats["cagr_pct"],
            "max_dd_pct": stats["max_dd_pct"],
            "max_dd_abs": stats["max_dd_abs"],
            "profit_factor": stats["profit_factor"],
            "sharpe": stats["sharpe"],
            "sortino": stats["sortino"],
            "calmar": stats["calmar"],
            "sqn": stats["sqn"],
            "trades": stats["trades"],
            "win_rate": stats["win_rate"],
            "expectancy": stats["expectancy"],
            "recovery_factor": stats["recovery_factor"],
            "ulcer_index": stats["ulcer_index"],
            "avg_bars_held": stats["avg_bars_held"],
        }
    )
    return row


def _expand_grid(param_grid: Dict[str, Sequence]) -> List[Dict[str, Any]]:
    keys = list(param_grid)
    combos = list(itertools.product(*(list(param_grid[k]) for k in keys)))
    return [dict(zip(keys, values)) for values in combos]


def _optimize_combos(
    backtest,
    combos: List[Dict[str, Any]],
    objective: Objective,
    n_jobs: int,
    progress: bool,
    constraint: Optional[Callable[[Any], bool]],
    min_trades: int,
    keep_best_result: bool,
) -> OptimizationResult:
    metric = objective if isinstance(objective, str) else getattr(
        objective, "__name__", "objective"
    )
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    best_score, best_params, best_result = -np.inf, {}, None
    total = len(combos)

    if n_jobs and n_jobs > 1:
        if callable(objective):
            raise ValueError("With n_jobs>1 the objective must be a metric name")
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                                 initargs=(backtest,)) as pool:
            futures = {pool.submit(_run_worker, params): params for params in combos}
            for k, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if progress:
                    print(f"  {k}/{total} combinations", end="\r", flush=True)
        # Worker processes do not ship BacktestResult objects back, so the best
        # combination is re-run once at the end.
        frame = pd.DataFrame(rows)
        valid = frame[frame["trades"] >= min_trades]
        if valid.empty:
            valid = frame
        ascending = objective in _MINIMIZE
        valid = valid.sort_values(objective, ascending=ascending)
        best_params = {k: valid.iloc[0][k] for k in combos[0]} if len(valid) else {}
        if keep_best_result and best_params:
            best_result = backtest.run(**best_params)
        frame = frame.sort_values(objective, ascending=ascending).reset_index(drop=True)
    else:
        for k, params in enumerate(combos, 1):
            result = backtest.run(**params)
            rows.append(_row(params, result))
            if result.stats["trades"] >= min_trades and (
                constraint is None or constraint(result)
            ):
                score = _score(result, objective)
                if score > best_score:
                    best_score, best_params = score, dict(params)
                    if keep_best_result:
                        best_result = result
            if progress:
                print(f"  {k}/{total} combinations · best {metric}={best_score:,.4f}",
                      end="\r", flush=True)
        frame = pd.DataFrame(rows)
        if isinstance(objective, str) and objective in frame.columns:
            frame = frame.sort_values(objective, ascending=objective in _MINIMIZE)
        frame = frame.reset_index(drop=True)

    if progress:
        print(" " * 80, end="\r")

    return OptimizationResult(
        results=frame,
        param_names=list(combos[0]) if combos else [],
        metric=metric,
        best_params=best_params,
        best_result=best_result,
        elapsed=time.perf_counter() - started,
    )


def grid_search(
    backtest,
    param_grid: Dict[str, Sequence],
    objective: Objective = "net_profit",
    n_jobs: int = 1,
    progress: bool = True,
    constraint: Optional[Callable[[Any], bool]] = None,
    min_trades: int = 1,
    keep_best_result: bool = True,
) -> OptimizationResult:
    """Try every combination in the parameter dictionary.

    ::

        opt = grid_search(bt, {"fast": [10, 20, 30], "slow": [50, 100]},
                          objective="sharpe", min_trades=30)
        opt.top(10)
        opt.plot()
    """
    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("The parameter grid is empty")
    return _optimize_combos(backtest, combos, objective, n_jobs, progress,
                            constraint, min_trades, keep_best_result)


def random_search(
    backtest,
    param_space: Dict[str, Sequence],
    n_iter: int = 100,
    objective: Objective = "net_profit",
    seed: Optional[int] = 42,
    n_jobs: int = 1,
    progress: bool = True,
    constraint: Optional[Callable[[Any], bool]] = None,
    min_trades: int = 1,
    keep_best_result: bool = True,
) -> OptimizationResult:
    """Sample ``n_iter`` random combinations from the parameter space.

    Each value can be a list (discrete choice) or a ``(min, max)`` tuple for
    continuous sampling.
    """
    rng = np.random.default_rng(seed)
    combos = []
    seen = set()
    for _ in range(n_iter * 5):
        if len(combos) >= n_iter:
            break
        params = {}
        for key, space in param_space.items():
            if isinstance(space, tuple) and len(space) == 2 and all(
                isinstance(v, (int, float)) for v in space
            ):
                low, high = space
                value = rng.uniform(low, high)
                params[key] = int(round(value)) if isinstance(low, int) else float(value)
            else:
                options = list(space)
                params[key] = options[int(rng.integers(len(options)))]
        signature = tuple(sorted(params.items()))
        if signature in seen:
            continue
        seen.add(signature)
        combos.append(params)
    return _optimize_combos(backtest, combos, objective, n_jobs, progress,
                            constraint, min_trades, keep_best_result)


# ---------------------------------------------------------------- WALK FORWARD
@dataclass
class WalkForwardResult:
    """Per-window results plus the chained out-of-sample curve."""

    windows: pd.DataFrame
    oos_equity: Optional[pd.Series]
    oos_trades: pd.DataFrame
    metric: str
    efficiency: float
    stats: Dict[str, Any] = field(default_factory=dict)

    def plot(self, **kwargs):
        from .plotting import plot_walk_forward

        return plot_walk_forward(self, **kwargs)

    def to_html(self, path: str = "walkforward_report.html", **kwargs) -> str:
        """Write a standalone HTML report of the walk-forward run."""
        from .report import walk_forward_report

        return walk_forward_report(self, path, **kwargs)

    def __repr__(self) -> str:
        return (f"<WalkForwardResult {len(self.windows)} windows, "
                f"efficiency {self.efficiency:.2f}, "
                f"OOS net {self.stats.get('net_profit', 0):,.2f}>")


def walk_forward(
    backtest,
    param_grid: Dict[str, Sequence],
    in_sample_bars: int = 5000,
    out_sample_bars: int = 1000,
    step_bars: Optional[int] = None,
    objective: Objective = "sharpe",
    anchored: bool = False,
    n_jobs: int = 1,
    progress: bool = True,
    min_trades: int = 5,
    compound: bool = True,
) -> WalkForwardResult:
    """Optimize on a rolling window and always validate out of sample.

    ``anchored=True`` keeps the start fixed and grows the in-sample window
    (anchored walk-forward); otherwise the whole window slides forward.
    """
    from .engine import Backtest

    data = backtest.data
    step = step_bars or out_sample_bars
    metric = objective if isinstance(objective, str) else "objective"

    rows = []
    oos_curves: List[pd.Series] = []
    oos_trades: List[pd.DataFrame] = []
    balance = backtest.config.initial_balance

    start = 0
    window = 0
    while start + in_sample_bars + out_sample_bars <= len(data):
        is_slice = data.iloc[(0 if anchored else start): start + in_sample_bars]
        oos_slice = data.iloc[start + in_sample_bars: start + in_sample_bars + out_sample_bars]
        window += 1
        if progress:
            print(f"  window {window}: IS {is_slice.index[0].date()}->{is_slice.index[-1].date()} "
                  f"| OOS {oos_slice.index[0].date()}->{oos_slice.index[-1].date()}")

        is_bt = Backtest(
            is_slice, backtest.strategy_class, backtest.spec, backtest.profile,
            replace(backtest.config, initial_balance=balance if compound else
                    backtest.config.initial_balance),
            strategy_params=backtest.strategy_params, warmup_bars=backtest.warmup_bars,
        )
        opt = grid_search(is_bt, param_grid, objective=objective, n_jobs=n_jobs,
                          progress=False, min_trades=min_trades, keep_best_result=False)
        best_params = opt.best_params or {}
        is_metric = float(opt.results.iloc[0][metric]) if metric in opt.results else np.nan

        oos_bt = Backtest(
            oos_slice, backtest.strategy_class, backtest.spec, backtest.profile,
            replace(backtest.config,
                    initial_balance=balance if compound else backtest.config.initial_balance),
            strategy_params={**backtest.strategy_params, **best_params},
            warmup_bars=backtest.warmup_bars,
        )
        oos = oos_bt.run()
        oos_metric = float(oos.stats.get(metric, np.nan)) if isinstance(objective, str) else _score(
            oos, objective
        )

        rows.append(
            {
                "window": window,
                "is_start": is_slice.index[0], "is_end": is_slice.index[-1],
                "oos_start": oos_slice.index[0], "oos_end": oos_slice.index[-1],
                "is_metric": is_metric, "oos_metric": oos_metric,
                "oos_net_profit": oos.stats["net_profit"],
                "oos_trades": oos.stats["trades"],
                "oos_max_dd_pct": oos.stats["max_dd_pct"],
                **{f"param_{k}": v for k, v in best_params.items()},
            }
        )
        oos_curves.append(oos.equity_curve)
        if not oos.trades.empty:
            frame = oos.trades.copy()
            frame["window"] = window
            oos_trades.append(frame)
        if compound:
            balance = float(oos.equity_curve.iloc[-1])
        start += step

    if not rows:
        raise ValueError(
            "Not enough data for a single window: reduce in_sample_bars/out_sample_bars"
        )

    windows = pd.DataFrame(rows)
    curve = pd.concat(oos_curves).sort_index() if oos_curves else None
    trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()

    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = windows["oos_metric"] / windows["is_metric"].replace(0, np.nan)
    efficiency = float(np.nanmean(ratios.replace([np.inf, -np.inf], np.nan)))

    from .metrics import compute_stats

    stats = compute_stats(
        trades if not trades.empty else pd.DataFrame(),
        curve if curve is not None else pd.Series(dtype=float),
        initial_balance=backtest.config.initial_balance,
    )
    if progress:
        print(f"  -> {len(windows)} windows · efficiency {efficiency:.2f} · "
              f"OOS net {stats['net_profit']:,.2f}")

    return WalkForwardResult(windows=windows, oos_equity=curve, oos_trades=trades,
                             metric=metric, efficiency=efficiency, stats=stats)


# ----------------------------------------------------------------- comparisons
def compare(results: Dict[str, Any], metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Side-by-side table of several ``BacktestResult`` objects.

    ::

        compare({"conservative": r1, "aggressive": r2})
    """
    metrics = list(metrics or [
        "net_profit", "return_pct", "cagr_pct", "max_dd_pct", "profit_factor",
        "sharpe", "sortino", "calmar", "trades", "win_rate", "expectancy",
        "recovery_factor", "total_commission", "total_swap",
    ])
    rows = {}
    for name, result in results.items():
        stats = result.stats if hasattr(result, "stats") else result
        rows[name] = {m: stats.get(m) for m in metrics}
    return pd.DataFrame(rows).T


def sensitivity(backtest, param: str, values: Sequence, objective: str = "net_profit",
                **kwargs) -> pd.DataFrame:
    """Sweep a single parameter to see whether the result is stable."""
    opt = grid_search(backtest, {param: list(values)}, objective=objective, **kwargs)
    return opt.results.sort_values(param).reset_index(drop=True)


def stress_test(backtest, spread_multipliers: Sequence[float] = (1, 1.5, 2, 3),
                slippage_points: Sequence[float] = (0, 5, 10, 20),
                commission_multipliers: Sequence[float] = (1, 1.5, 2),
                progress: bool = True) -> pd.DataFrame:
    """Re-run the backtest with progressively worse execution costs.

    Shows how much safety margin the strategy has against conditions worse
    than the current broker's. ``slippage_points=0`` keeps the base model.
    """
    from .engine import Backtest
    from .execution import FixedSlippage

    rows = []
    base_spread = backtest.spec.spread_points
    total = len(spread_multipliers) * len(slippage_points) * len(commission_multipliers)
    k = 0
    for sm in spread_multipliers:
        for slip in slippage_points:
            for cm in commission_multipliers:
                k += 1
                spec = type(backtest.spec).from_dict(backtest.spec.to_dict())
                spec.spread_points = base_spread * sm
                spec.commission_per_lot *= cm
                spec.commission_per_deal *= cm
                spec.commission_percent *= cm
                config = replace(
                    backtest.config,
                    spread=spec.spread_points,
                    slippage=FixedSlippage(slip) if slip else backtest.config.slippage,
                )
                result = Backtest(backtest.data, backtest.strategy_class, spec,
                                  backtest.profile, config,
                                  strategy_params=backtest.strategy_params,
                                  warmup_bars=backtest.warmup_bars).run()
                rows.append(
                    {
                        "spread_x": sm, "slippage_points": slip, "commission_x": cm,
                        "net_profit": result.stats["net_profit"],
                        "return_pct": result.stats["return_pct"],
                        "profit_factor": result.stats["profit_factor"],
                        "max_dd_pct": result.stats["max_dd_pct"],
                        "sharpe": result.stats["sharpe"],
                        "trades": result.stats["trades"],
                    }
                )
                if progress:
                    print(f"  {k}/{total} stress scenarios", end="\r", flush=True)
    if progress:
        print(" " * 60, end="\r")
    return pd.DataFrame(rows)
