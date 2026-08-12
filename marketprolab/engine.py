"""The backtest engine: ties together data, instrument, broker and strategy."""

from __future__ import annotations

import json
import os
import time as _time
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

from . import metrics as _metrics
from .broker import Broker, SimulationConfig
from .broker_profile import BrokerProfile
from .data import add_gap_points, prepare_bars, resample_bars, slice_dates
from .enums import Timeframe
from .strategy import Strategy
from .symbol import SymbolSpec


def _infer_timeframe_seconds(index: pd.DatetimeIndex, default: int = 60) -> int:
    if len(index) < 3:
        return default
    deltas = np.diff(index.values[: min(len(index), 5000)]).astype("timedelta64[s]").astype(float)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return default
    return int(np.median(deltas))


class BacktestResult:
    """The outcome of a backtest: trades, curves, metrics and charts."""

    def __init__(
        self,
        trades: pd.DataFrame,
        equity: pd.DataFrame,
        stats: Dict[str, Any],
        data: pd.DataFrame,
        spec: SymbolSpec,
        config: SimulationConfig,
        profile: BrokerProfile,
        strategy: Strategy,
        broker: Broker,
        elapsed: float = 0.0,
    ):
        self.trades = trades
        self.equity = equity
        self.stats = stats
        self.data = data
        self.spec = spec
        self.config = config
        self.profile = profile
        self.strategy = strategy
        self.broker = broker
        self.elapsed = elapsed
        self.params = strategy.params() if strategy else {}

    # ------------------------------------------------------------------ access
    @property
    def equity_curve(self) -> pd.Series:
        return self.equity["equity"]

    @property
    def balance_curve(self) -> pd.Series:
        return self.equity["balance"]

    @property
    def drawdown(self) -> pd.DataFrame:
        return _metrics.drawdown_series(self.equity_curve)

    @property
    def rejections(self) -> pd.DataFrame:
        return pd.DataFrame(self.broker.rejections)

    @property
    def events(self) -> pd.DataFrame:
        return pd.DataFrame(self.broker.events)

    def monthly_table(self) -> pd.DataFrame:
        return _metrics.monthly_table(self.equity_curve)

    def returns(self, freq: str = "ME") -> pd.Series:
        return _metrics.periodic_returns(self.equity_curve, freq)

    def rolling(self, window: int = 500, metric: str = "sharpe") -> pd.Series:
        return _metrics.rolling_metric(self.equity_curve, window, metric)

    def __getitem__(self, key: str):
        return self.stats[key]

    def __repr__(self) -> str:
        return (
            f"<BacktestResult {self.spec.symbol} trades={self.stats.get('trades', 0)} "
            f"net={self.stats.get('net_profit', 0):,.2f} "
            f"PF={self.stats.get('profit_factor', 0):.2f} "
            f"DD={self.stats.get('max_dd_pct', 0):.2f}%>"
        )

    # ---------------------------------------------------------------- reports
    def report(self, print_it: bool = True) -> str:
        """Plain-text report."""
        text = _metrics.format_stats(self.stats)
        header = (
            f"Symbol: {self.spec.symbol} | Broker: {self.profile.name} | "
            f"Strategy: {type(self.strategy).__name__} | Params: {self.params}\n"
            f"Elapsed: {self.elapsed:.2f}s\n"
        )
        out = header + text
        if print_it:
            print(out)
        return out

    def to_html(self, path: str = "backtest_report.html", **kwargs) -> str:
        """Write a standalone HTML report with charts and tables.

        Everything is inlined (charts as base64 PNG), so the file works offline
        and can be emailed or committed as-is.
        """
        from .report import backtest_report

        return backtest_report(self, path, **kwargs)

    def to_dict(self) -> dict:
        clean = {}
        for key, value in self.stats.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta, datetime)):
                clean[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                clean[key] = float(value)
            else:
                clean[key] = value
        return {"symbol": self.spec.symbol, "params": self.params, "stats": clean}

    def save(self, folder: str, prefix: str = "backtest", html: bool = False) -> str:
        """Save trades, curve, metrics and specifications into a folder."""
        os.makedirs(folder, exist_ok=True)
        self.trades.to_csv(os.path.join(folder, f"{prefix}_trades.csv"), index=False)
        self.equity.to_csv(os.path.join(folder, f"{prefix}_equity.csv"))
        with open(os.path.join(folder, f"{prefix}_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        self.spec.save(os.path.join(folder, f"{prefix}_symbol.json"))
        self.profile.save(os.path.join(folder, f"{prefix}_broker.json"))
        with open(os.path.join(folder, f"{prefix}_report.txt"), "w", encoding="utf-8") as fh:
            fh.write(self.report(print_it=False))
        if html:
            self.to_html(os.path.join(folder, f"{prefix}_report.html"))
        return folder

    # ----------------------------------------------------------------- charts
    def plot(self, **kwargs):
        from .plotting import plot_dashboard

        return plot_dashboard(self, **kwargs)

    def plot_equity(self, **kwargs):
        from .plotting import plot_equity

        return plot_equity(self, **kwargs)

    def plot_drawdown(self, **kwargs):
        from .plotting import plot_drawdown

        return plot_drawdown(self, **kwargs)

    def plot_trades(self, **kwargs):
        from .plotting import plot_price_trades

        return plot_price_trades(self, **kwargs)

    def plot_monthly(self, **kwargs):
        from .plotting import plot_monthly_heatmap

        return plot_monthly_heatmap(self, **kwargs)

    def plot_distribution(self, **kwargs):
        from .plotting import plot_trade_distribution

        return plot_trade_distribution(self, **kwargs)

    def save_charts(self, folder: str, **kwargs):
        from .plotting import save_all_charts

        return save_all_charts(self, folder, **kwargs)

    # ----------------------------------------------------------- quick montecarlo
    def montecarlo(self, n: int = 1000, **kwargs):
        from .montecarlo import monte_carlo

        return monte_carlo(self, n_simulations=n, **kwargs)


class Backtest:
    """Configure and run a backtest.

    ::

        bt = Backtest(
            data=bars,                        # OHLC DataFrame
            strategy=MyStrategy,              # a Strategy subclass
            symbol=my_spec,                   # SymbolSpec for the instrument
            broker=my_profile,                # BrokerProfile (optional)
            config=SimulationConfig(...),     # realism (optional)
        )
        result = bt.run()
        result.report()
        result.plot()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Union[Type[Strategy], Strategy],
        symbol: SymbolSpec,
        broker: Optional[BrokerProfile] = None,
        config: Optional[SimulationConfig] = None,
        start: Union[str, datetime, None] = None,
        end: Union[str, datetime, None] = None,
        timeframe: Union[str, int, Timeframe, None] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        warmup_bars: int = 0,
        drop_closed_bars: bool = False,
        progress: bool = False,
    ):
        self.spec = symbol
        self.profile = broker or BrokerProfile()
        self.config = config or SimulationConfig()
        self.strategy_class = strategy if isinstance(strategy, type) else type(strategy)
        self.strategy_params = dict(strategy_params or {})
        self.warmup_bars = warmup_bars
        self.progress = progress

        df = prepare_bars(data) if not _is_prepared(data) else data.copy()
        df = slice_dates(df, start, end)
        if timeframe is not None:
            df = resample_bars(df, timeframe)
        if drop_closed_bars and self.spec.quote_sessions.days:
            df = df[self.spec.quote_sessions.filter_index(df.index)]
        if df.empty:
            raise ValueError("No data left after applying the date range / sessions")

        self.timeframe_seconds = _infer_timeframe_seconds(df.index)
        self.data = add_gap_points(df, self.spec.point)

    # ---------------------------------------------------------------------- run
    def run(self, **override_params) -> BacktestResult:
        params = {**self.strategy_params, **override_params}
        started = _time.perf_counter()

        broker = Broker(self.spec, self.config, self.profile, self.timeframe_seconds)
        strategy = self.strategy_class(**params)
        strategy._bind(broker, self.data)
        strategy.init()

        times = self.data.index.to_pydatetime()
        opens = self.data["open"].to_numpy()
        highs = self.data["high"].to_numpy()
        lows = self.data["low"].to_numpy()
        closes = self.data["close"].to_numpy()
        volumes = self.data["volume"].to_numpy()
        spreads = (
            self.data["spread"].to_numpy() if "spread" in self.data.columns
            else np.full(len(self.data), np.nan)
        )
        gaps = self.data["gap_points"].to_numpy()

        n = len(self.data)
        respect = self.config.respect_sessions and bool(self.spec.trade_sessions.days)
        report_every = max(1, n // 20)
        seen_trades = 0
        bar: dict = {}

        for i in range(n):
            bar = {
                "time": times[i], "open": opens[i], "high": highs[i], "low": lows[i],
                "close": closes[i], "volume": volumes[i], "spread": spreads[i],
                "gap_points": gaps[i],
            }
            broker.begin_bar(i, bar)

            if len(broker.trades) > seen_trades:
                for trade in broker.trades[seen_trades:]:
                    strategy.on_trade_closed(trade)
                seen_trades = len(broker.trades)

            if broker.stopped:
                broker.end_bar()
                break

            if i >= self.warmup_bars and (
                not respect or self.spec.trade_sessions.is_open(broker.bar_close_time)
            ):
                strategy._i = i
                strategy.on_bar()

            broker.end_bar()

            if len(broker.trades) > seen_trades:
                for trade in broker.trades[seen_trades:]:
                    strategy.on_trade_closed(trade)
                seen_trades = len(broker.trades)

            if self.progress and i % report_every == 0:
                pct = i / n * 100
                print(f"  {pct:5.1f}%  {times[i]}  equity={broker.equity:,.2f}",
                      end="\r", flush=True)

        broker.finalize(bar)
        strategy.on_finish()
        if self.progress:
            print(" " * 70, end="\r")

        trades_df = pd.DataFrame([t.to_dict() for t in broker.trades])
        equity_df = pd.DataFrame(
            {
                "equity": broker.curve_equity,
                "balance": broker.curve_balance,
                "margin": broker.curve_margin,
                "exposure": broker.curve_exposure,
            },
            index=pd.DatetimeIndex(broker.curve_time, name="time"),
        )
        if equity_df.empty:
            equity_df = pd.DataFrame(
                {"equity": [self.config.initial_balance], "balance": [self.config.initial_balance],
                 "margin": [0.0], "exposure": [0.0]},
                index=pd.DatetimeIndex([self.data.index[0]], name="time"),
            )

        stats = _metrics.compute_stats(
            trades_df, equity_df["equity"], equity_df["balance"],
            initial_balance=self.config.initial_balance,
            exposure=equity_df["exposure"],
        )
        stats["rejections"] = len(broker.rejections)
        stats["stop_reason"] = broker.stop_reason

        return BacktestResult(
            trades=trades_df, equity=equity_df, stats=stats, data=self.data,
            spec=broker.spec, config=self.config, profile=self.profile,
            strategy=strategy, broker=broker,
            elapsed=_time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ helpers
    def optimize(self, param_grid: Dict[str, List[Any]], **kwargs):
        from .optimize import grid_search

        return grid_search(self, param_grid, **kwargs)

    def walk_forward(self, param_grid: Dict[str, List[Any]], **kwargs):
        from .optimize import walk_forward

        return walk_forward(self, param_grid, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<Backtest {self.spec.symbol} {len(self.data):,} bars "
            f"{self.data.index[0]} -> {self.data.index[-1]} "
            f"tf={self.timeframe_seconds}s>"
        )


def _is_prepared(df: pd.DataFrame) -> bool:
    return (
        isinstance(df.index, pd.DatetimeIndex)
        and {"open", "high", "low", "close"}.issubset(df.columns)
    )
