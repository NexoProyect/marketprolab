"""marketprolab - test, optimize and visualise trading strategies in Python.

Built to reproduce **your broker's real conditions**, whatever the instrument
and whoever the broker: every instrument is described by a :class:`SymbolSpec`
(everything MetaTrader 5 shows in its *Specification* window) and every account
by a :class:`BrokerProfile`. Simulation realism - spread, slippage, latency,
intrabar path - is configured separately with :class:`SimulationConfig`.

Quick start::

    from marketprolab import Backtest, Strategy, SymbolSpec, SimulationConfig, indicators

    class Cross(Strategy):
        fast, slow = 20, 50
        def init(self):
            self.f = self.I(indicators.sma, self.close.full, self.fast)
            self.s = self.I(indicators.sma, self.close.full, self.slow)
        def on_bar(self):
            if self.f[-1] > self.s[-1] and not self.has_position("buy"):
                self.close_all(); self.buy(0.1)
            elif self.f[-1] < self.s[-1] and not self.has_position("sell"):
                self.close_all(); self.sell(0.1)

    bt = Backtest(bars, Cross, my_symbol)
    r = bt.run()
    r.report()
    r.plot()
    r.to_html("report.html")
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    data,
    execution,
    indicators,
    metrics,
    montecarlo,
    optimize,
    plotting,
    presets,
    report,
)
from .broker import Broker, SimulationConfig
from .broker_profile import BrokerProfile, SymbolRegistry, broker_profile_preset
from .data import (
    data_quality_report,
    get_bars,
    load_bars,
    load_mt5_bars,
    load_mt5_ticks,
    prepare_bars,
    read_csv_bars,
    resample_bars,
    save_bars,
    ticks_to_bars,
)
from .engine import Backtest, BacktestResult
from .enums import (
    CalcMode,
    ChartMode,
    DealReason,
    ExecutionMode,
    ExpirationMode,
    FillingMode,
    GTCMode,
    IntrabarModel,
    MarginMode,
    OrderType,
    PositionType,
    PriceType,
    SwapType,
    Timeframe,
    TradeMode,
)
from .execution import (
    CallableSlippage,
    CallableSpread,
    DataSpread,
    FixedLatency,
    FixedSlippage,
    FixedSpread,
    GapSlippage,
    NoSlippage,
    RandomLatency,
    RandomSlippage,
    RandomSpread,
    SessionSpread,
    VolatilitySlippage,
    VolatilitySpread,
)
from .metrics import compute_stats, format_stats, monthly_table
from .montecarlo import monte_carlo, monte_carlo_bars, required_capital
from .optimize import compare, grid_search, random_search, sensitivity, stress_test, walk_forward
from .orders import Order, OrderResult, Position, Trade
from .plotting import set_theme
from .presets import get_preset
from .report import (
    backtest_report,
    combined_report,
    comparison_report,
    montecarlo_report,
    optimization_report,
    walk_forward_report,
)
from .sessions import SessionSpec
from .strategy import FunctionStrategy, SignalStrategy, Strategy
from .symbol import SymbolSpec

__all__ = [
    "__version__",
    # core
    "Backtest", "BacktestResult", "Broker", "SimulationConfig",
    "Strategy", "FunctionStrategy", "SignalStrategy",
    "SymbolSpec", "SessionSpec", "BrokerProfile", "SymbolRegistry",
    # orders
    "Order", "OrderResult", "Position", "Trade",
    # enums
    "OrderType", "PositionType", "DealReason", "Timeframe", "CalcMode", "ChartMode",
    "ExecutionMode", "ExpirationMode", "FillingMode", "GTCMode", "IntrabarModel",
    "MarginMode", "PriceType", "SwapType", "TradeMode",
    # execution models
    "FixedSpread", "DataSpread", "RandomSpread", "SessionSpread", "VolatilitySpread",
    "CallableSpread", "CallableSlippage",
    "NoSlippage", "FixedSlippage", "RandomSlippage", "VolatilitySlippage", "GapSlippage",
    "FixedLatency", "RandomLatency",
    # data
    "load_mt5_bars", "load_mt5_ticks", "read_csv_bars", "ticks_to_bars", "prepare_bars",
    "resample_bars", "save_bars", "load_bars", "get_bars", "data_quality_report",
    # analysis
    "compute_stats", "format_stats", "monthly_table",
    "grid_search", "random_search", "walk_forward", "compare", "sensitivity", "stress_test",
    "monte_carlo", "monte_carlo_bars", "required_capital",
    "set_theme", "get_preset", "broker_profile_preset",
    # html reports
    "backtest_report", "optimization_report", "walk_forward_report", "montecarlo_report",
    "comparison_report", "combined_report",
    # modules
    "data", "execution", "indicators", "metrics", "montecarlo", "optimize",
    "plotting", "presets", "report",
]
