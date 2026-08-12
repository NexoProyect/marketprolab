"""Example 2 - Real data: from the MT5 terminal and from a tick CSV.

Three ways to get data:
  A) Download bars from the MetaTrader 5 terminal.
  B) Read the symbol specification straight from the terminal.
  C) Turn a huge tick CSV (several GB) into bars with the real spread.

    python examples/02_mt5_and_tick_data.py
"""

import os

from marketprolab import (
    Backtest,
    BrokerProfile,
    DataSpread,
    RandomLatency,
    RandomSlippage,
    SignalStrategy,
    SimulationConfig,
    SymbolRegistry,
    SymbolSpec,
    Timeframe,
    data_quality_report,
    indicators,
    load_mt5_bars,
    presets,
    ticks_to_bars,
)

SYMBOL = "XAUUSDz"
TICK_CSV = "data/XAUUSDz.csv"       # MT5 tick export (tab separated)
CACHE = ".cache/XAUUSDz_M5.parquet"


# ──────────────────────────────────── A) Bars from the MT5 terminal
def from_mt5():
    bars = load_mt5_bars(SYMBOL, Timeframe.M5, start="2024-01-01", end="2026-01-01")
    print(bars.tail())
    print(data_quality_report(bars, Timeframe.M5))
    return bars


# ──────────────────────────────────── B) Specification from the terminal
def spec_from_mt5() -> SymbolSpec:
    """Reads digits, contract size, swaps, volumes, etc. from the live broker."""
    spec = SymbolSpec.from_mt5(SYMBOL)
    print(spec.summary())

    # Store it in a catalogue so the terminal is not needed next time
    registry = SymbolRegistry("symbols/my_broker")
    registry.add(spec)
    print("Stored symbols:", registry.list())
    return spec


# ──────────────────────────────────── C) Bars built from a tick CSV
def from_ticks():
    """Convert ticks into M5 bars with the spread measured tick by tick.

    Reads in chunks, so multi-gigabyte files are fine. The result is cached to
    parquet: the second run is instant.
    """
    bars = ticks_to_bars(
        TICK_CSV,
        timeframe=Timeframe.M5,
        price="bid",           # MT5 charts are drawn on the bid
        digits=3,              # so the spread can be expressed in points
        cache=CACHE,
        chunksize=2_000_000,
        progress=True,
    )
    print(bars.head())
    print("median spread (points):", bars["spread"].median())
    return bars


if __name__ == "__main__":
    # Use the tick CSV when it is there; otherwise pull from the terminal.
    bars = from_ticks() if os.path.exists(TICK_CSV) else from_mt5()

    # Specification from the terminal when available, otherwise by hand.
    try:
        spec = spec_from_mt5()
    except Exception as exc:
        print(f"MT5 unavailable ({exc}); falling back to a manual specification")
        spec = presets.xauusd(symbol=SYMBOL)

    config = SimulationConfig(
        initial_balance=10_000,
        # With tick data every bar carries the REAL spread: use it.
        spread=DataSpread(fallback_points=spec.spread_points),
        slippage=RandomSlippage(mean_points=8, sigma_points=12),
        latency=RandomLatency(mean_ms=180),
        seed=1,
    )

    # Signals computed outside the strategy (vectorised)
    rsi = indicators.rsi(bars["close"], 14)
    longs = rsi < 30
    shorts = rsi > 70

    bt = Backtest(
        bars,
        SignalStrategy,
        spec,
        BrokerProfile(name="my broker", leverage=100, stop_out_level=50),
        config,
        strategy_params={
            "long_signal": longs,
            "short_signal": shorts,
            "lots": 0.05,
            "sl_points": 2000,
            "tp_points": 3000,
        },
        warmup_bars=50,
        progress=True,
    )
    result = bt.run()
    result.report()
    result.to_html("results/example2/report.html")
    print("HTML report: results/example2/report.html")
