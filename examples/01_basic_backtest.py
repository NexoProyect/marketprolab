"""Example 1 - A complete backtest, end to end.

Defines an instrument, a broker profile, a simple strategy, and runs the
backtest under realistic conditions.

    python examples/01_basic_backtest.py
"""

import numpy as np
import pandas as pd

from marketprolab import (
    Backtest,
    BrokerProfile,
    CalcMode,
    RandomLatency,
    RandomSlippage,
    RandomSpread,
    SessionSpec,
    SimulationConfig,
    Strategy,
    SwapType,
    SymbolSpec,
    indicators,
)

# ─────────────────────────────────────────── 1. Data
# Synthetic data here so the example runs without MT5. In real use:
#   from marketprolab import load_mt5_bars
#   bars = load_mt5_bars("EURUSD", "M15", "2023-01-01", "2025-01-01")
rng = np.random.default_rng(42)
n = 20_000
index = pd.date_range("2023-01-02", periods=n, freq="15min")
close = 1.0800 + np.cumsum(rng.normal(0, 0.00035, n))
high = close + np.abs(rng.normal(0, 0.0003, n))
low = close - np.abs(rng.normal(0, 0.0003, n))
open_ = np.r_[close[0], close[:-1]]
bars = pd.DataFrame(
    {
        "open": open_,
        "high": np.maximum.reduce([high, open_, close]),
        "low": np.minimum.reduce([low, open_, close]),
        "close": close,
        "volume": rng.integers(100, 900, n),
    },
    index=index,
).rename_axis("time")


# ─────────────────────────────────────────── 2. The instrument (YOUR broker, YOUR symbol)
symbol = SymbolSpec(
    symbol="EURUSD",
    name="Euro vs US Dollar",
    category="Forex",
    digits=5,
    contract_size=100_000,
    margin_currency="EUR",
    profit_currency="USD",
    calc_mode=CalcMode.FOREX,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    stops_level=0,
    spread_points=12,              # average spread, in points
    commission_per_lot=3.5,        # per lot, per side
    swap_type=SwapType.POINTS,
    swap_long=-7.2,
    swap_short=-1.4,
    swap_rate_days={0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0},  # triple on Wednesday
    leverage=100,
    trade_sessions=SessionSpec.from_dict(
        {
            "sunday": "22:05-24:00",
            "monday": "00:00-24:00",
            "tuesday": "00:00-24:00",
            "wednesday": "00:00-24:00",
            "thursday": "00:00-24:00",
            "friday": "00:00-20:55",
        }
    ),
    timezone="Etc/GMT-3",
)

# ─────────────────────────────────────────── 3. The account / broker
broker = BrokerProfile(
    name="My broker",
    account_currency="USD",
    leverage=100,
    stop_out_level=50.0,      # forced liquidation below 50% margin level
    margin_call_level=100.0,
)

# ─────────────────────────────────────────── 4. Simulation realism
config = SimulationConfig(
    initial_balance=10_000,
    spread=RandomSpread(mean_points=12, sigma=0.30),            # floating spread
    slippage=RandomSlippage(mean_points=2, sigma_points=3),     # execution slippage
    latency=RandomLatency(mean_ms=140, spike_probability=0.01),  # real-world delay
    respect_sessions=True,
    apply_swap=True,
    seed=7,
)


# ─────────────────────────────────────────── 5. The strategy
class MovingAverageCross(Strategy):
    """Moving-average cross with an ATR stop and a trailing stop."""

    fast = 20
    slow = 60
    atr_period = 14
    atr_sl = 2.0
    atr_tp = 4.0
    risk_pct = 1.0

    def init(self):
        self.ma_fast = self.I(indicators.ema, self.close.full, self.fast)
        self.ma_slow = self.I(indicators.ema, self.close.full, self.slow)
        self.atr = self.I(
            indicators.atr, self.high.full, self.low.full, self.close.full, self.atr_period
        )

    def on_bar(self):
        if np.isnan(self.ma_slow[-1]) or np.isnan(self.atr[-1]):
            return

        sl_distance = self.atr[-1] * self.atr_sl
        tp_distance = self.atr[-1] * self.atr_tp
        sl_points = sl_distance / self.point
        lots = self.volume_for_risk_pct(self.risk_pct, sl_points)
        if lots <= 0:
            return

        crossed_up = self.ma_fast[-1] > self.ma_slow[-1] and self.ma_fast[-2] <= self.ma_slow[-2]
        crossed_down = self.ma_fast[-1] < self.ma_slow[-1] and self.ma_fast[-2] >= self.ma_slow[-2]

        if crossed_up:
            self.close_all()
            self.buy(lots, sl=self.bid - sl_distance, tp=self.bid + tp_distance,
                     comment="cross up")
        elif crossed_down:
            self.close_all()
            self.sell(lots, sl=self.ask + sl_distance, tp=self.ask - tp_distance,
                      comment="cross down")

        # Management: break-even at 1 ATR, then trail at the stop distance
        self.break_even(trigger_points=self.atr[-1] / self.point, offset_points=5)
        self.trailing_stop(distance_points=sl_points)


# ─────────────────────────────────────────── 6. Run
if __name__ == "__main__":
    bt = Backtest(
        data=bars,
        strategy=MovingAverageCross,
        symbol=symbol,
        broker=broker,
        config=config,
        warmup_bars=100,
        progress=True,
    )
    result = bt.run()
    result.report()

    print("\nLast trades:")
    print(result.trades.tail()[["type", "volume", "open_price", "close_price",
                                "net_profit", "reason"]])

    # Save everything: csv + json + txt + charts + an HTML report
    result.save("results/example1", html=True)
    result.save_charts("results/example1/charts")
    print("\nSaved to results/example1 (open backtest_report.html)")

    # Show a chart on screen
    import matplotlib.pyplot as plt

    result.plot()
    plt.show()
