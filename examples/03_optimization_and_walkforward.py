"""Example 3 - Optimization, walk-forward and stress testing.

The full validation cycle for a strategy:
  1. Grid search over the parameters.
  2. Pick a stable region, not the peak (overfitting).
  3. Walk-forward: optimize in sample, validate out of sample.
  4. Stress: does it survive double the spread and more slippage?

Windows note: with ``n_jobs > 1`` the script must be guarded with
``if __name__ == "__main__":`` (done below) and the strategy must live in an
importable module.

    python examples/03_optimization_and_walkforward.py
"""

import numpy as np
import pandas as pd

from marketprolab import (
    Backtest,
    BrokerProfile,
    SimulationConfig,
    Strategy,
    grid_search,
    indicators,
    presets,
    random_search,
    sensitivity,
    stress_test,
    walk_forward,
)


class ChannelBreakout(Strategy):
    """Break of the N-bar high/low, filtered by a slow trend."""

    channel = 40
    trend_filter = 200
    sl_atr = 2.0
    risk_pct = 0.5

    def init(self):
        self.top = self.I(indicators.highest, self.high.full, self.channel)
        self.bottom = self.I(indicators.lowest, self.low.full, self.channel)
        self.trend = self.I(indicators.ema, self.close.full, self.trend_filter)
        self.atr = self.I(indicators.atr, self.high.full, self.low.full, self.close.full, 14)

    def on_bar(self):
        if np.isnan(self.trend[-1]) or np.isnan(self.atr[-1]):
            return
        sl_distance = self.atr[-1] * self.sl_atr
        lots = self.volume_for_risk_pct(self.risk_pct, sl_distance / self.point)
        if lots <= 0:
            return

        breaks_up = self.close[-1] > self.top[-2]
        breaks_down = self.close[-1] < self.bottom[-2]
        bullish = self.close[-1] > self.trend[-1]

        if breaks_up and bullish and not self.has_position("buy"):
            self.close_all()
            self.buy(lots, sl=self.bid - sl_distance, tp=self.bid + sl_distance * 2)
        elif breaks_down and not bullish and not self.has_position("sell"):
            self.close_all()
            self.sell(lots, sl=self.ask + sl_distance, tp=self.ask - sl_distance * 2)


def demo_data(n=30_000):
    rng = np.random.default_rng(11)
    index = pd.date_range("2022-01-03", periods=n, freq="15min")
    close = 1900 + np.cumsum(rng.normal(0, 0.9, n)) + np.sin(np.arange(n) / 800) * 120
    high = close + np.abs(rng.normal(0, 0.7, n))
    low = close - np.abs(rng.normal(0, 0.7, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([high, open_, close]),
         "low": np.minimum.reduce([low, open_, close]), "close": close,
         "volume": rng.integers(50, 800, n)},
        index=index,
    ).rename_axis("time")


def main():
    bars = demo_data()
    spec = presets.xauusd(symbol="XAUUSDz")
    broker = BrokerProfile(name="demo", leverage=100, stop_out_level=50,
                           commission_per_lot=4.0)
    config = SimulationConfig(initial_balance=10_000, spread=25, slippage=4,
                              latency=150, seed=3)

    bt = Backtest(bars, ChannelBreakout, spec, broker, config, warmup_bars=250)

    # ── 1. Full grid ───────────────────────────────────────────────────────
    print("\n=== GRID SEARCH ===")
    opt = grid_search(
        bt,
        {"channel": [20, 30, 40, 60, 80], "sl_atr": [1.5, 2.0, 3.0]},
        objective="sharpe",     # or "net_profit", "calmar", "profit_factor"...
        min_trades=30,          # drop combinations with too few trades
        n_jobs=1,               # raise to 4-8 if you have spare cores
        progress=True,
    )
    print(opt)
    print(opt.top(8)[["channel", "sl_atr", "sharpe", "net_profit", "max_dd_pct", "trades"]])
    opt.plot()   # channel x sl_atr heatmap

    # ── 2. A stable region rather than the peak ────────────────────────────
    print("\n=== TOP BY NEIGHBOURHOOD (less overfitting) ===")
    print(opt.stable_top(5)[["channel", "sl_atr", "sharpe", "neighbourhood_score"]])

    # ── 3. Single-parameter sensitivity ────────────────────────────────────
    print("\n=== SENSITIVITY ===")
    print(sensitivity(bt, "channel", [20, 30, 40, 50, 60, 80, 100],
                      objective="net_profit", progress=False)
          [["channel", "net_profit", "profit_factor", "trades"]])

    # ── 4. Random search over a wider space ────────────────────────────────
    print("\n=== RANDOM SEARCH ===")
    rnd = random_search(
        bt,
        {"channel": (20, 120), "trend_filter": (100, 400), "sl_atr": (1.0, 4.0)},
        n_iter=40, objective="calmar", min_trades=30, progress=True,
    )
    print(rnd.best_params)

    # ── 5. Walk-forward ────────────────────────────────────────────────────
    print("\n=== WALK-FORWARD ===")
    wf = walk_forward(
        bt,
        {"channel": [20, 40, 60], "sl_atr": [1.5, 2.0, 3.0]},
        in_sample_bars=8_000,
        out_sample_bars=2_000,
        objective="sharpe",
        anchored=False,     # True keeps the start fixed and grows the window
        compound=True,      # capital carries over between windows
        min_trades=10,
    )
    print(wf)
    print(wf.windows[["window", "is_metric", "oos_metric", "oos_net_profit", "oos_trades"]])
    wf.plot()

    # ── 6. Cost stress test ────────────────────────────────────────────────
    print("\n=== COST STRESS ===")
    table = stress_test(
        bt,
        spread_multipliers=(1, 1.5, 2, 3),
        slippage_points=(0, 5, 15),     # 0 keeps the base model
        commission_multipliers=(1, 2),
    )
    print(table.pivot_table(index="spread_x", columns="slippage_points", values="net_profit"))

    # ── 7. HTML reports ────────────────────────────────────────────────────
    opt.to_html("results/example3/optimization.html")
    wf.to_html("results/example3/walkforward.html")
    print("\nHTML reports written to results/example3/")

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
