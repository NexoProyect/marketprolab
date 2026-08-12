"""Example 4 - Monte Carlo: how much drawdown should you be ready to sit through?

A backtest is a single sample. Monte Carlo reshuffles and resamples the trades
to estimate the realistic range of outcomes, the bad-tail drawdown, and the
minimum capital you need.

    python examples/04_monte_carlo.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketprolab import (
    Backtest,
    BrokerProfile,
    SimulationConfig,
    Strategy,
    indicators,
    monte_carlo,
    monte_carlo_bars,
    presets,
    required_capital,
)


class MeanReversion(Strategy):
    period = 20
    deviations = 2.0
    sl_points = 2500
    tp_points = 1500
    lots = 0.05

    def init(self):
        self.upper, self.middle, self.lower = self.I(
            indicators.bollinger, self.close.full, self.period, self.deviations
        )

    def on_bar(self):
        if np.isnan(self.middle[-1]) or self.has_position():
            return
        if self.close[-1] < self.lower[-1]:
            self.buy(self.lots,
                     sl=self.bid - self.sl_points * self.point,
                     tp=self.bid + self.tp_points * self.point)
        elif self.close[-1] > self.upper[-1]:
            self.sell(self.lots,
                      sl=self.ask + self.sl_points * self.point,
                      tp=self.ask - self.tp_points * self.point)


def demo_data(n=25_000):
    rng = np.random.default_rng(5)
    index = pd.date_range("2023-01-02", periods=n, freq="15min")
    close = 1950 + np.cumsum(rng.normal(0, 0.8, n))
    high = close + np.abs(rng.normal(0, 0.6, n))
    low = close - np.abs(rng.normal(0, 0.6, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([high, open_, close]),
         "low": np.minimum.reduce([low, open_, close]), "close": close,
         "volume": rng.integers(50, 700, n)},
        index=index,
    ).rename_axis("time")


if __name__ == "__main__":
    bars = demo_data()
    spec = presets.xauusd(symbol="XAUUSDz")
    bt = Backtest(bars, MeanReversion, spec,
                  BrokerProfile(name="demo", leverage=100, stop_out_level=50),
                  SimulationConfig(initial_balance=10_000, spread=25, slippage=5,
                                   latency=150, seed=9),
                  warmup_bars=50)
    result = bt.run()
    result.report()

    # ── 1. Monte Carlo over the trades ─────────────────────────────────────
    mc = monte_carlo(
        result,
        n_simulations=5_000,
        method="bootstrap",       # shuffle | bootstrap | block | normal
        dd_threshold_pct=20,      # probability of exceeding this drawdown
        ruin_level_pct=50,        # "ruin" = losing half the account
        seed=1,
    )
    mc.summary()
    mc.plot()

    # ── 2. Pessimistic scenario: missed signals and worse costs ────────────
    mc_bad = monte_carlo(
        result,
        n_simulations=5_000,
        method="bootstrap",
        skip_probability=0.10,       # you miss 1 signal in 10
        extra_cost_per_trade=3.0,    # 3 units more cost per trade
        pnl_noise_pct=15,            # +/-15% noise on every result
        dd_threshold_pct=20,
    )
    print("\n--- Pessimistic scenario ---")
    mc_bad.summary()

    # ── 3. Capital requirement ─────────────────────────────────────────────
    print("\n--- Suggested capital ---")
    print(required_capital(mc, confidence=95, safety_factor=1.5))
    print(required_capital(mc, confidence=99, safety_factor=2.0))

    # ── 4. Monte Carlo over PRICE (slower, more honest) ────────────────────
    # Builds synthetic price series and re-runs the whole strategy on each.
    print("\n--- Monte Carlo over the price data ---")
    table = monte_carlo_bars(bt, n_simulations=20, method="block", block_size=48)
    print(table.describe()[["net_profit", "max_dd_pct", "profit_factor", "trades"]])

    mc.to_html("results/example4/montecarlo.html")
    print("\nHTML report: results/example4/montecarlo.html")
    plt.show()
