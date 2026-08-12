"""Example 6 - Exporting HTML reports.

Every result object can write a standalone ``.html`` file: charts are embedded
as base64 PNGs and the CSS is inlined, so the report opens offline, survives
being emailed, and needs no server.

    python examples/06_html_reports.py
"""

import numpy as np
import pandas as pd

from marketprolab import (
    Backtest,
    BrokerProfile,
    SimulationConfig,
    Strategy,
    combined_report,
    comparison_report,
    grid_search,
    indicators,
    monte_carlo,
    presets,
    walk_forward,
)


class Cross(Strategy):
    fast = 20
    slow = 60
    risk_pct = 1.0
    sl_points = 3000
    tp_points = 6000

    def init(self):
        self.f = self.I(indicators.sma, self.close.full, self.fast)
        self.s = self.I(indicators.sma, self.close.full, self.slow)

    def on_bar(self):
        if np.isnan(self.s[-1]):
            return
        price = self.bid
        lots = self.volume_for_risk_pct(self.risk_pct, self.sl_points)
        if self.f[-1] > self.s[-1] and self.f[-2] <= self.s[-2]:
            self.close_all()
            self.buy(lots, sl=price - self.sl_points * self.point,
                     tp=price + self.tp_points * self.point)
        elif self.f[-1] < self.s[-1] and self.f[-2] >= self.s[-2]:
            self.close_all()
            self.sell(lots, sl=price + self.sl_points * self.point,
                      tp=price - self.tp_points * self.point)


def demo_data(n=12_000):
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 2000 + rng.normal(0, 1.2, n).cumsum() + np.sin(np.arange(n) / 300) * 40
    high = close + np.abs(rng.normal(0, 0.8, n))
    low = close - np.abs(rng.normal(0, 0.8, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": np.maximum.reduce([high, open_, close]),
         "low": np.minimum.reduce([low, open_, close]), "close": close,
         "volume": rng.integers(50, 500, n)},
        index=index,
    ).rename_axis("time")


def main():
    bars = demo_data()
    bt = Backtest(
        bars, Cross, presets.xauusd(symbol="XAUUSDz"),
        BrokerProfile(name="demo broker", leverage=100, stop_out_level=50,
                      commission_per_lot=4.0),
        SimulationConfig(initial_balance=10_000, spread=25, slippage=4, latency=150, seed=1),
        warmup_bars=80,
    )
    result = bt.run()
    opt = grid_search(bt, {"fast": [10, 20, 30], "slow": [50, 80]},
                      objective="sharpe", min_trades=5, progress=False)
    wf = walk_forward(bt, {"fast": [10, 20], "slow": [50, 80]},
                      in_sample_bars=5_000, out_sample_bars=2_000,
                      objective="sharpe", min_trades=2, progress=False)
    mc = monte_carlo(result, n_simulations=5_000, method="bootstrap", dd_threshold_pct=20)

    # ── One report per artefact ────────────────────────────────────────────
    print(result.to_html("results/html/backtest.html"))
    print(opt.to_html("results/html/optimization.html"))
    print(wf.to_html("results/html/walkforward.html"))
    print(mc.to_html("results/html/montecarlo.html"))

    # ── Everything on a single page ────────────────────────────────────────
    print(combined_report(result, "results/html/overall.html", opt=opt, wf=wf, mc=mc))

    # ── Comparing several runs ─────────────────────────────────────────────
    fast_variant = bt.run(fast=10, slow=80)
    print(comparison_report(
        {"baseline (20/60)": result, "faster (10/80)": fast_variant},
        "results/html/comparison.html",
    ))

    # ── Options ────────────────────────────────────────────────────────────
    # theme="dark"          dark page and dark charts
    # charts=False          tables only, a much smaller file
    # max_trades=1000       how many trade rows to include
    # dpi=160               sharper charts, bigger file
    # open_browser=True     open it as soon as it is written
    print(result.to_html("results/html/backtest_dark.html", theme="dark", dpi=140))
    print(result.to_html("results/html/backtest_light_tables.html", charts=False))

    # save(..., html=True) writes csv + json + txt + the HTML report together
    result.save("results/html/full_run", html=True)
    print("\nAll reports written to results/html/")


if __name__ == "__main__":
    main()
