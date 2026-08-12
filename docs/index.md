---
title: marketprolab
description: Backtesting under real broker conditions
---

# marketprolab

Test, optimize and visualise trading strategies in Python, under **your
broker's real conditions** - whatever the instrument, whoever the broker.

[Usage guide](usage.md) · [Packaging and publishing](packaging.md) ·
[Source on GitHub](https://github.com/NexoProyect/marketprolab) ·
[Wiki: recipes and FAQ](https://github.com/NexoProyect/marketprolab/wiki)

---

## The idea

Most backtesting libraries assume a world with no spread, no swap, no margin and
instant execution. This one starts from the opposite premise: you describe the
instrument exactly as it appears in MetaTrader 5's *Specification* window,
describe your account, and the simulator behaves accordingly.

```python
from marketprolab import Backtest, Strategy, SimulationConfig, indicators

class Cross(Strategy):
    fast, slow = 20, 50

    def init(self):
        self.f = self.I(indicators.ema, self.close.full, self.fast)
        self.s = self.I(indicators.ema, self.close.full, self.slow)

    def on_bar(self):
        if self.f[-1] > self.s[-1] and not self.has_position("buy"):
            self.close_all()
            self.buy(0.10, sl=self.bid - 3.0, tp=self.bid + 6.0)

r = Backtest(bars, Cross, my_symbol, my_broker, config).run()
r.report()                 # console
r.plot()                   # dashboard
r.to_html("report.html")   # standalone HTML report
r.montecarlo(5000).plot()
```

## Install

```bash
pip install git+https://github.com/NexoProyect/marketprolab.git
```

Python >= 3.9. `MetaTrader5` is an optional Windows-only extra
(`pip install "marketprolab[mt5]"`): the library runs on CSVs or DataFrames on
any operating system.

## What it simulates

- **The full instrument specification**: digits, contract size, margin and profit
  currencies, calculation mode, chart mode, execution type, GTC, filling and
  expiration policies, permitted order types, minimum/maximum/step volume,
  position limits, expiry date.
- **Real costs**: spread (fixed, floating, taken from real ticks, or driven by
  volatility), commission per lot / per deal / percentage, and swap in points,
  money or annual percentage, with triple swap on whichever day your broker says.
- **Margin and leverage**: initial and maintenance margin, margin level, margin
  call, and a stop-out that genuinely liquidates positions.
- **Microstructure**: slippage (fixed, random, volatility-driven, gap-driven) and
  execution latency with spikes, which shifts the fill inside the bar.
- **Schedules**: quote and trade sessions per weekday with intraday breaks,
  session-end and weekend flattening.
- **Broker rules**: hedging or netting, FIFO, position and pending-order limits,
  minimum stop distance.

## What you get

- A report with 50+ metrics: CAGR, drawdown and its duration, Sharpe, Sortino,
  Calmar, SQN, Ulcer, profit factor, expectancy, payoff, streaks, MAE/MFE,
  exposure, long/short breakdown and exits by reason.
- Charts out of the box, in light and dark themes with colourblind-validated
  palettes.
- **Standalone HTML reports** for backtests, optimizations, walk-forward runs and
  Monte Carlo - charts embedded, no server, no internet, one file.
- **Optimization**: grid, random, sensitivity analysis, and ranking by
  neighbourhood stability instead of the peak.
- **Walk-forward** with a rolling or anchored window, and OOS/IS efficiency.
- **Monte Carlo** over trades and over price, with risk of ruin and a suggested
  minimum capital.
- **Cost stress tests**: does it still win with double the spread?
- **Look-ahead is impossible**: reading a future bar raises `IndexError`, and no
  order is ever filled on the bar that generated the signal.

## Documentation

| Page | What is in it |
|---|---|
| [Usage guide](usage.md) | From install to Monte Carlo, plus a section on how the simulation works inside |
| [Packaging and publishing](packaging.md) | How the library is assembled, installed locally, and shipped to PyPI |
| [Wiki](https://github.com/NexoProyect/marketprolab/wiki) | Task-oriented recipes, FAQ and troubleshooting |
| [Examples](https://github.com/NexoProyect/marketprolab/tree/main/examples) | Six runnable scripts |

## A warning

A backtest does not predict the future. It reproduces what **would have
happened** under assumptions you chose, and the quality of the result never
exceeds the quality of those assumptions or of your data. Be pessimistic about
spread, slippage and latency: getting filled cheaply in the simulator is the
most common way to lose money live.

---

MIT licensed. Built by [NexoProyect](https://github.com/NexoProyect).
