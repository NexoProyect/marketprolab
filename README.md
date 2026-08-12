# marketprolab

Test, optimize and visualise trading strategies in Python, under **your broker's
real conditions** - whatever the instrument, whoever the broker.

Most backtesting libraries assume a world with no spread, no swap, no margin and
instant execution. This one starts from the opposite premise: you describe the
instrument exactly as it appears in MetaTrader 5's *Specification* window,
describe your account, and the simulator behaves accordingly.

```python
from marketprolab import Backtest, Strategy, SymbolSpec, SimulationConfig, indicators

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

## What it simulates

- **The full instrument specification**: digits, contract size, margin and profit
  currencies, calculation mode, chart mode, execution type, GTC, filling and
  expiration policies, permitted order types, minimum/maximum/step volume,
  position limits, expiry date.
- **Real costs**: spread (fixed, floating, taken from real ticks, or driven by
  volatility), commission per lot / per deal / percentage, and swap in points,
  money or annual percentage, with triple swap on whichever day your broker says.
- **Margin and leverage**: initial and maintenance margin, margin level, margin
  call, and a **stop-out** that genuinely liquidates positions.
- **Microstructure**: slippage (fixed, random, volatility-driven, gap-driven) and
  execution **latency** with spikes, which shifts the fill inside the bar.
- **Schedules**: quote and trade sessions per weekday with intraday breaks,
  session-end and weekend flattening.
- **Broker rules**: hedging or netting, FIFO, position and pending-order limits,
  minimum stop distance.

## What you get

- A report with 50+ metrics: CAGR, drawdown and its duration, Sharpe, Sortino,
  Calmar, SQN, Ulcer, profit factor, expectancy, payoff, streaks, MAE/MFE,
  exposure, long/short breakdown and exits by reason.
- Charts out of the box: equity, drawdown, trades over price, outcome
  distribution, monthly heatmap and a full dashboard. Light and dark themes with
  colourblind-validated palettes.
- **Standalone HTML reports** for backtests, optimizations, walk-forward runs and
  Monte Carlo - charts embedded, no server, no internet, one file.
- **Optimization**: grid, random, sensitivity analysis, and ranking by
  neighbourhood stability instead of the peak (which is nearly always overfitting).
- **Walk-forward** with a rolling or anchored window, and OOS/IS efficiency.
- **Monte Carlo** over trades (shuffle, bootstrap, blocks) and over price, with
  risk of ruin and a suggested minimum capital.
- **Cost stress tests**: does it still win with double the spread?
- **Look-ahead is impossible**: reading a future bar raises `IndexError`, and no
  order is ever filled on the bar that generated the signal.

## Install

```bash
pip install -e .            # from the project folder
pip install -e ".[all]"     # + MetaTrader5 (Windows) and parquet
```

Python >= 3.9. `MetaTrader5` is optional: the library runs on CSVs or DataFrames
on any operating system.

## Data

```python
from marketprolab import load_mt5_bars, read_csv_bars, ticks_to_bars

bars = load_mt5_bars("XAUUSDz", "M15", "2024-01-01", "2026-01-01")   # from the terminal
bars = read_csv_bars("data.csv", timeframe="M15")                     # from a CSV
bars = ticks_to_bars("ticks.csv", "M5", digits=3,                     # from ticks (GB-scale)
                     cache=".cache/gold_m5.parquet")
```

`ticks_to_bars` reads in chunks, so a multi-gigabyte CSV will not exhaust your
RAM, and it derives each bar's real spread from `ask - bid`.

## Documentation

- [**Usage guide**](docs/usage.md) - from install to Monte Carlo, everything in
  between, and a section on how the simulation works inside.
- [**Packaging and publishing**](docs/packaging.md) - how the library is
  assembled, how to install it locally, and how to ship it to PyPI.
- [**Runnable examples**](examples/) - six complete scripts.

## Status

Version 0.1.0. Test suite runs with `pytest -q`.

## A warning

A backtest does not predict the future. It reproduces what **would have
happened** under assumptions you chose, and the quality of the result never
exceeds the quality of those assumptions or of your data. Be pessimistic about
spread, slippage and latency: getting filled cheaply in the simulator is the
most common way to lose money live.

## Licence

MIT.
