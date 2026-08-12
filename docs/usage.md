# marketprolab - usage guide

A backtesting library that reproduces **your broker's real conditions**,
whatever the instrument and whoever the broker. Nothing is tied to a particular
market: you describe the instrument and the account, and the simulator behaves
accordingly.

---

## Contents

1. [Install](#1-install)
2. [The three objects that control everything](#2-the-three-objects-that-control-everything)
3. [Describing an instrument (`SymbolSpec`)](#3-describing-an-instrument-symbolspec)
4. [Describing the broker (`BrokerProfile`)](#4-describing-the-broker-brokerprofile)
5. [Simulation realism (`SimulationConfig`)](#5-simulation-realism-simulationconfig)
6. [Data: MT5, bar CSVs and tick CSVs](#6-data-mt5-bar-csvs-and-tick-csvs)
7. [Writing a strategy](#7-writing-a-strategy)
8. [Running the backtest and reading the report](#8-running-the-backtest-and-reading-the-report)
9. [Charts](#9-charts)
10. [HTML reports](#10-html-reports)
11. [Optimization](#11-optimization)
12. [Walk-forward](#12-walk-forward)
13. [Monte Carlo](#13-monte-carlo)
14. [Stress tests and comparisons](#14-stress-tests-and-comparisons)
15. [How the simulation works inside](#15-how-the-simulation-works-inside)
16. [Common mistakes](#16-common-mistakes)
17. [API quick reference](#17-api-quick-reference)

---

## 1. Install

```bash
# From the project folder, editable (what you want while developing)
pip install -e .

# With MetaTrader 5 and parquet support
pip install -e ".[all]"
```

Requires Python >= 3.9, `numpy`, `pandas`, `matplotlib`. The `MetaTrader5`
package is **optional** and Windows-only: the library works without it as long
as you feed it data from a CSV or a DataFrame.

Quick check:

```python
import marketprolab as mpl
print(mpl.__version__)
```

---

## 2. The three objects that control everything

| Object | What it describes | Example fields |
|---|---|---|
| `SymbolSpec` | **The instrument**: what MetaTrader 5 shows in its *Specification* window | digits, contract size, currencies, swap, volumes, sessions |
| `BrokerProfile` | **The account**: what changes between brokers for the same instrument | leverage, stop-out, commission, netting/hedging, timezone |
| `SimulationConfig` | **Simulation quality**: what does not come from the broker | spread, slippage, latency, intrabar path, seed |

They come together like this:

```python
bt = Backtest(data=bars, strategy=MyStrategy, symbol=spec,
              broker=profile, config=config)
result = bt.run()
```

`BrokerProfile.apply()` runs automatically on the `SymbolSpec` at startup: the
broker's leverage, commission and stress multipliers override whatever the
instrument carries.

---

## 3. Describing an instrument (`SymbolSpec`)

Three equivalent ways.

### a) By hand - works for any market

```python
from marketprolab import SymbolSpec, SessionSpec, CalcMode, SwapType

spec = SymbolSpec(
    # Identity
    symbol="XAUUSDz",
    name="XAU/USD, Gold vs US Dollar",
    category="Metals",
    asset_type="cfd",                 # spot | futures | cfd | etf | stock

    # Contract
    digits=3,                         # decimals in the quote
    contract_size=100,                # units per lot
    margin_currency="XAU",            # margin (collateral) currency
    profit_currency="USD",            # profit / quote currency
    calc_mode=CalcMode.FOREX,         # FOREX | CFD | CFD_INDEX | FUTURES | CRYPTO...
    chart_mode="bid",                 # chart drawn on bid or last
    tick_size=None,                   # None = one point

    # Trading
    execution="market",               # market | instant | request | exchange
    gtc_mode="gtc",                   # gtc | daily
    filling_modes=["fok", "ioc"],     # accepted filling policies
    expiration_modes=["all"],
    trade_mode="full",                # full | close_only | long_only | disabled
    stops_level=0,                    # minimum SL/TP distance, in points
    freeze_level=0,

    # Volume
    volume_min=0.01, volume_max=200, volume_step=0.01,
    volume_limit=0,                   # max aggregate volume (0 = no limit)

    # Costs and financing
    spread_points=25, spread_float=True,
    commission_per_lot=0.0,           # per lot, per side
    commission_per_deal=0.0,          # flat per deal
    commission_percent=0.0,           # % of notional, per side
    swap_type=SwapType.POINTS,        # POINTS | MONEY | PERCENT_ANNUAL | PERCENT_CURRENT
    swap_long=-531.6, swap_short=0.0,
    swap_rate_days={0:1, 1:1, 2:3, 3:1, 4:1, 5:0, 6:0},   # triple on Wednesday
    swap_rollover_hour=0,

    # Margin and leverage
    leverage=100,
    margin_initial_rate=1.0,
    margin_maintenance_rate=1.0,
    margin_initial_per_lot=None,      # flat override (typical for futures)
    max_leverage=None,

    # Schedule
    quote_sessions=SessionSpec.from_dict({
        "sunday":    "22:01-24:00",
        "monday":    "00:00-20:58, 22:00-24:00",
        "tuesday":   "00:00-20:58, 22:00-24:00",
        "wednesday": "00:00-20:58, 22:00-24:00",
        "thursday":  "00:00-20:58, 22:00-24:00",
        "friday":    "00:00-20:58",
    }),
    trade_sessions=...,               # usually the same schedule
    timezone="Etc/GMT-3",             # server timezone, informational

    # Other
    expiration_date=None,             # futures
    position_limit=0, pending_orders_limit=0,
    country_restrictions=[], notes="",
)

print(spec.summary())
```

### b) Read from the MetaTrader 5 terminal

```python
spec = SymbolSpec.from_mt5("XAUUSDz")     # needs MT5 installed and running
spec.save("symbols/XAUUSDz.json")
```

It brings back digits, contract size, currencies, volumes, swaps, filling
modes, stops level and the account leverage. **Sessions and the average spread
are worth reviewing by hand**, because MT5 only exposes the current spread.

### c) From a saved JSON or a catalogue

```python
from marketprolab import SymbolSpec, SymbolRegistry

spec = SymbolSpec.load("symbols/XAUUSDz.json")

registry = SymbolRegistry("symbols/my_broker", profile=my_profile)
registry.import_from_mt5(["EURUSD", "XAUUSDz", "US500"])   # dump several at once
spec = registry.get("EURUSD")        # already carrying the broker's conditions
```

### Starting templates (optional)

```python
from marketprolab import presets
spec = presets.xauusd(symbol="XAUUSDz")     # metal
spec = presets.forex_major("GBPUSD")
spec = presets.index_cfd("US500")
spec = presets.crypto("BTCUSD")
spec = presets.energy("USOIL")
```

These are only starting points with sensible values: **tune them to your broker.**

### Useful instrument methods

```python
spec.point                      # 10**-digits
spec.pip                        # 10 points when digits is 3 or 5
spec.normalize_price(1994.4567) # round to the tick
spec.normalize_volume(0.237)    # snap to the step and min/max -> 0.23
spec.point_value(volume=1)      # money moved by one point with one lot
spec.profit(+1, 1.0, 2000, 2010)      # gross profit of a long
spec.margin_required(1.0, 2000)       # initial margin for one lot
spec.commission(1.0, 2000)            # commission for one side
spec.swap_cost(+1, 1.0, 2000, when)   # swap for one rollover
```

---

## 4. Describing the broker (`BrokerProfile`)

```python
from marketprolab import BrokerProfile, MarginMode

profile = BrokerProfile(
    name="My ECN broker",
    account_currency="USD",
    leverage=100,
    margin_mode=MarginMode.HEDGING,   # HEDGING or NETTING
    margin_call_level=100.0,          # % margin level
    stop_out_level=50.0,              # % (or money if stop_out_in_money=True)
    server_timezone="Etc/GMT-3",
    swap_rollover_hour=0,
    swap_triple_weekday=2,            # Wednesday

    commission_per_lot=3.5,           # overrides the instrument's
    min_stops_level_points=None,
    hedging_allowed=True,
    max_positions=0,                  # 0 = no limit
    max_pending_orders=0,
    max_volume_total=0.0,
    weekend_close=False,              # flatten before the weekend
    fifo_only=False,

    # Multipliers for stressing a backtest
    spread_multiplier=1.0,
    swap_multiplier=1.0,
    commission_multiplier=1.0,
)
```

Comparing one instrument across two brokers is then trivial:

```python
for profile in (broker_a, broker_b):
    r = Backtest(bars, MyStrategy, spec, profile, config).run()
    print(profile.name, r.stats["net_profit"], r.stats["max_dd_pct"])
```

Templates: `broker_profile_preset("retail_ecn" | "retail_standard" | "prop_firm" | "us_regulated")`.

---

## 5. Simulation realism (`SimulationConfig`)

```python
from marketprolab import (SimulationConfig, RandomSpread, DataSpread,
                          RandomSlippage, VolatilitySlippage, RandomLatency,
                          IntrabarModel)

config = SimulationConfig(
    initial_balance=10_000,

    # Spread: a number (fixed points), "data", or a model
    spread=RandomSpread(mean_points=25, sigma=0.35),

    # Slippage: a number (fixed adverse points), or a model
    slippage=RandomSlippage(mean_points=3, sigma_points=4, only_adverse=False),

    # Latency: milliseconds, or a model
    latency=RandomLatency(mean_ms=140, sigma=0.6, spike_probability=0.01),

    # Intrabar path when SL and TP are both touched on the same bar
    intrabar=IntrabarModel.PESSIMISTIC,   # PESSIMISTIC | OPTIMISTIC | OHLC | OLHC
    same_bar_exit=True,          # a position can close on its own bar
    allow_same_bar_fill=False,   # False = never filled on the signal's bar
    latency_price_drift=True,    # price moves while the order travels
    slippage_on_limit=False,     # limit orders do not slip adversely

    # Sessions
    respect_sessions=True,       # no trading outside trading hours
    close_on_session_end=False,
    close_on_weekend=False,

    # Costs
    apply_swap=True,
    apply_commission=True,

    # Profit currency -> account currency conversion
    profit_currency_rate=1.0,    # number, pandas.Series by date, or callable(dt)

    # Risk and end of test
    stop_on_bankruptcy=True,
    bankruptcy_equity=0.0,
    close_open_positions_at_end=True,

    seed=42,                     # exact reproducibility
    verbose=False,
)
```

### Spread models

| Model | When to use it |
|---|---|
| `FixedSpread(points)` | Clean comparisons, or a genuinely fixed spread |
| `DataSpread()` | You have a real `spread` column (MT5 bars or ticks) |
| `RandomSpread(mean, sigma)` | Generic floating spread, lognormal |
| `SessionSpread(normal, wide, wide_hours)` | Widens at rollover and in thin hours |
| `VolatilitySpread(base, range_factor)` | Proportional to the bar's range |

### Slippage models

| Model | When to use it |
|---|---|
| `NoSlippage()` | Theoretical baseline |
| `FixedSlippage(points)` | Constant worst case |
| `RandomSlippage(mean, sigma)` | Realistic for market execution |
| `VolatilitySlippage(range_factor, volume_factor, stop_multiplier)` | Fast bars and big lots fill worse; stops worse still |
| `GapSlippage(normal, gap_factor)` | Punishes opening gaps (Sunday-night stops) |

Convention: **positive means a worse price for you**. A long enters more
expensive and exits cheaper; a short the other way round.

### Latency models

| Model | When to use it |
|---|---|
| `FixedLatency(milliseconds)` | Constant delay (a stable VPS) |
| `RandomLatency(mean_ms, sigma, spike_probability, spike_ms)` | Lognormal with spikes: what actually happens |

Latency shifts *when* the order executes. With `latency_price_drift=True`, the
fill price is interpolated inside the bar according to the delay, so "losing
300 ms" costs money on a directional bar.

---

## 6. Data: MT5, bar CSVs and tick CSVs

Canonical internal format: a `DatetimeIndex` named `time` and columns
`open, high, low, close, volume` (plus an optional `spread` in points).

### From the MetaTrader 5 terminal

```python
from marketprolab import load_mt5_bars, load_mt5_ticks, Timeframe

bars = load_mt5_bars("XAUUSDz", Timeframe.M15, start="2023-01-01", end="2026-01-01")
bars = load_mt5_bars("XAUUSDz", "H1", count=50_000)     # the last N bars
ticks = load_mt5_ticks("XAUUSDz", "2026-01-01", "2026-01-05")
```

### From a bar CSV

```python
from marketprolab import read_csv_bars
bars = read_csv_bars("data.csv", start="2024-01-01", timeframe="M15")
```

It recognises the MT5 export layout
(`<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`),
separated by tabs, semicolons or commas.

### From a tick CSV, even a multi-gigabyte one

```python
from marketprolab import ticks_to_bars

bars = ticks_to_bars(
    "data/XAUUSDz.csv",          # <DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>
    timeframe="M5",
    price="bid",                 # bid | ask | mid | last
    digits=3,                    # so the spread can be expressed in points
    cache=".cache/XAUUSDz_M5.parquet",   # second run is instant
    chunksize=2_000_000,
)
```

It reads in chunks and aggregates as it goes, so memory stays flat. It also
computes the **real average spread of each bar** from `ask - bid`, which you can
then feed to `DataSpread()`. That is the most faithful way to backtest.

### Utilities

```python
from marketprolab import resample_bars, save_bars, load_bars, data_quality_report

h1 = resample_bars(bars, "H1")
save_bars(h1, ".cache/gold_h1.parquet")
print(data_quality_report(bars, "M5"))   # gaps, duplicates, impossible OHLC
```

Run `data_quality_report` **before** trusting any result: a CSV with gaps or
invalid bars produces beautiful, false backtests.

---

## 7. Writing a strategy

```python
from marketprolab import Strategy, indicators
import numpy as np

class MyStrategy(Strategy):
    # Class attributes are the tunable parameters
    fast = 20
    slow = 50
    risk_pct = 1.0

    def init(self):
        """Runs once. Pre-compute indicators here (vectorised)."""
        self.ma_f = self.I(indicators.ema, self.close.full, self.fast)
        self.ma_s = self.I(indicators.ema, self.close.full, self.slow)
        self.atr  = self.I(indicators.atr, self.high.full, self.low.full,
                           self.close.full, 14)

    def on_bar(self):
        """Runs at the close of every bar."""
        if np.isnan(self.ma_s[-1]):
            return
        sl = self.atr[-1] * 2
        lots = self.volume_for_risk_pct(self.risk_pct, sl / self.point)

        if self.ma_f[-1] > self.ma_s[-1] and self.ma_f[-2] <= self.ma_s[-2]:
            self.close_all()
            self.buy(lots, sl=self.bid - sl, tp=self.bid + sl * 2)

        self.trailing_stop(distance_points=sl / self.point)

    def on_trade_closed(self, trade):
        """Optional: called whenever a trade closes."""

    def on_finish(self):
        """Optional: called when the backtest ends."""
```

> **Reserved names.** The engine puts `open`, `high`, `low`, `close`, `volume`,
> `data` and `broker` on every strategy instance. A parameter cannot use those
> names - use `lots` instead of `volume`, for instance. Doing it anyway raises a
> clear `AttributeError` instead of failing silently.

### Look-ahead-free data access

`self.close[-1]` is the current bar, `self.close[-2]` the previous one. Reading
the future raises `IndexError`, so **you cannot cheat by accident**.

```python
self.open, self.high, self.low, self.close, self.volume   # safe views
self.close.array        # all visible history as numpy
self.close.full         # the complete array - ONLY inside init()
self.close.series(100)  # last 100 bars as a pandas.Series
self.i                  # index of the current bar
self.now                # datetime of the current bar
```

### Sending orders

```python
self.buy(lots, sl=..., tp=..., comment="", magic=0, tag="")
self.sell(lots, sl=..., tp=...)
self.buy_limit(price, lots); self.sell_limit(price, lots)
self.buy_stop(price, lots);  self.sell_stop(price, lots)

self.close()                        # close the most recent position
self.close(position, volume=0.05)   # partial close
self.close_all()                    # everything
self.close_all(only=lambda p: p.is_long)
self.modify(position, sl=new_sl, tp=None)   # None removes the level
self.cancel_all()
```

Every order returns an `OrderResult`; when `result.ok` is `False`,
`result.retcode` says why (`no_margin`, `invalid_volume`, `market_closed`,
`invalid_stops`, `max_positions`, ...). All rejections land in
`result.rejections`.

### Account state

```python
self.equity, self.balance, self.free_margin, self.margin_level
self.bid, self.ask, self.spread_points, self.point, self.pip
self.positions      # list of open Position objects
self.position       # the most recent one (or None)
self.orders         # pending orders
self.has_position("buy")
```

### Position management, already written

```python
self.trailing_stop(distance_points=300, step_points=50)
self.break_even(trigger_points=200, offset_points=10)
self.volume_for_risk(risk_amount=100, stop_points=300)
self.volume_for_risk_pct(risk_pct=1.0, stop_points=300)
```

### Strategies without writing a class

```python
from marketprolab import FunctionStrategy, SignalStrategy

# a) a plain function
def logic(ctx):
    if ctx.close[-1] > ctx.close[-2]:
        ctx.buy(0.1)
Backtest(bars, FunctionStrategy.of(logic), spec).run()

# b) boolean signal arrays you already computed
Backtest(bars, SignalStrategy, spec, strategy_params={
    "long_signal": longs, "short_signal": shorts,
    "lots": 0.1, "sl_points": 300, "tp_points": 600,
}).run()
```

### Bundled indicators

`sma, ema, wma, rma, stdev, rsi, atr, true_range, bollinger, macd, stochastic,
adx, highest, lowest, roc, zscore, supertrend, crossover, crossunder`.

Any function that takes arrays and returns an array (or a tuple of arrays) works
with `self.I(...)`, including anything from `talib` or `pandas_ta`.

---

## 8. Running the backtest and reading the report

```python
from marketprolab import Backtest

bt = Backtest(
    data=bars,
    strategy=MyStrategy,
    symbol=spec,
    broker=profile,
    config=config,
    start="2024-01-01",        # trim by date
    end="2025-12-31",
    timeframe=None,            # resample the bars if given ("H1", "M30"...)
    strategy_params={"fast": 15},
    warmup_bars=200,           # no trading until bar 200
    drop_closed_bars=False,    # drop bars outside quoting hours
    progress=True,
)

r = bt.run()
r.report()
```

`BacktestResult` holds:

```python
r.trades        # DataFrame: one row per closed trade
r.equity        # DataFrame: equity, balance, margin, exposure per bar
r.stats         # dict with every metric
r.drawdown      # DataFrame: peak, dd_abs, dd_pct
r.rejections    # rejected orders and why
r.events        # full log of opens, closes, stop-outs...
r.monthly_table()          # year x month table
r.returns("ME")            # monthly / "W" / "YE" returns
r.rolling(500, "sharpe")   # rolling metric
r.save("results/run1", html=True)   # csv + json + txt + specs (+ HTML)
r.to_html("report.html")
r.montecarlo(n=5000)
```

Metrics in `r.stats`: net profit, return, CAGR, max drawdown (absolute, % and
duration), recovery factor, Sharpe, Sortino, Calmar, SQN, Ulcer index, annual
volatility, exposure, number of trades, win rate, profit factor, expectancy,
payoff, largest win/loss, streaks, average duration, trades per month,
commission, swap, average slippage, Kelly, a long/short breakdown, and exits by
reason (SL, TP, stop-out, session close).

---

## 9. Charts

```python
r.plot()               # full dashboard
r.plot_equity()        # equity + balance + drawdown panel
r.plot_drawdown()      # underwater
r.plot_trades()        # price with entries and exits
r.plot_distribution()  # P&L histogram, cumulative, duration, MAE/MFE
r.plot_monthly()       # year x month heatmap
r.save_charts("results/charts")   # save them all as png
```

They all return a matplotlib `Figure`, so you can keep editing it.

```python
from marketprolab import set_theme
set_theme("dark")      # or "light"
```

Palettes are validated for colour blindness and no chart uses a secondary Y
axis (two magnitudes go into stacked panels sharing the X axis).

Also available in `marketprolab.plotting`: `plot_returns_distribution`,
`plot_rolling`, `plot_montecarlo`, `plot_optimization`, `plot_walk_forward`.

---

## 10. HTML reports

Every result object can export a **standalone** `.html` file. Charts are
embedded as base64 PNGs and the CSS is inlined, so the report opens offline,
can be emailed, and needs no server or internet connection.

```python
r.to_html("report.html")                  # one backtest
opt.to_html("optimization.html")          # a grid or random search
wf.to_html("walkforward.html")            # a walk-forward run
mc.to_html("montecarlo.html")             # a Monte Carlo simulation
```

Everything on a single page:

```python
from marketprolab import combined_report, comparison_report

combined_report(r, "overall.html", opt=opt, wf=wf, mc=mc)   # omit what you don't have
comparison_report({"baseline": r1, "with filter": r2}, "comparison.html")
```

Options, shared by all of them:

| Argument | Default | What it does |
|---|---|---|
| `theme` | `"light"` | `"light"` or `"dark"` - page and charts together |
| `charts` | `True` | `False` gives tables only and a much smaller file |
| `max_trades` | `300` | How many trade rows to include |
| `dpi` | `120` | Chart resolution; higher means sharper and heavier |
| `open_browser` | `False` | Open the file as soon as it is written |
| `title` | auto | The page title |

A backtest report contains: headline cards (net profit, drawdown, profit factor,
Sharpe, win rate, expectancy, CAGR), the equity and drawdown chart, the full
metric table, trade analysis charts, the monthly returns table, the complete
setup (instrument, broker, strategy parameters, simulation settings), the trade
list and a rejection summary.

An optimization report adds the parameter surface, the neighbourhood-stability
ranking and the top combinations, plus the full detail of the best run.

```python
r.save("results/run1", html=True)   # writes the HTML alongside the CSVs
```

Typical size is 300-900 KB with charts, or ~50 KB with `charts=False`.

---

## 11. Optimization

```python
from marketprolab import grid_search, random_search, sensitivity

opt = grid_search(
    bt,
    {"fast": [10, 20, 30], "slow": [50, 100, 200]},
    objective="sharpe",     # any stats key, or a callable(result) -> float
    min_trades=30,          # drop combinations with too few trades
    constraint=lambda r: r.stats["max_dd_pct"] > -25,   # extra filter
    n_jobs=1,               # >1 uses processes (see the warning below)
)

opt.best_params        # {"fast": 20, "slow": 100}
opt.best_result        # the BacktestResult of the winning combination
opt.top(10)            # the ten best
opt.stable_top(10)     # ranked by NEIGHBOURHOOD average, not by the peak
opt.plot()             # heatmap for two parameters, bars for one
opt.to_html("optimization.html")
```

`stable_top()` matters: an isolated maximum in the grid is almost always
overfitting. A combination surrounded by good neighbours is far more likely to
survive in live trading.

```python
# Random search: better when there are many parameters
random_search(bt, {"fast": (5, 50), "slow": (50, 300)}, n_iter=200,
              objective="calmar")

# Single-parameter sweep to check stability
sensitivity(bt, "fast", [10, 15, 20, 25, 30])
```

Common objectives: `net_profit`, `sharpe`, `sortino`, `calmar`,
`profit_factor`, `sqn`, `recovery_factor`, `expectancy`. Risk metrics
(`max_dd_pct`, `ulcer_index`, ...) are minimised automatically.

> **Windows and `n_jobs > 1`:** parallelism uses processes, so the strategy must
> live in an importable module (not a notebook or an interactive `__main__`) and
> the script needs `if __name__ == "__main__":`. With `n_jobs > 1` the objective
> must also be a metric **name**, not a callable.

---

## 12. Walk-forward

Optimizing over the whole history and admiring the result proves nothing.
Walk-forward optimizes on one window and validates **always** on data the
optimization never saw.

```python
from marketprolab import walk_forward

wf = walk_forward(
    bt,
    {"fast": [10, 20, 30], "slow": [50, 100, 200]},
    in_sample_bars=10_000,
    out_sample_bars=2_500,
    step_bars=None,        # defaults to out_sample_bars
    objective="sharpe",
    anchored=False,        # True keeps the start fixed and grows the IS window
    compound=True,         # capital carries over between windows
    min_trades=10,
)

wf.efficiency     # average OOS/IS: >0.5 is decent, <0.3 smells of overfitting
wf.windows        # one row per window, with the parameters it chose
wf.oos_equity     # chained out-of-sample curve
wf.stats          # metrics of that OOS curve
wf.plot()
wf.to_html("walkforward.html")
```

---

## 13. Monte Carlo

A backtest is **one** sample. Monte Carlo estimates the range of outcomes and,
above all, the drawdown you must be prepared to sit through.

```python
from marketprolab import monte_carlo, monte_carlo_bars, required_capital

mc = monte_carlo(
    r,                        # BacktestResult, trades DataFrame, or P&L array
    n_simulations=5_000,
    method="bootstrap",       # shuffle | bootstrap | block | normal
    block_size=10,            # for method="block": preserves streaks
    skip_probability=0.10,    # you miss 1 signal in 10
    pnl_noise_pct=15,         # +/-15% noise on every result
    extra_cost_per_trade=3.0, # costs worse than simulated
    ruin_level_pct=50,        # "ruin" = losing half the account
    dd_threshold_pct=20,      # probability of exceeding this drawdown
    compound_risk=False,      # True = risk proportional to capital
    seed=1,
)

mc.summary()
mc.plot()
mc.to_html("montecarlo.html")
mc.percentiles["dd_p95"]     # the bad-tail drawdown
mc.risk_of_ruin
print(required_capital(mc, confidence=95, safety_factor=1.5))
```

| Method | What it assumes |
|---|---|
| `shuffle` | Same trades, different order. Measures sequence risk. |
| `bootstrap` | Resampling with replacement. Some trades repeat, others vanish. |
| `block` | Block bootstrap: preserves streaks (more realistic with autocorrelation). |
| `normal` | Draws P&L from a normal with your mean and sigma. The most optimistic: it ignores the tails. |

And the most honest of all, though slow: **Monte Carlo over the price**, which
generates synthetic series and re-runs the whole strategy.

```python
table = monte_carlo_bars(bt, n_simulations=50, method="block", block_size=48)
print(table.describe())
```

---

## 14. Stress tests and comparisons

```python
from marketprolab import stress_test, compare

table = stress_test(
    bt,
    spread_multipliers=(1, 1.5, 2, 3),
    slippage_points=(0, 5, 15),      # 0 keeps the base model
    commission_multipliers=(1, 2),
)
print(table.pivot_table(index="spread_x", columns="slippage_points",
                        values="net_profit"))

compare({"conservative": r1, "aggressive": r2, "other broker": r3})
```

If a strategy only wins with the tightest spread and zero slippage, it does not
win.

---

## 15. How the simulation works inside

Exactly what happens on every bar:

1. The bar's **spread** and the account-currency exchange rate are computed.
2. **Swaps** are charged if the rollover hour has been crossed (with the day's
   multiplier: triple on Wednesday, zero at the weekend).
3. Pending orders that have reached their **expiry** are removed.
4. Orders sitting in the **latency queue** whose execution instant falls inside
   this bar are filled.
5. The bar is checked for **pending-order triggers** (stop/limit).
6. **SL and TP** are checked for every open position.
7. Mark to market: equity, margin, MAE/MFE.
8. The **stop-out** (and bankruptcy) check runs.
9. Your strategy's `on_bar()` is called - it sees this bar's **close** and may
   send orders, which get queued behind the configured latency.
10. Session-end / weekend closes happen and the curve is recorded.

Details that matter:

- Data prices are **bid**. `ask = bid + spread`. Longs open at the ask and close
  at the bid; shorts the other way round. You always pay the spread.
- **Nothing is ever filled on the bar that generated the signal** (unless you set
  `allow_same_bar_fill=True` with zero latency). This removes the most common
  source of falsely brilliant backtests.
- If SL and TP are both touched inside a bar, `intrabar` decides which wins. The
  default is `PESSIMISTIC`: **always the stop**. That is the prudent assumption.
- If price gaps past the SL, the fill is at the open, not at the stop level -
  exactly as in real trading.
- Stops suffer more slippage than limits (the volatility models apply
  `stop_multiplier`).
- Opening commission is deducted from the balance on entry, closing commission
  on exit. Swap accrues on the position and settles when it closes.
- With `margin_mode=NETTING` or `hedging_allowed=False`, an opposite order
  **closes** the existing position instead of opening a new one.
- The stop-out closes the worst-losing position, one at a time, until the margin
  level recovers.

---

## 16. Common mistakes

| Symptom | Usual cause |
|---|---|
| `IndexError: look-ahead` | You are reading a future bar from `on_bar`. Use negative indices. |
| `AttributeError: reserved name(s)` | A strategy parameter is called `open/high/low/close/volume/data/broker`. Rename it (`lots`). |
| Zero trades | Volume below `volume_min`, closed session, or not enough margin. Check `r.rejections`. |
| Everything rejected with `no_margin` | Initial balance too small for the instrument's `contract_size`, or low leverage. |
| Results that look too good | `allow_same_bar_fill=True`, no spread, no slippage, or `intrabar=OPTIMISTIC`. |
| Enormous swap | Wrong `swap_type`. Check whether your broker quotes it in points, money or annual %. |
| Profits off by orders of magnitude | `contract_size` or `digits` are wrong. Verify with `spec.point_value(1)`. |
| The backtest takes forever | You are running M1 or ticks over years. Resample with `resample_bars` or trim the dates. |
| Different results on every run | Set `seed` in `SimulationConfig`. |
| Garbled characters in the console | It is the Windows console (cp1252). Use `chcp 65001` or save the report to a file. |

---

## 17. API quick reference

```
marketprolab
├── Backtest, BacktestResult              engine and result
├── Strategy, FunctionStrategy, SignalStrategy
├── SymbolSpec, SessionSpec               the instrument
├── BrokerProfile, SymbolRegistry         the account and the catalogue
├── SimulationConfig, Broker              realism and execution
├── Order, Position, Trade, OrderResult
├── data        load_mt5_bars, load_mt5_ticks, read_csv_bars, ticks_to_bars,
│               resample_bars, save_bars, load_bars, get_bars, data_quality_report
├── execution   FixedSpread, DataSpread, RandomSpread, SessionSpread, VolatilitySpread,
│               NoSlippage, FixedSlippage, RandomSlippage, VolatilitySlippage,
│               GapSlippage, FixedLatency, RandomLatency
├── indicators  sma, ema, rsi, atr, bollinger, macd, adx, stochastic, supertrend...
├── metrics     compute_stats, format_stats, monthly_table, drawdown_series
├── optimize    grid_search, random_search, walk_forward, sensitivity,
│               stress_test, compare
├── montecarlo  monte_carlo, monte_carlo_bars, required_capital, confidence_intervals
├── plotting    set_theme, plot_equity, plot_drawdown, plot_price_trades,
│               plot_trade_distribution, plot_monthly_heatmap, plot_dashboard,
│               plot_montecarlo, plot_optimization, plot_walk_forward, save_all_charts
├── report      backtest_report, optimization_report, walk_forward_report,
│               montecarlo_report, comparison_report, combined_report
└── presets     xauusd, forex_major, index_cfd, crypto, energy
```

Runnable examples live in [`examples/`](../examples):

| File | What it shows |
|---|---|
| `01_basic_backtest.py` | The full cycle, end to end |
| `02_mt5_and_tick_data.py` | Real terminal data and tick-built bars |
| `03_optimization_and_walkforward.py` | Grid, stability, walk-forward, stress |
| `04_monte_carlo.py` | Monte Carlo over trades and over price |
| `05_any_instrument.py` | Five different markets and two brokers |
| `06_html_reports.py` | Every flavour of HTML export |
