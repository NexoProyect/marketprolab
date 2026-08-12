"""The viral "4-Hour Range Breakout" strategy on EURUSD, tested honestly.

THE STRATEGY (as circulated)
----------------------------
1. Mark the high and low of the **first 4-hour candle of the day in New York
   time** (00:00-04:00 NY). That is the range.
2. Wait for a 5-minute candle to **close outside** that range.
3. If a later 5-minute candle then **closes back inside** the range, fade the
   breakout:
       - closed above the high, then back inside  ->  SELL
       - closed below the low, then back inside   ->  BUY
4. Time filter: if price stays outside for more than 75 minutes without coming
   back, the setup is dead.
5. Risk: a small percentage per trade. The original does not specify a stop, so
   both usual conventions are implemented here (range extreme, or an ATR
   multiple).

WHAT THE ORIGINAL SOURCE ADMITS
-------------------------------
Its own backtests on EURUSD were **not profitable**. This script exists to check
that claim under realistic conditions rather than to sell the idea.

WHAT THIS SCRIPT ADDS (because the original is silent about it)
---------------------------------------------------------------
- A monitoring window: setups are only armed between 04:00 and 17:00 NY.
- Open positions are flattened at 17:00 NY, so nothing is carried overnight.
- Real costs: the broker's own per-bar spread (floored), commission, swap,
  slippage and execution latency.
Every one of those is a named constant below - change them and see what happens.

USAGE
-----
    python backtest.py                  # run the backtest, print and export
    python backtest.py --optimize       # small parameter grid on top
    python backtest.py --walk-forward   # optimize in sample, validate out
    python backtest.py --montecarlo     # outcome distribution
    python backtest.py --all --open     # everything, then open the report
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from marketprolab import (
    Backtest,
    BacktestResult,
    BrokerProfile,
    DataSpread,
    RandomLatency,
    RandomSlippage,
    SessionSpec,
    SimulationConfig,
    Strategy,
    SymbolSpec,
    combined_report,
    data_quality_report,
    grid_search,
    indicators,
    monte_carlo,
    walk_forward,
)

# ════════════════════════════════════════════════════════════ CONFIGURATION

SYMBOL = "EURUSDz"          # the broker's name for EURUSD (Exness suffixes with z)
TIMEFRAME = "M5"            # the strategy is executed on 5-minute candles
YEARS = 3
INITIAL_BALANCE = 10_000.0

# The server clock the data is stamped with. Exness MT5 runs on GMT+0, which is
# why the New York conversion below can be exact. Check yours before trusting
# any of this - see the note at the bottom of the file.
SERVER_TZ = "UTC"
NY_TZ = "America/New_York"

# Costs. The account is an Exness "Zero" type: the raw feed reports a median
# spread of 0.0 points, which no one actually gets filled at. We floor it at
# MIN_SPREAD_POINTS and add the commission such accounts charge.
MIN_SPREAD_POINTS = 2.0     # 0.2 pips floor on the broker's own spread series
COMMISSION_PER_LOT = 3.5    # per lot, per side -> $7 round turn
SLIPPAGE_MEAN_POINTS = 1.5
SLIPPAGE_SIGMA_POINTS = 2.0
LATENCY_MEAN_MS = 150

CACHE_DIR = ".cache"
OUTPUT_DIR = "results/viral_4h_range"


# ════════════════════════════════════════════════════════════ THE STRATEGY


class FourHourRangeBreakout(Strategy):
    """Fade a failed breakout of the first New York 4-hour range.

    The range is built from the 00:00-04:00 NY window and only becomes visible
    to the strategy once that window has closed, so there is no look-ahead.
    """

    # --- setup
    range_start_hour_ny = 0       # first 4H candle of the NY day starts here
    range_hours = 4               # ... and lasts this long
    monitor_until_hour_ny = 17    # stop arming new setups at the NY close
    flat_at_hour_ny = 17          # flatten anything still open (None = never)
    invalidate_minutes = 75       # the time filter from the original rules

    # --- risk
    risk_pct = 0.5                # percentage of equity risked per trade
    sl_mode = "range"             # "range" (breakout extreme) or "atr"
    tp_mode = "opposite"          # "opposite" (other end of range) or "rr"
    sl_buffer_points = 20         # cushion beyond the extreme, in points
    atr_period = 14
    atr_sl_mult = 1.5
    rr = 1.5                      # reward-to-risk when tp_mode == "rr"

    # --- filters
    min_range_points = 0          # skip degenerate ranges (0 = no filter)
    max_range_points = 0          # skip absurdly wide ranges (0 = no filter)
    one_trade_per_day = True
    allow_rearm = False           # retry the same side after an invalidation

    # ---------------------------------------------------------------- setup
    def init(self) -> None:
        index = self.data.index
        ny = index.tz_localize(SERVER_TZ).tz_convert(NY_TZ)

        self.ny_minute = (ny.hour * 60 + ny.minute).to_numpy()
        self.day_id = ny.strftime("%Y-%m-%d").to_numpy()

        window_start = self.range_start_hour_ny * 60
        window_end = window_start + self.range_hours * 60
        in_window = (self.ny_minute >= window_start) & (self.ny_minute < window_end)

        # The range of each NY day, computed once, vectorised.
        frame = pd.DataFrame(
            {
                "day": self.day_id,
                "high": self.data["high"].to_numpy(),
                "low": self.data["low"].to_numpy(),
            }
        )[in_window]
        grouped = frame.groupby("day").agg(
            high=("high", "max"), low=("low", "min"), bars=("high", "size")
        )

        # A partial window (holiday, late open) does not produce a usable range.
        expected_bars = self.range_hours * 12          # twelve M5 bars per hour
        grouped.loc[grouped["bars"] < expected_bars * 0.5, ["high", "low"]] = np.nan

        self.range_high = grouped["high"].reindex(self.day_id).to_numpy()
        self.range_low = grouped["low"].reindex(self.day_id).to_numpy()

        # Hide the range from bars inside the formation window: at 02:00 the
        # 00:00-04:00 range does not exist yet.
        self.range_high = np.where(in_window, np.nan, self.range_high)
        self.range_low = np.where(in_window, np.nan, self.range_low)

        if self.sl_mode == "atr":
            self.atr = self.I(indicators.atr, self.high.full, self.low.full,
                              self.close.full, self.atr_period)

        self._reset_day(None)
        self.setups_seen = 0
        self.setups_expired = 0

    def _reset_day(self, day) -> None:
        self.current_day = day
        self.armed_side = 0          # +1 armed above the high, -1 below the low
        self.armed_at = None         # timestamp of the breakout close
        self.done_up = False
        self.done_down = False
        self.traded_today = 0

    # ------------------------------------------------------------- the loop
    def on_bar(self) -> None:
        i = self.i
        day = self.day_id[i]
        if day != self.current_day:
            self._reset_day(day)

        minute = self.ny_minute[i]
        high, low = self.range_high[i], self.range_low[i]

        # Flatten at the New York close, whatever else is going on.
        if self.flat_at_hour_ny is not None and minute >= self.flat_at_hour_ny * 60:
            if self.positions:
                self.close_all()
            return

        if np.isnan(high) or minute >= self.monitor_until_hour_ny * 60:
            return

        width_points = (high - low) / self.point
        if self.min_range_points and width_points < self.min_range_points:
            return
        if self.max_range_points and width_points > self.max_range_points:
            return

        close = self.close[-1]
        now = self.time_index[i]

        # 1. The time filter: a breakout that never comes back is dead.
        if self.armed_side and (now - self.armed_at) > timedelta(minutes=self.invalidate_minutes):
            self.setups_expired += 1
            if self.armed_side > 0:
                self.done_up = not self.allow_rearm
            else:
                self.done_down = not self.allow_rearm
            self.armed_side, self.armed_at = 0, None

        # 2. Back inside the range after a breakout -> fade it.
        if self.armed_side and low <= close <= high:
            side = -self.armed_side          # broke up -> sell, broke down -> buy
            if self.armed_side > 0:
                self.done_up = True
            else:
                self.done_down = True
            self.armed_side, self.armed_at = 0, None
            self._enter(side, high, low)
            return

        # 3. A close outside the range arms the setup.
        if close > high and not self.done_up and self.armed_side == 0:
            self.armed_side, self.armed_at = 1, now
            self.setups_seen += 1
        elif close < low and not self.done_down and self.armed_side == 0:
            self.armed_side, self.armed_at = -1, now
            self.setups_seen += 1

    # ---------------------------------------------------------------- entry
    def _enter(self, side: int, high: float, low: float) -> None:
        if self.one_trade_per_day and self.traded_today:
            return
        if self.has_position():
            return

        price = self.bid if side < 0 else self.ask

        if self.sl_mode == "atr":
            if np.isnan(self.atr[-1]):
                return
            sl_distance = self.atr[-1] * self.atr_sl_mult
        else:
            # The stop goes beyond the extreme that was broken.
            extreme = high if side < 0 else low
            sl_distance = abs(price - extreme) + self.sl_buffer_points * self.point

        if sl_distance <= 0:
            return

        if self.tp_mode == "opposite":
            target = low if side < 0 else high      # mean-revert to the far end
            tp_distance = abs(price - target)
        else:
            tp_distance = sl_distance * self.rr
        if tp_distance <= 0:
            return

        lots = self.volume_for_risk_pct(self.risk_pct, sl_distance / self.point)
        if lots <= 0:
            return

        if side < 0:
            self.sell(lots, sl=price + sl_distance, tp=price - tp_distance,
                      comment="fade breakout up", tag="short")
        else:
            self.buy(lots, sl=price - sl_distance, tp=price + tp_distance,
                     comment="fade breakout down", tag="long")
        self.traded_today += 1

    def on_finish(self) -> None:
        self.log("setups", seen=self.setups_seen, expired=self.setups_expired)


# ════════════════════════════════════════════════════════════ PLUMBING


def load_data(years: int = YEARS, refresh: bool = False) -> pd.DataFrame:
    """M5 bars for the last N years, from MT5, cached to parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{SYMBOL}_{TIMEFRAME}_{years}y.parquet")

    if os.path.exists(cache) and not refresh:
        bars = pd.read_parquet(cache)
        print(f"Data from cache: {len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}")
        return bars

    from marketprolab import load_mt5_bars

    end = datetime.now()
    start = end - timedelta(days=365 * years + 5)
    print(f"Downloading {SYMBOL} {TIMEFRAME} from MetaTrader 5 "
          f"({start.date()} -> {end.date()}) ...")
    bars = load_mt5_bars(SYMBOL, TIMEFRAME, start=start, end=end)
    bars.to_parquet(cache)
    print(f"Downloaded {len(bars):,} bars and cached to {cache}")
    return bars


def build_symbol() -> SymbolSpec:
    """The instrument, read from the terminal and completed by hand."""
    try:
        spec = SymbolSpec.from_mt5(SYMBOL)
        source = "MetaTrader 5"
    except Exception as exc:                                   # no terminal
        print(f"MT5 unavailable ({exc}); using a hand-written EURUSD spec")
        from marketprolab import presets

        spec = presets.forex_major("EURUSD")
        source = "manual preset"

    # from_mt5 cannot read these; they come from the account terms and the
    # Specification window.
    spec.commission_per_lot = COMMISSION_PER_LOT
    spec.spread_points = max(spec.spread_points, MIN_SPREAD_POINTS)

    # Forex hours on a GMT+0 server: Sunday night to Friday evening.
    sessions = SessionSpec.from_dict(
        {
            "sunday": "22:00-24:00",
            "monday": "00:00-24:00",
            "tuesday": "00:00-24:00",
            "wednesday": "00:00-24:00",
            "thursday": "00:00-24:00",
            "friday": "00:00-21:00",
        },
        timezone=SERVER_TZ,
    )
    spec.quote_sessions = sessions
    spec.trade_sessions = sessions
    spec.timezone = SERVER_TZ

    print(f"Instrument loaded from {source}")
    return spec


def build_backtest(bars: pd.DataFrame, spec: SymbolSpec, progress: bool = True,
                   **params) -> Backtest:
    # The broker's own spread series, floored: a reported 0.0 is not a fill.
    bars = bars.copy()
    if "spread" in bars.columns:
        bars["spread"] = bars["spread"].clip(lower=MIN_SPREAD_POINTS)

    profile = BrokerProfile(
        name="Exness Zero (EURUSDz)",
        account_currency="USD",
        leverage=100,
        stop_out_level=50.0,
        commission_per_lot=COMMISSION_PER_LOT,
        server_timezone=SERVER_TZ,
        swap_triple_weekday=2,
    )

    config = SimulationConfig(
        initial_balance=INITIAL_BALANCE,
        spread=DataSpread(fallback_points=MIN_SPREAD_POINTS),
        slippage=RandomSlippage(mean_points=SLIPPAGE_MEAN_POINTS,
                                sigma_points=SLIPPAGE_SIGMA_POINTS),
        latency=RandomLatency(mean_ms=LATENCY_MEAN_MS, spike_probability=0.01),
        respect_sessions=True,
        apply_swap=True,
        seed=42,
    )

    return Backtest(
        data=bars,
        strategy=FourHourRangeBreakout,
        symbol=spec,
        broker=profile,
        config=config,
        strategy_params=params or None,
        warmup_bars=288,          # one full day
        progress=progress,
    )


# ════════════════════════════════════════════════════════════ ANALYSIS


def describe_setups(result: BacktestResult) -> None:
    """What the state machine saw, beyond the trades it took."""
    events = result.events
    row = events[events["kind"] == "strategy"] if len(events) else pd.DataFrame()
    if len(row):
        seen = int(row.iloc[-1].get("seen", 0))
        expired = int(row.iloc[-1].get("expired", 0))
        taken = result.stats["trades"]
        print("\n--- Setup funnel ---")
        print(f"  Breakouts armed          : {seen:,}")
        print(f"  Killed by the 75' filter : {expired:,}"
              f"  ({expired / max(seen, 1) * 100:.1f}%)")
        print(f"  Reached a trade          : {taken:,}"
              f"  ({taken / max(seen, 1) * 100:.1f}%)")


def describe_trades(result: BacktestResult) -> None:
    trades = result.trades
    if trades.empty:
        print("\nNo trades. Check result.rejections.")
        return

    print("\n--- By direction ---")
    by_side = trades.groupby("type")["net_profit"].agg(
        trades="size", net="sum", avg="mean",
        wins=lambda s: int((s > 0).sum()),
    )
    by_side["win_rate_%"] = (by_side["wins"] / by_side["trades"] * 100).round(1)
    print(by_side.round(2).to_string())

    print("\n--- By exit reason ---")
    by_reason = trades.groupby("reason")["net_profit"].agg(
        trades="size", net="sum", avg="mean")
    print(by_reason.round(2).to_string())

    print("\n--- Cost drag ---")
    gross = trades["profit"].sum()
    costs = trades["commission"].sum() - trades["swap"].sum()
    print(f"  Gross P&L before costs : {gross:>12,.2f}")
    print(f"  Commission             : {trades['commission'].sum():>12,.2f}")
    print(f"  Swap                   : {-trades['swap'].sum():>12,.2f}")
    print(f"  Net P&L                : {gross - costs:>12,.2f}")
    if gross > 0:
        print(f"  Costs ate {costs / gross * 100:.0f}% of the gross profit.")


# ════════════════════════════════════════════════════════════ MAIN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, default=YEARS)
    parser.add_argument("--refresh", action="store_true", help="re-download the data")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--montecarlo", action="store_true")
    parser.add_argument("--all", action="store_true", help="every analysis")
    parser.add_argument("--open", action="store_true", help="open the HTML report")
    args = parser.parse_args()
    if args.all:
        args.optimize = args.walk_forward = args.montecarlo = True

    bars = load_data(args.years, refresh=args.refresh)
    print(data_quality_report(bars, TIMEFRAME))

    spec = build_symbol()
    bt = build_backtest(bars, spec)
    print(bt)

    print("\nRunning the backtest ...")
    result = bt.run()
    result.report()
    describe_setups(result)
    describe_trades(result)

    opt = wf = mc = None

    if args.optimize:
        print("\n=== OPTIMIZATION ===")
        opt = grid_search(
            bt,
            {
                "invalidate_minutes": [45, 75, 120],
                "sl_mode": ["range", "atr"],
                "tp_mode": ["opposite", "rr"],
            },
            objective="net_profit",
            min_trades=50,
        )
        print(opt)
        print(opt.results[["invalidate_minutes", "sl_mode", "tp_mode", "net_profit",
                           "profit_factor", "win_rate", "max_dd_pct", "trades"]]
              .round(2).to_string(index=False))

    if args.walk_forward:
        print("\n=== WALK-FORWARD ===")
        wf = walk_forward(
            bt,
            {"invalidate_minutes": [45, 75, 120], "sl_mode": ["range", "atr"]},
            in_sample_bars=40_000,      # roughly six months of M5
            out_sample_bars=20_000,     # validated on the next three
            objective="net_profit",
            min_trades=20,
        )
        print(wf)

    if args.montecarlo and result.stats["trades"] > 10:
        print("\n=== MONTE CARLO ===")
        mc = monte_carlo(result, n_simulations=5_000, method="bootstrap",
                         dd_threshold_pct=20, ruin_level_pct=50)
        mc.summary()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.save(OUTPUT_DIR)
    result.save_charts(os.path.join(OUTPUT_DIR, "charts"))
    report_path = combined_report(
        result, os.path.join(OUTPUT_DIR, "report.html"),
        opt=opt, wf=wf, mc=mc,
        title="Viral 4-hour range breakout - EURUSD",
        open_browser=args.open,
    )
    print(f"\nHTML report: {report_path}")

    verdict = "PROFITABLE" if result.stats["net_profit"] > 0 else "NOT PROFITABLE"
    print(f"\nVerdict over {args.years} years of {SYMBOL} {TIMEFRAME}: {verdict} "
          f"({result.stats['net_profit']:+,.2f} on {INITIAL_BALANCE:,.0f}, "
          f"{result.stats['trades']:,} trades)")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# A WORD ON TIME ZONES
#
# This strategy is anchored to New York midnight, so the whole thing hinges on
# converting the data's timestamps correctly. Two things have to be right:
#
#   1. SERVER_TZ must match the clock your broker stamps bars with. Exness MT5
#      uses GMT+0, so it is "UTC" here. Most brokers use GMT+2/+3 and shift with
#      European daylight saving, which no fixed offset can express - if that is
#      yours, pass a named zone such as "Europe/Helsinki" instead of "Etc/GMT-3".
#   2. New York daylight saving is handled for you by the tz database, which is
#      the entire reason the conversion is done with pandas rather than by
#      adding a constant number of hours.
#
# Get either wrong and you are marking a range that is one hour off for half the
# year - which, for a strategy whose only input is a range, means you are
# testing something else entirely.
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    sys.exit(main())
