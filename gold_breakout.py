"""The "Gold Breakout Trader" TradingView indicator, turned into a real strategy.

WHAT THE INDICATOR DOES
-----------------------
Every two hours (or once a day at a chosen hour, in UTC+1) it takes the close of
that candle as a reference and paints a ladder of levels around it:

        ref + 21.0   TP3          (mid of the third zone)
        ref + 16.0   TP2
        ref + 11.0   TP1
        ref +  7.0   Buy Stop     <- gray box top (+5) plus the buffer (+2)
        ref +  5.0   gray box top
        ref          reference close
        ref -  5.0   gray box bottom
        ref -  7.0   Sell Stop
        ref - 11.0   TP1
        ref - 16.0   TP2
        ref - 21.0   TP3

and alerts when price closes above the Buy Stop or below the Sell Stop. Those
alerts are the entries; every distance above is a script input.

WHAT THE INDICATOR DOES NOT SAY
-------------------------------
Where the stop-loss goes, how much to risk, and which of the three TP zones to
use. Those are choices, so all of them are parameters here:

    sl_mode   "opposite" (the other stop line)  |  "box" (far box edge)  |  "atr"
    tp_target "tp1" | "tp2" | "tp3" | "scale" (a third at each, one shared stop)

USAGE
-----
    python gold_breakout.py                 # single run with the indicator's defaults
    python gold_breakout.py --sweep         # parameter sweep + the top-5 chart
    python gold_breakout.py --sweep --wide  # a much larger sweep
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketprolab import (
    Backtest,
    BrokerProfile,
    DataSpread,
    RandomLatency,
    RandomSlippage,
    SessionSpec,
    SimulationConfig,
    Strategy,
    SwapType,
    SymbolSpec,
    grid_search,
    indicators,
    plotting,
)

# ════════════════════════════════════════════════════════════ CONFIGURATION

SYMBOL = "XAUUSDc"          # gold on this broker: 1 ounce per lot, 3 digits
TIMEFRAME = "M5"
YEARS = 3
INITIAL_BALANCE = 10_000.0

SERVER_TZ_OFFSET_H = 0      # the data's clock relative to UTC (Exness runs GMT+0)
INDICATOR_TZ_OFFSET_H = 1   # the Pine script adds one hour: "UTC+1", no DST

MIN_SPREAD_POINTS = 100.0   # 0.10 USD floor on the broker's own spread series
COMMISSION_PER_LOT = 0.0    # this account type pays through the spread
SLIPPAGE_MEAN_POINTS = 30.0     # 0.03 USD - gold fills are not free
SLIPPAGE_SIGMA_POINTS = 40.0
LATENCY_MEAN_MS = 150

CACHE_DIR = ".cache"
OUTPUT_DIR = "results/gold_breakout"


# ════════════════════════════════════════════════════════════ THE STRATEGY


class GoldBreakoutTrader(Strategy):
    """Trade the Buy Stop / Sell Stop alerts of the Gold Breakout Trader indicator."""

    # --- when the ladder is drawn (indicator inputs)
    every_2_hours = True          # enable2HourPlot
    trigger_hour = 7              # used when every_2_hours is False
    trigger_minute = 0

    # --- the ladder itself, in USD (indicator inputs)
    box_height = 10.0             # boxHeight
    stop_buffer = 2.0             # stopLineBuffer
    stop_to_tp_gap = 2.0          # stopToTPGap
    tp_zone_gap = 1.0             # tpZoneGap
    tp1_height = 4.0
    tp2_height = 4.0
    tp3_height = 4.0

    # --- the trading decisions the indicator leaves open
    sl_mode = "opposite"          # "opposite" | "box" | "atr"
    tp_target = "tp1"             # "tp1" | "tp2" | "tp3" | "scale"
    atr_period = 14
    atr_sl_mult = 2.0
    risk_pct = 0.5                # percentage of equity risked per trade
    one_position = True           # no pyramiding
    breakeven_after_tp1 = False   # only meaningful with tp_target="scale"

    # ---------------------------------------------------------------- setup
    def init(self) -> None:
        shifted = self.data.index + pd.Timedelta(
            hours=INDICATOR_TZ_OFFSET_H - SERVER_TZ_OFFSET_H
        )
        hour, minute = shifted.hour.to_numpy(), shifted.minute.to_numpy()

        if self.every_2_hours:
            self.is_trigger = (minute == 0) & (hour % 2 == 0)
        else:
            self.is_trigger = (hour == self.trigger_hour) & (minute == self.trigger_minute)

        if self.sl_mode == "atr":
            self.atr = self.I(indicators.atr, self.high.full, self.low.full,
                              self.close.full, self.atr_period)

        # The ladder, measured from the reference close.
        half = self.box_height / 2.0
        self.d_stop = half + self.stop_buffer
        tp1_bot = self.d_stop + self.stop_to_tp_gap
        self.d_tp1 = tp1_bot + self.tp1_height / 2.0
        tp2_bot = tp1_bot + self.tp1_height + self.tp_zone_gap
        self.d_tp2 = tp2_bot + self.tp2_height / 2.0
        tp3_bot = tp2_bot + self.tp2_height + self.tp_zone_gap
        self.d_tp3 = tp3_bot + self.tp3_height / 2.0
        self.d_box = half

        self.buy_stop = np.nan
        self.sell_stop = np.nan
        self.ref = np.nan
        self.prev_close = np.nan
        self.moved_to_be = False
        self.signals = 0

    # ------------------------------------------------------------- the loop
    def on_bar(self) -> None:
        close = self.close[-1]

        # A new trigger redraws the ladder, exactly as the indicator does.
        if self.is_trigger[self.i]:
            self.ref = close
            self.buy_stop = close + self.d_stop
            self.sell_stop = close - self.d_stop
            self.prev_close = close
            return

        if np.isnan(self.buy_stop) or np.isnan(self.prev_close):
            self.prev_close = close
            return

        crossed_up = self.prev_close <= self.buy_stop < close
        crossed_down = self.prev_close >= self.sell_stop > close
        self.prev_close = close

        if self.breakeven_after_tp1:
            self._maybe_move_to_breakeven()

        if not (crossed_up or crossed_down):
            return
        if self.one_position and self.has_position():
            return

        self.signals += 1
        self._enter(1 if crossed_up else -1)

    # ---------------------------------------------------------------- entry
    def _enter(self, side: int) -> None:
        ref = self.ref
        price = self.ask if side > 0 else self.bid

        # Stop-loss: whichever convention was selected.
        if self.sl_mode == "atr":
            if np.isnan(self.atr[-1]):
                return
            sl = price - side * self.atr[-1] * self.atr_sl_mult
        elif self.sl_mode == "box":
            sl = ref - side * self.d_box          # the far edge of the gray box
        else:
            sl = ref - side * self.d_stop         # the opposite stop line

        sl_distance = abs(price - sl)
        if sl_distance <= 0:
            return

        targets = {
            "tp1": [self.d_tp1],
            "tp2": [self.d_tp2],
            "tp3": [self.d_tp3],
            "scale": [self.d_tp1, self.d_tp2, self.d_tp3],
        }[self.tp_target]

        lots = self.volume_for_risk_pct(self.risk_pct, sl_distance / self.point)
        if lots <= 0:
            return

        # "scale" splits the risk across three tickets sharing one stop, which is
        # how a three-zone ladder is actually traded.
        each = self.spec.normalize_volume(lots / len(targets))
        if each <= 0:
            return

        self.moved_to_be = False
        for k, distance in enumerate(targets, start=1):
            self.broker.send(
                self._order_type(side),
                each,
                sl=sl,
                tp=ref + side * distance,
                comment=f"breakout tp{k}",
                tag=f"tp{k}",
            )

    def _order_type(self, side: int):
        from marketprolab import OrderType

        return OrderType.BUY if side > 0 else OrderType.SELL

    # ------------------------------------------------------------ management
    def _maybe_move_to_breakeven(self) -> None:
        """Once the TP1 ticket is gone, protect the rest at entry."""
        positions = self.positions
        if self.moved_to_be or not positions:
            return
        if any(p.tag == "tp1" for p in positions):
            return
        for p in positions:
            if (p.open_price - p.sl) * p.sign < 0:      # stop already beyond entry
                continue
            self.broker.modify(p, sl=p.open_price)
        self.moved_to_be = True

    def on_finish(self) -> None:
        self.log("signals", count=self.signals)


# ════════════════════════════════════════════════════════════ PLUMBING


def load_data(years: int = YEARS, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{SYMBOL}_{TIMEFRAME}_{years}y.parquet")
    if os.path.exists(cache) and not refresh:
        bars = pd.read_parquet(cache)
        print(f"Data from cache: {len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}")
        return bars

    from marketprolab import load_mt5_bars

    end = datetime.now()
    start = end - timedelta(days=365 * years + 5)
    print(f"Downloading {SYMBOL} {TIMEFRAME} ({start.date()} -> {end.date()}) ...")
    bars = load_mt5_bars(SYMBOL, TIMEFRAME, start=start, end=end)
    bars.to_parquet(cache)
    print(f"Cached {len(bars):,} bars to {cache}")
    return bars


def build_symbol() -> SymbolSpec:
    try:
        spec = SymbolSpec.from_mt5(SYMBOL)
    except Exception as exc:
        print(f"MT5 unavailable ({exc}); using a hand-written gold spec")
        from marketprolab import presets

        spec = presets.xauusd(symbol=SYMBOL, contract_size=1.0)

    # The terminal reports gold's swap as mode 1 = points. Sanity-check it:
    # one ounce financed overnight should cost cents, not hundreds of dollars.
    spec.swap_type = SwapType.POINTS
    spec.commission_per_lot = COMMISSION_PER_LOT
    spec.spread_points = max(spec.spread_points, MIN_SPREAD_POINTS)

    sessions = SessionSpec.from_dict(
        {
            "sunday": "22:05-24:00",
            "monday": "00:00-20:58, 22:05-24:00",
            "tuesday": "00:00-20:58, 22:05-24:00",
            "wednesday": "00:00-20:58, 22:05-24:00",
            "thursday": "00:00-20:58, 22:05-24:00",
            "friday": "00:00-20:58",
        }
    )
    spec.quote_sessions = spec.trade_sessions = sessions
    return spec


def build_backtest(bars: pd.DataFrame, spec: SymbolSpec, progress: bool = True,
                   **params) -> Backtest:
    bars = bars.copy()
    if "spread" in bars.columns:
        bars["spread"] = bars["spread"].clip(lower=MIN_SPREAD_POINTS)

    profile = BrokerProfile(
        name=f"Exness ({SYMBOL})",
        account_currency="USD",
        leverage=100,
        stop_out_level=50.0,
        commission_per_lot=COMMISSION_PER_LOT,
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
        data=bars, strategy=GoldBreakoutTrader, symbol=spec, broker=profile,
        config=config, strategy_params=params or None, warmup_bars=60,
        progress=progress,
    )


# ════════════════════════════════════════════════════════════ THE TOP-5 CHART


def label_for(row: pd.Series, params: list) -> str:
    parts = []
    for p in params:
        value = row[p]
        if p == "box_height":
            parts.append(f"box {value:g}")
        elif p == "stop_buffer":
            parts.append(f"buf {value:g}")
        elif p == "tp_target":
            parts.append(str(value).upper())
        elif p == "sl_mode":
            parts.append(f"SL {value}")
        else:
            parts.append(f"{p}={value}")
    return " · ".join(parts)


def plot_top5(curves: list, baseline=None, path: str = "top5.png",
              subtitle: str = "") -> str:
    """One chart: the balance curve of the five best configurations."""
    t = plotting.set_theme("light")
    fig, ax = plt.subplots(figsize=(13, 7))

    if baseline is not None:
        curve, name = baseline
        ax.plot(curve.index, curve.values, color=t["ink_muted"], linewidth=1.4,
                linestyle=(0, (5, 3)), zorder=2, label=f"{name} (indicator defaults)")

    ends = []
    for k, (curve, name, _stats) in enumerate(curves):
        colour = t["series"][k % len(t["series"])]
        ax.plot(curve.index, curve.values, color=colour, linewidth=1.9, zorder=3,
                label=f"#{k + 1}  {name}")
        ends.append({"value": float(curve.iloc[-1]), "y": float(curve.iloc[-1]),
                     "colour": colour, "x": curve.index[-1]})

    # Nudge the end labels apart so they stay readable when curves finish
    # within a few dollars of each other.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    gap = span * 0.030
    ends.sort(key=lambda e: e["y"])
    for previous, current in zip(ends, ends[1:]):
        if current["y"] - previous["y"] < gap:
            current["y"] = previous["y"] + gap
    for end in ends:
        ax.annotate(f"{end['value']:,.0f}", xy=(end["x"], end["y"]), xytext=(9, 0),
                    textcoords="offset points", color=end["colour"], fontsize=9,
                    fontweight="600", va="center", annotation_clip=False)

    ax.axhline(INITIAL_BALANCE, color=t["ink_muted"], linewidth=1.0,
               linestyle=(0, (2, 3)), alpha=0.8, zorder=1)
    ax.set_ylabel(f"Equity (USD, {INITIAL_BALANCE:,.0f} start)")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    plotting._dates(ax)
    ax.margins(x=0.07)
    ax.legend(loc="upper left", ncols=1, fontsize=9)

    plotting._finish(fig, "Gold Breakout Trader - five best configurations", subtitle)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"\nChart: {os.path.abspath(path)}")
    return path


def save_curves(curves: list, baseline, path: str) -> None:
    """Keep the winners' curves so the chart can be redrawn without re-running."""
    frame = pd.DataFrame({name: curve for curve, name, _ in curves})
    frame["__baseline__"] = baseline[0]
    frame.to_parquet(path)


# ════════════════════════════════════════════════════════════ MAIN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, default=YEARS)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="parameter sweep + top-5 chart")
    parser.add_argument("--wide", action="store_true", help="a much larger sweep")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    bars = load_data(args.years, refresh=args.refresh)
    spec = build_symbol()
    print(f"{spec.symbol}: {spec.contract_size:g} oz/lot, point value "
          f"{spec.point_value(1):.3f} USD, swap {spec.swap_type.value} {spec.swap_long:g}")

    bt = build_backtest(bars, spec)
    print(bt)

    # ── the indicator's own defaults ───────────────────────────────────────
    print("\n=== BASELINE (the indicator's default inputs) ===")
    baseline = bt.run()
    baseline.report()

    if not args.sweep:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        baseline.to_html(os.path.join(OUTPUT_DIR, "baseline.html"))
        return 0

    # ── the sweep ──────────────────────────────────────────────────────────
    bt.progress = False
    if args.wide:
        grid = {
            "box_height": [4.0, 6.0, 10.0, 16.0, 24.0, 36.0],
            "stop_buffer": [1.0, 2.0, 5.0],
            "tp_target": ["tp1", "tp2", "tp3", "scale"],
            "sl_mode": ["opposite", "box", "atr"],
        }
    else:
        grid = {
            "box_height": [6.0, 10.0, 16.0, 24.0],
            "tp_target": ["tp1", "tp2", "tp3", "scale"],
            "sl_mode": ["opposite", "box", "atr"],
        }
    total = int(np.prod([len(v) for v in grid.values()]))
    print(f"\n=== SWEEP: {total} configurations ===")

    opt = grid_search(bt, grid, objective="net_profit", min_trades=40, progress=True,
                      keep_best_result=False)

    cols = list(grid) + ["net_profit", "return_pct", "profit_factor", "win_rate",
                         "max_dd_pct", "calmar", "trades"]
    print("\nAll configurations, best first:")
    print(opt.results[cols].round(2).to_string(index=False))

    print("\nRanked by neighbourhood stability instead of the peak:")
    stable = opt.stable_top(5)
    print(stable[[*grid, "net_profit", "neighbourhood_score"]].round(2).to_string(index=False))

    # ── re-run the winners to collect their curves ─────────────────────────
    top = opt.results.head(args.top)
    print(f"\nRe-running the top {len(top)} to collect their equity curves ...")
    curves = []
    for _, row in top.iterrows():
        params = {p: (row[p] if not isinstance(row[p], np.floating) else float(row[p]))
                  for p in grid}
        result = bt.run(**params)
        curves.append((result.equity_curve, label_for(row, list(grid)), result.stats))
        print(f"  {label_for(row, list(grid)):<42} "
              f"net {result.stats['net_profit']:>10,.0f}  "
              f"PF {result.stats['profit_factor']:.2f}  "
              f"DD {result.stats['max_dd_pct']:.1f}%  "
              f"{result.stats['trades']:>4} trades")

    subtitle = (f"{SYMBOL} {TIMEFRAME} · {bars.index[0].date()} to {bars.index[-1].date()} · "
                f"{total} configurations tested · real spread, slippage, swap and latency")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reference = (baseline.equity_curve, "box 10 · TP1 · SL opposite")
    save_curves(curves, reference, os.path.join(OUTPUT_DIR, "top5_curves.parquet"))
    plot_top5(curves, baseline=reference,
              path=os.path.join(OUTPUT_DIR, "top5_equity.png"), subtitle=subtitle)

    opt.to_html(os.path.join(OUTPUT_DIR, "sweep.html"),
                title="Gold Breakout Trader - parameter sweep")
    print(f"Sweep report: {os.path.join(OUTPUT_DIR, 'sweep.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
