"""The simulated broker: execution, costs, margin and position management.

It is agnostic to both instrument and broker: all behaviour comes from
:class:`~marketprolab.symbol.SymbolSpec` (instrument conditions),
:class:`~marketprolab.broker_profile.BrokerProfile` (account conditions) and
:class:`SimulationConfig` (simulation realism).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np

from .broker_profile import BrokerProfile
from .enums import (
    DealReason,
    FillingMode,
    IntrabarModel,
    MarginMode,
    OrderState,
    OrderType,
    PositionType,
)
from .execution import (
    LatencyModel,
    SlippageModel,
    SpreadModel,
    make_latency,
    make_slippage,
    make_spread,
)
from .orders import Order, OrderRequest, OrderResult, Position, Trade
from .symbol import SymbolSpec


@dataclass
class SimulationConfig:
    """Simulation realism settings (neither broker nor instrument related)."""

    initial_balance: float = 10_000.0

    # Microstructure models
    spread: object = None       # number (points), "data", or a SpreadModel
    slippage: object = None     # number (points), or a SlippageModel
    latency: object = None      # milliseconds, or a LatencyModel

    # How price is assumed to travel inside a bar
    intrabar: IntrabarModel = IntrabarModel.PESSIMISTIC
    same_bar_exit: bool = True         # a position may close on its own bar
    allow_same_bar_fill: bool = False  # fill on the bar the order is sent
    latency_price_drift: bool = True   # price moves while the order travels
    slippage_on_limit: bool = False    # limit orders usually do not slip adversely

    # Session rules
    respect_sessions: bool = True      # do not trade outside trading hours
    close_on_session_end: bool = False
    close_on_weekend: bool = False

    # Costs
    apply_swap: bool = True
    apply_commission: bool = True

    # Currency conversion: profit currency -> account currency.
    # A number, a pandas.Series indexed by date, or a callable(dt) -> float
    profit_currency_rate: object = 1.0

    # Risk and end of test
    stop_on_bankruptcy: bool = True
    bankruptcy_equity: float = 0.0
    close_open_positions_at_end: bool = True

    # Reproducibility and diagnostics
    seed: Optional[int] = 42
    verbose: bool = False
    log_rejections: bool = True


class Broker:
    """The execution engine. Normally built for you by
    :class:`~marketprolab.engine.Backtest`."""

    def __init__(
        self,
        spec: SymbolSpec,
        config: Optional[SimulationConfig] = None,
        profile: Optional[BrokerProfile] = None,
        timeframe_seconds: int = 60,
    ):
        self.profile = profile or BrokerProfile()
        self.spec = self.profile.apply(spec)
        self.config = config or SimulationConfig()
        self.timeframe_seconds = max(1, int(timeframe_seconds))

        self.rng = np.random.default_rng(self.config.seed)
        self.spread_model: SpreadModel = make_spread(
            self.config.spread if self.config.spread is not None else self.spec.spread_points
        )
        self.slippage_model: SlippageModel = make_slippage(self.config.slippage)
        self.latency_model: LatencyModel = make_latency(self.config.latency)
        for model in (self.spread_model, self.slippage_model, self.latency_model):
            model.reset(self.rng)

        # ---- account state
        self.balance = float(self.config.initial_balance)
        self.equity = self.balance
        self.margin = 0.0
        self.credit = 0.0

        self.positions: Dict[int, Position] = {}
        self.orders: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.rejections: List[dict] = []
        self.events: List[dict] = []

        self._ticket = 0
        self._queue: List[tuple] = []     # (execute_at, OrderRequest, close_ticket, modify)
        self._bar: dict = {}
        self._index: int = -1
        self._spread_points: float = 0.0
        self._rate: float = 1.0
        self._last_rollover: Optional[datetime] = None
        self.stopped: bool = False
        self.stop_reason: str = ""

        # Result curves
        self.curve_time: List[datetime] = []
        self.curve_equity: List[float] = []
        self.curve_balance: List[float] = []
        self.curve_margin: List[float] = []
        self.curve_exposure: List[float] = []

    # ================================================================ utilities
    def _next_ticket(self) -> int:
        self._ticket += 1
        return self._ticket

    @property
    def time(self) -> datetime:
        return self._bar["time"]

    @property
    def bar_close_time(self) -> datetime:
        return self._bar["time"] + timedelta(seconds=self.timeframe_seconds)

    @property
    def spread_points(self) -> float:
        return self._spread_points

    @property
    def spread_price(self) -> float:
        return self._spread_points * self.spec.point

    @property
    def bid(self) -> float:
        return self._bar["close"]

    @property
    def ask(self) -> float:
        return self._bar["close"] + self.spread_price

    @property
    def free_margin(self) -> float:
        return self.equity - self.margin

    @property
    def margin_level(self) -> float:
        """Margin level in %. ``inf`` when there are no positions."""
        return float("inf") if self.margin <= 0 else self.equity / self.margin * 100.0

    @property
    def open_volume(self) -> float:
        return sum(p.volume for p in self.positions.values())

    @property
    def net_volume(self) -> float:
        return sum(p.volume * p.sign for p in self.positions.values())

    def _log(self, kind: str, **data) -> None:
        entry = {"time": self._bar.get("time"), "kind": kind, **data}
        self.events.append(entry)
        if self.config.verbose:
            print(f"[{entry['time']}] {kind}: {data}")

    def _reject(self, reason: str, **data) -> OrderResult:
        if self.config.log_rejections:
            self.rejections.append({"time": self._bar.get("time"), "reason": reason, **data})
        return OrderResult(ok=False, retcode=reason, comment=reason)

    # ================================================================ bar cycle
    def begin_bar(self, index: int, bar: dict) -> None:
        """Open the bar: spread, swap, queued fills, exits."""
        self._index = index
        self._bar = bar
        self._spread_points = max(0.0, float(self.spread_model(bar, self.spec)))
        self._rate = self._resolve_rate(bar["time"])

        if self.config.apply_swap:
            self._apply_swaps(bar["time"])
        self._expire_orders(bar["time"])
        self._execute_queue(bar)
        self._trigger_pending(bar)
        self._check_stops(bar)
        self._mark_to_market(bar)
        self._check_stop_out(bar)

    def end_bar(self) -> None:
        """Close the bar: session handling, curve recording."""
        bar = self._bar
        if self.config.close_on_session_end and self.positions:
            if self.spec.trade_sessions.is_close_of_session(
                self.bar_close_time, tolerance_min=max(1, self.timeframe_seconds // 60)
            ):
                self.close_all(reason=DealReason.SESSION_CLOSE)
        if (self.config.close_on_weekend or self.profile.weekend_close) and self.positions:
            if self.spec.trade_sessions.is_last_session_of_week(self.bar_close_time):
                self.close_all(reason=DealReason.SESSION_CLOSE)

        self._mark_to_market(bar)
        self.curve_time.append(bar["time"])
        self.curve_equity.append(self.equity)
        self.curve_balance.append(self.balance)
        self.curve_margin.append(self.margin)
        self.curve_exposure.append(self.open_volume)

    def finalize(self, bar: dict) -> None:
        """Flatten whatever is still open when the test ends."""
        self._bar = bar
        if self.config.close_open_positions_at_end:
            self.close_all(reason=DealReason.END_OF_TEST)
        for order in list(self.orders.values()):
            order.state = OrderState.CANCELED
            self.orders.pop(order.ticket, None)
        self._mark_to_market(bar)

    # --------------------------------------------------------------- currency
    def _resolve_rate(self, when: datetime) -> float:
        rate = self.config.profit_currency_rate
        if callable(rate):
            return float(rate(when))
        if hasattr(rate, "asof"):          # pandas.Series
            try:
                value = rate.asof(when)
                return float(value) if value == value else 1.0
            except Exception:
                return 1.0
        return float(rate)

    # ------------------------------------------------------------------- swap
    def _apply_swaps(self, now: datetime) -> None:
        hour = self.spec.swap_rollover_hour
        rollover = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if self._last_rollover is None:
            # Anchor the first rollover to the broker's clock, not the first bar.
            self._last_rollover = rollover if now >= rollover else rollover - timedelta(days=1)
            return
        while self._last_rollover + timedelta(days=1) <= now:
            self._last_rollover += timedelta(days=1)
            when = self._last_rollover
            for pos in self.positions.values():
                cost = self.spec.swap_cost(pos.sign, pos.volume, pos.open_price, when, self._rate)
                pos.swap += cost

    # ---------------------------------------------------------- latency queue
    def _execute_queue(self, bar: dict) -> None:
        if not self._queue:
            return
        bar_end = bar["time"] + timedelta(seconds=self.timeframe_seconds)
        remaining = []
        for execute_at, request, close_ticket, modify in self._queue:
            if execute_at >= bar_end:
                remaining.append((execute_at, request, close_ticket, modify))
                continue
            frac = 0.0
            if self.config.latency_price_drift and execute_at > bar["time"]:
                elapsed = (execute_at - bar["time"]).total_seconds()
                frac = min(1.0, elapsed / self.timeframe_seconds)
            ref = bar["open"] + (bar["close"] - bar["open"]) * frac
            if modify is not None:
                self._do_modify(modify, request)
            elif close_ticket is not None:
                self._do_close(close_ticket, ref, DealReason.CLIENT, request.volume)
            else:
                self._do_market(request, ref, bar)
        self._queue = remaining

    # --------------------------------------------------------- pending orders
    def _expire_orders(self, now: datetime) -> None:
        for ticket, order in list(self.orders.items()):
            if order.expiration and now >= order.expiration:
                order.state = OrderState.EXPIRED
                self.orders.pop(ticket, None)
                self._log("order_expired", ticket=ticket)

    def _trigger_pending(self, bar: dict) -> None:
        if not self.orders:
            return
        spread = self.spread_price
        high_bid, low_bid, open_bid = bar["high"], bar["low"], bar["open"]
        high_ask, low_ask, open_ask = high_bid + spread, low_bid + spread, open_bid + spread

        for ticket, order in list(self.orders.items()):
            if order.time_execute and bar["time"] + timedelta(
                seconds=self.timeframe_seconds
            ) <= order.time_execute:
                continue
            level = order.price
            if level is None:
                continue
            fill: Optional[float] = None
            kind = "stop"

            if order.order_type is OrderType.BUY_STOP and high_ask >= level:
                fill = max(level, open_ask) if open_ask > level else level
            elif order.order_type is OrderType.SELL_STOP and low_bid <= level:
                fill = min(level, open_bid) if open_bid < level else level
            elif order.order_type is OrderType.BUY_LIMIT and low_ask <= level:
                fill, kind = (min(level, open_ask) if open_ask < level else level), "limit"
            elif order.order_type is OrderType.SELL_LIMIT and high_bid >= level:
                fill, kind = (max(level, open_bid) if open_bid > level else level), "limit"

            if fill is None:
                continue

            side = 1 if order.order_type.is_buy else -1
            slip = 0.0
            if kind == "stop" or self.config.slippage_on_limit:
                slip = self.slippage_model(bar, self.spec, side, order.volume, kind)
            price = fill + side * slip * self.spec.point
            result = self._open_position(
                PositionType.BUY if order.order_type.is_buy else PositionType.SELL,
                order.volume,
                price,
                requested=level,
                slippage_points=slip,
                sl=order.sl,
                tp=order.tp,
                comment=order.comment,
                magic=order.magic,
                tag=order.tag,
                bar=bar,
            )
            if result.ok:
                order.state = OrderState.FILLED
                self.orders.pop(ticket, None)
            elif result.retcode == "no_margin":
                order.state = OrderState.REJECTED
                self.orders.pop(ticket, None)

    # ----------------------------------------------------------------- SL / TP
    def _check_stops(self, bar: dict) -> None:
        if not self.positions:
            return
        spread = self.spread_price
        for pos in list(self.positions.values()):
            if not self.config.same_bar_exit and pos.open_bar == self._index:
                continue
            hit_sl = hit_tp = False
            sl_price = tp_price = None

            if pos.is_long:
                # A long's SL/TP are evaluated against the bid
                if pos.sl is not None and bar["low"] <= pos.sl:
                    hit_sl, sl_price = True, min(pos.sl, bar["open"])
                if pos.tp is not None and bar["high"] >= pos.tp:
                    hit_tp, tp_price = True, max(pos.tp, bar["open"])
            else:
                # A short's SL/TP are evaluated against the ask
                if pos.sl is not None and bar["high"] + spread >= pos.sl:
                    hit_sl, sl_price = True, max(pos.sl, bar["open"] + spread)
                if pos.tp is not None and bar["low"] + spread <= pos.tp:
                    hit_tp, tp_price = True, min(pos.tp, bar["open"] + spread)

            if not (hit_sl or hit_tp):
                continue

            first = self._resolve_intrabar_order(hit_sl, hit_tp, pos)
            if first == "sl":
                slip = self.slippage_model(bar, self.spec, pos.sign, pos.volume, "sl")
                price = sl_price - pos.sign * slip * self.spec.point
                self._close_position(pos, price, DealReason.SL, bar, slippage=slip)
            else:
                # Take profits are limit orders: no adverse slippage
                self._close_position(pos, tp_price, DealReason.TP, bar, slippage=0.0)

    def _resolve_intrabar_order(self, hit_sl: bool, hit_tp: bool, pos: Position) -> str:
        if hit_sl and not hit_tp:
            return "sl"
        if hit_tp and not hit_sl:
            return "tp"
        model = self.config.intrabar
        if model is IntrabarModel.PESSIMISTIC:
            return "sl"
        if model is IntrabarModel.OPTIMISTIC:
            return "tp"
        # OHLC: O->H->L->C  |  OLHC: O->L->H->C
        up_first = model is IntrabarModel.OHLC
        if pos.is_long:
            return "tp" if up_first else "sl"
        return "sl" if up_first else "tp"

    # ----------------------------------------------------------- mark to market
    def _mark_to_market(self, bar: dict) -> None:
        spread = self.spread_price
        floating = 0.0
        for pos in self.positions.values():
            close_price = bar["close"] if pos.is_long else bar["close"] + spread
            pos.profit = self.spec.profit(pos.sign, pos.volume, pos.open_price,
                                          close_price, self._rate)
            floating += pos.profit + pos.swap

            # MAE / MFE from the bar extremes
            best_price = bar["high"] if pos.is_long else bar["low"] + spread
            worst_price = bar["low"] if pos.is_long else bar["high"] + spread
            best = self.spec.profit(pos.sign, pos.volume, pos.open_price, best_price, self._rate)
            worst = self.spec.profit(pos.sign, pos.volume, pos.open_price, worst_price, self._rate)
            pos.mfe = max(pos.mfe, best)
            pos.mae = min(pos.mae, worst)
            pos.mfe_points = max(pos.mfe_points,
                                 (best_price - pos.open_price) * pos.sign / self.spec.point)
            pos.mae_points = min(pos.mae_points,
                                 (worst_price - pos.open_price) * pos.sign / self.spec.point)

        self.equity = self.balance + self.credit + floating
        self.margin = sum(p.margin for p in self.positions.values())

    # -------------------------------------------------------------- stop out
    def _check_stop_out(self, bar: dict) -> None:
        if self.stopped:
            return
        if self.config.stop_on_bankruptcy and self.equity <= self.config.bankruptcy_equity:
            self.close_all(reason=DealReason.STOP_OUT)
            self.stopped = True
            self.stop_reason = "bankruptcy"
            self._log("bankruptcy", equity=self.equity)
            return
        if not self.positions:
            return
        level = self.profile.stop_out_level
        if level <= 0:
            return
        guard = 0
        while self.positions and guard < 1000:
            breached = (
                self.free_margin <= level if self.profile.stop_out_in_money
                else self.margin_level < level
            )
            if not breached:
                break
            worst = min(self.positions.values(), key=lambda p: p.profit + p.swap)
            spread = self.spread_price
            price = bar["close"] if worst.is_long else bar["close"] + spread
            slip = self.slippage_model(bar, self.spec, worst.sign, worst.volume, "stop_out")
            price -= worst.sign * slip * self.spec.point
            self._close_position(worst, price, DealReason.STOP_OUT, bar, slippage=slip)
            self._mark_to_market(bar)
            guard += 1
            self._log("stop_out", ticket=worst.ticket, margin_level=self.margin_level)

    # ============================================================== trading API
    def buy(self, volume: float = None, sl=None, tp=None, **kwargs) -> OrderResult:
        return self.send(OrderType.BUY, volume, sl=sl, tp=tp, **kwargs)

    def sell(self, volume: float = None, sl=None, tp=None, **kwargs) -> OrderResult:
        return self.send(OrderType.SELL, volume, sl=sl, tp=tp, **kwargs)

    def buy_limit(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.send(OrderType.BUY_LIMIT, volume, price=price, **kwargs)

    def sell_limit(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.send(OrderType.SELL_LIMIT, volume, price=price, **kwargs)

    def buy_stop(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.send(OrderType.BUY_STOP, volume, price=price, **kwargs)

    def sell_stop(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.send(OrderType.SELL_STOP, volume, price=price, **kwargs)

    def send(
        self,
        order_type: OrderType,
        volume: Optional[float] = None,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: Optional[float] = None,
        filling: Optional[FillingMode] = None,
        expiration: Optional[datetime] = None,
        comment: str = "",
        magic: int = 0,
        tag: str = "",
        **meta,
    ) -> OrderResult:
        """Send an order. Market orders go through the latency queue."""
        volume = self.spec.volume_min if volume is None else volume
        volume = self.spec.normalize_volume(volume)

        ok, why = self._validate(order_type, volume, price, sl, tp, filling)
        if not ok:
            return self._reject(why, order_type=order_type.name, volume=volume, price=price)

        request = OrderRequest(
            order_type=order_type, volume=volume, price=price, sl=sl, tp=tp,
            deviation=deviation, filling=filling, expiration=expiration,
            comment=comment, magic=magic, tag=tag, meta=meta,
        )

        if order_type.is_pending:
            ticket = self._next_ticket()
            order = Order(
                ticket=ticket, order_type=order_type, volume=volume, price=price,
                sl=sl, tp=tp, time_setup=self.bar_close_time,
                time_execute=self.bar_close_time + timedelta(
                    seconds=self.latency_model("pending")
                ),
                expiration=expiration or self._gtc_expiration(),
                filling=filling, deviation=deviation,
                comment=comment, magic=magic, tag=tag, meta=meta,
            )
            self.orders[ticket] = order
            self._log("order_placed", ticket=ticket, type=order_type.name, price=price)
            return OrderResult(ok=True, retcode="placed", ticket=ticket, order=order)

        # Market order
        latency = self.latency_model("market")
        if self.config.allow_same_bar_fill and latency <= 0:
            return self._do_market(request, self._bar["close"], self._bar)
        execute_at = self.bar_close_time + timedelta(seconds=latency)
        self._queue.append((execute_at, request, None, None))
        return OrderResult(ok=True, retcode="queued", price=self._bar["close"],
                           volume=volume, comment="queued behind latency")

    def _gtc_expiration(self) -> Optional[datetime]:
        from .enums import GTCMode

        if self.spec.gtc_mode is GTCMode.GTC:
            return None
        return self._bar["time"].replace(hour=23, minute=59, second=59)

    # ------------------------------------------------------------- validation
    def _validate(self, order_type, volume, price, sl, tp, filling):
        if self.stopped:
            return False, "trading_stopped"
        if volume <= 0 or not self.spec.is_valid_volume(volume):
            return False, "invalid_volume"
        if not self.spec.can_open(order_type):
            return False, "trade_disabled"
        if self.config.respect_sessions and not self.spec.trade_sessions.is_open(
            self.bar_close_time
        ):
            return False, "market_closed"
        if filling is not None and filling not in self.spec.filling_modes:
            return False, "unsupported_filling"
        if order_type.is_pending and price is None:
            return False, "missing_price"
        if self.profile.max_positions and len(self.positions) >= self.profile.max_positions:
            return False, "max_positions"
        if self.profile.max_pending_orders and len(self.orders) >= self.profile.max_pending_orders:
            return False, "max_pending_orders"
        limit = self.profile.max_volume_total or self.spec.volume_limit or self.spec.position_limit
        if limit and self.open_volume + volume > limit + 1e-9:
            return False, "volume_limit"
        ref = price if price is not None else self._bar.get("close")
        if ref is not None:
            if not self.spec.check_stops_distance(ref, sl):
                return False, "invalid_stops"
            if not self.spec.check_stops_distance(ref, tp):
                return False, "invalid_stops"
        return True, "ok"

    # ------------------------------------------------------- market execution
    def _do_market(self, request: OrderRequest, reference: float, bar: dict) -> OrderResult:
        side = 1 if request.order_type.is_buy else -1
        spread = self.spread_price
        slip = self.slippage_model(bar, self.spec, side, request.volume, "market")
        base = reference + spread if side > 0 else reference
        price = self.spec.normalize_price(base + side * slip * self.spec.point)

        if request.deviation is not None:
            requested = request.price if request.price is not None else reference
            if abs(price - requested) > request.deviation * self.spec.point:
                return self._reject("price_off_deviation", price=price, requested=requested)

        # Netting: an opposite order closes the existing position
        if self.profile.margin_mode is MarginMode.NETTING or not self.profile.hedging_allowed:
            opposite = [
                p for p in self.positions.values()
                if (p.is_long and side < 0) or (p.is_short and side > 0)
            ]
            remaining = request.volume
            for pos in sorted(opposite, key=lambda p: p.open_time):
                if remaining <= 1e-9:
                    break
                closing = min(pos.volume, remaining)
                close_price = bar["close"] if pos.is_long else bar["close"] + spread
                self._close_position(pos, close_price, DealReason.OPPOSITE, bar,
                                     slippage=slip, volume=closing)
                remaining -= closing
            if remaining <= 1e-9:
                return OrderResult(ok=True, retcode="netted", price=price, volume=request.volume)
            request.volume = self.spec.normalize_volume(remaining)

        return self._open_position(
            PositionType.BUY if side > 0 else PositionType.SELL,
            request.volume, price,
            requested=reference, slippage_points=slip,
            sl=request.sl, tp=request.tp, comment=request.comment,
            magic=request.magic, tag=request.tag, bar=bar, meta=request.meta,
        )

    def _open_position(self, ptype: PositionType, volume: float, price: float,
                       requested: float, slippage_points: float,
                       sl=None, tp=None, comment="", magic=0, tag="",
                       bar: dict = None, meta: dict = None) -> OrderResult:
        bar = bar or self._bar
        price = self.spec.normalize_price(price)
        margin = self.spec.margin_required(volume, price, self._rate, self.profile.leverage)
        if margin > self.free_margin + 1e-9:
            return self._reject("no_margin", required=margin, free=self.free_margin)

        commission = (
            self.spec.commission(volume, price, self._rate) if self.config.apply_commission else 0.0
        )
        ticket = self._next_ticket()
        pos = Position(
            ticket=ticket, symbol=self.spec.symbol, type=ptype, volume=volume,
            open_time=bar["time"], open_price=price,
            sl=self.spec.normalize_price(sl) if sl is not None else None,
            tp=self.spec.normalize_price(tp) if tp is not None else None,
            commission=commission, margin=margin, comment=comment, magic=magic, tag=tag,
            open_bar=self._index, requested_price=requested, slippage_points=slippage_points,
            meta=meta or {},
        )
        self.balance -= commission
        self.positions[ticket] = pos
        self._mark_to_market(bar)
        self._log("open", ticket=ticket, type=ptype.name, volume=volume, price=price,
                  slippage=slippage_points)
        return OrderResult(ok=True, retcode="filled", ticket=ticket, position=pos,
                           price=price, volume=volume)

    # ---------------------------------------------------------------- closing
    def close(self, position, volume: Optional[float] = None,
              reason: DealReason = DealReason.CLIENT) -> OrderResult:
        """Close a position, fully or partially, honouring latency."""
        ticket = position.ticket if isinstance(position, Position) else int(position)
        if ticket not in self.positions:
            return self._reject("position_not_found", ticket=ticket)
        latency = self.latency_model("close")
        if self.config.allow_same_bar_fill and latency <= 0:
            return self._do_close(ticket, self._bar["close"], reason, volume)
        request = OrderRequest(order_type=OrderType.SELL, volume=volume or 0.0)
        self._queue.append(
            (self.bar_close_time + timedelta(seconds=latency), request, ticket, None)
        )
        return OrderResult(ok=True, retcode="queued", ticket=ticket)

    def close_all(self, reason: DealReason = DealReason.CLIENT,
                  only: Optional[Callable[[Position], bool]] = None) -> int:
        """Close every position immediately (optionally filtered)."""
        count = 0
        bar = self._bar
        spread = self.spread_price
        for pos in list(self.positions.values()):
            if only and not only(pos):
                continue
            price = bar["close"] if pos.is_long else bar["close"] + spread
            slip = self.slippage_model(bar, self.spec, pos.sign, pos.volume, "market")
            price -= pos.sign * slip * self.spec.point
            self._close_position(pos, price, reason, bar, slippage=slip)
            count += 1
        return count

    def _do_close(self, ticket: int, reference: float, reason: DealReason,
                  volume: Optional[float]) -> OrderResult:
        pos = self.positions.get(ticket)
        if pos is None:
            return self._reject("position_not_found", ticket=ticket)
        bar = self._bar
        spread = self.spread_price
        price = reference if pos.is_long else reference + spread
        slip = self.slippage_model(bar, self.spec, pos.sign, pos.volume, "market")
        price -= pos.sign * slip * self.spec.point
        trade = self._close_position(pos, price, reason, bar, slippage=slip,
                                     volume=volume if volume else None)
        return OrderResult(ok=True, retcode="closed", ticket=ticket, price=price,
                           volume=trade.volume if trade else 0.0)

    def _close_position(self, pos: Position, price: float, reason: DealReason,
                        bar: dict, slippage: float = 0.0,
                        volume: Optional[float] = None) -> Optional[Trade]:
        price = self.spec.normalize_price(price)
        volume = pos.volume if volume is None else min(volume, pos.volume)
        volume = self.spec.normalize_volume(volume) if volume < pos.volume else pos.volume
        if volume <= 0:
            return None
        ratio = volume / pos.volume

        profit = self.spec.profit(pos.sign, volume, pos.open_price, price, self._rate)
        swap = pos.swap * ratio
        open_comm = pos.commission * ratio
        close_comm = (
            self.spec.commission(volume, price, self._rate) if self.config.apply_commission else 0.0
        )
        total_comm = open_comm + close_comm
        net = profit + swap - total_comm

        self.balance += profit + swap - close_comm

        close_time = bar["time"] + timedelta(seconds=self.timeframe_seconds)
        trade = Trade(
            ticket=pos.ticket, symbol=pos.symbol, type=pos.type, volume=volume,
            open_time=pos.open_time, open_price=pos.open_price,
            close_time=close_time, close_price=price,
            profit=profit, commission=total_comm, swap=swap, net_profit=net,
            reason=reason, sl=pos.sl, tp=pos.tp,
            bars_held=self._index - pos.open_bar,
            duration_s=(close_time - pos.open_time).total_seconds(),
            mae=pos.mae * ratio, mfe=pos.mfe * ratio,
            mae_points=pos.mae_points, mfe_points=pos.mfe_points,
            slippage_points=pos.slippage_points + slippage,
            balance_after=self.balance, equity_after=self.equity,
            comment=pos.comment, magic=pos.magic, tag=pos.tag, meta=dict(pos.meta),
        )
        self.trades.append(trade)

        if volume >= pos.volume - 1e-9:
            self.positions.pop(pos.ticket, None)
        else:
            pos.volume -= volume
            pos.swap -= swap
            pos.commission -= open_comm
            pos.margin *= 1 - ratio

        self._mark_to_market(bar)
        self._log("close", ticket=pos.ticket, price=price, net=net, reason=reason.name)
        return trade

    # ------------------------------------------------------------- modifying
    def modify(self, position, sl: Optional[float] = "keep",
               tp: Optional[float] = "keep") -> OrderResult:
        """Modify SL/TP. Pass ``None`` to remove a level."""
        ticket = position.ticket if isinstance(position, Position) else int(position)
        pos = self.positions.get(ticket)
        if pos is None:
            return self._reject("position_not_found", ticket=ticket)
        request = OrderRequest(order_type=OrderType.BUY, volume=pos.volume,
                               sl=sl, tp=tp, meta={"modify": True})
        latency = self.latency_model("modify")
        if self.config.allow_same_bar_fill and latency <= 0:
            return self._do_modify(ticket, request)
        self._queue.append((self.bar_close_time + timedelta(seconds=latency),
                            request, None, ticket))
        return OrderResult(ok=True, retcode="queued", ticket=ticket)

    def _do_modify(self, ticket: int, request: OrderRequest) -> OrderResult:
        pos = self.positions.get(ticket)
        if pos is None:
            return self._reject("position_not_found", ticket=ticket)
        reference = self._bar["close"]
        if request.sl != "keep":
            if request.sl is not None and not self.spec.check_stops_distance(reference, request.sl):
                return self._reject("invalid_stops", sl=request.sl)
            pos.sl = self.spec.normalize_price(request.sl) if request.sl is not None else None
        if request.tp != "keep":
            if request.tp is not None and not self.spec.check_stops_distance(reference, request.tp):
                return self._reject("invalid_stops", tp=request.tp)
            pos.tp = self.spec.normalize_price(request.tp) if request.tp is not None else None
        self._log("modify", ticket=ticket, sl=pos.sl, tp=pos.tp)
        return OrderResult(ok=True, retcode="modified", ticket=ticket, position=pos)

    def cancel(self, order) -> OrderResult:
        ticket = order.ticket if isinstance(order, Order) else int(order)
        if ticket not in self.orders:
            return self._reject("order_not_found", ticket=ticket)
        self.orders.pop(ticket).state = OrderState.CANCELED
        self._log("cancel", ticket=ticket)
        return OrderResult(ok=True, retcode="canceled", ticket=ticket)

    def cancel_all(self) -> int:
        count = len(self.orders)
        for ticket in list(self.orders):
            self.cancel(ticket)
        return count

    # ---------------------------------------------------------- sizing helpers
    def volume_for_risk(self, risk_amount: float, stop_points: float,
                        price: Optional[float] = None) -> float:
        """Lot size that risks ``risk_amount`` of the account for a given stop."""
        if stop_points <= 0:
            return 0.0
        value_per_point = self.spec.point_value(1.0, self._rate)
        if value_per_point <= 0:
            return 0.0
        raw = risk_amount / (stop_points * value_per_point)
        return self.spec.normalize_volume(raw)

    def volume_for_risk_pct(self, risk_pct: float, stop_points: float) -> float:
        """Same as :meth:`volume_for_risk` but as a percentage of equity."""
        return self.volume_for_risk(self.equity * risk_pct / 100.0, stop_points)

    def points_for_money(self, money: float, volume: float) -> float:
        value = self.spec.point_value(volume, self._rate)
        return money / value if value else 0.0

    # ------------------------------------------------------------------ queries
    def get_positions(self, magic: Optional[int] = None, tag: Optional[str] = None
                      ) -> List[Position]:
        items = list(self.positions.values())
        if magic is not None:
            items = [p for p in items if p.magic == magic]
        if tag is not None:
            items = [p for p in items if p.tag == tag]
        return items

    def has_position(self, direction: Optional[str] = None) -> bool:
        if direction is None:
            return bool(self.positions)
        want = PositionType.BUY if direction.lower().startswith("b") else PositionType.SELL
        return any(p.type is want for p in self.positions.values())
