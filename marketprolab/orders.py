"""Order, position and closed-trade structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .enums import DealReason, FillingMode, OrderState, OrderType, PositionType


@dataclass
class OrderRequest:
    """A request sent to the simulated broker (mirrors ``MqlTradeRequest``)."""

    order_type: OrderType
    volume: float
    price: Optional[float] = None      # pending orders only
    sl: Optional[float] = None
    tp: Optional[float] = None
    stop_limit: Optional[float] = None
    deviation: Optional[float] = None  # max accepted deviation, in points
    filling: Optional[FillingMode] = None
    expiration: Optional[datetime] = None
    comment: str = ""
    magic: int = 0
    tag: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """A live order on the server (pending, or market queued behind latency)."""

    ticket: int
    order_type: OrderType
    volume: float
    price: Optional[float]
    sl: Optional[float] = None
    tp: Optional[float] = None
    stop_limit: Optional[float] = None
    state: OrderState = OrderState.PLACED
    time_setup: Optional[datetime] = None
    time_execute: Optional[datetime] = None   # earliest execution instant (latency)
    expiration: Optional[datetime] = None
    filling: Optional[FillingMode] = None
    deviation: Optional[float] = None
    comment: str = ""
    magic: int = 0
    tag: str = ""
    close_position: Optional[int] = None      # set when the order closes a position
    modify_position: Optional[int] = None     # set when the order modifies SL/TP
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        return self.order_type.is_pending


@dataclass
class Position:
    """An open position."""

    ticket: int
    symbol: str
    type: PositionType
    volume: float
    open_time: datetime
    open_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0            # floating, refreshed every bar
    margin: float = 0.0
    comment: str = ""
    magic: int = 0
    tag: str = ""
    open_bar: int = 0
    requested_price: float = 0.0   # intended price before slippage
    slippage_points: float = 0.0
    mae: float = 0.0               # maximum adverse excursion (money)
    mfe: float = 0.0               # maximum favourable excursion (money)
    mae_points: float = 0.0
    mfe_points: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return self.type.sign

    @property
    def is_long(self) -> bool:
        return self.type is PositionType.BUY

    @property
    def is_short(self) -> bool:
        return self.type is PositionType.SELL


@dataclass
class Trade:
    """A closed trade (what ends up in the report)."""

    ticket: int
    symbol: str
    type: PositionType
    volume: float
    open_time: datetime
    open_price: float
    close_time: datetime
    close_price: float
    profit: float                  # gross, before costs
    commission: float
    swap: float
    net_profit: float              # profit - commission + swap
    reason: DealReason
    sl: Optional[float] = None
    tp: Optional[float] = None
    bars_held: int = 0
    duration_s: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    mae_points: float = 0.0
    mfe_points: float = 0.0
    slippage_points: float = 0.0
    balance_after: float = 0.0
    equity_after: float = 0.0
    comment: str = ""
    magic: int = 0
    tag: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.net_profit > 0

    @property
    def return_points(self) -> float:
        return (self.close_price - self.open_price) * self.type.sign

    def to_dict(self) -> dict:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "type": "buy" if self.type is PositionType.BUY else "sell",
            "volume": self.volume,
            "open_time": self.open_time,
            "open_price": self.open_price,
            "close_time": self.close_time,
            "close_price": self.close_price,
            "sl": self.sl,
            "tp": self.tp,
            "profit": self.profit,
            "commission": self.commission,
            "swap": self.swap,
            "net_profit": self.net_profit,
            "reason": self.reason.name,
            "bars_held": self.bars_held,
            "duration_s": self.duration_s,
            "mae": self.mae,
            "mfe": self.mfe,
            "mae_points": self.mae_points,
            "mfe_points": self.mfe_points,
            "slippage_points": self.slippage_points,
            "balance_after": self.balance_after,
            "equity_after": self.equity_after,
            "comment": self.comment,
            "magic": self.magic,
            "tag": self.tag,
        }


@dataclass
class OrderResult:
    """The broker's answer to a request (mirrors ``MqlTradeResult``)."""

    ok: bool
    retcode: str
    ticket: Optional[int] = None
    order: Optional[Order] = None
    position: Optional[Position] = None
    price: Optional[float] = None
    volume: float = 0.0
    comment: str = ""

    def __bool__(self) -> bool:
        return self.ok
