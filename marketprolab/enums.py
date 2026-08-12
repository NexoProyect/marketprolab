"""Enumerations mirroring MetaTrader 5 terminology."""

from __future__ import annotations

from enum import Enum, IntEnum


class OrderType(IntEnum):
    """Order types (same values as MT5's ORDER_TYPE_*)."""

    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5
    BUY_STOP_LIMIT = 6
    SELL_STOP_LIMIT = 7

    @property
    def is_buy(self) -> bool:
        return self in (
            OrderType.BUY,
            OrderType.BUY_LIMIT,
            OrderType.BUY_STOP,
            OrderType.BUY_STOP_LIMIT,
        )

    @property
    def is_sell(self) -> bool:
        return not self.is_buy

    @property
    def is_pending(self) -> bool:
        return self not in (OrderType.BUY, OrderType.SELL)


class PositionType(IntEnum):
    BUY = 0
    SELL = 1

    @property
    def sign(self) -> int:
        """+1 for longs, -1 for shorts."""
        return 1 if self is PositionType.BUY else -1


class OrderState(IntEnum):
    PLACED = 0
    PARTIAL = 1
    FILLED = 2
    CANCELED = 3
    REJECTED = 4
    EXPIRED = 5


class DealReason(IntEnum):
    CLIENT = 0
    SL = 1
    TP = 2
    STOP_OUT = 3
    SESSION_CLOSE = 4
    EXPIRY = 5
    END_OF_TEST = 6
    OPPOSITE = 7


class ExecutionMode(str, Enum):
    """Broker execution model."""

    MARKET = "market"
    INSTANT = "instant"
    REQUEST = "request"
    EXCHANGE = "exchange"


class FillingMode(str, Enum):
    """Order filling policy."""

    FOK = "fok"          # Fill or Kill
    IOC = "ioc"          # Immediate or Cancel
    RETURN = "return"    # Return / partial fills allowed
    BOC = "boc"          # Book or Cancel (passive)


class GTCMode(str, Enum):
    GTC = "gtc"                 # Good till cancelled
    DAILY = "daily"             # Cancelled at end of day
    DAILY_NO_STOPS = "daily_excluding_stops"


class ExpirationMode(str, Enum):
    ALL = "all"
    DAY = "day"
    SPECIFIED = "specified"
    SPECIFIED_DAY = "specified_day"
    GTC = "gtc"


class CalcMode(str, Enum):
    """Profit and margin calculation model."""

    FOREX = "forex"
    FOREX_NO_LEVERAGE = "forex_no_leverage"
    CFD = "cfd"
    CFD_INDEX = "cfd_index"
    CFD_LEVERAGE = "cfd_leverage"
    FUTURES = "futures"
    EXCH_STOCKS = "exch_stocks"
    CRYPTO = "crypto"


class ChartMode(str, Enum):
    BID = "bid"
    LAST = "last"


class TradeMode(str, Enum):
    DISABLED = "disabled"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    CLOSE_ONLY = "close_only"
    FULL = "full"


class SwapType(str, Enum):
    POINTS = "points"                    # swap quoted in points
    MONEY = "money"                      # money per lot, in profit currency
    MARGIN_CURRENCY = "margin_currency"  # money per lot, in margin currency
    PERCENT_ANNUAL = "percent_annual"    # annual % of notional
    PERCENT_CURRENT = "percent_current"  # daily % of notional


class MarginMode(str, Enum):
    NETTING = "netting"
    HEDGING = "hedging"


class Timeframe(IntEnum):
    """Minutes per bar. Maps to MT5's TIMEFRAME_* via :func:`to_mt5`."""

    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6
    M10 = 10
    M12 = 12
    M15 = 15
    M20 = 20
    M30 = 30
    H1 = 60
    H2 = 120
    H3 = 180
    H4 = 240
    H6 = 360
    H8 = 480
    H12 = 720
    D1 = 1440
    W1 = 10080
    MN1 = 43200

    @property
    def minutes(self) -> int:
        return int(self.value)

    @property
    def pandas_freq(self) -> str:
        if self is Timeframe.MN1:
            return "MS"
        if self is Timeframe.W1:
            return "W-MON"
        if self is Timeframe.D1:
            return "1D"
        return f"{self.minutes}min"

    def to_mt5(self) -> int:
        """Return the matching TIMEFRAME_* constant from the MetaTrader5 package."""
        import MetaTrader5 as mt5  # lazy import: only when actually used

        return getattr(mt5, f"TIMEFRAME_{self.name}")

    @classmethod
    def parse(cls, value) -> "Timeframe":
        if isinstance(value, Timeframe):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).upper()]


class PriceType(str, Enum):
    BID = "bid"
    ASK = "ask"
    MID = "mid"


class IntrabarModel(str, Enum):
    """How price is assumed to travel inside a bar."""

    PESSIMISTIC = "pessimistic"  # adverse extreme first (worst case)
    OPTIMISTIC = "optimistic"    # favourable extreme first
    OHLC = "ohlc"                # always O -> H -> L -> C
    OLHC = "olhc"                # always O -> L -> H -> C
