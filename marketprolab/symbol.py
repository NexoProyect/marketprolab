"""Full instrument specification, as published by the broker in MetaTrader 5."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .enums import (
    CalcMode,
    ChartMode,
    ExecutionMode,
    ExpirationMode,
    FillingMode,
    GTCMode,
    OrderType,
    SwapType,
    TradeMode,
)
from .sessions import SessionSpec

# MT5's SYMBOL_SWAP_MODE enumeration, mapped to this library's swap types.
# Getting it wrong is expensive: on a one-ounce gold contract, reading -531.6 as
# money instead of points overstates the overnight cost a thousandfold.
MT5_SWAP_MODES = {
    0: SwapType.POINTS,          # DISABLED - the values are zero anyway
    1: SwapType.POINTS,          # POINTS
    2: SwapType.MONEY,           # CURRENCY_SYMBOL
    3: SwapType.MARGIN_CURRENCY,  # CURRENCY_MARGIN
    4: SwapType.MONEY,           # CURRENCY_DEPOSIT
    5: SwapType.PERCENT_ANNUAL,  # INTEREST_CURRENT
    6: SwapType.PERCENT_ANNUAL,  # INTEREST_OPEN
    7: SwapType.PERCENT_ANNUAL,  # REOPEN_CURRENT
    8: SwapType.PERCENT_ANNUAL,  # REOPEN_BID
}


@dataclass
class SymbolSpec:
    """Every contract specification the simulator needs.

    Defaults describe a generic 5-digit CFD; the usual workflow is to fill in
    the fields by copying MetaTrader 5's *Specification* window, or to call
    :meth:`from_mt5` to read them straight from the terminal.
    """

    # ---------------------------------------------------- instrument identity
    symbol: str
    name: str = ""
    category: str = ""            # Metals, Indices, Crypto, Forex, Energies...
    asset_type: str = "cfd"       # spot | futures | cfd | etf | stock
    description: str = ""

    # ------------------------------------------------ contract specifications
    digits: int = 5
    contract_size: float = 100_000.0
    margin_currency: str = "EUR"      # margin (collateral) currency
    profit_currency: str = "USD"      # profit / quote currency
    base_currency: str = ""           # falls back to margin_currency if empty
    calc_mode: CalcMode = CalcMode.FOREX
    chart_mode: ChartMode = ChartMode.BID
    tick_size: Optional[float] = None  # None => one point (10**-digits)

    # ------------------------------------------------- trading and execution
    execution: ExecutionMode = ExecutionMode.MARKET
    gtc_mode: GTCMode = GTCMode.GTC
    filling_modes: List[FillingMode] = field(
        default_factory=lambda: [FillingMode.FOK, FillingMode.IOC]
    )
    expiration_modes: List[ExpirationMode] = field(
        default_factory=lambda: [ExpirationMode.ALL]
    )
    allowed_orders: List[OrderType] = field(default_factory=lambda: list(OrderType))
    trade_mode: TradeMode = TradeMode.FULL
    stops_level: int = 0          # minimum SL/TP distance from price, in points
    freeze_level: int = 0         # frozen zone around price, in points

    # ------------------------------------------------------------------ volume
    volume_min: float = 0.01
    volume_max: float = 200.0
    volume_step: float = 0.01
    volume_limit: float = 0.0     # max aggregate volume per symbol (0 = no limit)

    # -------------------------------------------------------- costs and swaps
    spread_points: float = 0.0    # average spread in points (when no real data)
    spread_float: bool = True
    commission_per_lot: float = 0.0     # per lot, per side (round turn = x2)
    commission_per_deal: float = 0.0    # flat per deal
    commission_percent: float = 0.0     # % of notional, per side
    swap_type: SwapType = SwapType.POINTS
    swap_long: float = 0.0
    swap_short: float = 0.0
    swap_rate_days: Dict[int, int] = field(
        default_factory=lambda: {0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0}
    )
    swap_rollover_hour: int = 0   # server hour when swap is charged
    swap_year_days: int = 360     # day count basis for annual percentage swaps

    # ------------------------------------------------------ margin & leverage
    leverage: float = 100.0
    margin_initial_rate: float = 1.0        # initial margin multiplier
    margin_maintenance_rate: float = 1.0    # maintenance margin multiplier
    margin_initial_per_lot: Optional[float] = None      # flat override (account ccy)
    margin_maintenance_per_lot: Optional[float] = None  # flat override
    margin_hedged_per_lot: float = 0.0
    max_leverage: Optional[float] = None

    # --------------------------------------------------------------- schedule
    quote_sessions: SessionSpec = field(default_factory=SessionSpec.always_open)
    trade_sessions: SessionSpec = field(default_factory=SessionSpec.always_open)
    timezone: str = ""            # reference timezone of the server

    # ------------------------------------------------------------------ other
    expiration_date: Optional[datetime] = None
    position_limit: float = 0.0        # max aggregate volume (0 = no limit)
    pending_orders_limit: int = 0      # max pending orders (0 = no limit)
    country_restrictions: List[str] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------- properties
    def __post_init__(self) -> None:
        if not self.base_currency:
            self.base_currency = self.margin_currency
        if not self.name:
            self.name = self.symbol
        if self.tick_size is None:
            self.tick_size = self.point
        if isinstance(self.quote_sessions, dict):
            self.quote_sessions = SessionSpec.from_dict(self.quote_sessions)
        if isinstance(self.trade_sessions, dict):
            self.trade_sessions = SessionSpec.from_dict(self.trade_sessions)

    @property
    def point(self) -> float:
        """Value of one point in price units (10**-digits)."""
        return 10.0 ** (-self.digits)

    @property
    def pip(self) -> float:
        """Value of one pip. For 3/5-digit symbols a pip is 10 points."""
        return self.point * (10 if self.digits in (3, 5) else 1)

    @property
    def points_per_pip(self) -> int:
        return 10 if self.digits in (3, 5) else 1

    # ------------------------------------------------------------ conversions
    def normalize_price(self, price: float) -> float:
        """Round to the symbol's tick size / digits."""
        if self.tick_size and self.tick_size > 0:
            return round(round(price / self.tick_size) * self.tick_size, self.digits)
        return round(price, self.digits)

    def normalize_volume(self, volume: float) -> float:
        """Snap volume to the step and clamp to [volume_min, volume_max]."""
        if volume <= 0:
            return 0.0
        step = self.volume_step or 0.01
        vol = math.floor(round(volume / step, 8)) * step
        vol = max(self.volume_min, min(self.volume_max, vol))
        decimals = max(0, -int(math.floor(math.log10(step))))
        return round(vol, decimals + 2)

    def is_valid_volume(self, volume: float) -> bool:
        if volume < self.volume_min - 1e-12 or volume > self.volume_max + 1e-12:
            return False
        step = self.volume_step or 0.01
        return abs(round(volume / step) * step - volume) < 1e-8

    def points_to_price(self, points: float) -> float:
        return points * self.point

    def price_to_points(self, price_delta: float) -> float:
        return price_delta / self.point

    def pips_to_price(self, pips: float) -> float:
        return pips * self.pip

    # ----------------------------------------------------------- value & cost
    def point_value(self, volume: float = 1.0, rate: float = 1.0) -> float:
        """Money value of one point for ``volume`` lots, in account currency.

        ``rate`` is the profit-currency -> account-currency exchange rate.
        """
        return self.contract_size * self.point * volume * rate

    def tick_value(self, volume: float = 1.0, rate: float = 1.0) -> float:
        return self.contract_size * (self.tick_size or self.point) * volume * rate

    def profit(self, position_type_sign: int, volume: float, open_price: float,
               close_price: float, rate: float = 1.0) -> float:
        """Gross profit in account currency."""
        return (close_price - open_price) * position_type_sign * volume * self.contract_size * rate

    def notional(self, volume: float, price: float, rate: float = 1.0) -> float:
        """Notional value of the position in account currency."""
        return volume * self.contract_size * price * rate

    def commission(self, volume: float, price: float, rate: float = 1.0) -> float:
        """Commission for one side (open or close)."""
        cost = self.commission_per_lot * volume + self.commission_per_deal
        if self.commission_percent:
            cost += self.notional(volume, price, rate) * self.commission_percent / 100.0
        return cost

    def margin_required(self, volume: float, price: float, rate: float = 1.0,
                        leverage: Optional[float] = None,
                        maintenance: bool = False) -> float:
        """Initial (or maintenance) margin in account currency."""
        override = self.margin_maintenance_per_lot if maintenance else self.margin_initial_per_lot
        if override is not None:
            return override * volume
        rate_mult = self.margin_maintenance_rate if maintenance else self.margin_initial_rate
        lev = leverage if leverage is not None else self.leverage
        if self.max_leverage:
            lev = min(lev, self.max_leverage)
        lev = lev if lev and lev > 0 else 1.0

        if self.calc_mode in (CalcMode.FOREX, CalcMode.CFD_LEVERAGE, CalcMode.CRYPTO):
            base = volume * self.contract_size * price * rate / lev
        elif self.calc_mode is CalcMode.FOREX_NO_LEVERAGE:
            base = volume * self.contract_size * price * rate
        elif self.calc_mode in (CalcMode.CFD, CalcMode.CFD_INDEX, CalcMode.EXCH_STOCKS):
            base = volume * self.contract_size * price * rate / lev
        elif self.calc_mode is CalcMode.FUTURES:
            base = volume * (self.margin_initial_per_lot or 0.0)
        else:
            base = volume * self.contract_size * price * rate / lev
        return base * rate_mult

    def swap_cost(self, position_type_sign: int, volume: float, price: float,
                  when: datetime, rate: float = 1.0) -> float:
        """Swap accrued at one rollover (negative = cost)."""
        multiplier = self.swap_rate_days.get(when.weekday(), 1)
        if not multiplier:
            return 0.0
        raw = self.swap_long if position_type_sign > 0 else self.swap_short
        if raw == 0:
            return 0.0

        if self.swap_type is SwapType.POINTS:
            value = raw * self.point * self.contract_size * volume * rate
        elif self.swap_type in (SwapType.MONEY, SwapType.MARGIN_CURRENCY):
            value = raw * volume * rate
        elif self.swap_type is SwapType.PERCENT_ANNUAL:
            value = self.notional(volume, price, rate) * (raw / 100.0) / self.swap_year_days
        elif self.swap_type is SwapType.PERCENT_CURRENT:
            value = self.notional(volume, price, rate) * (raw / 100.0)
        else:
            value = 0.0
        return value * multiplier

    # ------------------------------------------------------------- validation
    def check_stops_distance(self, price: float, stop: Optional[float]) -> bool:
        """Does this SL/TP respect the broker's ``stops_level``?"""
        if stop is None or not self.stops_level:
            return True
        return abs(price - stop) >= self.stops_level * self.point - 1e-12

    def can_open(self, order_type: OrderType) -> bool:
        if self.trade_mode in (TradeMode.DISABLED, TradeMode.CLOSE_ONLY):
            return False
        if self.trade_mode is TradeMode.LONG_ONLY and order_type.is_sell:
            return False
        if self.trade_mode is TradeMode.SHORT_ONLY and order_type.is_buy:
            return False
        return order_type in self.allowed_orders

    # ------------------------------------------------------------- serialising
    def to_dict(self) -> dict:
        data = asdict(self)
        data["calc_mode"] = self.calc_mode.value
        data["chart_mode"] = self.chart_mode.value
        data["execution"] = self.execution.value
        data["gtc_mode"] = self.gtc_mode.value
        data["trade_mode"] = self.trade_mode.value
        data["swap_type"] = self.swap_type.value
        data["filling_modes"] = [f.value for f in self.filling_modes]
        data["expiration_modes"] = [e.value for e in self.expiration_modes]
        data["allowed_orders"] = [o.name for o in self.allowed_orders]
        data["quote_sessions"] = {str(k): v for k, v in self.quote_sessions.days.items()}
        data["trade_sessions"] = {str(k): v for k, v in self.trade_sessions.days.items()}
        if self.expiration_date:
            data["expiration_date"] = self.expiration_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolSpec":
        data = dict(data)
        if "calc_mode" in data:
            data["calc_mode"] = CalcMode(data["calc_mode"])
        if "chart_mode" in data:
            data["chart_mode"] = ChartMode(data["chart_mode"])
        if "execution" in data:
            data["execution"] = ExecutionMode(data["execution"])
        if "gtc_mode" in data:
            data["gtc_mode"] = GTCMode(data["gtc_mode"])
        if "trade_mode" in data:
            data["trade_mode"] = TradeMode(data["trade_mode"])
        if "swap_type" in data:
            data["swap_type"] = SwapType(data["swap_type"])
        if "filling_modes" in data:
            data["filling_modes"] = [FillingMode(f) for f in data["filling_modes"]]
        if "expiration_modes" in data:
            data["expiration_modes"] = [ExpirationMode(e) for e in data["expiration_modes"]]
        if "allowed_orders" in data:
            data["allowed_orders"] = [
                o if isinstance(o, OrderType) else OrderType[o] for o in data["allowed_orders"]
            ]
        for key in ("quote_sessions", "trade_sessions"):
            if key in data and isinstance(data[key], dict):
                data[key] = SessionSpec.from_dict(
                    {int(k): v for k, v in data[key].items()}
                )
        if data.get("swap_rate_days"):
            data["swap_rate_days"] = {int(k): int(v) for k, v in data["swap_rate_days"].items()}
        if isinstance(data.get("expiration_date"), str):
            data["expiration_date"] = datetime.fromisoformat(data["expiration_date"])
        return cls(**data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SymbolSpec":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -------------------------------------------------------------------- MT5
    @classmethod
    def from_mt5(cls, symbol: str, initialize: bool = True, **overrides) -> "SymbolSpec":
        """Read the specification straight from a running MetaTrader 5 terminal."""
        import MetaTrader5 as mt5

        if initialize and not mt5.initialize():
            raise RuntimeError(f"Could not initialise MT5: {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol unknown to the terminal: {symbol}")

        calc_map = {
            0: CalcMode.FOREX, 1: CalcMode.FUTURES, 2: CalcMode.CFD,
            3: CalcMode.CFD_INDEX, 4: CalcMode.CFD_LEVERAGE, 5: CalcMode.FOREX_NO_LEVERAGE,
        }
        trade_map = {
            0: TradeMode.DISABLED, 1: TradeMode.LONG_ONLY, 2: TradeMode.SHORT_ONLY,
            3: TradeMode.CLOSE_ONLY, 4: TradeMode.FULL,
        }
        exec_map = {
            0: ExecutionMode.REQUEST, 1: ExecutionMode.INSTANT,
            2: ExecutionMode.MARKET, 3: ExecutionMode.EXCHANGE,
        }
        swap_map = MT5_SWAP_MODES

        filling = []
        mode = getattr(info, "filling_mode", 0)
        if mode & 1:
            filling.append(FillingMode.FOK)
        if mode & 2:
            filling.append(FillingMode.IOC)
        if mode & 4:
            filling.append(FillingMode.BOC)
        if not filling:
            filling = [FillingMode.FOK]

        account = mt5.account_info()
        leverage = float(getattr(account, "leverage", 100)) if account else 100.0

        spec = cls(
            symbol=info.name,
            name=getattr(info, "description", "") or info.name,
            category=getattr(info, "path", "").split("\\")[0],
            description=getattr(info, "description", ""),
            digits=int(info.digits),
            contract_size=float(info.trade_contract_size),
            margin_currency=getattr(info, "currency_margin", ""),
            profit_currency=getattr(info, "currency_profit", ""),
            base_currency=getattr(info, "currency_base", ""),
            calc_mode=calc_map.get(int(getattr(info, "trade_calc_mode", 0)), CalcMode.CFD),
            chart_mode=(ChartMode.BID if int(getattr(info, "chart_mode", 0)) == 0
                        else ChartMode.LAST),
            tick_size=float(getattr(info, "trade_tick_size", 0)) or None,
            execution=exec_map.get(int(getattr(info, "trade_exemode", 2)), ExecutionMode.MARKET),
            filling_modes=filling,
            trade_mode=trade_map.get(int(getattr(info, "trade_mode", 4)), TradeMode.FULL),
            stops_level=int(getattr(info, "trade_stops_level", 0)),
            freeze_level=int(getattr(info, "trade_freeze_level", 0)),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            volume_limit=float(getattr(info, "volume_limit", 0.0)),
            spread_points=float(getattr(info, "spread", 0)),
            spread_float=bool(getattr(info, "spread_float", True)),
            swap_type=swap_map.get(int(getattr(info, "swap_mode", 1)), SwapType.POINTS),
            swap_long=(0.0 if int(getattr(info, "swap_mode", 1)) == 0
                       else float(getattr(info, "swap_long", 0.0))),
            swap_short=(0.0 if int(getattr(info, "swap_mode", 1)) == 0
                        else float(getattr(info, "swap_short", 0.0))),
            swap_rate_days={0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0},
            leverage=leverage,
            margin_initial_rate=float(getattr(info, "margin_initial", 0.0)) or 1.0,
            margin_maintenance_rate=float(getattr(info, "margin_maintenance", 0.0)) or 1.0,
        )
        # Triple swap on the weekday the broker reports (0=Sunday in MT5).
        rollover3 = int(getattr(info, "swap_rollover3days", 3))
        mt5_to_py = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        spec.swap_rate_days[mt5_to_py.get(rollover3, 2)] = 3

        for key, value in overrides.items():
            setattr(spec, key, value)
        spec.__post_init__()
        return spec

    # ---------------------------------------------------------------- summary
    def summary(self) -> str:
        lines = [
            f"=== {self.symbol} - {self.name} ===",
            f"Category           : {self.category or '-'}   Type: {self.asset_type}",
            f"Digits / point     : {self.digits} / {self.point:g}",
            f"Contract size      : {self.contract_size:g}",
            f"Margin / profit ccy: {self.margin_currency} / {self.profit_currency}",
            f"Calc / chart mode  : {self.calc_mode.value} / {self.chart_mode.value}",
            f"Execution          : {self.execution.value}   GTC: {self.gtc_mode.value}",
            f"Filling            : {', '.join(f.value for f in self.filling_modes)}",
            f"Trading            : {self.trade_mode.value}   Stops level: {self.stops_level} pts",
            f"Volume             : min {self.volume_min} / max {self.volume_max}"
            f" / step {self.volume_step}",
            f"Spread             : {self.spread_points:g} pts"
            f" ({'floating' if self.spread_float else 'fixed'})",
            f"Commission         : {self.commission_per_lot:g}/lot"
            f" + {self.commission_per_deal:g}/deal"
            f" + {self.commission_percent:g}%",
            f"Swap ({self.swap_type.value:<14}) long {self.swap_long:g}"
            f" / short {self.swap_short:g}",
            f"Swap days          : {self.swap_rate_days}",
            f"Leverage           : 1:{self.leverage:g}"
            f"   initial margin x{self.margin_initial_rate:g}",
            "Trade sessions:",
            self.trade_sessions.describe(),
        ]
        return "\n".join(lines)
