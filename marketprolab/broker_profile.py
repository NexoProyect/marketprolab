"""**Account / broker** conditions, independent of the instrument.

The same symbol behaves differently at two brokers: leverage, stop-out level,
commission, execution quality and server timezone all change. `BrokerProfile`
captures that and can be applied to any `SymbolSpec`; `SymbolRegistry` keeps a
JSON catalogue of your symbols per broker.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from .enums import MarginMode
from .symbol import SymbolSpec


@dataclass
class BrokerProfile:
    """Account conditions at a specific broker.

    Only the fields you fill in (i.e. not ``None``) override what the
    `SymbolSpec` carries; everything else is left untouched.
    """

    name: str = "generic"
    account_currency: str = "USD"
    leverage: float = 100.0
    margin_mode: MarginMode = MarginMode.HEDGING
    margin_call_level: float = 100.0      # % margin level
    stop_out_level: float = 50.0          # % margin level
    stop_out_in_money: bool = False       # True => stop_out_level is money, not %
    server_timezone: str = "Etc/GMT-3"    # broker server time
    swap_rollover_hour: int = 0           # server hour when swap is charged
    swap_triple_weekday: Optional[int] = 2  # 0=Monday ... 2=Wednesday (None = leave as is)

    # Default commissions (applied to the symbol when it defines none)
    commission_per_lot: Optional[float] = None
    commission_per_deal: Optional[float] = None
    commission_percent: Optional[float] = None

    # Broker trading rules
    min_stops_level_points: Optional[int] = None
    hedging_allowed: bool = True
    max_positions: int = 0        # 0 = no limit
    max_pending_orders: int = 0
    max_volume_total: float = 0.0
    weekend_close: bool = False   # flatten everything before the weekend
    fifo_only: bool = False       # FIFO rule (US brokers)

    # Cost multipliers, handy for stress-testing a backtest
    spread_multiplier: float = 1.0
    swap_multiplier: float = 1.0
    commission_multiplier: float = 1.0

    notes: str = ""

    # ------------------------------------------------------------------- apply
    def apply(self, spec: SymbolSpec, inplace: bool = False) -> SymbolSpec:
        """Return the symbol with this broker's conditions applied."""
        target = spec if inplace else SymbolSpec.from_dict(spec.to_dict())

        target.leverage = self.leverage
        target.swap_rollover_hour = self.swap_rollover_hour
        if self.commission_per_lot is not None:
            target.commission_per_lot = self.commission_per_lot
        if self.commission_per_deal is not None:
            target.commission_per_deal = self.commission_per_deal
        if self.commission_percent is not None:
            target.commission_percent = self.commission_percent
        if self.min_stops_level_points is not None:
            target.stops_level = max(target.stops_level, self.min_stops_level_points)
        if self.swap_triple_weekday is not None:
            days = {d: (0 if d in (5, 6) else 1) for d in range(7)}
            days.update({d: v for d, v in target.swap_rate_days.items() if d in (5, 6)})
            days[self.swap_triple_weekday] = 3
            target.swap_rate_days = days
        if not target.timezone:
            target.timezone = self.server_timezone

        # Stress multipliers
        target.commission_per_lot *= self.commission_multiplier
        target.commission_per_deal *= self.commission_multiplier
        target.commission_percent *= self.commission_multiplier
        target.swap_long *= self.swap_multiplier
        target.swap_short *= self.swap_multiplier
        target.spread_points *= self.spread_multiplier
        return target

    # -------------------------------------------------------------- serialising
    def to_dict(self) -> dict:
        data = asdict(self)
        data["margin_mode"] = self.margin_mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "BrokerProfile":
        data = dict(data)
        if "margin_mode" in data:
            data["margin_mode"] = MarginMode(data["margin_mode"])
        return cls(**data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BrokerProfile":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def summary(self) -> str:
        return "\n".join(
            [
                f"=== Broker: {self.name} ===",
                f"Account currency   : {self.account_currency}",
                f"Leverage           : 1:{self.leverage:g}   mode: {self.margin_mode.value}",
                f"Margin call        : {self.margin_call_level:g}%   "
                f"Stop out: {self.stop_out_level:g}{'' if self.stop_out_in_money else '%'}",
                f"Server timezone    : {self.server_timezone}   "
                f"rollover {self.swap_rollover_hour:02d}:00",
                f"Multipliers        : spread x{self.spread_multiplier:g}, "
                f"swap x{self.swap_multiplier:g}, commission x{self.commission_multiplier:g}",
            ]
        )


class SymbolRegistry:
    """An on-disk symbol catalogue: one JSON file per symbol inside a folder.

    ::

        reg = SymbolRegistry("./symbols/mybroker")
        reg.add(my_spec)
        spec = reg.get("XAUUSDz")
        reg.list()
    """

    def __init__(self, directory: str, profile: Optional[BrokerProfile] = None):
        self.directory = directory
        self.profile = profile
        os.makedirs(directory, exist_ok=True)
        self._cache: Dict[str, SymbolSpec] = {}

    def _path(self, symbol: str) -> str:
        return os.path.join(self.directory, f"{symbol}.json")

    def add(self, spec: SymbolSpec, overwrite: bool = True) -> str:
        path = self._path(spec.symbol)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)
        spec.save(path)
        self._cache[spec.symbol] = spec
        return path

    def get(self, symbol: str, apply_profile: bool = True) -> SymbolSpec:
        if symbol not in self._cache:
            path = self._path(symbol)
            if not os.path.exists(path):
                raise KeyError(f"No stored specification for {symbol} in {self.directory}")
            self._cache[symbol] = SymbolSpec.load(path)
        spec = self._cache[symbol]
        if apply_profile and self.profile is not None:
            return self.profile.apply(spec)
        return spec

    def list(self) -> List[str]:
        return sorted(
            f[:-5] for f in os.listdir(self.directory) if f.endswith(".json")
        )

    def import_from_mt5(self, symbols: Iterable[str], **overrides) -> List[str]:
        """Dump the specification of several symbols from the MT5 terminal."""
        saved = []
        for sym in symbols:
            spec = SymbolSpec.from_mt5(sym, **overrides)
            self.add(spec)
            saved.append(sym)
        return saved


def broker_profile_preset(kind: str = "retail_ecn", **overrides) -> BrokerProfile:
    """Starting-point profiles. Templates only - tune them to your real broker."""
    presets = {
        "retail_ecn": dict(
            name="ECN retail", leverage=100, stop_out_level=50.0,
            commission_per_lot=3.5, spread_multiplier=1.0,
        ),
        "retail_standard": dict(
            name="Standard retail", leverage=200, stop_out_level=20.0,
            commission_per_lot=0.0, spread_multiplier=1.6,
        ),
        "prop_firm": dict(
            name="Prop firm", leverage=100, stop_out_level=0.0,
            commission_per_lot=3.0, hedging_allowed=True, weekend_close=True,
        ),
        "us_regulated": dict(
            name="US regulated", leverage=50, stop_out_level=100.0,
            hedging_allowed=False, fifo_only=True, commission_per_lot=5.0,
        ),
    }
    if kind not in presets:
        raise KeyError(f"Unknown profile: {kind}. Available: {sorted(presets)}")
    data = presets[kind]
    data.update(overrides)
    return BrokerProfile(**data)
