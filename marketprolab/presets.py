"""Ready-made specifications for common instrument families.

They are starting points only: copy them and adjust the values to **your**
broker (spreads, swaps and commissions differ between brokers and over time).
"""

from __future__ import annotations

from typing import Callable, Dict

from .enums import (
    CalcMode,
    ChartMode,
    ExecutionMode,
    ExpirationMode,
    FillingMode,
    GTCMode,
    SwapType,
    TradeMode,
)
from .sessions import SessionSpec
from .symbol import SymbolSpec

# Typical schedule for a GMT+2/+3 server.
_METALS_SESSIONS = SessionSpec.from_dict(
    {
        "sunday": "22:01-24:00",
        "monday": "00:00-20:58, 22:00-24:00",
        "tuesday": "00:00-20:58, 22:00-24:00",
        "wednesday": "00:00-20:58, 22:00-24:00",
        "thursday": "00:00-20:58, 22:00-24:00",
        "friday": "00:00-20:58",
    }
)

_FOREX_SESSIONS = SessionSpec.from_dict(
    {
        "sunday": "22:00-24:00",
        "monday": "00:00-24:00",
        "tuesday": "00:00-24:00",
        "wednesday": "00:00-24:00",
        "thursday": "00:00-24:00",
        "friday": "00:00-21:00",
    }
)

_CRYPTO_SESSIONS = SessionSpec.always_open()


def xauusd(symbol: str = "XAUUSD", leverage: float = 100.0, **overrides) -> SymbolSpec:
    """Gold against the US dollar: 100 oz contract, 3 digits, swap in points."""
    spec = SymbolSpec(
        symbol=symbol,
        name="XAU/USD, Gold vs US Dollar",
        category="Metals",
        asset_type="cfd",
        digits=3,
        contract_size=100.0,
        margin_currency="XAU",
        profit_currency="USD",
        base_currency="XAU",
        calc_mode=CalcMode.FOREX,
        chart_mode=ChartMode.BID,
        execution=ExecutionMode.MARKET,
        gtc_mode=GTCMode.GTC,
        filling_modes=[FillingMode.FOK, FillingMode.IOC],
        expiration_modes=[ExpirationMode.ALL],
        trade_mode=TradeMode.FULL,
        stops_level=0,
        volume_min=0.01,
        volume_max=200.0,
        volume_step=0.01,
        spread_points=25.0,          # ~0.025 per ounce; tune it
        spread_float=True,
        commission_per_lot=0.0,
        swap_type=SwapType.POINTS,
        swap_long=-531.6,
        swap_short=0.0,
        swap_rate_days={0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0},
        leverage=leverage,
        margin_initial_rate=1.0,
        margin_maintenance_rate=1.0,
        quote_sessions=_METALS_SESSIONS,
        trade_sessions=_METALS_SESSIONS,
        timezone="Etc/GMT-3",
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    spec.__post_init__()
    return spec


def forex_major(symbol: str = "EURUSD", digits: int = 5, leverage: float = 100.0,
                **overrides) -> SymbolSpec:
    """Standard currency pair: 100,000 units, 5 digits."""
    spec = SymbolSpec(
        symbol=symbol,
        name=f"{symbol[:3]}/{symbol[3:6]}",
        category="Forex",
        asset_type="spot",
        digits=digits,
        contract_size=100_000.0,
        margin_currency=symbol[:3],
        profit_currency=symbol[3:6],
        base_currency=symbol[:3],
        calc_mode=CalcMode.FOREX,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=10.0,
        commission_per_lot=3.5,
        swap_type=SwapType.POINTS,
        swap_long=-5.0,
        swap_short=-2.0,
        leverage=leverage,
        quote_sessions=_FOREX_SESSIONS,
        trade_sessions=_FOREX_SESSIONS,
        timezone="Etc/GMT-3",
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    spec.__post_init__()
    return spec


def index_cfd(symbol: str = "US500", digits: int = 2, contract_size: float = 1.0,
              leverage: float = 200.0, **overrides) -> SymbolSpec:
    """Index CFD: one contract equals one index unit."""
    spec = SymbolSpec(
        symbol=symbol,
        name=f"{symbol} index CFD",
        category="Indices",
        asset_type="cfd",
        digits=digits,
        contract_size=contract_size,
        margin_currency="USD",
        profit_currency="USD",
        calc_mode=CalcMode.CFD_INDEX,
        volume_min=0.1,
        volume_max=500.0,
        volume_step=0.1,
        spread_points=60.0,
        swap_type=SwapType.PERCENT_ANNUAL,
        swap_long=-6.5,
        swap_short=1.0,
        leverage=leverage,
        quote_sessions=SessionSpec.from_dict(
            {d: "00:05-23:10" for d in ("monday", "tuesday", "wednesday", "thursday", "friday")}
        ),
        trade_sessions=SessionSpec.from_dict(
            {d: "00:05-23:10" for d in ("monday", "tuesday", "wednesday", "thursday", "friday")}
        ),
        timezone="Etc/GMT-3",
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    spec.__post_init__()
    return spec


def crypto(symbol: str = "BTCUSD", digits: int = 2, contract_size: float = 1.0,
           leverage: float = 20.0, **overrides) -> SymbolSpec:
    """24/7 crypto with an annual percentage swap."""
    spec = SymbolSpec(
        symbol=symbol,
        name=f"{symbol[:3]}/{symbol[3:]}",
        category="Crypto",
        asset_type="cfd",
        digits=digits,
        contract_size=contract_size,
        margin_currency="USD",
        profit_currency="USD",
        calc_mode=CalcMode.CRYPTO,
        volume_min=0.01,
        volume_max=50.0,
        volume_step=0.01,
        spread_points=3000.0,
        swap_type=SwapType.PERCENT_ANNUAL,
        swap_long=-20.0,
        swap_short=-20.0,
        swap_rate_days=dict.fromkeys(range(7), 1),
        leverage=leverage,
        quote_sessions=_CRYPTO_SESSIONS,
        trade_sessions=_CRYPTO_SESSIONS,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    spec.__post_init__()
    return spec


def energy(symbol: str = "USOIL", digits: int = 3, contract_size: float = 1000.0,
           leverage: float = 100.0, **overrides) -> SymbolSpec:
    """Oil CFD: 1,000 barrels per lot."""
    spec = SymbolSpec(
        symbol=symbol,
        name="Crude Oil CFD",
        category="Energies",
        asset_type="cfd",
        digits=digits,
        contract_size=contract_size,
        margin_currency="USD",
        profit_currency="USD",
        calc_mode=CalcMode.CFD,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=30.0,
        swap_type=SwapType.POINTS,
        swap_long=-25.0,
        swap_short=-12.0,
        leverage=leverage,
        quote_sessions=_METALS_SESSIONS,
        trade_sessions=_METALS_SESSIONS,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    spec.__post_init__()
    return spec


PRESETS: Dict[str, Callable[..., SymbolSpec]] = {
    "xauusd": xauusd,
    "gold": xauusd,
    "forex": forex_major,
    "index": index_cfd,
    "crypto": crypto,
    "energy": energy,
}


def get_preset(name: str, **kwargs) -> SymbolSpec:
    """``get_preset("xauusd", symbol="XAUUSDz", leverage=200)``."""
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown preset: {name}. Available: {sorted(PRESETS)}")
    return PRESETS[key](**kwargs)
