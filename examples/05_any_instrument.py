"""Example 5 - Describing any instrument at any broker.

The library knows nothing about any particular market: everything it needs comes
from your `SymbolSpec`. Here are five very different instruments defined by hand
(forex, metal, index, crypto and a future), the same instrument under two
different brokers, and how to keep a reusable catalogue.

    python examples/05_any_instrument.py
"""

from datetime import datetime

from marketprolab import (
    BrokerProfile,
    CalcMode,
    ChartMode,
    ExecutionMode,
    FillingMode,
    GTCMode,
    MarginMode,
    SessionSpec,
    SwapType,
    SymbolRegistry,
    SymbolSpec,
    TradeMode,
    broker_profile_preset,
)

# ══════════════════════════════════════════ 1. A currency pair
eurusd = SymbolSpec(
    symbol="EURUSD",
    name="Euro vs US Dollar",
    category="Forex",
    asset_type="spot",
    digits=5,
    contract_size=100_000,
    margin_currency="EUR",
    profit_currency="USD",
    calc_mode=CalcMode.FOREX,
    chart_mode=ChartMode.BID,
    execution=ExecutionMode.MARKET,
    gtc_mode=GTCMode.GTC,
    filling_modes=[FillingMode.FOK, FillingMode.IOC],
    trade_mode=TradeMode.FULL,
    stops_level=10,                 # SL/TP at least 10 points away from price
    volume_min=0.01, volume_max=100, volume_step=0.01,
    spread_points=8, spread_float=True,
    commission_per_lot=3.5,
    swap_type=SwapType.POINTS, swap_long=-7.5, swap_short=-1.2,
    swap_rate_days={0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0},
    leverage=100,
)

# ══════════════════════════════════════════ 2. A metal (100-ounce contract)
xauusd = SymbolSpec(
    symbol="XAUUSD",
    name="Gold vs US Dollar",
    category="Metals",
    digits=3,
    contract_size=100,             # 100 ounces per lot
    margin_currency="XAU",
    profit_currency="USD",
    calc_mode=CalcMode.FOREX,
    volume_min=0.01, volume_max=200, volume_step=0.01,
    spread_points=25,
    swap_type=SwapType.POINTS, swap_long=-531.6, swap_short=0.0,
    swap_rate_days={0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0},
    leverage=100,
    trade_sessions=SessionSpec.from_dict(
        {
            "sunday": "22:01-24:00",
            "monday": "00:00-20:58, 22:00-24:00",
            "tuesday": "00:00-20:58, 22:00-24:00",
            "wednesday": "00:00-20:58, 22:00-24:00",
            "thursday": "00:00-20:58, 22:00-24:00",
            "friday": "00:00-20:58",
        }
    ),
    timezone="Etc/GMT-3",
)

# ══════════════════════════════════════════ 3. An index with percentage swap
us500 = SymbolSpec(
    symbol="US500",
    name="S&P 500 index CFD",
    category="Indices",
    asset_type="cfd",
    digits=2,
    contract_size=1,               # one contract = one index point
    margin_currency="USD",
    profit_currency="USD",
    calc_mode=CalcMode.CFD_INDEX,
    volume_min=0.1, volume_max=500, volume_step=0.1,
    spread_points=60,
    swap_type=SwapType.PERCENT_ANNUAL,   # financing as an annual % of notional
    swap_long=-6.8, swap_short=1.2,
    swap_year_days=360,
    leverage=200,
    trade_sessions=SessionSpec.from_dict(
        {d: "00:05-22:00, 22:05-23:55" for d in
         ("monday", "tuesday", "wednesday", "thursday", "friday")}
    ),
)

# ══════════════════════════════════════════ 4. A 24/7 crypto
btcusd = SymbolSpec(
    symbol="BTCUSD",
    name="Bitcoin vs US Dollar",
    category="Crypto",
    digits=2,
    contract_size=1,
    margin_currency="USD",
    profit_currency="USD",
    calc_mode=CalcMode.CRYPTO,
    volume_min=0.01, volume_max=50, volume_step=0.01,
    spread_points=3500,
    swap_type=SwapType.PERCENT_ANNUAL, swap_long=-20, swap_short=-20,
    swap_rate_days=dict.fromkeys(range(7), 1),   # weekends included
    leverage=20,
    trade_sessions=SessionSpec.always_open(),
)

# ══════════════════════════════════════════ 5. A future with an expiry
mini_dax = SymbolSpec(
    symbol="FDXM_U6",
    name="Mini-DAX Future Sep 2026",
    category="Futures",
    asset_type="futures",
    digits=1,
    contract_size=5,                       # 5 EUR per index point
    margin_currency="EUR",
    profit_currency="EUR",
    calc_mode=CalcMode.FUTURES,
    margin_initial_per_lot=2_400.0,        # flat margin per contract
    margin_maintenance_per_lot=2_000.0,
    volume_min=1, volume_max=50, volume_step=1,
    spread_points=10,
    commission_per_deal=2.5,
    swap_type=SwapType.MONEY, swap_long=0.0, swap_short=0.0,
    expiration_date=datetime(2026, 9, 18),
    position_limit=20,
    trade_sessions=SessionSpec.from_dict(
        {d: "08:00-22:00" for d in
         ("monday", "tuesday", "wednesday", "thursday", "friday")}
    ),
)


# ══════════════════════════════════════════ The same instrument, two brokers
broker_a = BrokerProfile(
    name="Broker A (ECN)",
    account_currency="USD",
    leverage=100,
    stop_out_level=50,
    commission_per_lot=3.0,
    spread_multiplier=1.0,
    margin_mode=MarginMode.HEDGING,
)

broker_b = BrokerProfile(
    name="Broker B (standard: no commission, wider spread)",
    account_currency="USD",
    leverage=30,                 # European retail regulation
    stop_out_level=50,
    commission_per_lot=0.0,
    spread_multiplier=2.2,
    swap_multiplier=1.4,
    margin_mode=MarginMode.NETTING,
    hedging_allowed=False,
)

if __name__ == "__main__":
    for spec in (eurusd, xauusd, us500, btcusd, mini_dax):
        print(spec.summary())
        print(f"  margin for 1 lot at price 100: {spec.margin_required(1, 100):,.2f}")
        print(f"  value of one point (1 lot)   : {spec.point_value(1):,.4f}")
        print()

    print("=== The same gold at two brokers ===")
    for profile in (broker_a, broker_b):
        applied = profile.apply(xauusd)
        print(profile.summary())
        print(f"  effective spread  : {applied.spread_points:.1f} points")
        print(f"  commission/lot    : {applied.commission_per_lot:.2f}")
        print(f"  long swap         : {applied.swap_long:.1f}")
        print(f"  margin 1 lot@2000 : {applied.margin_required(1, 2000):,.2f}")
        print()

    # Ready-made profiles (templates: adjust to your real broker)
    print(broker_profile_preset("prop_firm", leverage=100).summary())

    # ── On-disk catalogue ──────────────────────────────────────────────────
    registry = SymbolRegistry("symbols/my_broker", profile=broker_a)
    for spec in (eurusd, xauusd, us500, btcusd, mini_dax):
        registry.add(spec)
    print("\nCatalogue saved:", registry.list())

    # Reload later, already carrying the broker's conditions
    gold = registry.get("XAUUSD")
    print("Gold reloaded with Broker A conditions:",
          gold.commission_per_lot, gold.leverage)
