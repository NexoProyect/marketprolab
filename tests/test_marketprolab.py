"""Test suite. Run with:  pytest -q"""

from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from marketprolab import (  # noqa: E402
    Backtest,
    BrokerProfile,
    CalcMode,
    FixedLatency,
    FixedSlippage,
    FixedSpread,
    MarginMode,
    SessionSpec,
    SimulationConfig,
    Strategy,
    SwapType,
    SymbolSpec,
    combined_report,
    comparison_report,
    grid_search,
    indicators,
    monte_carlo,
    prepare_bars,
    resample_bars,
)


# ------------------------------------------------------------------- fixtures
@pytest.fixture
def spec():
    return SymbolSpec(
        symbol="TEST",
        digits=3,
        contract_size=100,
        margin_currency="XAU",
        profit_currency="USD",
        calc_mode=CalcMode.FOREX,
        volume_min=0.01,
        volume_max=100,
        volume_step=0.01,
        spread_points=0,
        leverage=100,
        swap_type=SwapType.POINTS,
        swap_long=-100,
        swap_short=-50,
    )


@pytest.fixture
def bars():
    """20 flat bars at 2000 with a 1-point range, so the maths is exact."""
    index = pd.date_range("2024-01-01", periods=20, freq="1h")
    close = np.full(20, 2000.0)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 100.0},
        index=index,
    ).rename_axis("time")


@pytest.fixture
def trending_bars():
    index = pd.date_range("2024-01-01", periods=500, freq="1h")
    close = 2000 + np.arange(500) * 0.5
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=index,
    ).rename_axis("time")


# ----------------------------------------------------------------- SymbolSpec
def test_point_and_pip(spec):
    assert spec.point == pytest.approx(0.001)
    assert spec.pip == pytest.approx(0.01)      # 3 digits -> a pip is 10 points


def test_volume_normalisation(spec):
    assert spec.normalize_volume(0.234) == pytest.approx(0.23)
    assert spec.normalize_volume(0.001) == pytest.approx(0.01)   # raised to the minimum
    assert spec.normalize_volume(999) == pytest.approx(100.0)    # clamped to the maximum
    assert spec.is_valid_volume(0.05)
    assert not spec.is_valid_volume(0.055)


def test_margin(spec):
    # 1 lot = 100 units at 2000 with 1:100 leverage -> 2000 of margin
    assert spec.margin_required(1.0, 2000.0) == pytest.approx(2000.0)
    assert spec.margin_required(0.5, 2000.0) == pytest.approx(1000.0)
    spec.margin_initial_per_lot = 500.0
    assert spec.margin_required(2.0, 2000.0) == pytest.approx(1000.0)


def test_profit_and_point_value(spec):
    # +1.000 in price, 1 lot, contract 100 -> 100 USD
    assert spec.profit(1, 1.0, 2000.0, 2001.0) == pytest.approx(100.0)
    assert spec.profit(-1, 1.0, 2000.0, 2001.0) == pytest.approx(-100.0)
    assert spec.point_value(1.0) == pytest.approx(0.1)


def test_swap_in_points_and_triple_day(spec):
    monday = datetime(2024, 1, 1)
    wednesday = datetime(2024, 1, 3)
    saturday = datetime(2024, 1, 6)
    # -100 points * 0.001 * 100 * 1 lot = -10
    assert spec.swap_cost(1, 1.0, 2000.0, monday) == pytest.approx(-10.0)
    assert spec.swap_cost(1, 1.0, 2000.0, wednesday) == pytest.approx(-30.0)  # triple
    assert spec.swap_cost(1, 1.0, 2000.0, saturday) == pytest.approx(0.0)


def test_annual_percentage_swap():
    spec = SymbolSpec(symbol="IDX", digits=2, contract_size=1,
                      swap_type=SwapType.PERCENT_ANNUAL, swap_long=-3.6,
                      swap_year_days=360)
    # 1 lot at 5000 -> notional 5000; -3.6%/360 = -0.01% a day = -0.5
    assert spec.swap_cost(1, 1.0, 5000.0, datetime(2024, 1, 1)) == pytest.approx(-0.5)


def test_serialisation_round_trip(spec, tmp_path):
    spec.trade_sessions = SessionSpec.from_dict({"monday": "08:00-17:00"})
    path = tmp_path / "spec.json"
    spec.save(str(path))
    other = SymbolSpec.load(str(path))
    assert other.symbol == spec.symbol
    assert other.contract_size == spec.contract_size
    assert other.swap_rate_days == spec.swap_rate_days
    assert other.trade_sessions.days[0] == [(480, 1020)]


# -------------------------------------------------------------------- sessions
def test_sessions():
    s = SessionSpec.from_dict({"monday": "00:00-20:58, 22:00-24:00", "friday": "00:00-20:58"})
    assert s.is_open(datetime(2024, 1, 1, 10, 0))       # Monday 10:00
    assert not s.is_open(datetime(2024, 1, 1, 21, 30))  # Monday 21:30 (break)
    assert s.is_open(datetime(2024, 1, 1, 22, 30))
    assert not s.is_open(datetime(2024, 1, 2, 10, 0))   # Tuesday: undefined
    assert s.is_last_session_of_week(datetime(2024, 1, 5, 15, 0))


# ------------------------------------------------------------------------ data
def test_prepare_and_resample():
    index = pd.date_range("2024-01-01", periods=60, freq="1min")
    df = pd.DataFrame({"<OPEN>": 1.0, "<HIGH>": 2.0, "<LOW>": 0.5, "<CLOSE>": 1.5,
                       "<TICKVOL>": 10}, index=index)
    prepared = prepare_bars(df)
    assert list(prepared.columns[:5]) == ["open", "high", "low", "close", "volume"]
    h1 = resample_bars(prepared, "H1")
    assert len(h1) == 1
    assert h1["volume"].iloc[0] == 600


# ------------------------------------------------------------------- execution
class BuyOnce(Strategy):
    lots = 1.0
    sl = None
    tp = None

    def init(self):
        self.done = False

    def on_bar(self):
        if not self.done:
            self.buy(self.lots, sl=self.sl, tp=self.tp)
            self.done = True


def _config(**kwargs):
    base = dict(initial_balance=100_000, spread=FixedSpread(0), slippage=FixedSlippage(0),
                latency=FixedLatency(0), respect_sessions=False, apply_swap=False,
                close_open_positions_at_end=True)
    base.update(kwargs)
    return SimulationConfig(**base)


def test_simple_buy_without_costs(spec, bars):
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100), _config()).run()
    assert r.stats["trades"] == 1
    trade = r.trades.iloc[0]
    assert trade["type"] == "buy"
    assert trade["net_profit"] == pytest.approx(0.0)     # flat price, no costs
    assert r.stats["final_equity"] == pytest.approx(100_000)


def test_spread_costs_money(spec, bars):
    """With a 10-point spread, opening and closing a long costs 10 points."""
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(spread=FixedSpread(10))).run()
    # 10 points * 0.001 * 100 * 1 lot = 1.0
    assert r.trades.iloc[0]["net_profit"] == pytest.approx(-1.0, abs=1e-6)


def test_commission_per_lot(spec, bars):
    spec.commission_per_lot = 5.0
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100), _config()).run()
    assert r.trades.iloc[0]["commission"] == pytest.approx(10.0)   # both sides
    assert r.trades.iloc[0]["net_profit"] == pytest.approx(-10.0)


def test_slippage_worsens_the_price(spec, bars):
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(slippage=FixedSlippage(20))).run()
    trade = r.trades.iloc[0]
    assert trade["open_price"] > 2000.0     # the long enters higher
    assert trade["close_price"] < 2000.0    # and exits lower
    assert trade["net_profit"] < 0


def test_latency_delays_execution(spec, trending_bars):
    """With one hour of latency on hourly bars, the entry slips to the next bar."""
    without = Backtest(trending_bars.iloc[:10], BuyOnce, spec,
                       BrokerProfile(leverage=100), _config()).run()
    with_latency = Backtest(trending_bars.iloc[:10], BuyOnce, spec,
                            BrokerProfile(leverage=100),
                            _config(latency=FixedLatency(3600 * 1000),
                                    latency_price_drift=False)).run()
    assert with_latency.trades.iloc[0]["open_price"] > without.trades.iloc[0]["open_price"]


def test_stop_loss_fires(spec):
    index = pd.date_range("2024-01-01", periods=5, freq="1h")
    close = np.array([2000.0, 2000.0, 1990.0, 1980.0, 1980.0])
    bars = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0}, index=index).rename_axis("time")
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(), strategy_params={"sl": 1995.0}).run()
    assert r.trades.iloc[0]["reason"] == "SL"
    assert r.trades.iloc[0]["net_profit"] < 0


def test_take_profit_fires(spec):
    index = pd.date_range("2024-01-01", periods=5, freq="1h")
    close = np.array([2000.0, 2000.0, 2010.0, 2020.0, 2020.0])
    bars = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0}, index=index).rename_axis("time")
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(), strategy_params={"tp": 2005.0}).run()
    assert r.trades.iloc[0]["reason"] == "TP"
    assert r.trades.iloc[0]["net_profit"] > 0


def test_swap_accrues(spec):
    """Three days holding a position means the rollovers get charged."""
    index = pd.date_range("2024-01-01", periods=72, freq="1h")
    close = np.full(72, 2000.0)
    bars = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0}, index=index).rename_axis("time")
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(apply_swap=True)).run()
    # Two rollovers are crossed: Tuesday the 2nd (x1 = -10) and Wednesday the 3rd (x3 = -30)
    assert r.trades.iloc[0]["swap"] == pytest.approx(-40.0)


def test_no_swap_without_crossing_rollover(spec, bars):
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(apply_swap=True)).run()
    assert r.trades.iloc[0]["swap"] == pytest.approx(0.0)


def test_insufficient_margin_is_rejected(spec, bars):
    """1 lot at 2000 needs 2,000 of margin; with 500 of balance it cannot open."""
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(initial_balance=500)).run()
    assert r.stats["trades"] == 0
    assert any(x["reason"] == "no_margin" for x in r.broker.rejections)


def test_invalid_volume_is_rejected(spec, bars):
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100), _config(),
                 strategy_params={"lots": 0.0}).run()
    assert r.stats["trades"] == 0


def test_closed_session_blocks_trading(spec, bars):
    spec.trade_sessions = SessionSpec.from_dict({"monday": "12:00-13:00"})
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 _config(respect_sessions=True)).run()
    assert r.stats["trades"] <= 1
    if r.stats["trades"]:
        assert 11 <= r.trades.iloc[0]["open_time"].hour <= 13


def test_netting_closes_the_opposite(spec, bars):
    class BuyThenSell(Strategy):
        def init(self):
            self.step = 0

        def on_bar(self):
            self.step += 1
            if self.step == 1:
                self.buy(1.0)
            elif self.step == 5:
                self.sell(1.0)

    profile = BrokerProfile(leverage=100, margin_mode=MarginMode.NETTING,
                            hedging_allowed=False)
    r = Backtest(bars, BuyThenSell, spec, profile, _config()).run()
    assert r.stats["trades"] == 1                      # the sell closed the buy
    assert r.trades.iloc[0]["reason"] == "OPPOSITE"


def test_stop_out_liquidates(spec):
    """A sharp drop on a heavily leveraged account triggers the stop out."""
    index = pd.date_range("2024-01-01", periods=30, freq="1h")
    close = np.linspace(2000, 1400, 30)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1.0}, index=index).rename_axis("time")
    r = Backtest(bars, BuyOnce, spec, BrokerProfile(leverage=100, stop_out_level=50),
                 _config(initial_balance=2_100), strategy_params={"lots": 1.0}).run()
    assert r.trades.iloc[0]["reason"] in ("STOP_OUT", "END_OF_TEST")


def test_no_look_ahead(spec, trending_bars):
    """Peeking at a future bar must raise IndexError."""
    class Cheater(Strategy):
        def on_bar(self):
            if self.i == 5:
                _ = self.close[10]

    with pytest.raises(IndexError):
        Backtest(trending_bars, Cheater, spec, BrokerProfile(), _config()).run()


def test_risk_based_sizing(spec, bars):
    class Risk(Strategy):
        def init(self):
            self.done = False

        def on_bar(self):
            if not self.done:
                # 1% of 100,000 = 1,000; 100-point stop; point value 0.1 per lot
                assert self.volume_for_risk_pct(1.0, 100) == pytest.approx(100.0)
                self.done = True

    Backtest(bars, Risk, spec, BrokerProfile(), _config()).run()


# --------------------------------------------------------------------- metrics
def test_basic_metrics(spec, trending_bars):
    r = Backtest(trending_bars, BuyOnce, spec, BrokerProfile(leverage=100), _config()).run()
    s = r.stats
    assert s["trades"] == 1
    assert s["net_profit"] > 0
    assert s["final_equity"] == pytest.approx(s["initial_balance"] + s["net_profit"])
    assert 0 <= s["win_rate"] <= 100
    assert s["max_dd_pct"] <= 0
    assert len(r.equity) == len(trending_bars)


# ------------------------------------------------------------------ indicators
def test_indicators():
    values = pd.Series(np.arange(1, 101, dtype=float))
    assert indicators.sma(values, 10)[-1] == pytest.approx(95.5)
    assert np.isnan(indicators.sma(values, 10)[:9]).all()
    assert indicators.rsi(values, 14)[-1] > 99          # a permanently rising series
    assert indicators.atr(values, values, values, 14)[-1] == pytest.approx(1.0, abs=0.1)
    up = indicators.crossover(np.array([1, 2, 3.0]), np.array([2, 2, 2.0]))
    assert up.tolist() == [False, False, True]


# ----------------------------------------------------------------- monte carlo
def test_monte_carlo():
    pnl = np.array([100.0, -50, 200, -80, 30, -120, 90, 40, -30, 60] * 5)
    mc = monte_carlo(pnl, n_simulations=200, initial_balance=10_000, method="bootstrap")
    assert mc.paths.shape == (200, len(pnl) + 1)
    assert 0 <= mc.probability_of_loss <= 1
    assert mc.percentiles["dd_p95"] <= mc.percentiles["dd_p50"]  # the worse tail
    assert mc.summary(print_it=False)


def test_monte_carlo_is_reproducible():
    pnl = np.random.default_rng(0).normal(5, 50, 100)
    a = monte_carlo(pnl, n_simulations=100, seed=1, initial_balance=1000)
    b = monte_carlo(pnl, n_simulations=100, seed=1, initial_balance=1000)
    assert np.allclose(a.final_equity, b.final_equity)


# ---------------------------------------------------------------- optimization
class ThresholdStrategy(Strategy):
    threshold = 10

    def init(self):
        self.done = False

    def on_bar(self):
        if not self.done and self.i >= self.threshold:
            self.buy(0.1)
            self.done = True


def test_grid_search(spec, trending_bars):
    bt = Backtest(trending_bars, ThresholdStrategy, spec, BrokerProfile(leverage=100), _config())
    opt = grid_search(bt, {"threshold": [10, 50, 100]}, objective="net_profit", progress=False)
    assert len(opt.results) == 3
    assert opt.best_params["threshold"] == 10   # entering earlier wins in an uptrend


def test_reproducibility(spec, trending_bars):
    from marketprolab import RandomSlippage, RandomSpread

    cfg = dict(initial_balance=100_000, spread=RandomSpread(20), respect_sessions=False,
               slippage=RandomSlippage(3, 3), seed=123)
    a = Backtest(trending_bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 SimulationConfig(**cfg)).run()
    b = Backtest(trending_bars, BuyOnce, spec, BrokerProfile(leverage=100),
                 SimulationConfig(**cfg)).run()
    assert a.stats["net_profit"] == pytest.approx(b.stats["net_profit"])


# --------------------------------------------------------------- HTML reports
def test_html_reports_are_self_contained(spec, trending_bars, tmp_path):
    bt = Backtest(trending_bars, ThresholdStrategy, spec, BrokerProfile(leverage=100), _config())
    result = bt.run()
    path = result.to_html(str(tmp_path / "report.html"))
    content = open(path, encoding="utf-8").read()
    assert "<title>" in content
    assert "data:image/png;base64" in content     # charts are embedded
    assert "http://" not in content and "https://" not in content   # no external refs
    assert "Net profit" in content


def test_optimization_and_combined_reports(spec, trending_bars, tmp_path):
    bt = Backtest(trending_bars, ThresholdStrategy, spec, BrokerProfile(leverage=100), _config())
    result = bt.run()
    opt = grid_search(bt, {"threshold": [10, 50]}, objective="net_profit", progress=False)
    assert opt.to_html(str(tmp_path / "opt.html"))
    mc = monte_carlo(np.array([10.0, -5, 20, -8] * 10), n_simulations=50, initial_balance=1000)
    assert mc.to_html(str(tmp_path / "mc.html"))
    path = combined_report(result, str(tmp_path / "all.html"), opt=opt, mc=mc)
    content = open(path, encoding="utf-8").read()
    assert "Optimization" in content and "Monte Carlo" in content
    assert comparison_report({"a": result, "b": result}, str(tmp_path / "cmp.html"))


def test_dark_theme_report(spec, trending_bars, tmp_path):
    from marketprolab import plotting

    bt = Backtest(trending_bars, ThresholdStrategy, spec, BrokerProfile(leverage=100), _config())
    result = bt.run()
    result.to_html(str(tmp_path / "dark.html"), theme="dark")
    assert plotting._ACTIVE == "light"   # the theme is restored afterwards


def test_mt5_swap_mode_mapping():
    """MT5 mode 1 is POINTS, not money.

    Reading it as money turned a one-ounce gold contract's overnight cost from
    -0.53 USD into -531.60 USD - a thousandfold error that quietly destroys any
    backtest holding positions overnight.
    """
    from marketprolab.symbol import MT5_SWAP_MODES

    assert MT5_SWAP_MODES[1] is SwapType.POINTS
    assert MT5_SWAP_MODES[2] is SwapType.MONEY
    assert MT5_SWAP_MODES[5] is SwapType.PERCENT_ANNUAL

    gold = SymbolSpec(symbol="XAUUSD", digits=3, contract_size=1.0,
                      swap_type=MT5_SWAP_MODES[1], swap_long=-531.6)
    nightly = gold.swap_cost(1, 1.0, 4000.0, datetime(2024, 1, 2))
    assert nightly == pytest.approx(-0.5316)          # cents, not hundreds
    assert abs(nightly) * 365 / 4000 < 0.10           # under 10% annualised
