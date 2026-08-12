"""Strategy base class and look-ahead-free data access."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .broker import Broker
from .orders import Order, OrderResult, Position, Trade

# Names the engine puts on every strategy instance; a parameter cannot use them.
RESERVED_NAMES = ("open", "high", "low", "close", "volume", "data", "broker")


class _ArrayView:
    """A view of an array that only exposes data up to the current bar.

    ``self.close[-1]`` is the current bar, ``self.close[-2]`` the previous one.
    Reading ahead raises ``IndexError``, so accidental look-ahead is impossible.
    """

    __slots__ = ("_array", "_owner", "name")

    def __init__(self, array: np.ndarray, owner: "Strategy", name: str = ""):
        self._array = np.asarray(array)
        self._owner = owner
        self.name = name

    def __len__(self) -> int:
        return self._owner._i + 1

    def __getitem__(self, key):
        end = self._owner._i + 1
        if isinstance(key, slice):
            return self._array[:end][key]
        if key >= 0:
            if key >= end:
                raise IndexError(f"look-ahead: index {key} >= current bar {end - 1}")
            return self._array[key]
        if -key > end:
            raise IndexError(f"not enough history: {key} with {end} bars")
        return self._array[end + key]

    @property
    def array(self) -> np.ndarray:
        """All visible history as a numpy array."""
        return self._array[: self._owner._i + 1]

    @property
    def full(self) -> np.ndarray:
        """The complete array (this one *does* include the future). Use it in ``init``."""
        return self._array

    def series(self, n: Optional[int] = None) -> pd.Series:
        data = self.array if n is None else self.array[-n:]
        idx = self._owner.data.index[: self._owner._i + 1]
        return pd.Series(data, index=idx[-len(data):], name=self.name)

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def __repr__(self) -> str:
        tail = self.array[-3:]
        return f"<{self.name or 'series'} ...{np.round(tail, 6).tolist()}>"


class Strategy:
    """Subclass this and implement :meth:`init` and :meth:`on_bar`.

    Minimal example::

        class Cross(Strategy):
            fast = 20
            slow = 50

            def init(self):
                self.ma_f = self.I(indicators.sma, self.close.full, self.fast)
                self.ma_s = self.I(indicators.sma, self.close.full, self.slow)

            def on_bar(self):
                if self.ma_f[-1] > self.ma_s[-1] and not self.has_position("buy"):
                    self.close_all()
                    self.buy(0.10, sl=self.close[-1] - 5, tp=self.close[-1] + 10)
    """

    # Class attributes are the tunable parameters; override them per run with
    # ``Backtest(..., strategy_params={"fast": 10})``.

    def __init__(self, **params):
        for key, value in params.items():
            if not hasattr(type(self), key):
                raise AttributeError(
                    f"{type(self).__name__} has no parameter '{key}'. "
                    f"Declare it as a class attribute."
                )
            setattr(self, key, value)
        self._params = dict(params)
        self.broker: Broker = None       # type: ignore[assignment]
        self.data: pd.DataFrame = None   # type: ignore[assignment]
        self._i: int = -1
        self._indicators: List[tuple] = []
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------- life cycle
    def _bind(self, broker: Broker, data: pd.DataFrame) -> None:
        clashes = [n for n in RESERVED_NAMES if n in vars(type(self))
                   or n in self._params]
        if clashes:
            raise AttributeError(
                f"{type(self).__name__} declares the reserved name(s) {clashes}. "
                f"These belong to the bar series exposed on the strategy "
                f"({', '.join(RESERVED_NAMES)}); rename your parameter, "
                f"e.g. 'lots' instead of 'volume'."
            )
        self.broker = broker
        self.data = data
        self.open = _ArrayView(data["open"].to_numpy(), self, "open")
        self.high = _ArrayView(data["high"].to_numpy(), self, "high")
        self.low = _ArrayView(data["low"].to_numpy(), self, "low")
        self.close = _ArrayView(data["close"].to_numpy(), self, "close")
        self.volume = _ArrayView(data["volume"].to_numpy(), self, "volume")
        self.time_index = data.index

    def init(self) -> None:
        """Runs once before the loop. Pre-compute your indicators here."""

    def on_bar(self) -> None:
        """Runs at the close of every bar. Your logic goes here."""

    def on_trade_closed(self, trade: Trade) -> None:
        """Called every time a trade is closed."""

    def on_finish(self) -> None:
        """Called when the backtest ends."""

    # ------------------------------------------------------------- indicators
    def I(self, func: Callable, *args, name: str = "", **kwargs) -> _ArrayView:  # noqa: E743
        """Pre-compute an indicator over the whole history, exposed without look-ahead.

        The computation happens once inside :meth:`init` (fast and vectorised),
        but the returned view only lets you read up to the current bar.
        """
        values = func(*args, **kwargs)
        if isinstance(values, tuple):
            views = tuple(
                _ArrayView(np.asarray(v, dtype="float64"), self, f"{name or func.__name__}[{k}]")
                for k, v in enumerate(values)
            )
            self._indicators.append((name or func.__name__, views))
            return views  # type: ignore[return-value]
        view = _ArrayView(np.asarray(values, dtype="float64"), self, name or func.__name__)
        self._indicators.append((view.name, view))
        return view

    # ------------------------------------------------------------ conveniences
    @property
    def i(self) -> int:
        """Index of the current bar."""
        return self._i

    @property
    def now(self) -> datetime:
        """Open time of the current bar."""
        return self.time_index[self._i].to_pydatetime()

    @property
    def bar(self) -> dict:
        return self.broker._bar

    @property
    def price(self) -> float:
        """Last known bid price."""
        return self.broker.bid

    @property
    def bid(self) -> float:
        return self.broker.bid

    @property
    def ask(self) -> float:
        return self.broker.ask

    @property
    def spread_points(self) -> float:
        return self.broker.spread_points

    @property
    def equity(self) -> float:
        return self.broker.equity

    @property
    def balance(self) -> float:
        return self.broker.balance

    @property
    def free_margin(self) -> float:
        return self.broker.free_margin

    @property
    def margin_level(self) -> float:
        return self.broker.margin_level

    @property
    def positions(self) -> List[Position]:
        return list(self.broker.positions.values())

    @property
    def position(self) -> Optional[Position]:
        """The most recent open position, or ``None``."""
        if not self.broker.positions:
            return None
        return max(self.broker.positions.values(), key=lambda p: p.ticket)

    @property
    def orders(self) -> List[Order]:
        return list(self.broker.orders.values())

    @property
    def trades(self) -> List[Trade]:
        return self.broker.trades

    @property
    def spec(self):
        return self.broker.spec

    @property
    def point(self) -> float:
        return self.broker.spec.point

    @property
    def pip(self) -> float:
        return self.broker.spec.pip

    def has_position(self, direction: Optional[str] = None) -> bool:
        return self.broker.has_position(direction)

    # ---------------------------------------------------------------- trading
    def buy(self, volume: float = None, **kwargs) -> OrderResult:
        return self.broker.buy(volume, **kwargs)

    def sell(self, volume: float = None, **kwargs) -> OrderResult:
        return self.broker.sell(volume, **kwargs)

    def buy_limit(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.broker.buy_limit(price, volume, **kwargs)

    def sell_limit(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.broker.sell_limit(price, volume, **kwargs)

    def buy_stop(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.broker.buy_stop(price, volume, **kwargs)

    def sell_stop(self, price: float, volume: float = None, **kwargs) -> OrderResult:
        return self.broker.sell_stop(price, volume, **kwargs)

    def close(self, position=None, volume: float = None) -> OrderResult:
        target = position or self.position
        if target is None:
            return OrderResult(ok=False, retcode="no_position")
        return self.broker.close(target, volume)

    def close_all(self, only: Optional[Callable[[Position], bool]] = None) -> int:
        return self.broker.close_all(only=only)

    def modify(self, position=None, sl="keep", tp="keep") -> OrderResult:
        target = position or self.position
        if target is None:
            return OrderResult(ok=False, retcode="no_position")
        return self.broker.modify(target, sl=sl, tp=tp)

    def cancel_all(self) -> int:
        return self.broker.cancel_all()

    # ----------------------------------------------------------------- sizing
    def volume_for_risk(self, risk_amount: float, stop_points: float) -> float:
        return self.broker.volume_for_risk(risk_amount, stop_points)

    def volume_for_risk_pct(self, risk_pct: float, stop_points: float) -> float:
        return self.broker.volume_for_risk_pct(risk_pct, stop_points)

    def points(self, price_delta: float) -> float:
        return price_delta / self.point

    # ------------------------------------------------------ position management
    def trailing_stop(self, distance_points: float, position: Optional[Position] = None,
                      step_points: float = 0.0) -> None:
        """Drag the stop ``distance_points`` behind price, never against you."""
        targets = [position] if position else self.positions
        step = step_points * self.point
        for pos in targets:
            if pos is None:
                continue
            distance = distance_points * self.point
            if pos.is_long:
                new_sl = self.bid - distance
                if pos.sl is None or new_sl > pos.sl + step:
                    self.broker.modify(pos, sl=new_sl)
            else:
                new_sl = self.ask + distance
                if pos.sl is None or new_sl < pos.sl - step:
                    self.broker.modify(pos, sl=new_sl)

    def break_even(self, trigger_points: float, offset_points: float = 0.0,
                   position: Optional[Position] = None) -> None:
        """Move the stop to break-even once the position gains ``trigger_points``."""
        targets = [position] if position else self.positions
        for pos in targets:
            if pos is None:
                continue
            gain = (self.bid - pos.open_price) * pos.sign / self.point
            if gain < trigger_points:
                continue
            target_sl = pos.open_price + pos.sign * offset_points * self.point
            if pos.sl is None or (target_sl - pos.sl) * pos.sign > 0:
                self.broker.modify(pos, sl=target_sl)

    # ------------------------------------------------------------------- misc
    def log(self, message: str, **data) -> None:
        self.broker._log("strategy", message=message, **data)

    def params(self) -> Dict[str, Any]:
        """Effective parameters (class attributes plus overrides)."""
        out = {}
        for key in dir(type(self)):
            if key.startswith("_"):
                continue
            value = getattr(type(self), key)
            if isinstance(value, (int, float, str, bool)) and not callable(value):
                out[key] = getattr(self, key)
        out.update(self._params)
        return out


class FunctionStrategy(Strategy):
    """Wrap a plain function as a strategy.

    ::

        def my_logic(ctx):
            if ctx.close[-1] > ctx.close[-2]:
                ctx.buy(0.1)

        Backtest(data, FunctionStrategy.of(my_logic), spec)
    """

    func: Optional[Callable] = None
    init_func: Optional[Callable] = None

    @classmethod
    def of(cls, func: Callable, init_func: Optional[Callable] = None) -> type:
        return type(
            f"FunctionStrategy_{func.__name__}",
            (FunctionStrategy,),
            {"func": staticmethod(func),
             "init_func": staticmethod(init_func) if init_func else None},
        )

    def init(self) -> None:
        if self.init_func:
            self.init_func(self)

    def on_bar(self) -> None:
        if self.func:
            self.func(self)


class SignalStrategy(Strategy):
    """A strategy driven by two boolean signal arrays.

    Handy when the signals are already computed in a DataFrame::

        Backtest(data, SignalStrategy, spec,
                 strategy_params={"long_signal": longs, "short_signal": shorts,
                                  "lots": 0.1, "sl_points": 300, "tp_points": 600})
    """

    long_signal: Optional[Sequence[bool]] = None
    short_signal: Optional[Sequence[bool]] = None
    exit_long_signal: Optional[Sequence[bool]] = None
    exit_short_signal: Optional[Sequence[bool]] = None
    lots: float = 0.01
    sl_points: float = 0.0
    tp_points: float = 0.0
    reverse: bool = True   # an opposite signal closes whatever is open

    def init(self) -> None:
        n = len(self.data)
        for attr in ("long_signal", "short_signal", "exit_long_signal", "exit_short_signal"):
            values = getattr(self, attr)
            if values is None:
                setattr(self, attr, np.zeros(n, dtype=bool))
            else:
                arr = np.asarray(values).astype(bool)
                if len(arr) != n:
                    raise ValueError(f"{attr} has {len(arr)} items but the data has {n}")
                setattr(self, attr, arr)

    def on_bar(self) -> None:
        i = self._i
        price = self.bid
        sl_dist = self.sl_points * self.point
        tp_dist = self.tp_points * self.point

        if self.exit_long_signal[i]:
            self.close_all(only=lambda p: p.is_long)
        if self.exit_short_signal[i]:
            self.close_all(only=lambda p: p.is_short)

        if self.long_signal[i]:
            if self.reverse:
                self.close_all(only=lambda p: p.is_short)
            if not self.has_position("buy"):
                self.buy(self.lots,
                         sl=price - sl_dist if self.sl_points else None,
                         tp=price + tp_dist if self.tp_points else None)
        elif self.short_signal[i]:
            if self.reverse:
                self.close_all(only=lambda p: p.is_long)
            if not self.has_position("sell"):
                self.sell(self.lots,
                          sl=price + sl_dist if self.sl_points else None,
                          tp=price - tp_dist if self.tp_points else None)
