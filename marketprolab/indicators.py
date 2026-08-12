"""Vectorised basic indicators (numpy/pandas), no external dependencies.

They all return arrays the same length as the input, with ``NaN`` during the
warm-up period, so they can be indexed with the same bar index as the data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype="float64"))


def sma(values, period: int) -> np.ndarray:
    """Simple moving average."""
    return _series(values).rolling(period).mean().to_numpy()


def ema(values, period: int) -> np.ndarray:
    """Exponential moving average."""
    return _series(values).ewm(span=period, adjust=False).mean().to_numpy()


def wma(values, period: int) -> np.ndarray:
    """Linearly weighted moving average."""
    weights = np.arange(1, period + 1)
    return (
        _series(values)
        .rolling(period)
        .apply(lambda w: np.dot(w, weights) / weights.sum(), raw=True)
        .to_numpy()
    )


def rma(values, period: int) -> np.ndarray:
    """Wilder's smoothed average (the one behind MT5's RSI and ATR)."""
    return _series(values).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


def stdev(values, period: int) -> np.ndarray:
    return _series(values).rolling(period).std(ddof=0).to_numpy()


def rsi(values, period: int = 14) -> np.ndarray:
    delta = _series(values).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0 * (avg_gain > 0)).to_numpy()


def true_range(high, low, close) -> np.ndarray:
    high, low, close = _series(high), _series(low), _series(close)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.to_numpy()


def atr(high, low, close, period: int = 14) -> np.ndarray:
    """Average true range."""
    return rma(true_range(high, low, close), period)


def bollinger(values, period: int = 20, deviations: float = 2.0
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(upper, middle, lower)``."""
    mid = sma(values, period)
    sd = stdev(values, period)
    return mid + deviations * sd, mid, mid - deviations * sd


def macd(values, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(macd_line, signal_line, histogram)``."""
    line = ema(values, fast) - ema(values, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(high, low, close, k_period: int = 14, d_period: int = 3
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ``(%K, %D)``."""
    high, low, close = _series(high), _series(low), _series(close)
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    return k.to_numpy(), k.rolling(d_period).mean().to_numpy()


def adx(high, low, close, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(adx, +DI, -DI)``."""
    high, low = _series(high), _series(low)
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = rma(true_range(high, low, close), period)
    plus_di = 100.0 * rma(plus_dm, period) / np.where(tr == 0, np.nan, tr)
    minus_di = 100.0 * rma(minus_dm, period) / np.where(tr == 0, np.nan, tr)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where(
        (plus_di + minus_di) == 0, np.nan, plus_di + minus_di
    )
    return rma(dx, period), plus_di, minus_di


def highest(values, period: int) -> np.ndarray:
    return _series(values).rolling(period).max().to_numpy()


def lowest(values, period: int) -> np.ndarray:
    return _series(values).rolling(period).min().to_numpy()


def roc(values, period: int = 1) -> np.ndarray:
    """Rate of change, in percent."""
    s = _series(values)
    return (s / s.shift(period) - 1.0).to_numpy() * 100.0


def zscore(values, period: int = 20) -> np.ndarray:
    s = _series(values)
    mean = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0).replace(0.0, np.nan)
    return ((s - mean) / sd).to_numpy()


def supertrend(high, low, close, period: int = 10, multiplier: float = 3.0
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ``(line, direction)`` with direction +1 bullish / -1 bearish."""
    high, low, close = _series(high), _series(low), _series(close)
    hl2 = (high + low) / 2.0
    band = multiplier * pd.Series(atr(high, low, close, period))
    upper, lower = (hl2 + band).to_numpy(), (hl2 - band).to_numpy()
    close_v = close.to_numpy()
    n = len(close_v)
    trend = np.ones(n)
    line = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        upper[i] = min(upper[i], upper[i - 1]) if close_v[i - 1] <= upper[i - 1] else upper[i]
        lower[i] = max(lower[i], lower[i - 1]) if close_v[i - 1] >= lower[i - 1] else lower[i]
        if close_v[i] > upper[i - 1]:
            trend[i] = 1
        elif close_v[i] < lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
        line[i] = lower[i] if trend[i] > 0 else upper[i]
    return line, trend


def crossover(a, b) -> np.ndarray:
    """True on the bar where ``a`` crosses above ``b``."""
    a, b = _series(a), _series(b)
    return ((a > b) & (a.shift(1) <= b.shift(1))).to_numpy()


def crossunder(a, b) -> np.ndarray:
    """True on the bar where ``a`` crosses below ``b``."""
    a, b = _series(a), _series(b)
    return ((a < b) & (a.shift(1) >= b.shift(1))).to_numpy()
