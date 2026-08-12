"""Data loading and preparation: MT5, bar CSVs and tick CSVs.

Canonical internal format: a ``DataFrame`` with a ``DatetimeIndex`` named
``time`` and columns ``open, high, low, close, volume`` (plus an optional
``spread`` in points and ``real_volume``). Prices are **bid** unless the symbol
is configured with ``chart_mode = last``.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from .enums import Timeframe

OHLC_COLUMNS = ["open", "high", "low", "close", "volume"]

_COLUMN_ALIASES = {
    "<date>": "date", "date": "date",
    "<time>": "time_", "time": "time_",
    "<datetime>": "time", "datetime": "time", "timestamp": "time",
    "<open>": "open", "open": "open", "o": "open",
    "<high>": "high", "high": "high", "h": "high",
    "<low>": "low", "low": "low", "l": "low",
    "<close>": "close", "close": "close", "c": "close",
    "<tickvol>": "volume", "tickvol": "volume", "tick_volume": "volume",
    "<vol>": "real_volume", "vol": "real_volume", "real_volume": "real_volume",
    "<volume>": "volume", "volume": "volume",
    "<spread>": "spread", "spread": "spread",
    "<bid>": "bid", "bid": "bid",
    "<ask>": "ask", "ask": "ask",
    "<last>": "last", "last": "last",
    "<flags>": "flags", "flags": "flags",
}


# --------------------------------------------------------------------- helpers
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
                              for c in df.columns})


def _build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        times = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns and "time_" in df.columns:
        times = pd.to_datetime(
            df["date"].astype(str) + " " + df["time_"].astype(str),
            format="mixed", errors="coerce",
        )
    elif "date" in df.columns:
        times = pd.to_datetime(df["date"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        times = df.index
    else:
        raise ValueError("No date/time column found in the data")
    out = df.drop(columns=[c for c in ("date", "time_", "time") if c in df.columns])
    out.index = pd.DatetimeIndex(times, name="time")
    return out[out.index.notna()].sort_index()


def prepare_bars(df: pd.DataFrame, spread_points: Optional[float] = None) -> pd.DataFrame:
    """Normalise any DataFrame into the canonical bar format."""
    df = _normalize_columns(df.copy())
    if not isinstance(df.index, pd.DatetimeIndex) or "date" in df.columns:
        df = _build_time_index(df)
    else:
        df.index = pd.DatetimeIndex(df.index, name="time")
        df = df.sort_index()

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Columns present: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if spread_points is not None and "spread" not in df.columns:
        df["spread"] = float(spread_points)

    keep = [c for c in ["open", "high", "low", "close", "volume", "real_volume", "spread"]
            if c in df.columns]
    df = df[keep].astype(dict.fromkeys(keep, "float64"))
    return df[~df.index.duplicated(keep="last")]


def slice_dates(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Trim by date (accepts ``str``, ``datetime`` or ``None``)."""
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def resample_bars(df: pd.DataFrame, timeframe: Union[str, int, Timeframe]) -> pd.DataFrame:
    """Aggregate bars into a higher timeframe."""
    tf = Timeframe.parse(timeframe)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "real_volume" in df.columns:
        agg["real_volume"] = "sum"
    if "spread" in df.columns:
        agg["spread"] = "mean"
    out = df.resample(tf.pandas_freq, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def add_gap_points(df: pd.DataFrame, point: float) -> pd.DataFrame:
    """Add a ``gap_points`` column: the jump between previous close and open."""
    df = df.copy()
    df["gap_points"] = (df["open"] - df["close"].shift(1)).abs() / point
    df["gap_points"] = df["gap_points"].fillna(0.0)
    return df


# ------------------------------------------------------------------------ MT5
def load_mt5_bars(
    symbol: str,
    timeframe: Union[str, int, Timeframe] = Timeframe.M15,
    start: Union[str, datetime, None] = None,
    end: Union[str, datetime, None] = None,
    count: Optional[int] = None,
    initialize: bool = True,
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
    terminal_path: Optional[str] = None,
) -> pd.DataFrame:
    """Download bars from the MetaTrader 5 terminal.

    ``count`` fetches the last N bars; with ``start``/``end`` the range is used
    instead. The terminal must be installed and the symbol visible.
    """
    import MetaTrader5 as mt5

    if initialize:
        kwargs = {}
        if terminal_path:
            kwargs["path"] = terminal_path
        if login:
            kwargs.update(login=login, password=password, server=server)
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"Could not initialise MT5: {mt5.last_error()}")

    tf = Timeframe.parse(timeframe).to_mt5()
    mt5.symbol_select(symbol, True)

    if count:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
    else:
        start_dt = pd.Timestamp(start or "1970-01-01").to_pydatetime()
        end_dt = pd.Timestamp(end or datetime.now()).to_pydatetime()
        rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume", "real_volume": "real_volume"})
    return prepare_bars(df)


def load_mt5_ticks(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    flags: str = "all",
    initialize: bool = True,
) -> pd.DataFrame:
    """Download ticks from the terminal (``time, bid, ask, last, volume``)."""
    import MetaTrader5 as mt5

    if initialize and not mt5.initialize():
        raise RuntimeError(f"Could not initialise MT5: {mt5.last_error()}")
    flag_map = {"all": mt5.COPY_TICKS_ALL, "info": mt5.COPY_TICKS_INFO,
                "trade": mt5.COPY_TICKS_TRADE}
    ticks = mt5.copy_ticks_range(
        symbol, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime(),
        flag_map.get(flags, mt5.COPY_TICKS_ALL),
    )
    if ticks is None or len(ticks) == 0:
        raise RuntimeError(f"MT5 returned no ticks for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time_msc"], unit="ms")
    return df.set_index("time")[["bid", "ask", "last", "volume"]]


# ------------------------------------------------------------------------ CSV
def read_csv_bars(
    path: str,
    sep: Optional[str] = None,
    start=None,
    end=None,
    timeframe: Union[str, int, Timeframe, None] = None,
    spread_points: Optional[float] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """Read a bar CSV (MT5 export format or any other) and normalise it."""
    if sep is None:
        sep = _sniff_sep(path)
    df = pd.read_csv(path, sep=sep, **read_kwargs)
    df = prepare_bars(df, spread_points=spread_points)
    df = slice_dates(df, start, end)
    if timeframe is not None:
        df = resample_bars(df, timeframe)
    return df


def _sniff_sep(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        head = fh.readline()
    for candidate in ("\t", ";", ","):
        if candidate in head:
            return candidate
    return ","


def ticks_to_bars(
    source: Union[str, pd.DataFrame],
    timeframe: Union[str, int, Timeframe] = Timeframe.M1,
    price: str = "bid",
    start=None,
    end=None,
    chunksize: int = 2_000_000,
    sep: Optional[str] = None,
    digits: Optional[int] = None,
    cache: Optional[str] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Turn a tick file (or DataFrame) into OHLC bars with the real spread.

    Built for huge files: it reads in chunks and aggregates as it goes, so a
    multi-gigabyte CSV fits comfortably in memory.

    Parameters
    ----------
    source : path to an MT5 tick CSV export, or an already loaded DataFrame.
    price : ``"bid"``, ``"ask"``, ``"mid"`` or ``"last"`` - which series forms the OHLC.
    cache : when given, the result is written to / read from that ``.parquet`` file.
    """
    tf = Timeframe.parse(timeframe)

    if cache and os.path.exists(cache):
        return slice_dates(pd.read_parquet(cache), start, end)

    if isinstance(source, pd.DataFrame):
        chunks: Iterable[pd.DataFrame] = [source]
    else:
        if sep is None:
            sep = _sniff_sep(source)
        chunks = pd.read_csv(source, sep=sep, chunksize=chunksize, low_memory=False)

    parts = []
    total_rows = 0
    for n, chunk in enumerate(chunks, 1):
        chunk = _normalize_columns(chunk)
        chunk = _build_time_index(chunk) if not isinstance(chunk.index, pd.DatetimeIndex) else chunk
        if start is not None:
            chunk = chunk[chunk.index >= pd.Timestamp(start)]
        if end is not None:
            chunk = chunk[chunk.index <= pd.Timestamp(end)]
        if chunk.empty:
            continue

        bid = chunk["bid"] if "bid" in chunk else None
        ask = chunk["ask"] if "ask" in chunk else None
        last = chunk["last"] if "last" in chunk else None
        if bid is not None:
            bid = bid.replace(0.0, np.nan).ffill()
        if ask is not None:
            ask = ask.replace(0.0, np.nan).ffill()
        if last is not None:
            last = last.replace(0.0, np.nan)

        if price == "bid" and bid is not None:
            series = bid
        elif price == "ask" and ask is not None:
            series = ask
        elif price == "mid" and bid is not None and ask is not None:
            series = (bid + ask) / 2.0
        elif price == "last" and last is not None:
            series = last.ffill()
        else:
            series = bid if bid is not None else last
        if series is None:
            raise ValueError("The tick file has no usable bid/ask/last columns")

        frame = pd.DataFrame({"price": series})
        if bid is not None and ask is not None:
            frame["spread_price"] = (ask - bid).clip(lower=0)
        frame["ticks"] = 1.0
        if "volume" in chunk:
            frame["real_volume"] = pd.to_numeric(chunk["volume"], errors="coerce").fillna(0.0)

        grouped = frame.resample(tf.pandas_freq, label="left", closed="left")
        agg = {"price": ["first", "max", "min", "last"], "ticks": "sum"}
        if "spread_price" in frame:
            agg["spread_price"] = "mean"
        if "real_volume" in frame:
            agg["real_volume"] = "sum"
        part = grouped.agg(agg)
        part.columns = ["_".join(c).strip("_") for c in part.columns.to_flat_index()]
        parts.append(part.dropna(subset=["price_first"]))

        total_rows += len(chunk)
        if progress:
            print(f"  chunk {n}: {total_rows:,} ticks processed "
                  f"(up to {chunk.index[-1]})", end="\r", flush=True)

    if progress:
        print()
    if not parts:
        raise ValueError("No bars were produced: check the date range")

    merged = pd.concat(parts)
    final_agg = {"price_first": "first", "price_max": "max",
                 "price_min": "min", "price_last": "last", "ticks_sum": "sum"}
    if "spread_price_mean" in merged.columns:
        final_agg["spread_price_mean"] = "mean"
    if "real_volume_sum" in merged.columns:
        final_agg["real_volume_sum"] = "sum"
    merged = merged.groupby(level=0).agg(final_agg).sort_index()

    out = pd.DataFrame(
        {
            "open": merged["price_first"],
            "high": merged["price_max"],
            "low": merged["price_min"],
            "close": merged["price_last"],
            "volume": merged["ticks_sum"],
        }
    )
    if "real_volume_sum" in merged:
        out["real_volume"] = merged["real_volume_sum"]
    if "spread_price_mean" in merged and digits:
        out["spread"] = merged["spread_price_mean"] / (10.0 ** -digits)
    elif "spread_price_mean" in merged:
        out["spread_price"] = merged["spread_price_mean"]

    out.index.name = "time"
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        out.to_parquet(cache)
    return out


# ---------------------------------------------------------------------- cache
def save_bars(df: pd.DataFrame, path: str) -> str:
    """Save bars to parquet (or csv, depending on the extension)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".csv"):
        df.to_csv(path)
    else:
        df.to_parquet(path)
    return path


def load_bars(path: str, start=None, end=None,
              timeframe: Union[str, int, Timeframe, None] = None) -> pd.DataFrame:
    """Load bars from parquet or csv, optionally resampling."""
    df = read_csv_bars(path) if path.endswith(".csv") else prepare_bars(pd.read_parquet(path))
    df = slice_dates(df, start, end)
    if timeframe is not None:
        df = resample_bars(df, timeframe)
    return df


def get_bars(
    symbol: str,
    timeframe: Union[str, int, Timeframe] = Timeframe.M15,
    start=None,
    end=None,
    cache_dir: str = "./.cache",
    source: str = "auto",
    csv_path: Optional[str] = None,
    refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Cached shortcut: try the local cache, then a CSV, then MT5."""
    tf = Timeframe.parse(timeframe)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_{tf.name}.parquet")

    if not refresh and os.path.exists(cache_file):
        return slice_dates(prepare_bars(pd.read_parquet(cache_file)), start, end)

    if source in ("auto", "csv") and csv_path:
        df = read_csv_bars(csv_path, timeframe=tf, **kwargs)
    else:
        df = load_mt5_bars(symbol, tf, start=start, end=end, **kwargs)

    save_bars(df, cache_file)
    return slice_dates(df, start, end)


# ------------------------------------------------------------------ diagnostics
def data_quality_report(df: pd.DataFrame, timeframe: Union[str, int, Timeframe]) -> dict:
    """Spot gaps, duplicates and impossible bars before trusting a backtest."""
    tf = Timeframe.parse(timeframe)
    expected = pd.Timedelta(minutes=tf.minutes)
    deltas = df.index.to_series().diff().dropna()
    gaps = deltas[deltas > expected]
    bad_ohlc = df[(df["high"] < df["low"]) |
                  (df["open"] > df["high"]) | (df["open"] < df["low"]) |
                  (df["close"] > df["high"]) | (df["close"] < df["low"])]
    return {
        "bars": len(df),
        "start": df.index[0] if len(df) else None,
        "end": df.index[-1] if len(df) else None,
        "duplicates": int(df.index.duplicated().sum()),
        "gaps": int(len(gaps)),
        "largest_gap": gaps.max() if len(gaps) else pd.Timedelta(0),
        "zero_volume_bars": int((df.get("volume", pd.Series(dtype=float)) == 0).sum()),
        "invalid_ohlc": int(len(bad_ohlc)),
        "nan_rows": int(df.isna().any(axis=1).sum()),
        "median_spread_points": float(df["spread"].median()) if "spread" in df else None,
    }
