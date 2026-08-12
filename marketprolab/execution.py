"""Spread, slippage and latency models.

They are all stateful callables; write your own as long as it honours the
signature of the matching base class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------- SPREAD
class SpreadModel:
    """Returns the spread **in points** for a given bar."""

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, bar: dict, spec) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class FixedSpread(SpreadModel):
    """Constant spread, in points."""

    points: float = 0.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec) -> float:
        return self.points


@dataclass
class DataSpread(SpreadModel):
    """Use the ``spread`` column from the data (MT5 exports it with the bars).

    Falls back to ``fallback_points`` when a bar carries no spread.
    """

    fallback_points: float = 0.0
    multiplier: float = 1.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec) -> float:
        value = bar.get("spread")
        if value is None or value != value:  # None or NaN
            return self.fallback_points
        return float(value) * self.multiplier


@dataclass
class RandomSpread(SpreadModel):
    """Lognormal spread around a mean, with a floor and a cap."""

    mean_points: float = 20.0
    sigma: float = 0.35
    min_points: float = 1.0
    max_points: float = 500.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec) -> float:
        value = self.mean_points * float(self.rng.lognormal(0.0, self.sigma))
        return float(np.clip(value, self.min_points, self.max_points))


@dataclass
class SessionSpread(SpreadModel):
    """Spread that widens outside liquid hours and around the rollover."""

    normal_points: float = 20.0
    wide_points: float = 120.0
    wide_hours: tuple = (21, 22, 23, 0)
    jitter: float = 0.2

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec) -> float:
        hour = bar["time"].hour
        base = self.wide_points if hour in self.wide_hours else self.normal_points
        if self.jitter:
            base *= 1.0 + float(self.rng.normal(0.0, self.jitter))
        return max(0.0, base)


@dataclass
class VolatilitySpread(SpreadModel):
    """Spread proportional to the bar's range (a volatility proxy)."""

    base_points: float = 15.0
    range_factor: float = 0.05
    max_points: float = 500.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec) -> float:
        rng_points = (bar["high"] - bar["low"]) / spec.point
        return float(min(self.base_points + self.range_factor * rng_points, self.max_points))


# ------------------------------------------------------------------ SLIPPAGE
class SlippageModel:
    """Returns slippage **in points**, already oriented.

    Convention: a positive value means a worse price for the trader.
    """

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, bar: dict, spec, side: int, volume: float, reason: str) -> float:
        raise NotImplementedError  # pragma: no cover


@dataclass
class NoSlippage(SlippageModel):
    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec, side, volume, reason) -> float:
        return 0.0


@dataclass
class FixedSlippage(SlippageModel):
    """Always the same adverse points - the classic pessimistic case."""

    points: float = 5.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec, side, volume, reason) -> float:
        return self.points


@dataclass
class RandomSlippage(SlippageModel):
    """Normal slippage: an adverse mean plus noise, optionally clipped at zero.

    ``mean_points`` shifts the distribution towards the bad side, which is what
    actually happens under market execution.
    """

    mean_points: float = 3.0
    sigma_points: float = 3.0
    only_adverse: bool = False
    max_points: float = 200.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec, side, volume, reason) -> float:
        value = float(self.rng.normal(self.mean_points, self.sigma_points))
        if self.only_adverse:
            value = abs(value)
        return float(np.clip(value, -self.max_points, self.max_points))


@dataclass
class VolatilitySlippage(SlippageModel):
    """Slippage proportional to bar range and traded volume.

    Models the fact that fast bars and large lot sizes fill worse.
    """

    range_factor: float = 0.03
    volume_factor: float = 0.0     # extra points per lot
    sigma: float = 0.5
    stop_multiplier: float = 2.0   # stops fill worse than limits
    max_points: float = 500.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec, side, volume, reason) -> float:
        rng_points = (bar["high"] - bar["low"]) / spec.point
        base = self.range_factor * rng_points + self.volume_factor * volume
        if reason in ("sl", "stop_out", "stop"):
            base *= self.stop_multiplier
        value = base * (1.0 + float(self.rng.normal(0.0, self.sigma)))
        return float(np.clip(value, 0.0, self.max_points))


@dataclass
class GapSlippage(SlippageModel):
    """Slippage only where the bar opens away from the previous close.

    Useful to stay honest about Sunday-night stop fills.
    """

    normal_points: float = 1.0
    gap_factor: float = 1.0

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec, side, volume, reason) -> float:
        gap = abs(bar.get("gap_points", 0.0) or 0.0)
        return self.normal_points + self.gap_factor * gap


# ------------------------------------------------------------------- LATENCY
class LatencyModel:
    """Returns the execution delay **in seconds**."""

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __call__(self, reason: str) -> float:
        raise NotImplementedError  # pragma: no cover


@dataclass
class FixedLatency(LatencyModel):
    """Constant latency (round trip plus broker processing)."""

    milliseconds: float = 120.0

    def reset(self, rng): self.rng = rng
    def __call__(self, reason: str = "market") -> float:
        return self.milliseconds / 1000.0


@dataclass
class RandomLatency(LatencyModel):
    """Lognormal latency with a fat tail - what a real VPS looks like."""

    mean_ms: float = 120.0
    sigma: float = 0.6
    min_ms: float = 20.0
    max_ms: float = 3000.0
    spike_probability: float = 0.01
    spike_ms: float = 2000.0

    def reset(self, rng): self.rng = rng
    def __call__(self, reason: str = "market") -> float:
        value = self.mean_ms * float(self.rng.lognormal(0.0, self.sigma))
        if self.spike_probability and self.rng.random() < self.spike_probability:
            value += self.spike_ms
        return float(np.clip(value, self.min_ms, self.max_ms)) / 1000.0


# --------------------------------------------------------------------- HELPERS
@dataclass
class CallableSpread(SpreadModel):
    """Wraps a plain ``func(bar, spec) -> points`` as a spread model."""

    func: object = None

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec) -> float:
        return float(self.func(bar, spec))


@dataclass
class CallableSlippage(SlippageModel):
    """Wraps a plain ``func(bar, spec, side, volume, reason) -> points``."""

    func: object = None

    def reset(self, rng): self.rng = rng
    def __call__(self, bar, spec, side, volume, reason) -> float:
        return float(self.func(bar, spec, side, volume, reason))


def make_spread(value) -> SpreadModel:
    """Accepts a number (fixed points), ``"data"``, a callable, or a model."""
    if isinstance(value, SpreadModel):
        return value
    if value is None:
        return FixedSpread(0.0)
    if isinstance(value, str):
        if value.lower() in ("data", "real", "csv"):
            return DataSpread()
        raise ValueError(f"Unknown spread model: {value}")
    if callable(value):
        return CallableSpread(value)
    return FixedSpread(float(value))


def make_slippage(value) -> SlippageModel:
    """Accepts a number (fixed points), ``None``, a callable, or a model."""
    if isinstance(value, SlippageModel):
        return value
    if value is None:
        return NoSlippage()
    if callable(value):
        return CallableSlippage(value)
    return FixedSlippage(float(value))


def make_latency(value) -> LatencyModel:
    """Accepts milliseconds (a number), ``None``, or a ready-made model."""
    if isinstance(value, LatencyModel):
        return value
    if value is None:
        return FixedLatency(0.0)
    return FixedLatency(float(value))
