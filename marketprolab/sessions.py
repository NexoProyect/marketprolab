"""Quote and trade sessions, per weekday."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

# 0 = Monday ... 6 = Sunday (matches datetime.weekday())
WEEKDAY_NAMES = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_RANGE_RE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def _to_minutes(h: int, m: int) -> int:
    return h * 60 + m


def parse_ranges(spec: str | Sequence) -> List[Tuple[int, int]]:
    """Turn ``"00:00-20:58, 22:00-24:00"`` into ``[(0, 1258), (1320, 1440)]``.

    Minutes are counted from midnight. ``24:00`` equals 1440.
    """
    if spec is None:
        return []
    if not isinstance(spec, str):
        out: List[Tuple[int, int]] = []
        for item in spec:
            if isinstance(item, str):
                out.extend(parse_ranges(item))
            else:
                a, b = item
                a = _to_minutes(a.hour, a.minute) if isinstance(a, time) else int(a)
                b = _to_minutes(b.hour, b.minute) if isinstance(b, time) else int(b)
                out.append((a, b))
        return sorted(out)
    ranges = [
        (_to_minutes(int(h1), int(m1)), _to_minutes(int(h2), int(m2)))
        for h1, m1, h2, m2 in _RANGE_RE.findall(spec)
    ]
    return sorted(ranges)


def format_ranges(ranges: Sequence[Tuple[int, int]]) -> str:
    return ", ".join(f"{a // 60:02d}:{a % 60:02d}-{b // 60:02d}:{b % 60:02d}" for a, b in ranges)


@dataclass
class SessionSpec:
    """A weekly schedule.

    ``days`` maps the weekday (0=Monday) to a list of ``(start_minute,
    end_minute)`` ranges. A day with no ranges is closed.

    Example::

        SessionSpec.from_dict({
            "sunday":   "22:01-24:00",
            "monday":   "00:00-20:58, 22:00-24:00",
            "friday":   "00:00-20:58",
        })
    """

    days: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    timezone: str | None = None  # informational: server timezone (e.g. "Etc/GMT-3")

    # -------------------------------------------------------------- factories
    @classmethod
    def from_dict(cls, mapping: Dict, timezone: str | None = None) -> "SessionSpec":
        days: Dict[int, List[Tuple[int, int]]] = {}
        for key, value in mapping.items():
            idx = key if isinstance(key, int) else WEEKDAY_NAMES[str(key).strip().lower()]
            days[idx] = parse_ranges(value)
        return cls(days=days, timezone=timezone)

    @classmethod
    def always_open(cls) -> "SessionSpec":
        return cls(days={d: [(0, 1440)] for d in range(7)})

    @classmethod
    def forex_default(cls, timezone: str | None = None) -> "SessionSpec":
        """Sunday 22:00 -> Friday 21:00 (typical GMT+2/+3 server)."""
        return cls.from_dict(
            {
                "sunday": "22:00-24:00",
                "monday": "00:00-24:00",
                "tuesday": "00:00-24:00",
                "wednesday": "00:00-24:00",
                "thursday": "00:00-24:00",
                "friday": "00:00-21:00",
            },
            timezone=timezone,
        )

    # ---------------------------------------------------------------- queries
    def is_open(self, dt: datetime) -> bool:
        """Is the market open at that instant?"""
        if not self.days:
            return True  # no definition => always open
        minutes = dt.hour * 60 + dt.minute
        for start, end in self.days.get(dt.weekday(), ()):
            if start <= minutes < end:
                return True
        return False

    def is_close_of_session(self, dt: datetime, tolerance_min: int = 1) -> bool:
        """Does that instant fall in the last ``tolerance_min`` of a range?"""
        minutes = dt.hour * 60 + dt.minute
        for _, end in self.days.get(dt.weekday(), ()):
            if end - tolerance_min <= minutes < end:
                return True
        return False

    def is_last_session_of_week(self, dt: datetime) -> bool:
        """True if this is the last open range of the week (Friday close)."""
        ranges = self.days.get(dt.weekday(), [])
        if not ranges:
            return False
        following = [d for d in range(dt.weekday() + 1, 7) if self.days.get(d)]
        if following:
            return False
        minutes = dt.hour * 60 + dt.minute
        return minutes >= ranges[-1][0]

    def next_open(self, dt: datetime, max_days: int = 14) -> datetime | None:
        """Next opening instant at or after ``dt``."""
        if not self.days:
            return dt
        cursor = dt
        for offset in range(max_days):
            day = (cursor + timedelta(days=offset)).date()
            weekday = (dt.weekday() + offset) % 7
            for start, _ in self.days.get(weekday, ()):
                candidate = datetime.combine(day, time(0, 0)) + timedelta(minutes=start)
                if candidate >= dt:
                    return candidate
        return None

    def filter_index(self, index) -> "Iterable[bool]":
        """Boolean mask for a pandas ``DatetimeIndex``."""
        import numpy as np

        if not self.days:
            return np.ones(len(index), dtype=bool)
        minutes = index.hour * 60 + index.minute
        weekdays = index.weekday
        mask = np.zeros(len(index), dtype=bool)
        for day, ranges in self.days.items():
            day_mask = weekdays == day
            if not day_mask.any():
                continue
            in_range = np.zeros(len(index), dtype=bool)
            for start, end in ranges:
                in_range |= (minutes >= start) & (minutes < end)
            mask |= day_mask & in_range
        return mask

    def describe(self) -> str:
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        lines = []
        for d in range(7):
            ranges = self.days.get(d, [])
            lines.append(f"  {names[d]:<10} {format_ranges(ranges) if ranges else 'closed'}")
        return "\n".join(lines)
