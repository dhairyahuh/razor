"""The one place that decides when a payment happens.

Payment volume is not uniform in time, and every part of the simulator has to agree about
how it is shaped. When the benign path draws days from a weekly x salary-window x drift
weighting and a bespoke attack generator draws them from ``rng.uniform(0, n_days)``, the two
populations differ in a way nobody intended: day-of-week alone carries signal, and the
defence scores points for learning the generator's implementation rather than fraud.

The same applies to time of day. Legitimate traffic is bimodal - a late-morning commerce
peak and a larger evening one - and a flat hourly draw is a different distribution wearing
the same column name. Attackers *do* prefer the small hours, so the night preference is
modelled explicitly, as a bias applied on top of the ordinary curve rather than as a
replacement for it. That way "2am" stays weak evidence, which is what it is in reality.

Centralising also settles a definition. ``is_salary_window`` was computed two different ways
- ``(day % 30) + 1`` when weighting days, and the real calendar day-of-month when writing
the feature - so the two disagreed whenever the window did not begin on the 1st.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

#: Relative volume by day of week, Monday first. Friday and Saturday peak, Sunday dips.
WEEKLY = np.array([0.95, 0.98, 1.00, 1.02, 1.12, 1.18, 0.88])

#: Multiplier applied inside the salary window.
SALARY_BUMP = 1.28

#: Per-day compound growth in digital payment volume across the window.
DAILY_DRIFT = 0.0022


def is_salary_window(day_of_month: np.ndarray) -> np.ndarray:
    """Month-end and month-start, when salaries land and bills are paid."""
    dom = np.asarray(day_of_month)
    return (dom <= 3) | (dom >= 29)


def day_weights(n_days: int, start_date: Optional[str] = None) -> np.ndarray:
    """Relative payment volume for each day of the simulation window.

    ``start_date`` anchors the weekly and month-end effects to the real calendar, so that the
    weighting agrees with the ``day_of_week`` and ``is_salary_window`` columns derived from
    the timestamps. Without it the two are consistent only when the window starts on a Monday
    the 1st.
    """
    days = np.arange(n_days)
    if start_date is not None:
        stamps = pd.Timestamp(start_date) + pd.to_timedelta(days, unit="D")
        dow = stamps.dayofweek.to_numpy()
        dom = stamps.day.to_numpy()
    else:
        dow = days % 7
        dom = (days % 30) + 1
    w = WEEKLY[dow] * np.where(is_salary_window(dom), SALARY_BUMP, 1.0) * (1.0 + DAILY_DRIFT * days)
    return w / w.sum()


def draw_days(rng: np.random.Generator, total: int, n_days: int,
              start_date: Optional[str] = None) -> np.ndarray:
    """Integer day indices sampled in proportion to payment volume."""
    return rng.choice(np.arange(n_days), size=total, p=day_weights(n_days, start_date))


def draw_base_day(rng: np.random.Generator, n_days: int, span_days: float = 0.0,
                  start_date: Optional[str] = None) -> float:
    """A single episode start day, sampled from the same weighting as benign volume.

    ``span_days`` is how long the episode runs, so it can be kept inside the window.
    """
    usable = max(int(np.floor(n_days - span_days)), 1)
    day = int(rng.choice(np.arange(usable), p=day_weights(usable, start_date)))
    return float(day) + float(rng.random())


def draw_hours(rng: np.random.Generator, total: int, night_bias: float = 0.0,
               night_owl: Optional[np.ndarray] = None) -> np.ndarray:
    """Hours-of-day from the bimodal legitimate profile, optionally pushed towards night.

    ``night_bias`` is the probability a given payment is moved into the 00:00-05:30 window.
    Attack generators pass a nonzero value; benign traffic reaches the same tail through
    ``night_owl`` customers, who are a genuine and unremarkable part of the population.
    """
    comp = rng.random(total)
    hour = np.empty(total)
    m1 = comp < 0.34
    hour[m1] = rng.normal(11.2, 2.3, int(m1.sum()))
    m2 = (comp >= 0.34) & (comp < 0.86)
    hour[m2] = rng.normal(19.8, 2.1, int(m2.sum()))
    m3 = comp >= 0.86
    hour[m3] = rng.uniform(6, 23, int(m3.sum()))

    p_night = np.full(total, float(night_bias))
    if night_owl is not None:
        p_night = np.maximum(p_night, 0.09 * np.asarray(night_owl, dtype=float))
    shift = rng.random(total) < p_night
    hour[shift] = rng.uniform(0, 5.5, int(shift.sum()))
    return np.clip(hour, 0, 23.999)


def hours_to_hms(rng: np.random.Generator, hour: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = hour.astype(int)
    minute = ((hour - h) * 60).astype(int)
    return h, minute, rng.integers(0, 60, len(hour))


def episode_seconds(rng: np.random.Generator, n_days: int, span_s: float,
                    night_bias: float = 0.0,
                    start_date: Optional[str] = None) -> float:
    """Seconds from the window start to an episode's first payment.

    Combines a day drawn from the benign volume weighting with a start hour drawn from the
    benign diurnal curve, so an attack episode sits in time where real traffic sits.
    """
    span_days = span_s / 86_400.0
    usable = max(int(np.floor(n_days - span_days)), 1)
    day = int(rng.choice(np.arange(usable), p=day_weights(usable, start_date)))
    hour = float(draw_hours(rng, 1, night_bias=night_bias)[0])
    return day * 86_400.0 + hour * 3_600.0
