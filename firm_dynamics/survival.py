from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinnedCounts:
    ages: np.ndarray
    totals: np.ndarray
    survivors: np.ndarray


@dataclass(frozen=True)
class SurvivalFractions:
    ages: np.ndarray
    fractions: np.ndarray


def add_analysis_columns(df, *, as_of="2023-01-01"):
    """Return a copy with age and dead indicator columns used in notebooks."""
    out = df.copy()
    out["age"] = (pd.to_datetime(as_of) - out["Entry Date"]).dt.days / 365.25
    out["Dead"] = 1 - out["status"]
    return out


def binned_counts(
    df,
    *,
    category=None,
    value=None,
    bin_width=0.2,
    age_col="age",
    status_col="status",
):
    """Count total and surviving firms in age bins."""
    df1 = _filtered(df, category, value)
    if df1.empty:
        return BinnedCounts(np.array([]), np.array([], dtype=int), np.array([], dtype=int))

    _require_columns(df1, [age_col, status_col])
    bins = _age_bins(df1[age_col], bin_width)
    df1 = df1.copy()
    df1["age_bin"] = pd.cut(df1[age_col], bins)

    grouped = df1.groupby("age_bin", observed=True)
    totals = grouped.size().to_numpy(dtype=int)
    survivors = grouped[status_col].sum().to_numpy(dtype=int)
    ages = grouped.size().index.map(lambda interval: interval.right).to_numpy(dtype=float)

    return BinnedCounts(ages=ages, totals=totals, survivors=survivors)


def survival_fractions(
    df,
    *,
    category=None,
    value=None,
    bin_width=0.2,
    age_col="age",
    status_col="status",
):
    """Compute survivor fraction in each populated age bin."""
    counts = binned_counts(
        df,
        category=category,
        value=value,
        bin_width=bin_width,
        age_col=age_col,
        status_col=status_col,
    )
    if counts.totals.size == 0:
        return SurvivalFractions(counts.ages, np.array([], dtype=float))

    return SurvivalFractions(
        ages=counts.ages,
        fractions=counts.survivors / counts.totals,
    )


def _filtered(df, category, value):
    if category is None:
        return df.copy()
    return df[df[category] == value].copy()


def _age_bins(ages, bin_width):
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    max_age = float(np.nanmax(ages))
    if max_age <= 0:
        return np.array([0.0, bin_width])

    upper = np.ceil(max_age / bin_width) * bin_width + bin_width
    return np.arange(0.0, upper, bin_width)


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
