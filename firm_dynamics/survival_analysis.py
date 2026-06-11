from firm_dynamics.data import prepare_df
from firm_dynamics.survival import (
    add_analysis_columns,
    binned_counts,
    survival_fractions,
)


def obtain_survival_fractions(df, category=None, filter_val=None):
    result = survival_fractions(df, category=category, value=filter_val)
    return result.fractions, result.ages


def obtain_total_alive_count(df, category=None, filter_val=None):
    result = binned_counts(df, category=category, value=filter_val)
    return result.totals, result.survivors


__all__ = [
    "add_analysis_columns",
    "binned_counts",
    "obtain_survival_fractions",
    "obtain_total_alive_count",
    "prepare_df",
    "survival_fractions",
]
