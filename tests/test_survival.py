import unittest

import numpy as np
import pandas as pd

from firm_dynamics import survival


class SurvivalTests(unittest.TestCase):
    def test_binned_counts_share_age_grid_with_survival_fraction(self):
        df = pd.DataFrame(
            {
                "age": [0.1, 0.2, 0.3, 0.6, 0.8],
                "status": [1, 0, 1, 1, 0],
                "Sector": ["A", "A", "A", "B", "B"],
            }
        )

        counts = survival.binned_counts(df, bin_width=0.5)
        fractions = survival.survival_fractions(df, bin_width=0.5)

        np.testing.assert_allclose(counts.ages, fractions.ages)
        np.testing.assert_array_equal(counts.totals, np.array([3, 2]))
        np.testing.assert_array_equal(counts.survivors, np.array([2, 1]))
        np.testing.assert_allclose(fractions.fractions, np.array([2 / 3, 1 / 2]))

    def test_binned_counts_can_filter_by_category(self):
        df = pd.DataFrame(
            {
                "age": [0.1, 0.2, 0.3, 0.6, 0.8],
                "status": [1, 0, 1, 1, 0],
                "Sector": ["A", "A", "A", "B", "B"],
            }
        )

        counts = survival.binned_counts(df, category="Sector", value="B", bin_width=0.5)

        np.testing.assert_array_equal(counts.totals, np.array([2]))
        np.testing.assert_array_equal(counts.survivors, np.array([1]))


if __name__ == "__main__":
    unittest.main()
