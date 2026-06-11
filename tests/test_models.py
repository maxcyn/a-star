import unittest

import numpy as np

from firm_dynamics.models import constant, hill, perturbation, power
from firm_dynamics.fitting import aic, bic, binomial_log_likelihood


class ModelTests(unittest.TestCase):
    def test_hill_survival_starts_at_one_and_decreases(self):
        ages = np.array([0.0, 1.0, 2.0, 3.0])

        survival = hill.survival(ages, mu_ub=0.2, mu_lb=0.05, k=2.0, m=4.0)

        self.assertAlmostEqual(survival[0], 1.0)
        self.assertTrue(np.all(np.diff(survival) <= 0.0))

    def test_constant_and_power_models_return_valid_probabilities(self):
        ages = np.array([0.0, 1.0, 2.0])

        constant_survival = constant.survival(ages, rate=0.1)
        power_survival = power.survival(ages, lam=0.2, alpha=0.5)

        self.assertTrue(np.all((0.0 < constant_survival) & (constant_survival <= 1.0)))
        self.assertTrue(np.all((0.0 < power_survival) & (power_survival <= 1.0)))

    def test_binomial_likelihood_clips_probabilities(self):
        survivors = np.array([10, 5])
        totals = np.array([10, 10])
        probabilities = np.array([1.0, 0.5])

        log_likelihood = binomial_log_likelihood(probabilities, survivors, totals)

        self.assertTrue(np.isfinite(log_likelihood))
        self.assertEqual(aic(log_likelihood, n_params=2), 4 - 2 * log_likelihood)
        self.assertAlmostEqual(bic(log_likelihood, n_params=2, n_obs=10), 2 * np.log(10) - 2 * log_likelihood)

    def test_perturbation_model_supports_quad_and_grid_methods(self):
        ages = np.array([0.0, 1.0, 2.0])
        params = perturbation.PerturbationParams(
            mu_ub=0.2,
            mu_lb=0.05,
            k=2.0,
            m=4.0,
            eps0=0.5,
            tau=0.8,
            lam=0.3,
            event_age=1.0,
        )

        quad_survival = perturbation.survival(ages, params, method="quad")
        grid_survival = perturbation.survival(ages, params, method="grid", grid_size=400)

        self.assertAlmostEqual(quad_survival[0], 1.0)
        self.assertTrue(np.all((0.0 < quad_survival) & (quad_survival <= 1.0)))
        np.testing.assert_allclose(quad_survival, grid_survival, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
