import unittest
import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm
from likelihood import observed_nll


class IntervalLikelihoodTests(unittest.TestCase):
    def test_independent_quadrature(self):
        for a, b in [
            (-2.0, 1.0),
            (-8.0, -7.0),
            (7.0, 8.0),
            (0.0, 0.000001),
            (-0.5, 0.5),
        ]:
            probability = quad(lambda x: norm.pdf(x), a, b, epsabs=1e-30, epsrel=1e-10)[
                0
            ]
            actual = observed_nll(
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
                torch.tensor([a], dtype=torch.float64),
                torch.tensor([b], dtype=torch.float64),
            ).item()
            self.assertAlmostEqual(actual, -np.log(probability), places=7)

    def test_positive_extreme_tail_is_finite(self):
        mean = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        result = observed_nll(
            mean,
            scale,
            torch.tensor([50.0], dtype=torch.float64),
            torch.tensor([51.0], dtype=torch.float64),
        )
        result.sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(scale.grad).all())

    def test_interval_shrink_matches_density_times_width(self):
        mean = torch.tensor([0.2], dtype=torch.float64)
        scale = torch.tensor([0.7], dtype=torch.float64)
        exact = observed_nll(
            mean,
            scale,
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        )
        width = 1e-6
        interval = observed_nll(
            mean,
            scale,
            torch.tensor([1.0 - width / 2], dtype=torch.float64),
            torch.tensor([1.0 + width / 2], dtype=torch.float64),
        )
        self.assertAlmostEqual(float(interval + np.log(width)), float(exact), places=6)

    def test_interval_derivative(self):
        lower = torch.tensor([-0.7, 0.2, 7.0], dtype=torch.float64)
        upper = torch.tensor([0.3, 0.6, 8.0], dtype=torch.float64)
        mean = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda m, s: observed_nll(m, s, lower, upper), (mean, scale)
            )
        )

    def test_reversed_bounds_rejected(self):
        with self.assertRaises(ValueError):
            observed_nll(
                torch.zeros(1), torch.ones(1), torch.tensor([2.0]), torch.tensor([1.0])
            )


if __name__ == "__main__":
    unittest.main()
