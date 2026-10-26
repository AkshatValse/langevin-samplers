"""Independent references for sampler accounting and Monte Carlo diagnostics."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langevin.diagnostics import (  # noqa: E402
    integrated_autocorr_time,
    moment_summary,
    variance_mc_standard_error,
)
from langevin import provenance  # noqa: E402
from langevin.samplers import mala, ula  # noqa: E402
from langevin.targets import GaussianTarget  # noqa: E402
from langevin.theory import ula_stationary_var, ula_w2_to_target  # noqa: E402


class CountingGaussian(GaussianTarget):
    def __init__(self, scales):
        super().__init__(scales)
        self.gradient_evaluations = 0

    def grad_log_prob(self, x):
        self.gradient_evaluations += x.shape[0]
        return super().grad_log_prob(x)


class SamplerTests(unittest.TestCase):
    def test_reported_gradient_evaluations_match_observed_calls(self):
        for sampler in (ula, mala):
            for steps in (0, 1, 19):
                with self.subTest(sampler=sampler.__name__, steps=steps):
                    target = CountingGaussian([1.0])
                    _, info = sampler(target, 0.1, steps, np.zeros((3, 1)),
                                      np.random.default_rng(7))
                    self.assertEqual(info["grad_evals"], target.gradient_evaluations)
                    self.assertEqual(info["grad_evals"], 3 * (steps + (sampler is mala)))

    def test_mala_gaussian_accept_reject_against_analytic_ratio(self):
        class FixedDraws:
            def standard_normal(self, shape):
                return np.full(shape, np.sqrt(2.0))

            def random(self, size):
                return np.array([0.2, 0.5])

        # For N(0,1), log alpha = -h*(y^2-x^2)/4 = -1 here.
        samples, info = mala(GaussianTarget([1]), 1.0, 1, np.zeros((2, 1)),
                             FixedDraws(), burn_in=0)
        np.testing.assert_allclose(samples[0, :, 0], [2, 0])
        self.assertEqual(info["acceptance_rate"], 0.5)

    def test_ula_stationary_variance_against_exact_ar1_law(self):
        scales, h = np.array([0.5, 1, 2]), 0.2
        samples, _ = ula(GaussianTarget(scales), h, 12_000,
                         np.zeros((48, 3)), np.random.default_rng(2026), thin=5)
        _, observed = moment_summary(samples)
        se = variance_mc_standard_error(samples)
        self.assertTrue(np.all(np.abs(observed - ula_stationary_var(scales, h)) < 5 * se))

    def test_theory_stability_and_first_order_wasserstein(self):
        np.testing.assert_allclose(ula_stationary_var([1], 1), [2])
        self.assertTrue(np.isnan(ula_stationary_var([1], 2)[0]))
        scales = np.array([0.5, 1, 2])
        h = 1e-6
        self.assertAlmostEqual(ula_w2_to_target(scales, h) / h,
                               np.sqrt(np.sum(scales ** -2)) / 4, places=5)


class DiagnosticsTests(unittest.TestCase):
    def test_geyer_pairs_start_at_lag_zero(self):
        with patch("langevin.diagnostics._autocorr",
                   return_value=np.array([1.0, 0.8, 0.5, 0.3, -0.2, -0.2])):
            self.assertAlmostEqual(integrated_autocorr_time(np.arange(6)), 4.2)

    def test_variance_se_is_invariant_to_location(self):
        samples = np.random.default_rng(2).normal(size=(100, 16, 3))
        np.testing.assert_allclose(variance_mc_standard_error(samples),
                                   variance_mc_standard_error(samples + 17), rtol=1e-12)

    def test_repeating_correlated_draws_does_not_create_precision(self):
        values = np.random.default_rng(3).normal(size=(1, 128, 2))
        short = variance_mc_standard_error(np.repeat(values, 16, axis=0))
        long = variance_mc_standard_error(np.repeat(values, 128, axis=0))
        np.testing.assert_allclose(long / short, 1, atol=0.001)

    def test_variance_se_calibrates_across_independent_ar1_experiments(self):
        # Stationary, correlated, nonzero-mean draws; each ensemble estimates
        # the same variance. Compare predicted SE with actual across-run spread.
        rng = np.random.default_rng(12)
        repetitions, steps, chains, rho = 160, 600, 48, 0.9
        traces = np.empty((steps, repetitions, chains))
        traces[0] = rng.normal(size=(repetitions, chains))
        for t in range(1, steps):
            traces[t] = rho * traces[t - 1] + np.sqrt(1 - rho**2) * rng.normal(
                size=(repetitions, chains))
        estimates, errors = [], []
        for r in range(repetitions):
            draws = traces[:, r, :, None] + 3.5
            estimates.append(moment_summary(draws)[1][0])
            errors.append(variance_mc_standard_error(draws)[0])
        estimates, errors = np.array(estimates), np.array(errors)
        ratio = np.sqrt(np.mean(errors**2)) / estimates.std(ddof=1)
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.2)
        coverage = np.mean(np.abs(estimates - 1) < 1.96 * errors)
        self.assertGreater(coverage, 0.89)
        self.assertLess(coverage, 0.995)

    def test_variance_se_validates_replications_and_finite_values(self):
        for shape in ((5,), (10, 1, 2), (1, 8, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                variance_mc_standard_error(np.zeros(shape))
        with self.assertRaises(ValueError):
            variance_mc_standard_error(np.full((2, 2, 1), np.nan))
        np.testing.assert_equal(variance_mc_standard_error(np.ones((10, 8, 2))), [0, 0])


class ProvenanceTests(unittest.TestCase):
    def test_generated_outputs_do_not_hide_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", tmp, *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            (root / "results").mkdir()
            (root / "results" / "run.json").write_text("{}")
            (root / "source.py").write_text("x = 1\n")
            git("add", ".")
            git("commit", "-qm", "fixture")
            sha = git("rev-parse", "HEAD")
            with patch.object(provenance, "PROJECT_ROOT", tmp):
                (root / "results" / "run.json").write_text('{"changed": true}')
                self.assertEqual(provenance.git_commit(), sha)
                (root / "new_source.py").write_text("x = 2\n")
                self.assertEqual(provenance.git_commit(), sha + "-dirty")


if __name__ == "__main__":
    unittest.main()
