"""Target distributions, each paired with the reference quantities it admits.

Convention
----------
A target exposes ``log_prob`` and ``grad_log_prob`` (the score).  Samplers are
written against the score convention standard in the sampling and diffusion
literature: the overdamped Langevin diffusion

    dX_t = grad log pi(X_t) dt + sqrt(2) dW_t

leaves ``pi`` invariant, and every sampler here is a discretization of it.

All targets are vectorized over a leading chain axis: ``x`` has shape
``(n_chains, dim)`` and ``grad_log_prob`` returns the same shape.
"""

import numpy as np


class GaussianTarget:
    """Zero-mean Gaussian with diagonal covariance ``diag(scales**2)``.

    Diagonal by construction, and that is the point.  ULA applied to a
    Gaussian acts independently on each coordinate, so the *discretized*
    chain's stationary law factorizes into one-dimensional AR(1) laws that are
    available in closed form (see :mod:`langevin.theory`).  The numerics can
    then be validated against an exact answer instead of against a longer run
    of themselves.
    """

    def __init__(self, scales):
        scales = np.atleast_1d(np.asarray(scales, dtype=float))
        if np.any(scales <= 0.0):
            raise ValueError("scales must be strictly positive")
        self.scales = scales
        self.var = scales**2
        self.dim = scales.size

    def grad_log_prob(self, x):
        return -x / self.var

    def log_prob(self, x):
        return -0.5 * np.sum(x * x / self.var, axis=-1)

    @property
    def reference_var(self):
        """Exact coordinatewise variance of the target."""
        return self.var

    def __repr__(self):
        return f"GaussianTarget(scales={self.scales.tolist()})"


class TwoComponentMixture:
    """Equal-weight mixture ``0.5 N(-mu e_0, s^2 I) + 0.5 N(+mu e_0, s^2 I)``.

    The modes are separated along the first coordinate.  For ``mu > s`` the
    density is not log-concave and the two wells are separated by a barrier,
    which is the regime where both samplers become metastable: they equilibrate
    rapidly *within* a mode and cross between modes on an exponentially longer
    timescale.  No closed form is available for the discretized chain here, so
    the reference is the exact mixture, whose moments are elementary:

        E[X] = 0,  Var(X_0) = s^2 + mu^2,  Var(X_i) = s^2  for i > 0,

    the first coordinate picking up the between-mode contribution by the law of
    total variance.
    """

    def __init__(self, mu, scale, dim=2):
        if scale <= 0.0:
            raise ValueError("scale must be strictly positive")
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.mu = float(mu)
        self.scale = float(scale)
        self.var = float(scale) ** 2
        self.dim = int(dim)
        self.centers = np.zeros((2, self.dim))
        self.centers[0, 0] = -self.mu
        self.centers[1, 0] = +self.mu

    def _log_component_kernels(self, x):
        """Unnormalized log component densities, shape ``(n_chains, 2)``."""
        d = x[:, None, :] - self.centers[None, :, :]
        return -0.5 * np.sum(d * d, axis=-1) / self.var

    def log_prob(self, x):
        lk = self._log_component_kernels(x)
        m = np.max(lk, axis=-1, keepdims=True)
        return (m[:, 0] + np.log(np.sum(np.exp(lk - m), axis=-1))) - np.log(2.0)

    def grad_log_prob(self, x):
        # Responsibilities w_k = pi_k(x) / sum_j pi_j(x), computed in log space.
        lk = self._log_component_kernels(x)
        m = np.max(lk, axis=-1, keepdims=True)
        w = np.exp(lk - m)
        w /= np.sum(w, axis=-1, keepdims=True)
        # The mixture score is the responsibility-weighted average of the
        # component scores, grad log pi_k(x) = -(x - c_k) / s^2.
        d = x[:, None, :] - self.centers[None, :, :]
        return -np.sum(w[:, :, None] * d, axis=1) / self.var

    @property
    def reference_var(self):
        v = np.full(self.dim, self.var)
        v[0] += self.mu**2
        return v

    def mode_label(self, x):
        """Index of the nearer mode (0 for the negative well, 1 for positive)."""
        return (x[..., 0] > 0.0).astype(np.int8)

    def __repr__(self):
        return (
            f"TwoComponentMixture(mu={self.mu}, scale={self.scale}, dim={self.dim})"
        )
