"""Gaussian reference laws and a large-separation mixture-barrier approximation.

Gaussian stationary quantities are exact. The mixture barrier is explicitly
asymptotic; it is used to compare the measured crossing rates across separations.
"""

import numpy as np


def ula_stability_limit(scales):
    """Largest step size for which ULA is geometrically ergodic on a Gaussian.

    ULA on ``N(0, diag(scales**2))`` is the coordinatewise AR(1) recursion

        X' = (1 - h / s^2) X + sqrt(2h) xi,       xi ~ N(0, 1),

    which is stable iff ``|1 - h / s^2| < 1``, i.e. ``0 < h < 2 s^2``.  With
    several scales present the binding constraint is the smallest one, so the
    admissible step size is set by the *narrowest* direction while the time to
    decorrelate is set by the widest.  That mismatch is the reason
    ill-conditioned targets are hard for Langevin methods, and it is visible
    directly in the AR(1) coefficients.
    """
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    return 2.0 * np.min(scales**2)


def ula_stationary_var(scales, h):
    """Exact stationary variance of ULA on ``N(0, diag(scales**2))``.

    Solving the AR(1) fixed point ``v = a^2 v + 2h`` with ``a = 1 - h / s^2``
    and ``1 - a^2 = (h / s^2)(2 - h / s^2)`` gives

        v(h) = s^2 / (1 - h / (2 s^2)).

    ULA therefore over-disperses at every positive step size, with bias

        v(h) - s^2 = (h / 2) / (1 - h / (2 s^2)) = h / 2 + O(h^2),

    whose leading term is ``h / 2`` *independently of the scale* ``s``.  The
    correction blows up as ``h`` approaches ``2 s^2``, where the chain ceases
    to be stationary at all.

    Returns ``nan`` in coordinates where ``h`` exceeds the stability limit.
    """
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    var = scales**2
    denom = 1.0 - h / (2.0 * var)
    out = np.where(denom > 0.0, var / np.where(denom > 0.0, denom, 1.0), np.nan)
    return out


def ula_variance_bias(scales, h):
    """Exact ``v(h) - s^2``, the stationary variance inflation of ULA."""
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    return ula_stationary_var(scales, h) - scales**2


def mixture_barrier(mu, scale=1.0):
    """Large-separation approximation to the mixture's log-density barrier.

    Along the separating coordinate the mixture log-density at a mode is
    ``log(1/2) - log Z`` up to ``O(e^{-2mu^2/s^2})``, while at the saddle
    ``x = 0`` both components contribute equally, giving ``-mu^2/(2 s^2) - log Z``.
    Approximating the actual modes by the component centers gives

        Delta(mu) = mu^2 / (2 s^2) - log 2,

    The exact modes are slightly shifted toward zero. Kramers-type asymptotics
    predict the leading exponential ``exp(-Delta)`` for transitions, with a
    separation-dependent prefactor. The discrete experiment counts sign
    crossings (including recrossings), so its fitted slope is a numerical
    comparison, not an exact transition-rate identity.
    """
    mu = np.asarray(mu, dtype=float)
    return mu**2 / (2.0 * float(scale) ** 2) - np.log(2.0)


def gaussian_w2(var_a, var_b):
    """2-Wasserstein distance between two zero-mean diagonal Gaussians.

    For diagonal covariances the optimal coupling is coordinatewise and

        W_2^2 = sum_i (sqrt(v_i) - sqrt(u_i))^2.
    """
    var_a = np.atleast_1d(np.asarray(var_a, dtype=float))
    var_b = np.atleast_1d(np.asarray(var_b, dtype=float))
    return float(np.sqrt(np.sum((np.sqrt(var_a) - np.sqrt(var_b)) ** 2)))


def ula_w2_to_target(scales, h):
    """Exact ``W_2(pi_h, pi)`` for ULA on a Gaussian target.

    Expanding ``sqrt(v_i) - s_i = s_i[(1 - h/(2 s_i^2))^{-1/2} - 1]``
    to first order gives ``h / (4 s_i)``, so

        W_2(pi_h, pi) = (h / 4) sqrt(sum_i s_i^{-2}) + O(h^2),

    i.e. the bias is *first order* in the step size.  This is the rate the
    log-log fit in ``scripts/01`` recovers, and the reason a Metropolis
    correction is worth its cost whenever an asymptotically exact answer is
    wanted.
    """
    return gaussian_w2(ula_stationary_var(scales, h), np.asarray(scales) ** 2)
