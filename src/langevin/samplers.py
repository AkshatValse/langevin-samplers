"""Unadjusted and Metropolis-adjusted Langevin samplers.

Both samplers are vectorized over independent chains: the state is an array of
shape ``(n_chains, dim)`` and one Python-level iteration advances every chain a
single step.  Returned traces have shape ``(n_kept, n_chains, dim)``.
"""

import numpy as np


def _prepare(x0, h, n_steps, burn_in, thin):
    x = np.array(x0, dtype=float, copy=True)
    if x.ndim != 2:
        raise ValueError("x0 must have shape (n_chains, dim)")
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    if not 0.0 <= burn_in < 1.0:
        raise ValueError("burn_in must lie in [0, 1)")
    if thin < 1:
        raise ValueError("thin must be at least 1")
    return x, int(burn_in * n_steps)


def ula(target, h, n_steps, x0, rng, burn_in=0.2, thin=1):
    """Unadjusted Langevin algorithm (Euler-Maruyama, no correction).

        X_{k+1} = X_k + h grad log pi(X_k) + sqrt(2h) xi_k,   xi_k ~ N(0, I).

    The chain is ergodic but its invariant law ``pi_h`` is *not* ``pi``.  The
    resulting O(h) bias is the quantity this repository measures; for Gaussian
    targets ``pi_h`` is known exactly (:func:`langevin.theory.ula_stationary_var`).

    For step sizes past the stability limit the iteration diverges rather than
    equilibrating; ``info["diverged"]`` reports this instead of returning
    meaningless moments.
    """
    x, n_burn = _prepare(x0, h, n_steps, burn_in, thin)
    sqrt_2h = np.sqrt(2.0 * h)
    kept = []
    diverged = False

    for k in range(n_steps):
        x = x + h * target.grad_log_prob(x) + sqrt_2h * rng.standard_normal(x.shape)
        if not np.all(np.isfinite(x)):
            diverged = True
            break
        if k >= n_burn and (k - n_burn) % thin == 0:
            kept.append(x.copy())

    info = {
        "sampler": "ula",
        "h": float(h),
        "n_steps": int(n_steps),
        "burn_in": float(burn_in),
        "thin": int(thin),
        "acceptance_rate": 1.0,
        "grad_evals_per_step": 1,
        "diverged": bool(diverged),
    }
    if diverged or not kept:
        return np.empty((0,) + x.shape), info
    return np.asarray(kept), info


def mala(target, h, n_steps, x0, rng, burn_in=0.2, thin=1):
    """Metropolis-adjusted Langevin algorithm.

    The ULA move is used as a proposal and accepted with probability

        alpha = min(1, [pi(y) q(x|y)] / [pi(x) q(y|x)]),
        q(y|x) = N(y; x + h grad log pi(x), 2h I),

    which restores exact reversibility with respect to ``pi``.  In stationarity
    MALA is unbiased at every step size, so its only error is Monte Carlo;
    what degrades as ``h`` grows is the acceptance rate, and with it the
    effective sample size per gradient evaluation.  That is the trade the
    ``cost`` columns in ``scripts/01`` are meant to expose.
    """
    x, n_burn = _prepare(x0, h, n_steps, burn_in, thin)
    n_chains = x.shape[0]
    sqrt_2h = np.sqrt(2.0 * h)
    four_h = 4.0 * h

    grad = target.grad_log_prob(x)
    logp = target.log_prob(x)
    kept = []
    n_accept = 0
    n_proposed = 0

    for k in range(n_steps):
        y = x + h * grad + sqrt_2h * rng.standard_normal(x.shape)
        grad_y = target.grad_log_prob(y)
        logp_y = target.log_prob(y)

        # Forward and reverse proposal log-densities, up to the shared
        # normalizing constant of N(., 2h I), which cancels in the ratio.
        log_fwd = -np.sum((y - x - h * grad) ** 2, axis=-1) / four_h
        log_bwd = -np.sum((x - y - h * grad_y) ** 2, axis=-1) / four_h

        log_alpha = (logp_y + log_bwd) - (logp + log_fwd)
        # A proposal that overflows to a non-finite log-density is rejected.
        log_alpha = np.where(np.isfinite(log_alpha), log_alpha, -np.inf)

        with np.errstate(divide="ignore"):
            accept = np.log(rng.random(n_chains)) < log_alpha

        x = np.where(accept[:, None], y, x)
        grad = np.where(accept[:, None], grad_y, grad)
        logp = np.where(accept, logp_y, logp)
        n_accept += int(np.count_nonzero(accept))
        n_proposed += n_chains

        if k >= n_burn and (k - n_burn) % thin == 0:
            kept.append(x.copy())

    info = {
        "sampler": "mala",
        "h": float(h),
        "n_steps": int(n_steps),
        "burn_in": float(burn_in),
        "thin": int(thin),
        "acceptance_rate": n_accept / n_proposed if n_proposed else float("nan"),
        "grad_evals_per_step": 2,
        "diverged": False,
    }
    if not kept:
        return np.empty((0,) + x.shape), info
    return np.asarray(kept), info


SAMPLERS = {"ula": ula, "mala": mala}
