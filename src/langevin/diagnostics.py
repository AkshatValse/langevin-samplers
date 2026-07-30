"""Monte Carlo diagnostics: autocorrelation time and effective sample size."""

import numpy as np


def _autocorr(chain):
    """Normalized autocorrelation of a 1-D series, via FFT."""
    n = chain.size
    c = chain - chain.mean()
    # Zero-pad to at least 2n to get the linear (not circular) correlation.
    n_fft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(c, n_fft)
    acov = np.fft.irfft(f * np.conjugate(f), n_fft)[:n]
    if acov[0] <= 0.0:
        return np.zeros(n)
    return acov / acov[0]


def integrated_autocorr_time(chain):
    """Geyer's initial monotone positive sequence estimator of ``tau_int``.

    Sums the autocorrelation in adjacent pairs ``Gamma_m = rho_{2m} + rho_{2m+1}``,
    which is positive and decreasing for a reversible chain, and truncates at
    the first violation.  Reversibility is what licenses the pairing, so this is
    the right estimator for MALA and a serviceable one for ULA.

    Returns ``tau >= 1``; ``ESS = N / tau``.
    """
    chain = np.asarray(chain, dtype=float).ravel()
    if chain.size < 4 or np.allclose(chain, chain[0]):
        return float(chain.size)
    rho = _autocorr(chain)
    n_pairs = (rho.size - 1) // 2
    if n_pairs < 1:
        return 1.0
    gamma = rho[1 : 2 * n_pairs + 1 : 2] + rho[2 : 2 * n_pairs + 2 : 2]
    # Truncate at the first non-positive pair sum.
    nonpos = np.nonzero(gamma <= 0.0)[0]
    cut = nonpos[0] if nonpos.size else gamma.size
    gamma = gamma[:cut]
    if gamma.size == 0:
        return 1.0
    # Enforce the monotone-decreasing property the estimator assumes.
    gamma = np.minimum.accumulate(gamma)
    tau = 1.0 + 2.0 * float(np.sum(gamma))
    return max(tau, 1.0)


def effective_sample_size(samples):
    """ESS of a ``(n_kept, n_chains, dim)`` trace, summed over chains.

    ``tau`` is estimated per chain and coordinate on the series as given and
    the ESS is ``n_kept / tau`` summed over chains, so the result counts
    effective *draws in the supplied trace*.  If the trace was thinned this is
    still the honest count -- thinning discards information, it does not create
    independent draws, so there is nothing to rescale by.  Returns one ESS per
    coordinate.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[0] < 4:
        return np.full(samples.shape[-1] if samples.ndim == 3 else 1, np.nan)
    n_kept, n_chains, dim = samples.shape
    ess = np.zeros(dim)
    for d in range(dim):
        total = 0.0
        for c in range(n_chains):
            tau = integrated_autocorr_time(samples[:, c, d])
            total += n_kept / tau
        ess[d] = total
    return ess


def moment_summary(samples):
    """Pooled mean and variance across the chain and coordinate axes.

    Returns ``(mean, var)``, each of shape ``(dim,)``.  The variance is pooled
    over all draws from all chains, which is the right estimator here because
    every chain targets the same law and is initialized from the same
    overdispersed distribution.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        dim = samples.shape[-1] if samples.ndim == 3 else 1
        return np.full(dim, np.nan), np.full(dim, np.nan)
    flat = samples.reshape(-1, samples.shape[-1])
    return flat.mean(axis=0), flat.var(axis=0, ddof=1)


def var_relative_mc_error(n_eff):
    """Relative standard error of a variance estimate from ``n_eff`` draws.

    For Gaussian data ``Var(s^2) / sigma^4 = 2 / (n - 1)``, so the relative
    error is ``sqrt(2 / n_eff)``.  Used to decide whether a measured bias is
    resolvable at the smallest step sizes rather than assuming it is.
    """
    n_eff = np.asarray(n_eff, dtype=float)
    return np.sqrt(2.0 / np.maximum(n_eff - 1.0, 1.0))
