# Langevin samplers: discretization bias against exact references

A numerical study of what you give up by dropping the Metropolis correction from
a Langevin sampler, set up so that every measured quantity has a closed-form
answer to be checked against.

**The math in one paragraph.** The overdamped Langevin diffusion
`dX_t = ∇log π(X_t) dt + √2 dW_t` leaves `π` invariant, so discretizing it gives
a sampler. The Euler–Maruyama discretization — the *unadjusted* Langevin
algorithm, `X_{k+1} = X_k + h ∇log π(X_k) + √(2h) ξ_k` — is ergodic, but its
invariant law `π_h` is not `π`; the gap is the discretization bias. Metropolizing
the same move (MALA) restores exact reversibility with respect to `π` at the cost
of an accept/reject step. On a Gaussian target `N(0, diag(s²))` the ULA chain is
a coordinatewise AR(1), `X' = (1 − h/s²)X + √(2h) ξ`, which is stationary iff
`0 < h < 2s²` and whose stationary variance solves `v = (1 − h/s²)²v + 2h`:

```
v(h) = s² / (1 − h/(2s²)),      v(h) − s² = (h/2)/(1 − h/(2s²)) = h/2 + O(h²).
```

So ULA over-disperses at every step size, the leading-order variance inflation is
`h/2` *independently of the scale* `s`, and the induced Wasserstein error is first
order, `W₂(π_h, π) = (h/4)·(Σ_i s_i⁻²)^{1/2} + O(h²)`. Because `π_h` is known
exactly, the simulations here can be validated against a formula rather than
against a longer run of themselves — the measured ULA variance is required to
agree with `v(h)` to within Monte Carlo error, and a failure indicts the sampler,
not the theory.

The step-size ceiling is set by the *narrowest* direction (`h < 2 min_i s_i²`)
while the time to decorrelate is set by the widest, which is exactly why
ill-conditioned targets are hard for Langevin methods and is visible directly in
the AR(1) coefficients.

## What the experiments show

1. **Gaussian bias** (`scripts/01`) — ULA and MALA over a decade and a half of
   step sizes on an anisotropic Gaussian (`s = 0.5, 1, 2`; 4:1 conditioning,
   stability limit `h* = 0.5`). Recovers the exact `v(h)` curve, the first-order
   `W₂` rate, and the acceptance-rate collapse that is the price of the
   correction.
2. **Metastability** (`scripts/02`) — a separated two-component mixture, sweeping
   the *barrier* `Δ(μ) = μ²/2s² − log 2` rather than the step size, with every
   chain started in the left well. Kramers' law predicts a crossing rate scaling
   like `exp(−Δ)`, and the measured rates recover that slope. Past `μ ≈ 4` both
   samplers stop crossing entirely and report the well they were started in.
   The Metropolis correction buys nothing here — MALA is unbiased *in
   stationarity*, and a chain that has not crossed is nowhere near stationarity.
   Bias is polynomial in `h` and fixable; mixing is exponential in `Δ` and is
   not, and correcting the first does nothing for the second.

## Results

**The sampler reproduces its own exact stationary law.** Across 10 step sizes and
3 coordinates, the measured ULA variance agrees with `v(h) = s²/(1 − h/2s²)` in
30/30 cases to within 3 Monte Carlo standard errors, at a Monte Carlo error of
0.14–0.29%. At `h = 0.4` — four fifths of the way to the stability limit — the
measured variances are `[1.2479, 1.2488, 4.2085]` against exact
`[1.25, 1.25, 4.2105]`. MALA at the same step size returns
`[0.2494, 1.0003, 4.0075]` against the target `[0.25, 1, 4]`, and its acceptance
rate falls from 0.999 to 0.595 over the same sweep. That is the whole trade in
one line: ULA's error is a bias you cannot average away, MALA's is a cost you pay
in rejected proposals.

**Mixing is exponential in the barrier, and the correction does not touch it.**
Fitting `log(crossing rate)` against `Δ = μ²/2s² − log 2` gives slope
**−0.948** for ULA (R² = 0.998) and **−1.032** for MALA (R² = 0.999), bracketing
the Kramers prediction of −1. By `μ = 5` (`Δ = 11.8`) ULA records zero crossings
with all 32 chains still in the well they started in. MALA, unbiased in
stationarity, is equally stuck — which is the point.

## Layout

```
src/langevin/     library
  targets.py      Gaussian and two-component mixture (score + reference moments)
  samplers.py     ULA and MALA, vectorized over independent chains
  theory.py       exact ULA stationary law, stability limit, Gaussian W2
  diagnostics.py  Geyer initial-positive-sequence ESS, pooled moments
  provenance.py   run sidecars (parameters, seed, resolved commit SHA)
scripts/          01 gaussian bias -> 02 multimodal -> 03 figures
results/          JSON outputs + .meta.json sidecars
figures/          PDFs
```

## Reproduce

```bash
uv sync
uv run python scripts/01_gaussian_bias.py --smoke   # ~10s pipeline check
uv run python scripts/01_gaussian_bias.py
uv run python scripts/02_multimodal.py
uv run python scripts/03_plot.py
```

Every run is seeded (`np.random.default_rng`, streams keyed by role and
configuration so each point reproduces independently of iteration order) and
writes a sidecar with its full parameter set, seed, and resolved git commit.
Stage 03 reads only saved results, so figures regenerate without re-sampling.
Pure NumPy; no autodiff framework.

## References

- Roberts, G. O. and Tweedie, R. L. (1996). Exponential convergence of Langevin
  distributions and their discrete approximations. *Bernoulli* 2(4), 341–363.
- Roberts, G. O. and Rosenthal, J. S. (1998). Optimal scaling of discrete
  approximations to Langevin diffusions. *JRSS-B* 60(1), 255–268.
- Dalalyan, A. S. (2017). Theoretical guarantees for approximate sampling from
  smooth and log-concave densities. *JRSS-B* 79(3), 651–676.
- Durmus, A. and Moulines, É. (2017). Nonasymptotic convergence analysis for the
  unadjusted Langevin algorithm. *Annals of Applied Probability* 27(3),
  1551–1587.
