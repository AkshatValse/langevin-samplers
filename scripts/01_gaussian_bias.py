"""Stage 01: measure ULA's discretization bias against its exact stationary law.

On ``N(0, diag(s^2))`` the ULA chain is a coordinatewise AR(1) whose stationary
variance is known in closed form, ``v(h) = s^2 / (1 - h/(2 s^2))``.  This stage
runs both samplers over a grid of step sizes and records, per coordinate:

  * the measured stationary variance,
  * the exact ULA prediction (a validation of the sampler, not of the theory),
  * the exact target variance (the bias proper),
  * effective sample size per gradient evaluation, and acceptance rate.

Writes ``results/gaussian_bias.json`` plus its sidecar.  Nothing is plotted
here; stage 03 reads this file from disk.

Usage:  uv run python scripts/01_gaussian_bias.py [--smoke]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from langevin.diagnostics import (  # noqa: E402
    effective_sample_size,
    moment_summary,
    var_relative_mc_error,
)
from langevin.provenance import save_json  # noqa: E402
from langevin.samplers import SAMPLERS  # noqa: E402
from langevin.targets import GaussianTarget  # noqa: E402
from langevin.theory import (  # noqa: E402
    gaussian_w2,
    ula_stability_limit,
    ula_stationary_var,
)

SEED = 20260729
SCALES = [0.5, 1.0, 2.0]  # 4:1 conditioning; stability limit is 2*min(s^2) = 0.5
N_H = 10
H_MIN, H_MAX = 5e-3, 4e-1

FULL = dict(n_chains=128, n_steps=200_000, thin=20, burn_in=0.2)
SMOKE = dict(n_chains=16, n_steps=4_000, thin=4, burn_in=0.2)


def run(config, smoke):
    target = GaussianTarget(SCALES)
    h_grid = np.logspace(np.log10(H_MIN), np.log10(H_MAX), N_H)
    h_star = ula_stability_limit(SCALES)

    records = []
    for i_s, name in enumerate(("ula", "mala")):
        sampler = SAMPLERS[name]
        for i_h, h in enumerate(h_grid):
            # Streams are keyed by (role, sampler, step size) rather than drawn
            # from one running generator, so each configuration reproduces on its
            # own and the two samplers share an identical overdispersed start.
            init_rng = np.random.default_rng([SEED, 1, i_h])
            chain_rng = np.random.default_rng([SEED, 2, i_s, i_h])
            x0 = init_rng.standard_normal((config["n_chains"], target.dim)) * 2.0
            rng = chain_rng
            t0 = time.perf_counter()
            samples, info = sampler(
                target,
                float(h),
                config["n_steps"],
                x0,
                rng,
                burn_in=config["burn_in"],
                thin=config["thin"],
            )
            elapsed = time.perf_counter() - t0

            mean, var = moment_summary(samples)
            if samples.size:
                ess = effective_sample_size(samples)
                w2_target = gaussian_w2(var, target.reference_var)
            else:
                ess = np.full(target.dim, np.nan)
                w2_target = float("nan")

            grad_evals = config["n_steps"] * config["n_chains"] * info["grad_evals_per_step"]
            rec = {
                "sampler": name,
                "h": float(h),
                "measured_var": var.tolist(),
                "measured_mean": mean.tolist(),
                "target_var": target.reference_var.tolist(),
                "ula_exact_var": ula_stationary_var(SCALES, float(h)).tolist(),
                "w2_to_target": w2_target,
                "ess": ess.tolist(),
                "ess_per_grad_eval": (ess / grad_evals).tolist(),
                "mc_rel_err_on_var": var_relative_mc_error(ess).tolist(),
                "acceptance_rate": info["acceptance_rate"],
                "diverged": info["diverged"],
                "seconds": elapsed,
            }
            records.append(rec)
            flag = "  DIVERGED" if info["diverged"] else ""
            print(
                f"[{name:4s}] h={h:7.4f}  var={np.array2string(var, precision=4)}"
                f"  acc={info['acceptance_rate']:.3f}  {elapsed:5.1f}s{flag}",
                flush=True,
            )

    payload = {
        "scales": SCALES,
        "h_grid": h_grid.tolist(),
        "ula_stability_limit": float(h_star),
        "records": records,
    }
    params = dict(config, scales=SCALES, h_grid=h_grid.tolist(), smoke=smoke)
    path = save_json("gaussian_bias.json", payload, params, SEED)
    print(f"\nwrote {os.path.basename(path)}  ({len(records)} runs)")
    print(f"ULA stability limit h* = {h_star:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = ap.parse_args()
    config = SMOKE if args.smoke else FULL
    print(f"scales={SCALES}  config={config}")
    run(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
