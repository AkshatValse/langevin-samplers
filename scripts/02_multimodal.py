"""Stage 02: mixing time is exponential in the barrier, and the correction does not help.

Stage 01 isolates discretization bias with mixing held easy.  This stage
isolates the opposite failure.  On the separated mixture

    pi = 0.5 N(-mu e_0, s^2 I) + 0.5 N(+mu e_0, s^2 I)

the log-density barrier between the wells is ``Delta(mu) = mu^2/(2 s^2) - log 2``,
and Kramers' law predicts a mean crossing rate scaling like ``exp(-Delta)``.  So
the barrier height, not the step size, is swept here: every chain starts in the
left well and the run reports how often it escapes.

The point of the comparison is that MALA's correction does nothing for this.
MALA is unbiased *in stationarity*, and a chain that has not crossed is nowhere
near stationarity; at large ``mu`` both samplers simply report the mode they
were started in.  Bias is polynomial in ``h`` and fixable; mixing is exponential
in ``Delta`` and is not.

Runs unthinned so crossings are counted exactly rather than sampled.

Writes ``results/multimodal.json`` plus its sidecar.

Usage:  uv run python scripts/02_multimodal.py [--smoke]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from langevin.diagnostics import moment_summary  # noqa: E402
from langevin.provenance import save_json  # noqa: E402
from langevin.samplers import SAMPLERS  # noqa: E402
from langevin.targets import TwoComponentMixture  # noqa: E402
from langevin.theory import mixture_barrier  # noqa: E402

SEED = 20260729
SCALE = 1.0
DIM = 2
H = 0.05  # well inside the stability limit 2*s^2 = 2, and near-unit acceptance
MU_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MIN_CROSSINGS_TO_FIT = 20

FULL = dict(n_chains=32, n_steps=150_000, thin=1, burn_in=0.2)
SMOKE = dict(n_chains=8, n_steps=3_000, thin=1, burn_in=0.2)


def crossing_stats(samples, target, n_kept):
    labels = target.mode_label(samples)  # (n_kept, n_chains)
    crossings = np.abs(np.diff(labels.astype(np.int16), axis=0)).sum(axis=0)
    occ_right = float(labels.mean())
    total = int(crossings.sum())
    return {
        "occupancy_right": occ_right,
        "occupancy_imbalance": abs(occ_right - 0.5),
        "crossings_mean": float(crossings.mean()),
        "crossings_total": total,
        "crossing_rate_per_step": total / (labels.shape[1] * max(n_kept - 1, 1)),
        "chains_that_never_crossed": int(np.count_nonzero(crossings == 0)),
        "n_chains": int(labels.shape[1]),
    }


def arrhenius_fit(records, sampler):
    """Fit ``log(rate) = a - b * Delta``; Kramers predicts ``b ~ 1``."""
    pts = [
        r for r in records
        if r["sampler"] == sampler and r["crossings_total"] >= MIN_CROSSINGS_TO_FIT
    ]
    if len(pts) < 3:
        return None
    delta = np.array([r["barrier"] for r in pts])
    lograte = np.log(np.array([r["crossing_rate_per_step"] for r in pts]))
    slope, intercept = np.polyfit(delta, lograte, 1)
    resid = lograte - (slope * delta + intercept)
    ss_tot = np.sum((lograte - lograte.mean()) ** 2)
    return {
        "n_points": len(pts),
        "mu_used": [r["mu"] for r in pts],
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def run(config, smoke):
    records = []
    for i_s, name in enumerate(("ula", "mala")):
        sampler = SAMPLERS[name]
        for i_mu, mu in enumerate(MU_GRID):
            target = TwoComponentMixture(mu=mu, scale=SCALE, dim=DIM)
            init_rng = np.random.default_rng([SEED, 11, i_mu])
            chain_rng = np.random.default_rng([SEED, 12, i_s, i_mu])
            # Every chain starts in the left well.  A sampler that mixes forgets
            # this; one that does not reports it as the answer.
            x0 = init_rng.standard_normal((config["n_chains"], DIM)) * SCALE
            x0[:, 0] -= mu

            t0 = time.perf_counter()
            samples, info = sampler(
                target, H, config["n_steps"], x0, chain_rng,
                burn_in=config["burn_in"], thin=config["thin"],
            )
            elapsed = time.perf_counter() - t0

            mean, var = moment_summary(samples)
            stats = crossing_stats(samples, target, samples.shape[0]) if samples.size else {}
            rec = {
                "sampler": name,
                "mu": float(mu),
                "h": H,
                "barrier": float(mixture_barrier(mu, SCALE)),
                "measured_mean": mean.tolist(),
                "measured_var": var.tolist(),
                "target_mean": [0.0] * DIM,
                "target_var": target.reference_var.tolist(),
                "acceptance_rate": info["acceptance_rate"],
                "seconds": elapsed,
                **stats,
            }
            records.append(rec)
            print(
                f"[{name:4s}] mu={mu:4.1f}  barrier={rec['barrier']:5.2f}"
                f"  occ_R={stats.get('occupancy_right', float('nan')):.3f}"
                f"  crossings={stats.get('crossings_total', -1):6d}"
                f"  never_crossed={stats.get('chains_that_never_crossed', -1):3d}/"
                f"{config['n_chains']}  {elapsed:5.1f}s",
                flush=True,
            )

    fits = {s: arrhenius_fit(records, s) for s in ("ula", "mala")}
    for s, f in fits.items():
        if f:
            print(f"\n{s.upper()} Arrhenius fit over mu={f['mu_used']}: "
                  f"log(rate) = {f['intercept']:.2f} - {-f['slope']:.3f}*Delta   "
                  f"(R^2={f['r_squared']:.4f}; Kramers predicts slope -1)")
        else:
            print(f"\n{s.upper()}: too few crossing-rich points to fit.")

    payload = {
        "scale": SCALE, "dim": DIM, "h": H, "mu_grid": MU_GRID,
        "arrhenius_fits": fits, "records": records,
    }
    params = dict(config, scale=SCALE, dim=DIM, h=H, mu_grid=MU_GRID, smoke=smoke)
    path = save_json("multimodal.json", payload, params, SEED)
    print(f"\nwrote {os.path.basename(path)}  ({len(records)} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = ap.parse_args()
    config = SMOKE if args.smoke else FULL
    print(f"mixture scale={SCALE} dim={DIM} h={H}  mu_grid={MU_GRID}  config={config}")
    run(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
