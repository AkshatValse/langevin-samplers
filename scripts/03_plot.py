"""Stage 03: figures and the validation table, from saved results only.

Reads ``results/*.json`` and writes PDFs to ``figures/``.  Re-running this never
requires re-sampling.

Usage:  uv run python scripts/03_plot.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from langevin.provenance import FIGURES_DIR, ensure_dirs, load_json, save_json  # noqa: E402
from langevin.theory import ula_stationary_var, ula_w2_to_target  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

STYLE = {
    "ula": dict(color="black", marker="o", ms=3.5, ls="none", label="ULA"),
    "mala": dict(color="black", marker="s", ms=3.5, ls="none", mfc="white", label="MALA"),
}


def _split(records, key="sampler"):
    return {n: [r for r in records if r[key] == n] for n in ("ula", "mala")}


def figure_variance(data):
    scales = np.array(data["scales"])
    by = _split(data["records"])
    h_fine = np.logspace(np.log10(min(data["h_grid"])), np.log10(max(data["h_grid"])), 400)

    h_star = data["ula_stability_limit"]
    h_lo, h_hi = min(data["h_grid"]), max(data["h_grid"])

    fig, axes = plt.subplots(1, len(scales), figsize=(3.0 * len(scales), 2.8), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        s = scales[i]
        exact = np.array([ula_stationary_var(scales, h)[i] for h in h_fine])
        ax.plot(h_fine, exact, color="black", lw=1.0, label=r"ULA exact $v(h)$")
        ax.axhline(s**2, color="black", lw=0.8, ls="--", label=r"target $s^2$")
        # One vertical line, at the *global* stability limit 2 min(s^2): the
        # narrowest direction is what actually caps the step size.
        ax.axvline(h_star, color="0.6", lw=0.8, ls=":",
                   label=rf"$h^*={h_star:g}$" if i == 0 else None)
        vals = [s**2]
        for name in ("ula", "mala"):
            rs = [r for r in by[name] if not r["diverged"]]
            ys = [r["measured_var"][i] for r in rs]
            vals += ys
            ax.plot([r["h"] for r in rs], ys, **STYLE[name])
        ax.set_xscale("log")
        ax.set_xlim(h_lo / 1.6, h_star * 1.25)
        lo, hi = min(vals), max(vals)
        pad = 0.12 * (hi - lo) if hi > lo else 0.1 * hi
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(rf"$s={s:g}$")
        ax.set_xlabel(r"step size $h$")
        if i == 0:
            ax.set_ylabel("stationary variance")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("ULA over-disperses by exactly $h/2 + O(h^2)$ per coordinate; MALA does not", y=1.04)
    path = os.path.join(FIGURES_DIR, "fig1_variance_vs_h.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_w2(data):
    scales = np.array(data["scales"])
    by = _split(data["records"])
    h_fine = np.logspace(np.log10(min(data["h_grid"])), np.log10(max(data["h_grid"])), 400)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(h_fine, [ula_w2_to_target(scales, h) for h in h_fine],
            color="black", lw=1.0, label=r"ULA exact $W_2(\pi_h,\pi)$")
    for name in ("ula", "mala"):
        rs = [r for r in by[name] if not r["diverged"] and np.isfinite(r["w2_to_target"])]
        ax.plot([r["h"] for r in rs], [r["w2_to_target"] for r in rs], **STYLE[name])

    h_ref = np.array([min(data["h_grid"]), max(data["h_grid"])])
    anchor = ula_w2_to_target(scales, h_ref[0])
    ax.plot(h_ref, anchor * (h_ref / h_ref[0]), color="0.55", lw=0.8, ls="--",
            label=r"slope $1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$W_2(\hat\pi, \pi)$")
    ax.set_title("ULA bias is first order in $h$; MALA sits at the Monte Carlo floor")
    ax.legend(frameon=False, fontsize=7)
    path = os.path.join(FIGURES_DIR, "fig2_w2_scaling.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_cost(data):
    by = _split(data["records"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    for name in ("ula", "mala"):
        rs = [r for r in by[name] if not r["diverged"]]
        # Worst coordinate: the slowest direction governs the run.
        ax1.plot([r["h"] for r in rs],
                 [min(r["ess_per_grad_eval"]) for r in rs], **STYLE[name])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"step size $h$")
    ax1.set_ylabel("ESS per gradient evaluation\n(slowest coordinate)")
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title("Sampling efficiency")

    rs = [r for r in by["mala"] if not r["diverged"]]
    ax2.plot([r["h"] for r in rs], [r["acceptance_rate"] for r in rs], **STYLE["mala"])
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel(r"step size $h$")
    ax2.set_ylabel("MALA acceptance rate")
    ax2.set_title("The price of the correction")
    path = os.path.join(FIGURES_DIR, "fig3_cost.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_multimodal(data):
    by = _split(data["records"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for name in ("ula", "mala"):
        rs = by[name]
        ax1.plot([r["mu"] for r in rs], [r["occupancy_right"] for r in rs], **STYLE[name])
    ax1.axhline(0.5, color="black", lw=0.8, ls="--", label="stationary value (0.5)")
    ax1.set_ylim(-0.03, 0.62)
    ax1.set_xlabel(r"mode separation $\mu$")
    ax1.set_ylabel("time in the right well")
    ax1.set_title("All chains started in the left well")
    ax1.legend(frameon=False, fontsize=7, loc="lower left")

    floor = 1.0 / (data["records"][0]["n_chains"] * 1e5)
    for name in ("ula", "mala"):
        rs = [r for r in by[name] if r["crossings_total"] > 0]
        ax2.plot([r["barrier"] for r in rs],
                 [max(r["crossing_rate_per_step"], floor) for r in rs], **STYLE[name])
        zero = [r for r in by[name] if r["crossings_total"] == 0]
        if zero:
            ax2.plot([r["barrier"] for r in zero], [floor] * len(zero),
                     color="black", marker="v", ms=4, ls="none", mfc="none",
                     label=f"{name.upper()}: no crossings observed")
    fit = (data.get("arrhenius_fits") or {}).get("mala")
    if fit:
        d = np.linspace(min(r["barrier"] for r in data["records"]),
                        max(r["barrier"] for r in data["records"]), 50)
        ax2.plot(d, np.exp(fit["intercept"] + fit["slope"] * d), color="0.55", lw=0.9, ls="--",
                 label=rf"fit: slope $={fit['slope']:.2f}$")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"barrier $\Delta = \mu^2/2s^2 - \log 2$")
    ax2.set_ylabel("crossing rate per step")
    ax2.set_title("Mixing is exponential in the barrier")
    ax2.legend(frameon=False, fontsize=6.5)
    fig.suptitle("The Metropolis correction fixes bias, not mixing", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(FIGURES_DIR, "fig4_multimodal.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def validation_table(data):
    """Check measured ULA variance against the exact AR(1) prediction.

    This validates the *implementation*: if the sampler is correct the
    discrepancy should sit inside the Monte Carlo error of the variance
    estimate, at every step size below the stability limit.
    """
    scales = np.array(data["scales"])
    rows = []
    print(f"\n{'h':>8}  {'coord':>5}  {'measured':>10}  {'exact':>10}  "
          f"{'rel.diff':>9}  {'MC err':>9}  {'within':>6}")
    for r in [x for x in data["records"] if x["sampler"] == "ula" and not x["diverged"]]:
        exact = ula_stationary_var(scales, r["h"])
        for i in range(len(scales)):
            meas, ex = r["measured_var"][i], exact[i]
            rel = (meas - ex) / ex
            mc = r["mc_rel_err_on_var"][i]
            ok = abs(rel) <= 3.0 * mc
            rows.append({"h": r["h"], "coord": i, "measured": meas, "exact": float(ex),
                         "rel_diff": float(rel), "mc_rel_err": float(mc), "within_3sigma": bool(ok)})
            print(f"{r['h']:8.4f}  {i:5d}  {meas:10.5f}  {ex:10.5f}  "
                  f"{rel:+9.2%}  {mc:9.2%}  {'yes' if ok else 'NO':>6}")
    n_ok = sum(x["within_3sigma"] for x in rows)
    print(f"\n{n_ok}/{len(rows)} coordinate-step-size pairs agree with the exact "
          f"ULA law within 3 Monte Carlo standard errors.")
    save_json("ula_validation.json", {"rows": rows, "n_within_3sigma": n_ok, "n_total": len(rows)},
              {"source": "gaussian_bias.json", "criterion": "|rel_diff| <= 3 * mc_rel_err"}, seed=0)
    return n_ok, len(rows)


def main():
    ensure_dirs()
    written = []
    gauss = load_json("gaussian_bias.json")
    written += [figure_variance(gauss), figure_w2(gauss), figure_cost(gauss)]
    validation_table(gauss)
    try:
        multi = load_json("multimodal.json")
    except FileNotFoundError:
        print("\nresults/multimodal.json not found - run scripts/02_multimodal.py")
    else:
        written.append(figure_multimodal(multi))
    print("\nwrote:")
    for p in written:
        print("  " + os.path.relpath(p, os.path.dirname(FIGURES_DIR)))


if __name__ == "__main__":
    main()
