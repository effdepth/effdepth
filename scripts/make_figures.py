"""Build the manuscript's figures from the run outputs.

    python scripts/make_figures.py [--runs results/runs] [--out paper/figures]

The run directory defaults to $EFFDEPTH_RUNS, else results/runs. A figure whose
inputs are missing is skipped with a notice.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from effdepth import kernels as K
from effdepth.plotting import CAT, FULL, INK, INK2, MARKERS, ordinal, plt, setup

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.environ.get("EFFDEPTH_RUNS", "results/runs"))
ap.add_argument("--out", default="paper/figures")
ap.add_argument("--only", nargs="*")
args = ap.parse_args()
RUNS, OUT = Path(args.runs), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
setup()


def load(name):
    p = RUNS / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"{OUT / name}.pdf")


FIGS = {}


def figure(fn):
    FIGS[fn.__name__] = fn
    return fn


@figure
def collapse():
    m = load("e1-collapse")
    if m is None:
        return print("collapse: no e1 run")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    th = np.linspace(0, np.pi / 2, 200)
    a.plot(th, K.kernel_T(np.maximum(th, 1e-12), 4.0), color=INK, lw=1.2, label=r"$K_4$", zorder=1)
    for i, alpha in enumerate([1.0, 0.5, 0.25]):
        eta = alpha ** 2 / (1 + alpha ** 2)
        L = int(round(4.0 / eta))
        thm = th[i * 4::12]
        a.plot(thm, K.rntk_discrete(thm, np.full(L, alpha)), ls="none", marker=MARKERS[i],
               color=CAT[i], label=rf"$\alpha={alpha:g}$, $L={L}$", zorder=2)
    a.set_xlabel(r"augmented angle $\vartheta$")
    a.set_ylabel("normalized kernel")
    a.legend(loc="upper right")
    kinds = ["constant", "power", "inv_sqrt_l", "stable", "random"]
    labels = {"constant": r"constant $\alpha$", "power": r"$\alpha=L^{-\gamma}$",
              "inv_sqrt_l": r"$\alpha_\ell=\ell^{-1/2}$", "stable": "Stable ResNet", "random": "random"}
    recs = m["records"]
    for i, k in enumerate(kinds):
        r = [x for x in recs if x["schedule"] == k]
        b.loglog([x["eta_max"] for x in r], [x["gap"] for x in r], ls="none", marker=MARKERS[i],
                 color=CAT[i], label=labels[k], alpha=0.9)
    e = np.geomspace(1e-8, 1, 50)
    b.loglog(e, m["max_gap_over_eta"] * e, ls="--", color=INK2, lw=1)
    b.set_xlabel(r"$\eta_{\max}$")
    b.set_ylabel(r"$\sup_\vartheta|\bar r-K_T|$")
    b.legend(loc="lower right", ncol=1)
    save(fig, "collapse")


@figure
def large():
    m = load("e2-large-depth")
    if m is None:
        return print("large: no e2 run")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    pr = m["profile"]
    v = np.array(pr["v"])
    cols = ordinal(4)
    for i, (T, d) in enumerate(pr["by_T"].items()):
        a.semilogx(d["v"], d["K"], ls="none", marker=MARKERS[i], color=cols[i],
                   label=rf"$T=10^{{{int(round(np.log10(float(T))))}}}$")
    a.semilogx(v, pr["F"], color=INK, lw=1.2, label=r"$F(3\pi/v)$")
    a.set_xlabel(r"$v=T\vartheta$")
    a.set_ylabel(r"$K_T(v/T)$")
    a.legend(loc="upper right")
    th = np.geomspace(0.02, np.pi / 2, 120)
    for i, T in enumerate([10.0, 100.0, 1000.0, 10000.0]):
        b.loglog(th, T * (K.kernel_T(th, T) - 0.25), color=cols[i], lw=1.1,
                 label=rf"$T=10^{{{int(round(np.log10(T)))}}}$")
    b.loglog(th, K.first_order_term(th), color=INK, lw=1.4, label=r"$h$")
    b.loglog(th, 3 * np.pi / (4 * th), color=INK2, lw=1, ls=":", label=r"$3\pi/(4\vartheta)$")
    b.set_xlabel(r"augmented angle $\vartheta$")
    b.set_ylabel(r"$T\,(K_T-\frac{1}{4})$")
    b.legend(loc="lower left")
    save(fig, "large")


@figure
def rates():
    m = load("e2-large-depth")
    if m is None:
        return print("rates: no e2 run")
    dg = m["degeneration"]
    j = 2  # augmented angle pi/2
    h = dg["h_fixed"][j]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    gam = ["0.0", "0.125", "0.25", "0.375"]
    for i, g in enumerate(gam):
        rows = sorted(dg["by_gamma"][g], key=lambda r: r["L"])
        L = np.array([r["L"] for r in rows], float)
        T = np.array([r["T"] for r in rows])
        y = np.array([r["minus_quarter_fixed"][j] for r in rows])
        lab = rf"$\gamma={float(g):g}$"
        a.loglog(L, y, ls="none", marker=MARKERS[i], color=CAT[i], label=lab)
        b.semilogx(L, T * y, ls="-", marker=MARKERS[i], color=CAT[i], label=lab, lw=1)
        if g != "0.0":
            kt = np.array([K.kernel_T(np.array([np.pi / 2]), t)[0] for t in T])
            a.loglog(L, np.where(kt > 0.25, kt - 0.25, np.nan), ls="--", color=CAT[i], lw=0.9)
        else:
            c = np.mean((y - 3 / 8 * np.log(L) / L) * L)
            a.loglog(L, 3 / 8 * np.log(L) / L + c / L, ls=":", color=CAT[i], lw=1)
    b.axhline(h, color=INK, lw=1, ls="--")
    a.set_xlabel(r"depth $L$")
    a.set_ylabel(r"$\bar r-\frac{1}{4}$")
    a.legend(loc="lower left")
    b.set_xlabel(r"depth $L$")
    b.set_ylabel(r"$T_L\,(\bar r-\frac{1}{4})$")
    save(fig, "rates")


@figure
def spectrum():
    m = load("e3-spectrum-d3")
    if m is None:
        return print("spectrum: no e3 run")
    k = np.array(m["k"], float)
    muh = np.array(m["mu_h"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    Ts = ["4", "16", "64", "256", "1024"]
    cols = ordinal(5)
    kk = k[1:]
    for i, T in enumerate(Ts):
        mu = np.array(m["mu"][T])[1:]
        a.loglog(kk, float(T) * mu, color=cols[i], lw=1.1, label=rf"$T={T}$")
        sel = kk >= 2
        b.loglog(kk[sel] / float(T), mu[sel] * kk[sel] ** 3 / (1 + float(T)), color=cols[i], lw=1.1,
                 label=rf"$T={T}$")
    a.loglog(kk, muh[1:], color=INK, lw=1.4, label=r"$\mu^h_k$")
    a.set_ylim(1e-4, 3)
    a.set_xlabel("degree $k$")
    a.set_ylabel(r"$T\mu_k(T)$")
    a.legend(loc="lower left", ncol=2)
    mu0 = np.array(m["mu"]["0"])[1:]
    cd = np.median((mu0 * kk ** 3)[kk >= 500])
    b.axhline(cd, color=INK2, ls="--", lw=1)
    b.set_xlabel(r"$k/T$")
    b.set_ylabel(r"$\mu_k(T)\,k^3/(1+T)$")
    b.legend(loc="lower right")
    save(fig, "spectrum")


def e4a_table(d, target, sigma):
    """mean over seeds: {n: {T: {interp, oracle, val_selected}}}"""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(str(RUNS / f"e4a-d{d}-n*-s*" / "metrics.json")):
        n = int(f.split("-n")[1].split("-")[0])
        r = json.loads(Path(f).read_text())["results"][target][sigma]
        for T, v in r.items():
            for key in ("interp", "oracle", "val_selected"):
                acc[n][float(T)][key].append(v[key])
    return {n: {T: {k: (np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0, len(x))
                    for k, x in v.items()} for T, v in sorted(t.items())} for n, t in sorted(acc.items())}


def phase_figure(d, name, ns_wanted):
    tab = e4a_table(d, "harm1", "0.5")
    el = load("e4b-eigenlearning")
    if not tab:
        return print(f"{name}: no e4a runs")
    ns = [n for n in ns_wanted if n in tab]
    cols = ordinal(len(ns))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.7))
    for i, n in enumerate(ns):
        Ts = np.array([T for T in tab[n] if T > 0])
        for ax, key, pkey in ((a, "interp", "interp"), (b, "val_selected", "optimal")):
            y = np.array([tab[n][T][key][0] for T in Ts])
            ax.loglog(Ts, y, ls="none", marker=MARKERS[i], color=cols[i], label=f"$n={n}$")
            if el is not None:
                pred = el["predictions"][str(d)]
                Tp = sorted(float(T) for T in pred if float(T) > 0)
                yp = [next(r[pkey] for r in pred[str(T) if str(T) in pred else f"{T}"]["by_target"]["harm1:0.5"]
                           if r["n"] == n) for T in Tp]
                ax.loglog(Tp, yp, color=cols[i], lw=1)
    for ax, ttl in ((a, "interpolation"), (b, "validation-selected stopping")):
        ax.set_xlabel("effective depth $T$")
        ax.set_ylabel("excess risk")
        ax.set_title(ttl, color=INK)
    b.legend(loc="upper left", ncol=1)
    ins = a.inset_axes([0.1, 0.61, 0.33, 0.35])  # above the tempered plateau
    for i, n in enumerate(ns):
        Ts = np.array([T for T in tab[n] if T > 0])
        ins.loglog(Ts / n, [tab[n][T]["interp"][0] for T in Ts], ls="none", marker=MARKERS[i],
                   color=cols[i], ms=2.5)
    ins.set_xlabel("$T/n$", fontsize=6, labelpad=0)
    ins.tick_params(labelsize=5)
    ins.set_xlim(1e-2, 1e2)
    save(fig, name)


@figure
def phase():
    phase_figure(3, "phase", (250, 1000, 4000, 16000, 32000))


@figure
def phased5():
    phase_figure(5, "phased5", (250, 1000, 4000, 16000))


@figure
def revisit():
    m = load("e5-source-protocol")
    if m is None:
        return print("revisit: no e5 run")
    R = m["results"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6))
    ep = np.array(m["epochs_curve"])
    for i, g in enumerate(m["gammas"]):
        r = R[f"{g}:5000"]
        c = np.array(r["curve_L5000"]).mean(0)
        tlab = f"{r['T']:.0f}" if r["T"] >= 100 else f"{r['T']:.3g}"
        a.semilogy(ep, c, color=CAT[i], marker=MARKERS[i], markevery=5, lw=1.1,
                   label=rf"$\gamma={g:g}$ ($T={tlab}$)")
        a.plot([2600], [np.mean(r["val_selected"])], marker=MARKERS[i], color=CAT[i], ms=6, clip_on=False)
    a.set_xlabel("epochs")
    a.set_ylabel("excess risk")
    a.set_title(r"$L=5000$; right markers: trained to convergence", color=INK)
    a.legend(loc="lower left", fontsize=6)
    for i, g in enumerate(m["gammas"]):
        pts = [(R[f"{g}:{L}"]["T"], np.mean(R[f"{g}:{L}"]["fixed_4500"]), np.mean(R[f"{g}:{L}"]["val_selected"]))
               for L in m["depths"]]
        T, f45, vs = map(np.array, zip(*pts))
        b.loglog(T, f45, ls="none", marker=MARKERS[i], color=CAT[i], mfc="none", label=rf"$\gamma={g:g}$")
        b.loglog(T, vs, ls="none", marker=MARKERS[i], color=CAT[i])
    b.axvline(160, color=INK2, ls=":", lw=1)
    b.text(160 * 1.15, 2e-3, "$T=n$", color=INK2, fontsize=7)
    b.set_xlabel("effective depth $T$")
    b.set_ylabel("excess risk")
    b.set_title("open: 4500 epochs; filled: validation-selected", color=INK)
    b.legend(loc="upper left", fontsize=6)
    save(fig, "revisit")




def published(name):
    vals = {}
    p = Path(__file__).resolve().parent.parent / "results" / "published" / name
    if p.exists():
        for line in p.read_text().splitlines():
            if line and not line.startswith("#"):
                fig, key, v = line.split(",")
                vals[key] = float(v)
    return vals


def merged(prefix):
    """A run split into units <prefix>-<kernel> (one kernel per job), merged into one."""
    res = {}
    for p in sorted(RUNS.glob(f"{prefix}-*")):
        m = load(p.name)
        if m:
            res.update(m["results"])
    return {"results": res} if res else None


@figure
def real():
    labels = {"e4c-mnist-n5000": ("MNIST", 5000), "e4c-mnist-n20000": ("MNIST", 20000),
              "e4c-cifar10-n10000": ("CIFAR-10", 10000), "e4c-cifar10-n45000": ("CIFAR-10", 45000)}
    runs = {k: load(k) or merged(k) for k in labels}
    if not any(runs.values()):
        return print("real: no e4c runs")
    edge = 3e-7  # T = 0 is drawn at the left edge of the T/n axis
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    for i, (name, m) in enumerate(runs.items()):  # colour fixed per dataset and size
        if not m:
            continue
        data, n = labels[name]
        ax = axes[0] if data == "MNIST" else axes[1]
        rows = sorted(((v.get("T", 0.0), v) for k, v in m["results"].items() if k.startswith("K_T")),
                      key=lambda x: x[0])
        x = np.array([max(t / n, edge) for t, _ in rows])
        ax.semilogx(x, [100 * v["val_acc"] for _, v in rows], color=CAT[i], marker=MARKERS[i], ms=3.5,
                    lw=1.1, label=f"$n={n}$")
        disc = [(v["T"] / n, 100 * v["val_acc"]) for k, v in m["results"].items() if k.startswith("K_L")]
        if disc:
            ax.semilogx(*zip(*sorted(disc)), ls="none", marker="*", ms=8, color=INK,
                        label=r"$\bar r$, $L=50$, $\alpha=50^{-\gamma}$")
        T = np.array([max(t, 0.03) for t, _ in rows])
        axes[2].semilogx(T, [v["min_eig"] for _, v in rows], color=CAT[i], marker=MARKERS[i], ms=3.5,
                         lw=1.1)  # colours and markers as in the first two panels
    for ax, title in zip(axes[:2], ("MNIST", "CIFAR-10")):
        ax.set_title(title, color=INK)
        ax.set_xlabel("$T/n$ ($T=0$ at the left edge)")
        ax.set_xlim(edge / 2, 8)
        ax.legend(loc="lower left", fontsize=6)
    axes[0].set_ylabel("test accuracy (%)")
    axes[2].axhline(0.75, color=INK2, ls="--", lw=1)
    axes[2].set_xlabel("effective depth $T$")
    axes[2].set_ylabel("smallest Gram eigenvalue")
    fig.subplots_adjust(wspace=0.38)
    save(fig, "real")


def e6a_runs():
    rs = {}
    for f in glob.glob(str(RUNS / "e6a-*" / "metrics.json")):
        L, T, m, s = Path(f).parent.name.split("-")[1:]
        rs[(int(L[1:]), float(T[1:]), int(m[1:]), int(s[1:]))] = json.loads(Path(f).read_text())
    return rs


@figure
def fcnets():
    rs = e6a_runs()
    if not rs:
        return print("fcnets: no e6a runs")
    depths = (8, 32, 128)
    cols = ordinal(3)
    fig = plt.figure(figsize=(FULL, 2.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.3, 1.15], wspace=0.1)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4)]
    for ax in axes[1:3]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)
    for ax, T in zip(axes[:3], (0.5, 2.0, 8.0)):
        for i, L in reversed(list(enumerate(depths))):  # small markers on top: equal-T curves coincide
            sel = [v for (l, t, m, s), v in rs.items() if l == L and t == T and m == 2048]
            if not sel:
                continue
            t = np.array(sel[0]["times"])
            ax.loglog(t, np.mean([v["kernel_risk"] for v in sel], 0), color=cols[i], lw=1.1)
            ax.loglog(t, np.mean([v["net_risk"] for v in sel], 0), ls="none", marker=MARKERS[i],
                      color=cols[i], ms=3.5, label=f"$L={L}$")
        ax.set_title(f"$T={T:g}$", color=INK)
        ax.set_xlabel("time $t$")
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].set_ylabel("excess risk")
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h[::-1], lab[::-1], loc="upper right", fontsize=6)
    ax = axes[3]
    for j, width in enumerate((512, 2048)):
        for i, L in enumerate(depths):
            Ts = sorted({t for (l, t, m, s) in rs if l == L and m == width})
            ratio = [np.mean([min(v["net_risk"]) / min(v["kernel_risk"])
                              for (l, t, m, s), v in rs.items() if l == L and t == T and m == width]) for T in Ts]
            ax.semilogx(Ts, ratio, ls="none", marker=MARKERS[i], color=CAT[j], ms=4,
                        label=f"width {width}" if i == 0 else None)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xlabel("effective depth $T$")
    ax.set_ylabel("best risk, network / kernel")
    ax.legend(loc="upper left", fontsize=6)
    save(fig, "fcnets")


@figure
def cifar():
    def step_runs(prefix):  # the main runs; the others use the decay Yu et al. state
        out = {}
        for p in sorted(RUNS.glob(f"{prefix}-*")):
            m, cfg = load(p.name), json.loads((p / "config.json").read_text())
            if m and cfg.get("schedule", "exp") == "step":
                out[p.name] = dict(m, bn=cfg.get("bn", False))
        return out

    rb = list(step_runs("e6b").values())
    rc = step_runs("e6c")
    if not rb and not rc:
        return print("cifar: no e6 runs")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    if rb:
        by = defaultdict(list)
        bn = []
        for m in rb:
            if m.get("diverged"):
                continue
            (bn if m["bn"] else by[round(m["T"], 4)]).append(100 * m["final_test_acc"])
        T = sorted(by)
        mu = [np.mean(by[t]) for t in T]
        sd = [np.std(by[t], ddof=1) if len(by[t]) > 1 else 0 for t in T]
        a.errorbar(T, mu, yerr=sd, color=CAT[0], marker="o", ms=4, lw=1.1, capsize=2,
                   label="ours, no normalization")
        a.set_xscale("log")
        if bn:
            a.axhline(np.mean(bn), color=CAT[0], ls="--", lw=1, label="ours, batch normalization")
        pub = published("yu2026_figures.csv")
        blocks = 16  # ResNet-34: base alpha = 1, scale alpha = 1/16, at their effective depths
        T_of = lambda al: blocks * al ** 2 / (1 + al ** 2)
        a.plot([T_of(1 / blocks), T_of(1.0)], [pub["cifar100_resnet34_scale_test_acc"],
               pub["cifar100_resnet34_base_test_acc"]], ls="none", marker="s", ms=5, mfc="none",
               color=INK2, label=r"Yu et al.: $\alpha=1/16$ and $\alpha=1$")
        a.axhline(pub["cifar100_resnet34_bn_test_acc"], color=INK2, ls=":", lw=1, label="Yu et al.: batch normalization")
        a.set_xlabel("effective depth $T$")
        a.set_ylabel("test accuracy (%)")
        a.set_title("CIFAR-100, ResNet-34", color=INK)
        a.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=6)
    if rc:
        cols = ordinal(3)
        for i, arch in enumerate(("cifar3", "cifar9", "cifar18")):
            by = defaultdict(list)
            for name, m in rc.items():
                if f"-{arch}-" in name and not m.get("diverged"):
                    by[m["T"]].append(100 * m["final_test_acc"])
            if not by:
                continue
            T = sorted(by)
            b.semilogx(T, [np.mean(by[t]) for t in T], color=cols[i], marker=MARKERS[i], lw=1.1,
                       label=f"{3 * int(arch[5:])} blocks")
        b.set_xlabel("effective depth $T$")
        b.set_ylabel("test accuracy (%)")
        b.set_title("CIFAR-10, depth at fixed $T$", color=INK)
        b.legend(loc="lower center", fontsize=6)
    save(fig, "cifar")


for name, fn in FIGS.items():
    if not args.only or name in args.only:
        fn()
