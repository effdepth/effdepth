"""Build the manuscript's tables and the number macros from the run outputs.

    python scripts/make_tables.py [--runs results/runs] [--out paper/tables]

Writes one .tex per table and numbers.tex, which defines a macro for every
number quoted in the text, so that no number in the manuscript is typed by hand.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from effdepth import kernels as K

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.environ.get("EFFDEPTH_RUNS", "results/runs"))
ap.add_argument("--out", default="paper/tables")
args = ap.parse_args()
RUNS, OUT = Path(args.runs), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
NUM = {}


def load(name):
    p = RUNS / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def num(name, value, fmt):
    assert name.isalpha(), f"macro name {name!r}: letters only"
    NUM[name] = format(value, fmt)


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print(OUT / f"{name}.tex")


# --- E1: effective-depth collapse ---------------------------------------------
m = load("e1-collapse")
if m:
    num("Cgap", m["max_gap_over_eta"], ".3f")
    labels = {"constant": r"constant $\alpha\in[2^{-5},2]$, $T\in[0.25,256]$",
              "power": r"$\alpha=L^{-\gamma}$, $\gamma\in\{\frac14,\frac12,\frac34,1\}$, $L\le4096$",
              "inv_sqrt_l": r"$\alpha_\ell=\ell^{-1/2}$, $L\le8192$",
              "stable": r"$\alpha_\ell=(\sqrt\ell\log(\ell+1))^{-1}$, $L\le8192$",
              "random": r"random $\eta_\ell$, $\eta_{\max}\le0.4$, $T\in\{1,16\}$"}
    rows = []
    for k, lab in labels.items():
        r = [x for x in m["records"] if x["schedule"] == k]
        rows.append(f"{lab} & {len(r)} & {max(x['eta_max'] for x in r):.3g} & "
                    f"{max(x['gap'] / x['eta_max'] for x in r):.3f} \\\\")
    write("collapse", "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
          "schedule & runs & largest $\\eta_{\\max}$ & $\\max\\,\\sup_\\vartheta|\\bar r-K_T|/\\eta_{\\max}$ \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

num("Cinf", K.C_INF, ".4f")

# --- E2: large effective depth ------------------------------------------------
m = load("e2-large-depth")
if m:
    fo = m["first_order"]
    h = np.array(fo["h"])
    T = np.array(fo["T"])
    big = [(t, np.array(v) - h) for t, v in zip(T, fo["T_times_K_minus_quarter"]) if t >= 1e4]
    num("FirstOrderCorr", np.mean([t * d.mean() for t, d in big]), ".1f")
    pr = m["profile"]
    v, F = np.array(pr["v"]), np.array(pr["F"])
    d = pr["by_T"]["10000.0"]
    err = np.abs(np.array(d["K"]) - np.interp(d["v"], v, F)).max()
    mant, ex = f"{err:.1e}".split("e")
    NUM["ProfileErr"] = f"{mant}\\times10^{{{int(ex)}}}"
    rows = sorted(m["degeneration"]["by_gamma"]["0.0"], key=lambda r: r["L"])
    big = [(r["L"], r["T"] * r["minus_quarter_fixed"][2]) for r in rows if r["L"] in (10 ** 5, 10 ** 6, 10 ** 7)]
    incs = [b[1] - a[1] for a, b in zip(big, big[1:])]
    num("DriftIncrement", np.mean(incs), ".2f")
    num("DriftPredicted", 3 * 0.5 / 8 * np.log(10), ".2f")

# --- E3: spectrum ---------------------------------------------------------------
for d, name in ((3, "Cthree"), (5, "Cfive")):
    m = load(f"e3-spectrum-d{d}")
    if m:
        k = np.array(m["k"], float)
        mu = np.array(m["mu"]["0"])
        sel = (k >= 300) & (k <= (1000 if d == 3 else 600))
        num(name, np.median(mu[sel] * k[sel] ** d), ".3f" if d == 3 else ".2f")


m = load("e3-spectrum-d3")
if m:
    defect = max(1 - v for T, v in m["trace"].items() if float(T) <= 1024)
    mant, ex = f"{defect:.1e}".split("e")
    NUM["TraceDefect"] = f"{mant}\\times10^{{{int(ex)}}}"

# --- E4a: phase diagram on the sphere -------------------------------------------
def e4a(d, target, sigma):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(str(RUNS / f"e4a-d{d}-n*-s*" / "metrics.json")):
        n = int(f.split("-n")[1].split("-")[0])
        r = json.loads(Path(f).read_text())["results"][target][sigma]
        for T, vals in r.items():
            for key in ("interp", "oracle", "val_selected"):
                acc[n][float(T)][key].append(vals[key])
    return {n: {T: {k: float(np.mean(x)) for k, x in v.items()} for T, v in t.items()}
            for n, t in sorted(acc.items())}


def pm(x):
    return f"{x:.4f}" if x < 0.1 else f"{x:.3f}"


tab = e4a(3, "harm1", "0.5")
if tab:
    num("Nmax", max(tab), "d")
    rows = []
    for n, t in tab.items():
        Tpos = [T for T in t if T > 0]
        Tb = min(Tpos, key=lambda T: t[T]["interp"])
        rows.append(f"{n} & {pm(t[0.0]['interp'])} & {pm(t[Tb]['interp'])} & {Tb:g} & "
                    f"{pm(t[0.0]['val_selected'])} & {pm(t[Tb]['val_selected'])} \\\\")
    write("phase", "\\begin{tabular}{@{}rrrrrr@{}}\n\\toprule\n"
          "& tempered & \\multicolumn{2}{c}{best interpolation} & \\multicolumn{2}{c}{stopped} \\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
          "$n$ & $T=0$ & risk & $T^\\ast$ & $T=0$ & $T^\\ast$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

if tab:
    ns = sorted(tab)
    temp = [tab[n][0.0]["interp"] for n in ns]
    num("TempLo", min(temp), ".3f")
    num("TempHi", max(temp), ".3f")
    best = {n: min(((T, v["interp"]) for T, v in tab[n].items() if T > 0), key=lambda x: x[1]) for n in ns}
    num("TstarFirst", best[ns[0]][0], "g")
    num("TstarLast", best[ns[-1]][0], "g")
    num("BestFirst", best[ns[0]][1], ".3f")
    num("BestLast", best[ns[-1]][1], ".3f")
    big = [n for n in ns if n >= 1000]
    num("TstarSlope", np.polyfit(np.log(big), np.log([best[n][0] for n in big]), 1)[0], ".2f")
    num("BestSlope", np.polyfit(np.log(big), np.log([best[n][1] for n in big]), 1)[0], ".2f")
    num("StopLast", tab[ns[-1]][0.0]["val_selected"], ".4f")
    for mult, name in ((1, "One"), (4, "Four")):
        vals = [tab[n][float(2 ** round(np.log2(n)) * mult)]["interp"] for n in ns
                if float(2 ** round(np.log2(n)) * mult) in tab[n]]
        num(f"Collapse{name}Lo", min(vals), ".2f")
        num(f"Collapse{name}Hi", max(vals), ".2f")
    lin = e4a(3, "linear", "0.1")
    t = lin[ns[-1]]
    num("LinTempered", t[0.0]["interp"], ".4f")
    num("LinBest", min(v["interp"] for T, v in t.items() if T > 0), ".4f")

# --- E5: the synthetic protocol of Yu et al. ----------------------------------------
m = load("e5-source-protocol")
if m:
    R = m["results"]
    end = lambda g: float(np.mean(np.array(R[f"{g}:5000"]["curve_L5000"])[:, -1]))
    num("SrcGzero", end(0.0), ".2f")
    num("SrcGquarter", end(0.25), ".2f")
    num("SrcGhalf", end(0.5), ".4f")
    num("SrcGzeroLong", np.mean(R["0.0:5000"]["val_selected"]), ".2f")
    num("SrcGquarterLong", np.mean(R["0.25:5000"]["val_selected"]), ".3f")
    num("SrcRatio", end(0.25) / np.mean(R["0.25:5000"]["val_selected"]), ".1f")

# --- E4c: real data ------------------------------------------------------------------
def real(prefix):
    """Results of a real-data run by kernel name, merged over its units if it was split."""
    m = load(prefix)
    if m:
        return m["results"]
    res = {}
    for p in sorted(RUNS.glob(f"{prefix}-*")):
        u = load(p.name)
        if u:
            res.update(u["results"])
    return res or None


def by_T(res):
    return dict(sorted((v.get("T", 0.0), v) for k, v in res.items() if k.startswith("K_T")))


def published(name):
    vals = {}
    p = Path(__file__).resolve().parent.parent / "results" / "published" / name
    for line in p.read_text().splitlines():
        if line and not line.startswith("#"):
            fig, key, v = line.split(",")
            vals[key] = float(v)
    return vals


pct = lambda x: 100 * x
for prefix, tag in (("e4c-mnist-n5000", "MnistS"), ("e4c-mnist-n20000", "MnistL"),
                    ("e4c-cifar10-n10000", "CifarS"), ("e4c-cifar10-n45000", "CifarL")):
    res = real(prefix)
    if not res:
        continue
    t = by_T(res)
    acc = {T: pct(v["val_acc"]) for T, v in t.items()}
    Tb = max(acc, key=acc.get)
    num(f"{tag}Zero", acc[0.0], ".1f")
    num(f"{tag}Best", acc[Tb], ".1f")
    num(f"{tag}BestT", Tb, "g")
    num(f"{tag}Last", acc[max(acc)], ".1f")
    num(f"{tag}LastT", max(acc), "g")
    for T, word in ((16, "Sixteen"), (64, "SixtyFour"), (256, "TwoFiftySix"), (1024, "ThousandTwentyFour")):
        if float(T) in acc:  # macro names cannot hold digits
            num(f"{tag}At{word}", acc[float(T)], ".1f")
    num(f"{tag}EigZero", t[0.0]["min_eig"], ".3f")
    num(f"{tag}EigLast", t[max(t)]["min_eig"], ".3f")
    # largest gap between interpolation and validation-selected stopping, in points
    num(f"{tag}StopGap", max(abs(pct(v["val_acc"] - v["interp_acc"])) for v in t.values()), ".1f")
    # squared error (labels have variance 0.9): interpolation, and the best stopping time
    num(f"{tag}MseInterpZero", t[0.0]["interp_mse"], ".3f")
    num(f"{tag}MseStopZero", min(t[0.0]["mse_curve"]), ".3f")
    Tm = min(t, key=lambda T: t[T]["interp_mse"])
    num(f"{tag}MseBest", t[Tm]["interp_mse"], ".3f")
    num(f"{tag}MseBestT", Tm, "g")
    num(f"{tag}MseLast", t[max(t)]["interp_mse"], ".3f")

# MNIST at equal T/n: n = 5000 at T against n = 20000 at 4T, from T = 16 on
s5, s20 = real("e4c-mnist-n5000"), real("e4c-mnist-n20000")
if s5 and s20:
    a5 = {T: pct(v["val_acc"]) for T, v in by_T(s5).items()}
    a20 = {T: pct(v["val_acc"]) for T, v in by_T(s20).items()}
    pairs = [T for T in a5 if T >= 16 and 4 * T in a20]
    num("MnistRatioGap", max(abs(a5[T] - a20[4 * T]) for T in pairs), ".1f")
    num("MnistRatioPairs", len(pairs), "d")

# CIFAR-10 at equal T/n: n = 10000 at T against n = 45000 at 4.5 T, the latter
# interpolated linearly in log T between grid points, from T = 64 on
c10, c45 = real("e4c-cifar10-n10000"), real("e4c-cifar10-n45000")
if c10 and c45:
    a10 = {T: pct(v["val_acc"]) for T, v in by_T(c10).items()}
    a45 = {T: pct(v["val_acc"]) for T, v in by_T(c45).items() if T > 0}
    grid = sorted(a45)
    gaps = []
    for T in (t for t in a10 if t >= 64):
        u = 4.5 * T
        lo = [g for g in grid if g <= u]
        hi = [g for g in grid if g >= u]
        if lo and hi:
            g0, g1 = lo[-1], hi[0]
            w = 0.0 if g1 == g0 else np.log(u / g0) / np.log(g1 / g0)
            gaps.append(abs(a10[T] - ((1 - w) * a45[g0] + w * a45[g1])))
    if gaps:
        num("CifarRatioGap", max(gaps), ".1f")

gaps = [float(NUM[k]) for k in NUM if k.endswith("StopGap")]
if gaps:
    num("RealStopGap", max(gaps), ".1f")

res = real("e4c-mnist-n20000")
if res and any(k.startswith("K_L50") for k in res):
    pub = published("yu2026_figures.csv")
    t = by_T(res)
    rows, gaps, pubgaps = [], [], []
    for k, v in sorted(((k, v) for k, v in res.items() if k.startswith("K_L50")), key=lambda kv: -kv[1]["alpha"]):
        a, T = v["alpha"], v["T"]
        g = abs(round(-np.log(a) / np.log(v["L"]), 4))  # alpha = 50^-gamma; abs: no "-0"
        kt = min(t, key=lambda x: abs(x - T))
        assert abs(kt - T) < 1e-4 * max(T, 1), (k, T, kt)
        gaps.append(abs(pct(v["val_acc"] - t[kt]["val_acc"])))
        pubgaps.append(abs(pub[f"mnist_L50_gamma{g:g}_test_acc"] - pct(v["val_acc"])))
        rows.append(f"${g:g}$ & {a:.4g} & {T:.3g} & {pub[f'mnist_L50_gamma{g:g}_test_acc']:.2f} & "
                    f"{pct(v['val_acc']):.2f} & {pct(t[kt]['val_acc']):.2f} \\\\")
    num("MnistDiscGap", max(gaps), ".2f")
    num("MnistPubGap", max(pubgaps), ".2f")
    write("mnist", "\\begin{tabular}{@{}rrrrrr@{}}\n\\toprule\n"
          "& & & \\multicolumn{3}{c}{test accuracy (\\%)} \\\\\n\\cmidrule(lr){4-6}\n"
          "$\\gamma$ & $\\alpha=50^{-\\gamma}$ & $T$ & published & $\\bar r$, $L=50$ & $K_T$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --- E6a: fully connected networks against their kernels ------------------------------
fc = defaultdict(list)
for f in glob.glob(str(RUNS / "e6a-*" / "metrics.json")):
    L, T, m, sd = Path(f).parent.name.split("-")[1:]
    fc[(int(L[1:]), float(T[1:]), int(m[1:]))].append(json.loads(Path(f).read_text()))
if fc:
    # best risk of the network over its kernel's, mean over seeds
    ratio = {k: float(np.mean([min(v["net_risk"]) / min(v["kernel_risk"]) for v in runs]))
             for k, runs in fc.items()}
    num("FcSmallDev", 100 * max(abs(r - 1) for (L, T, m), r in ratio.items() if T <= 8), ".0f")
    for T, word in ((16.0, "Sixteen"), (64.0, "SixtyFour")):  # the same T at both widths
        for width, tag in ((512, "Narrow"), (2048, "Wide")):
            r = [v for (L, t, m), v in ratio.items() if t == T and m == width]
            if r:
                num(f"Fc{word}{tag}", 100 * (r[0] - 1), ".0f")
    seeds = [min(v["net_risk"]) / min(v["kernel_risk"]) for (L, T, m), runs in fc.items() if T == 32 for v in runs]
    if seeds:  # T = 32: per-seed range over both widths
        num("FcThirtyTwoLo", 100 * (min(seeds) - 1), ".0f")
        num("FcThirtyTwoHi", 100 * (max(seeds) - 1), ".0f")
    num("FcWeightChange", 100 * max(v["relative_weight_change"][-1] for runs in fc.values() for v in runs), ".1f")
    # at a fixed effective depth, the spread of the network's best risk across depths
    spread = []
    for T in {k[1] for k in fc}:
        best = [np.mean([min(v["net_risk"]) for v in fc[k]]) for k in fc if k[1] == T and k[2] == 2048]
        if len(best) > 1:
            spread.append(max(best) / min(best) - 1)
    if spread:
        num("FcDepthSpread", 100 * max(spread), ".1f")

# --- E6b-c: convolutional networks on CIFAR -----------------------------------------------
pub = published("yu2026_figures.csv")
for key, tag in (("base", "Base"), ("scale", "Scale"), ("bn", "Bn")):
    num(f"PubC{tag}", pub[f"cifar100_resnet34_{key}_test_acc"], ".1f")
runs6 = []  # (config, metrics) of every finished convolutional run
for p in sorted(RUNS.glob("e6[bc]-*")):
    m = load(p.name)
    if m:
        runs6.append((json.loads((p / "config.json").read_text()), m))


def cell(arch, schedule):
    """{T or "bn": ([test %], [train %])} for one architecture and schedule."""
    out = defaultdict(lambda: ([], []))
    for cfg, m in runs6:
        if cfg["arch"] == arch and cfg.get("schedule", "exp") == schedule and not m.get("diverged"):
            k = "bn" if cfg.get("bn") else round(m["T"], 4)
            out[k][0].append(100 * m["final_test_acc"])
            out[k][1].append(100 * m["final_train_acc"])
    return out


mean = lambda v: float(np.mean(v))
num("CDiverged", sum(bool(m.get("diverged")) for _, m in runs6), "d")
c100 = cell("resnet34", "step")
if c100:
    Ts = sorted(k for k in c100 if k != "bn")
    Tb = max(Ts, key=lambda T: mean(c100[T][0]))
    num("CBest", mean(c100[Tb][0]), ".1f")
    num("CBestT", Tb, "g")
    num("CSmallT", Ts[0], ".2f")
    num("CSmall", mean(c100[Ts[0]][0]), ".1f")
    num("CLargeT", Ts[-1], "g")
    num("CLarge", mean(c100[Ts[-1]][0]), ".1f")
    num("COne", mean(c100[1.0][0]), ".1f")
    num("CBn", mean(c100["bn"][0]), ".1f")
    num("CGapSmall", mean(c100[Tb][0]) - mean(c100[Ts[0]][0]), ".1f")  # best T against alpha = 1/16
    num("CSd", max(float(np.std(v[0], ddof=1)) for v in c100.values() if len(v[0]) > 1), ".1f")
    num("CTrain", min(min(v[1]) for k, v in c100.items() if k != "bn"), ".1f")
e100 = cell("resnet34", "exp")  # the decay Yu et al. state, for the three networks they compare
if e100:
    Ts = sorted(k for k in e100 if k != "bn")
    num("ExpCSmall", mean(e100[Ts[0]][0]), ".1f")
    num("ExpCLarge", mean(e100[Ts[-1]][0]), ".1f")
    num("ExpCBn", mean(e100["bn"][0]), ".1f")
    num("ExpCTrain", min(min(v[1]) for k, v in e100.items() if k != "bn"), ".1f")
for arch, tag in (("cifar3", "Twenty"), ("cifar9", "FiftySix"), ("cifar18", "HundredTen")):
    c = cell(arch, "step")
    if not c:
        continue
    Ts = sorted(c)
    num(f"D{tag}Small", mean(c[Ts[0]][0]), ".1f")   # alpha = 1/B
    num(f"D{tag}SmallT", Ts[0], ".2f")
    num(f"D{tag}Large", mean(c[Ts[-1]][0]), ".1f")  # alpha = 1
    num(f"D{tag}LargeT", Ts[-1], "g")
    num(f"D{tag}LargeTrain", mean(c[Ts[-1]][1]), ".1f")
    for T, word in ((0.25, "Quarter"), (1.0, "One"), (4.0, "Four")):
        num(f"D{tag}{word}", mean(c[T][0]), ".1f")
gaps = []  # best effective depth against alpha = 1/B, per depth
for arch in ("cifar3", "cifar9", "cifar18"):
    c = cell(arch, "step")
    if c:
        gaps.append(max(mean(v[0]) for v in c.values()) - mean(c[min(c)][0]))
if gaps:
    num("DGapSmall", min(gaps), ".1f")
d_all = [v for arch in ("cifar3", "cifar9", "cifar18") for v in cell(arch, "step").values()]
if d_all:
    num("DTrainLo", min(min(v[1]) for v in d_all), ".0f")
    num("DTrainHi", max(max(v[1]) for v in d_all), ".0f")
for schedule, tag in (("exp", "Exp"), ("step", "Step")):
    c = cell("cifar3", schedule)
    if c:
        num(f"{tag}TrainTwentyLo", min(min(v[1]) for v in c.values()), ".0f")
        num(f"{tag}TrainTwentyHi", max(max(v[1]) for v in c.values()), ".0f")

# --- compute, from the wall time each run recorded ------------------------------------------
hours = {"cpu": 0.0, "gpu": 0.0}
for f in glob.glob(str(RUNS / "*" / "metrics.json")):
    env = Path(f).parent / "env.json"
    e = json.loads(env.read_text()) if env.exists() else {}
    gpu = e.get("gpu") or e.get("cuda")  # cuda is recorded only when a GPU was used
    hours["gpu" if gpu else "cpu"] += json.loads(Path(f).read_text()).get("wall_seconds", 0) / 3600
num("ComputeCPU", 10 * round(hours["cpu"] / 10), "d")
num("ComputeGPU", 10 * round(hours["gpu"] / 10), "d")

(OUT / "numbers.tex").write_text(
    "% generated by scripts/make_tables.py from the run outputs; do not edit\n"
    + "".join(f"\\newcommand{{\\Num{k}}}{{{v}}}\n" for k, v in sorted(NUM.items())))
print(OUT / "numbers.tex", len(NUM), "numbers")
