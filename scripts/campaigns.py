"""The computations behind the paper: one command per unit of work, with its resources.

    python scripts/campaigns.py <campaign> [--slurm]

Prints the command of every unit of a campaign, to run as it is, or, with
--slurm, wrapped in an sbatch line with the cores, memory and wall time it was
given in our runs (AMD EPYC CPUs with one BLAS thread, NVIDIA A100 GPUs; memory
is what the eigendecompositions need, the cores follow from it). Run `tables`
first: the regression units read the tabulated kernels from out/tables. Each run
writes its own directory under out/runs and exits at once if that directory
already holds metrics.json, so a campaign can be resubmitted after an
interruption. Build the tables and figures from new runs with
EFFDEPTH_RUNS=out/runs.
"""
import argparse
import shlex

E4_TS = [0] + [2.0 ** k for k in range(-4, 18)]
E4_SEEDS = {250: 5, 500: 5, 1000: 5, 2000: 5, 4000: 5, 8000: 3, 16000: 3, 32000: 1}
E4_RES = {250: (4, "7G", "1:00:00"), 500: (4, "7G", "1:00:00"), 1000: (4, "7G", "1:00:00"),
          2000: (4, "7G", "1:00:00"), 4000: (8, "14G", "2:00:00"), 8000: (16, "28G", "3:00:00"),
          16000: (32, "56G", "6:00:00"), 32000: (24, "45G", "3:00:00")}
SPLIT_N = 32000  # from this size on, one effective depth per unit
REAL_TS = [0, 0.0625, 0.25, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384]
SOURCE_ALPHA = [0.02, 0.05318296, 0.1414214, 0.3760603, 1.0]  # 50^-gamma, gamma = 1, 3/4, 1/2, 1/4, 0
SOURCE_T = [0.019992, 0.1410225, 0.9803922, 6.194967, 25.0]   # 50 alpha^2 / (1 + alpha^2)
GPU = "gpu"


def fmt(values):
    return " ".join(f"{v:g}" for v in values)


def tables():
    discrete = " ".join(f"50:{a:g}" for a in SOURCE_ALPHA)
    yield "tables-a", (4, "7G", "3:00:00"), f"scripts/kernel_tables.py --Ts 0 0.0625 0.125 0.25 0.5 1 2 4 --discrete {discrete}"
    yield "tables-b", (4, "7G", "3:00:00"), f"scripts/kernel_tables.py --Ts 8 16 32 64 128 256 {fmt(SOURCE_T)}"
    yield "tables-c", (4, "7G", "4:00:00"), "scripts/kernel_tables.py --Ts 512 1024 2048 4096 8192"
    yield "tables-d", (4, "7G", "6:00:00"), "scripts/kernel_tables.py --Ts 16384 32768 65536 131072"


def kernels():
    yield "e1-collapse", (2, "3G", "4:00:00"), "scripts/e1_collapse.py"
    yield "e2-large-depth", (4, "7G", "24:00:00"), "scripts/e2_large_depth.py"
    for d, kmax in ((3, 4000), (5, 2000)):
        yield f"e3-spectrum-d{d}", (8, "14G", "8:00:00"), f"scripts/e3_spectrum.py --d {d} --kmax {kmax}"
    for d, kmax in ((3, 4000), (5, 400)):  # the spectra the eigenlearning predictions use
        yield (f"e3-spectrum-d{d}-grid", (8, "14G", "8:00:00"),
               f"scripts/e3_spectrum.py --d {d} --kmax {kmax} --Ts {fmt(E4_TS)}")
    yield "e4b-eigenlearning", (4, "7G", "2:00:00"), "scripts/e4b_eigenlearning.py"
    yield "e5-source-protocol", (4, "7G", "4:00:00"), "scripts/e5_source_protocol.py --seeds 20 --lr 1e-4"


def e4a():
    for d, nmax in ((3, 32000), (5, 16000)):
        for n, seeds in E4_SEEDS.items():
            if n > nmax:
                continue
            for s in range(seeds):
                name = f"e4a-d{d}-n{n}-s{s}"
                cmd = f"scripts/e4_synthetic.py --d {d} --n {n} --seed {s}"
                if n < SPLIT_N:
                    yield name, E4_RES[n], f"{cmd} --Ts {fmt(E4_TS)}"
                else:
                    for t in E4_TS:
                        yield f"{name}-T{t:g}", E4_RES[n], f"{cmd} --Ts {t:g}"


def e4c():
    kernels_ = [f"K_T{t:g}" for t in REAL_TS]
    source = [f"K_L50_a{a:g}" for a in SOURCE_ALPHA] + [f"K_T{t:g}" for t in SOURCE_T]
    yield ("e4c-mnist-n5000", (8, "14G", "2:00:00"),
           f"scripts/e4_real.py --dataset mnist --n 5000 --n-val 10000 --kernels {' '.join(kernels_)}")
    for k in kernels_ + source:
        yield (f"e4c-mnist-n20000-{k}", (12, "22G", "1:30:00"),
               f"scripts/e4_real.py --dataset mnist --n 20000 --n-val 10000 --kernels {k}")
    yield ("e4c-cifar10-n10000", (16, "28G", "6:00:00"),
           f"scripts/e4_real.py --dataset cifar10 --n 10000 --n-val 10000 --kernels {' '.join(kernels_)}")
    for t in REAL_TS:
        yield (f"e4c-cifar10-n45000-T{t:g}", (40, "75G", "6:00:00"),
               f"scripts/e4_real.py --dataset cifar10 --n 45000 --n-val 5000 --kernels K_T{t:g}")


def e6a():
    for L, Ts in ((8, (0.5, 2, 4)), (32, (0.5, 2, 8, 16)), (128, (0.5, 2, 8, 32, 64))):
        for T in Ts:
            for m in (512, 2048):
                for s in (0, 1):
                    time = "3:00:00" if L * m > 100_000 else "1:00:00"  # depth 128, width 2048
                    yield (f"e6a-L{L}-T{T:g}-m{m}-s{s}", (8, "40G", time, GPU),
                           f"scripts/e6_fc_resnet.py --L {L} --T {T:g} --width {m} --seed {s}")


def e6b():
    """ResNet-34 on CIFAR-100; the decay Yu et al. state (exp) for the methods they compare."""
    specs = [("a0.0625", "--alpha 0.0625")] + [(f"T{T:g}", f"--T {T:g}") for T in (0.25, 1, 2, 4)]
    specs += [("a1", "--alpha 1"), ("bn", "--bn")]
    for schedule in ("step", "exp"):
        for tag, flag in specs:
            if schedule == "exp" and tag.startswith("T"):
                continue
            prefix = "e6b-c100-resnet34-" + ("step-" if schedule == "step" else "")
            for s in range(3):
                yield (f"{prefix}{tag}-s{s}", (8, "48G", "4:00:00", GPU),
                       f"scripts/e6_cifar.py --dataset cifar100 --arch resnet34 --schedule {schedule} {flag} --seed {s}")


def e6c():
    """CIFAR ResNets with 9, 27 and 54 blocks on CIFAR-10; the stated decay for 9 blocks."""
    for schedule in ("step", "exp"):
        for k in ((3, 9, 18) if schedule == "step" else (3,)):
            blocks = 3 * k
            specs = [("a1overB", f"--alpha {1 / blocks}")] + [(f"T{T:g}", f"--T {T:g}") for T in (0.25, 1, 4)]
            specs += [("a1", "--alpha 1")]
            prefix = f"e6c-c10-cifar{k}-" + ("step-" if schedule == "step" else "")
            for tag, flag in specs:
                for s in (0, 1):
                    yield (f"{prefix}{tag}-s{s}", (8, "40G", "2:00:00", GPU),
                           f"scripts/e6_cifar.py --dataset cifar10 --arch cifar{k} --schedule {schedule} {flag} --seed {s}")


CAMPAIGNS = {"tables": tables, "kernels": kernels, "e4a": e4a, "e4c": e4c,
             "e6a": e6a, "e6b": e6b, "e6c": e6c}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign", choices=list(CAMPAIGNS))
    ap.add_argument("--slurm", action="store_true", help="print sbatch lines with resources")
    a = ap.parse_args()
    for name, res, cmd in CAMPAIGNS[a.campaign]():
        if not name.startswith("tables"):
            cmd += f" --run-dir out/runs/{name}"
        line = f"python {cmd}"
        if a.slurm:
            cpus, mem, time = res[:3]
            gpu = " --gres=gpu:1" if len(res) > 3 else ""
            line = (f"sbatch --job-name={name} -c {cpus} --mem={mem} --time={time}{gpu} "
                    f"--wrap={shlex.quote(line)}")
        print(line)


if __name__ == "__main__":
    main()
