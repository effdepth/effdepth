# effdepth

Effective depth of wide residual networks: the neural tangent kernel (NTK) of a
wide residual network with branch scales $\alpha_1,\dots,\alpha_L$ depends on
its depth and its scales through one number, the effective depth
$T=\sum_\ell \alpha_\ell^2/(1+\alpha_\ell^2)$, up to an error proportional to the
largest $\alpha_\ell^2$. This repository holds the code that computes these
kernels, their large-$T$ expansion and their Mercer spectra; the kernel
regression experiments across effective depths and sample sizes (on the sphere,
MNIST and CIFAR-10); the finite networks trained across effective depths; the
outputs of every run; and the scripts that build the tables and figures.

## Installation

Python 3.10 to 3.13. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

On Linux this installs PyTorch 2.6 built for CUDA 12.4, which runs on any NVIDIA
GPU from Pascal on; elsewhere, the PyPI build. With pip: `pip install -e .[dev]`.
Only the network experiments need a GPU.

## Usage

```python
import numpy as np
from effdepth import kernels as K

alphas = np.full(64, 0.25)                # depth 64, constant branch scale 1/4
T = K.effective_depth(alphas)             # sum of alpha^2 / (1 + alpha^2)
theta = np.array([0.01, 0.1, 0.5, 1.0, np.pi / 2])   # augmented angles
print(f"effective depth T = {T:.4f}")
print("discrete kernel   ", np.round(K.rntk_discrete(theta, alphas), 5))
print("effective kernel  ", np.round(K.kernel_T(theta, T), 5))
print("same T, depth 1024", np.round(K.rntk_discrete(theta, np.full(1024, K.alpha_for_T(T, 1024))), 5))
```

prints

```
effective depth T = 3.7647
discrete kernel    [0.99251 0.92549 0.65444 0.40861 0.24372]
effective kernel   [0.99242 0.92471 0.65264 0.40716 0.24248]
same T, depth 1024 [0.99243 0.92476 0.65275 0.40725 0.24256]
```

The normalized NTK of depth 64 (`rntk_discrete`) is close to the
effective-depth kernel $K_T$ (`kernel_T`), and a network of depth 1024 with the
same effective depth is closer still. The augmented angle of two inputs $x,x'$
on the unit sphere is $\arccos((1+x\cdot x')/2)$ (`augmented_angle`).

| module | contents |
|---|---|
| `effdepth.kernels` | the discrete normalized NTK, the effective-depth kernel $K_T$, its phase and large-$T$ expansion |
| `effdepth.spectrum` | Mercer eigenvalues on the sphere (Funk–Hecke) |
| `effdepth.krr` | kernel gradient flow and gradient descent through one eigendecomposition |
| `effdepth.eigenlearning` | eigenlearning predictions of the excess risk |
| `effdepth.data` | uniform data on the sphere, targets, MNIST and CIFAR |
| `effdepth.nets` | fully connected and convolutional residual networks with branch scales |
| `effdepth.runio` | run directories with their JSON sidecars |

Run the tests with `uv run pytest`.

## Tables and figures

The outputs of every run are in `results/runs/` (one directory per run, with
`config.json`, `metrics.json` and the library versions in `env.json`), and the
values read from the figures of Yu et al. (2026) in `results/published/`.

```bash
uv run python scripts/make_tables.py     # paper/tables/*.tex
uv run python scripts/make_figures.py    # paper/figures/*.pdf
```

| output | content |
|---|---|
| `tables/collapse.tex` | distance between the discrete NTK and $K_T$ over branch-scale schedules |
| `tables/phase.tex` | kernel regression on the sphere: tempered, best and stopped risks by sample size |
| `tables/mnist.tex` | MNIST accuracies of the branch-scale schedules of Yu et al. at depth 50 |
| `tables/numbers.tex` | every other number quoted in the paper, as LaTeX macros |
| `figures/collapse.pdf` | the discrete NTK against $K_T$ at equal effective depth |
| `figures/large.pdf` | $K_T$ at large effective depth and its near-diagonal profile |
| `figures/rates.pdf` | degeneration rates of the schedules $\alpha=L^{-\gamma}$ |
| `figures/spectrum.pdf` | Mercer spectra of $K_T$ |
| `figures/phase.pdf`, `figures/phased5.pdf` | excess risk across sample size and effective depth on the 2- and 4-sphere |
| `figures/revisit.pdf` | the synthetic experiment of Yu et al. across depths, scales and training budgets |
| `figures/real.pdf` | kernel regression on MNIST and CIFAR-10 across effective depths |
| `figures/fcnets.pdf` | finite fully connected networks against their kernels |
| `figures/cifar.pdf` | convolutional networks on CIFAR-100 and CIFAR-10 across effective depths and depths |

## Rerunning the experiments

`scripts/campaigns.py` lists every unit of work with its command and the
resources it was given:

```bash
python scripts/campaigns.py tables            # kernel tables, needed by the regressions
python scripts/campaigns.py kernels           # collapse, large depth, spectra, eigenlearning, the synthetic protocol
python scripts/campaigns.py e4a --slurm       # phase diagram on the sphere, as sbatch lines
python scripts/campaigns.py e4c               # MNIST and CIFAR-10 kernels
python scripts/campaigns.py e6a               # fully connected networks (GPU)
python scripts/campaigns.py e6b               # ResNet-34 on CIFAR-100 (GPU)
python scripts/campaigns.py e6c               # CIFAR ResNets on CIFAR-10 (GPU)
```

The commands run as printed, locally or on any Slurm cluster (`--slurm` wraps
them in `sbatch` with cores, memory and wall time). Each run writes
`out/runs/<name>/` and is skipped when that directory already holds
`metrics.json`, so an interrupted campaign can be resubmitted. Build the tables
and figures from the new runs with `EFFDEPTH_RUNS=out/runs`. The kernel
computations took about 90 CPU-hours (AMD EPYC, one BLAS thread per run,
which was the fastest setting for the eigendecompositions), and the networks
about 40 GPU-hours on NVIDIA A100s (one run on an RTX 6000 Ada).

## Data

MNIST (LeCun et al., 1998) and CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) are
downloaded by torchvision from their distributors into `$EFFDEPTH_DATA`
(default `~/data/effdepth`) and are not redistributed here; their terms are those
of their distributors. The CIFAR page states no license and asks that the
technical report (Krizhevsky, 2009) be cited.

## References

- A. Krizhevsky. Learning multiple layers of features from tiny images.
  Technical report, University of Toronto, 2009.
- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner. Gradient-based learning applied
  to document recognition. *Proceedings of the IEEE* 86(11):2278–2324, 1998.
- Z. Yu, G. Chen, J. Lai, B. Li and S. Tian. Branch scaling manifests as implicit
  architectural regularization for improving generalization in
  overparameterized ResNets. *Proceedings of the 43rd International Conference
  on Machine Learning*, PMLR 306:150046–150091, 2026.

## License

MIT; see `LICENSE`.
