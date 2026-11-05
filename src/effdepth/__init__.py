"""Effective depth of wide residual networks: kernels, spectra and generalization."""

import os as _os

# Under Slurm, keep BLAS within the CPUs the job was given (OpenBLAS otherwise
# starts one thread per core of the node). Import effdepth before numpy.
_n = _os.environ.get("SLURM_CPUS_PER_TASK")
if _n:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_v, _n)

__version__ = "0.1.0"
