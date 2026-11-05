"""One directory per run, with small JSON sidecars beside any artifacts.

config.json  the arguments of the run
env.json     library versions, threads, host, and the commit the code came from
metrics.json the results; its presence marks the run as complete, so a resubmitted
             campaign skips it
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path


def _commit() -> dict:
    stamp = Path(".git_commit")
    if stamp.exists():
        return dict(line.split("=", 1) for line in stamp.read_text().split("\n") if "=" in line)
    try:
        import subprocess
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return {"commit": c.stdout.strip()}
    except Exception:
        return {}


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME", ""),
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "threads": {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                        "OPENBLAS_NUM_THREADS", "SLURM_CPUS_PER_TASK")},
        "commit": _commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for mod in ("numpy", "scipy", "torch", "torchvision", "mpmath"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except Exception:
        pass
    return env


class Run:
    def __init__(self, path, config: dict):
        self.path = Path(path)
        self.config = config

    @property
    def done(self) -> bool:
        return (self.path / "metrics.json").exists()

    def start(self):
        self.path.mkdir(parents=True, exist_ok=True)
        _write(self.path / "config.json", self.config)
        _write(self.path / "env.json", environment())
        self._t0 = time.time()

    def finish(self, metrics: dict):
        metrics = dict(metrics, wall_seconds=round(time.time() - self._t0, 1))
        _write(self.path / "metrics.json", metrics)


def _write(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(path)


def _default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def load(path) -> dict:
    return json.loads(Path(path).read_text())
