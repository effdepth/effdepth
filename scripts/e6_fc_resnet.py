"""E6a: finite-width fully connected ResNets trained by full-batch gradient descent.

The network (effdepth.nets.FCResNet) is scaled so that its infinite-width NTK is
the normalized RNTK; GD with step s for k steps is compared with the kernel flow
of the exact discrete kernel (same depth and alpha) at time t = s k, on the same
data. Several depths at the same effective depth T test the collapse at finite width.
"""
import effdepth  # noqa: F401
import argparse

import numpy as np
import torch

from effdepth import data, kernels as K, krr
from effdepth.nets import FCResNet
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--L", type=int, required=True)
ap.add_argument("--T", type=float, required=True)
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--n-test", type=int, default=4000)
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--target", default="harm1")
ap.add_argument("--sigma", type=float, default=0.5)
ap.add_argument("--step", type=float, default=1.0)
ap.add_argument("--t-max", type=float, default=3000.0)
args = ap.parse_args()

run = Run(args.run_dir, vars(args) | {"experiment": "e6_fc_resnet"})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"

alpha = K.alpha_for_T(args.T, args.L)
rng = np.random.default_rng(args.seed)
x_tr = data.sphere(args.n, args.d, rng)
x_te = data.sphere(args.n_test, args.d, rng)
tgt = (data.LinearTarget(args.d) if args.target == "linear"
       else data.HarmonicTarget(args.d, float(args.target[4:]), np.random.default_rng(777 + args.d)))
f_tr, f_te = tgt(x_tr), tgt(x_te)
y_tr = f_tr + args.sigma * rng.standard_normal(args.n)

checkpoints = np.array([t for t in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
                        if t <= args.t_max])

# kernel prediction with the exact discrete kernel at this (L, alpha)
grid = np.geomspace(1e-7, np.pi / 2, 4000)
kv = K.rntk_discrete(grid, np.full(args.L, alpha))


def gram(xa, xb):
    th = K.augmented_angle_from_dist2(np.clip(2 - 2 * xa @ xb.T, 0, 4))
    out = np.interp(np.log(np.maximum(th, grid[0])), np.log(grid), kv)
    out[th == 0] = 1.0
    return out


flow = krr.SpectralFlow(gram(x_tr, x_tr))
b_te = gram(x_te, x_tr) @ flow.q
kernel_risk = [float(np.mean((b_te @ flow.coefficients(y_tr, t) - f_te) ** 2)) for t in checkpoints]

gen = torch.Generator().manual_seed(10_000 + args.seed)
net = FCResNet(args.d, args.width, np.full(args.L, alpha), generator=gen).to(dev)
w0 = [p.detach().clone() for p in (net.W, net.V)]
xt = torch.tensor(x_tr, dtype=torch.float32, device=dev)
yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
xe = torch.tensor(x_te, dtype=torch.float32, device=dev)
fe = torch.tensor(f_te, dtype=torch.float32, device=dev)
opt = torch.optim.SGD(net.parameters(), lr=args.step)

net_risk, train_mse, rel_change = [], [], []
steps_done, ci = 0, 0
while ci < len(checkpoints):
    target_steps = int(round(checkpoints[ci] / args.step))
    while steps_done < target_steps:
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        steps_done += 1
    with torch.no_grad():
        net_risk.append(float(torch.mean((net(xe) - fe) ** 2)))
        train_mse.append(float(torch.mean((net(xt) - yt) ** 2)))
        rel_change.append(float(sum((p - q).norm() ** 2 for p, q in zip((net.W, net.V), w0)).sqrt()
                                / sum(q.norm() ** 2 for q in w0).sqrt()))
    print(f"t={checkpoints[ci]:g}: net {net_risk[-1]:.4f} kernel {kernel_risk[ci]:.4f} "
          f"train {train_mse[-1]:.4f} dW/W {rel_change[-1]:.2e}", flush=True)
    ci += 1

run.finish({"alpha": alpha, "times": checkpoints, "net_risk": net_risk, "kernel_risk": kernel_risk,
            "train_mse": train_mse, "relative_weight_change": rel_change,
            "target_variance": float(np.mean(f_te ** 2))})
