"""E6b-c: normalization-free convolutional ResNets on CIFAR, across effective depths.

Training: Adam with learning rate 3e-4 for 200 epochs, as in Yu et al. (2026,
Sec. 4.2.2), decayed either by 0.95 per epoch, the decay they state (--schedule
exp), or tenfold at epochs 30, 60 and 90, the schedule their training curves show
(--schedule step); random crop (padding 4) and horizontal flip; batch 128. Every
branch is scaled by alpha, set from --T through
alpha = sqrt(T/(B - T)) for B blocks, or given with --alpha; --bn trains the
standard batch-normalized network instead. The run resumes from its last
checkpoint when restarted.
"""
import effdepth  # noqa: F401
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from effdepth import data
from effdepth.kernels import effective_depth
from effdepth.nets import conv_resnet, n_blocks
from effdepth.runio import Run

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
ap.add_argument("--arch", default="resnet34")
ap.add_argument("--T", type=float)
ap.add_argument("--alpha", type=float)
ap.add_argument("--bn", action="store_true")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--decay", type=float, default=0.95)
ap.add_argument("--schedule", choices=["exp", "step"], default="exp")
ap.add_argument("--milestones", type=int, nargs="+", default=[30, 60, 90])
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

blocks = n_blocks(args.arch)
if args.bn:
    alpha = 1.0
elif args.alpha is not None:
    alpha = args.alpha
else:
    alpha = math.sqrt(args.T / (blocks - args.T))
T = effective_depth(np.full(blocks, alpha))
run = Run(args.run_dir, vars(args) | {"experiment": "e6_cifar", "alpha_used": alpha,
                                      "T_used": T, "blocks": blocks})
if run.done:
    raise SystemExit(f"{args.run_dir}: done")
run.start()
dev = "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)

x_tr, y_tr = data.image_dataset(args.dataset, train=True)
x_te, y_te = data.image_dataset(args.dataset, train=False)
mean = x_tr.mean(axis=(0, 2, 3), keepdims=True)
std = x_tr.std(axis=(0, 2, 3), keepdims=True)
to_dev = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32, device=dev)
x_tr, x_te = to_dev(x_tr), to_dev(x_te)
y_tr = torch.tensor(y_tr, device=dev)
y_te = torch.tensor(y_te, device=dev)
classes = int(y_tr.max()) + 1
n = len(x_tr)

model = conv_resnet(args.arch, alpha, classes, bn=args.bn).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.schedule == "exp":
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.decay)
else:
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=0.1)
ckpt = Path(args.run_dir) / "ckpt.pt"
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
start = 0
if ckpt.exists():
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    history, start = state["history"], state["epoch"]
    torch.set_rng_state(state["rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    print(f"resumed at epoch {start}", flush=True)

idx32 = torch.arange(32, device=dev)


def augment(x):
    b = len(x)
    p = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (b,), device=dev)
    j = torch.randint(0, 9, (b,), device=dev)
    rows = (i[:, None] + idx32)[:, :, None]
    cols = (j[:, None] + idx32)[:, None, :]
    out = p[torch.arange(b, device=dev)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def evaluate():
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(x_te), 1000):
            out = model(x_te[k:k + 1000].contiguous(memory_format=torch.channels_last)).float()
            loss += float(F.cross_entropy(out, y_te[k:k + 1000], reduction="sum"))
            correct += int((out.argmax(1) == y_te[k:k + 1000]).sum())
    model.train()
    return loss / len(x_te), correct / len(x_te)


diverged = False
for epoch in range(start, args.epochs):
    perm = torch.randperm(n, device=dev)
    tl, tc = 0.0, 0
    for k in range(0, n, args.batch):
        b = perm[k:k + args.batch]
        xb = augment(x_tr[b]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb).float()
        loss = F.cross_entropy(out, y_tr[b])
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tl += float(loss) * len(b)
        tc += int((out.argmax(1) == y_tr[b]).sum())
    if diverged:
        print(f"diverged at epoch {epoch}", flush=True)
        break
    sched.step()
    te_loss, te_acc = evaluate()
    for key, val in zip(history, (tl / n, tc / n, te_loss, te_acc)):
        history[key].append(val)
    print(f"epoch {epoch + 1}: train {tc / n:.4f} ({tl / n:.4f})  test {te_acc:.4f} ({te_loss:.4f})", flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "history": history, "epoch": epoch + 1, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, ckpt)

acc = history["test_acc"]
run.finish({"history": history, "diverged": diverged, "alpha": alpha, "T": T, "blocks": blocks,
            "final_test_acc": acc[-1] if acc else None, "best_test_acc": max(acc) if acc else None,
            "final_train_acc": history["train_acc"][-1] if acc else None})
ckpt.unlink(missing_ok=True)  # keep the run directory small once finished
