"""Networks: the fully connected ResNet of the theory, and normalization-free conv ResNets."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCResNet(nn.Module):
    """The fully connected ResNet of Huang et al. (2020) and Yu et al. (2026), mirrored.

    h0 = A x~ with x~ = (x, 1)/sqrt(2) and A fixed;
    h_l = (h_{l-1} + alpha_l V_l relu(sqrt(2/m) W_l h_{l-1}) / sqrt(m)) / sqrt(1 + alpha_l^2);
    f = v^T h_L / sqrt(2 T m), v fixed. The division by sqrt(1 + alpha^2) rescales
    every layer by a constant (relu is positively homogeneous), so the function is
    the theory's up to a constant factor, chosen with 1/sqrt(2 T m) so that the
    infinite-width NTK is the normalized kernel (value 1 on the diagonal): gradient
    descent with step s for k steps then follows the kernel flow at time t = s k.
    Two copies with identical initial weights, f = (f1 - f2)/sqrt(2), start at f = 0.
    """

    def __init__(self, d: int, width: int, alphas, generator: torch.Generator | None = None):
        super().__init__()
        self.alphas = [float(a) for a in alphas]
        self.T = sum(a * a / (1 + a * a) for a in self.alphas)
        m, L = width, len(self.alphas)
        g = generator
        A = torch.randn(m, d + 1, generator=g)
        v = torch.randn(m, generator=g)
        W = torch.randn(L, m, m, generator=g)
        V = torch.randn(L, m, m, generator=g)
        self.register_buffer("A", A)
        self.register_buffer("v", v)
        self.W = nn.Parameter(torch.stack([W, W.clone()]))     # (2, L, m, m): two mirrored copies
        self.V = nn.Parameter(torch.stack([V, V.clone()]))
        self.m = m

    def forward(self, x):
        xt = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) / math.sqrt(2)
        # unbind, not indexing: the backward of W[c, l] writes a zero tensor of W's full
        # size for every block, which makes a step cost grow like L^2 m^2
        Ws = [w.unbind(0) for w in self.W.unbind(0)]
        Vs = [v.unbind(0) for v in self.V.unbind(0)]
        out = []
        for c in range(2):
            h = xt @ self.A.T
            for l, a in enumerate(self.alphas):
                z = math.sqrt(2.0 / self.m) * h @ Ws[c][l].T
                h = (h + a * torch.relu(z) @ Vs[c][l].T / math.sqrt(self.m)) / math.sqrt(1 + a * a)
            out.append(h @ self.v / math.sqrt(2 * self.T * self.m))
        return (out[0] - out[1]) / math.sqrt(2)


class Block(nn.Module):
    """Post-activation basic block without normalization: relu(shortcut(x) + alpha F(x)).

    F = conv3x3 -> relu -> conv3x3, initialized with unit gain (He-normal first
    convolution, variance 1/fan_in for the second), so that at initialization F
    preserves the second moment of its input and alpha is the branch-to-skip ratio
    of the theory. With `bn`, the standard block: conv-BN-relu-conv-BN, alpha = 1.
    """

    def __init__(self, cin, cout, stride, alpha, bn=False):
        super().__init__()
        self.alpha, self.bn = alpha, bn
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not bn)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not bn)
        self.b1 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.b2 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=not bn),
                                       nn.BatchNorm2d(cout) if bn else nn.Identity())
        if not bn:
            nn.init.kaiming_normal_(self.c1.weight, mode="fan_in", nonlinearity="relu")
            nn.init.normal_(self.c2.weight, std=1 / math.sqrt(self.c2.weight[0].numel()))
            nn.init.zeros_(self.c1.bias)
            nn.init.zeros_(self.c2.bias)
            if not isinstance(self.short, nn.Identity):
                conv = self.short[0]
                nn.init.normal_(conv.weight, std=1 / math.sqrt(conv.weight[0].numel()))
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(self.short(x) + self.alpha * y)


class ConvResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem, stages of basic blocks, global pooling, linear head.

    resnet34: blocks (3, 4, 6, 3), widths (64, 128, 256, 512), as in the standard
    ResNet-34 used by Yu et al. (2026) on CIFAR-100. cifar: (n, n, n) blocks of
    widths (16, 32, 64) (He et al. 2016, Sec. 4.2). `alpha` multiplies every branch.
    """

    def __init__(self, blocks, widths, alpha, classes, bn=False):
        super().__init__()
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=not bn)
        self.stem_bn = nn.BatchNorm2d(widths[0]) if bn else nn.Identity()
        if not bn:
            nn.init.kaiming_normal_(self.stem.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(self.stem.bias)
        layers, cin = [], widths[0]
        for i, (nb, w) in enumerate(zip(blocks, widths)):
            for j in range(nb):
                layers.append(Block(cin, w, 2 if (j == 0 and i > 0) else 1, alpha, bn))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(cin, classes)
        self.n_blocks = len(layers)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        x = self.layers(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


ARCHS = {"resnet34": ((3, 4, 6, 3), (64, 128, 256, 512))}


def conv_resnet(arch: str, alpha: float, classes: int, bn: bool = False) -> ConvResNet:
    if arch in ARCHS:
        blocks, widths = ARCHS[arch]
    elif arch.startswith("cifar"):                 # cifar<n>: n blocks per stage, depth 6n+2
        n = int(arch[5:])
        blocks, widths = (n, n, n), (16, 32, 64)
    else:
        raise ValueError(arch)
    return ConvResNet(blocks, widths, alpha, classes, bn)


def n_blocks(arch: str) -> int:
    if arch in ARCHS:
        return sum(ARCHS[arch][0])
    return 3 * int(arch[5:])
