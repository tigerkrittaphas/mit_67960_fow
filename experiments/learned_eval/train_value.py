"""Train a small value net (NNUE-style MLP) for FoW chess leaves.

Input  : 12x64 piece planes + 2x64 visibility planes (side to move = "us")
Target : t = lam * sf + (1 - lam) * (z + 1) / 2     (win probability for us)
Output : weights in an .npz consumed by bots/*/valuenet.py (pure numpy).

  .venv-train/bin/python experiments/learned_eval/train_value.py runs/learned_eval/data.npz \
      --out runs/learned_eval/value_v1.npz --lam 0.5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, h1=256, h2=32):
        super().__init__()
        self.l1 = nn.Linear(14 * 64, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.l1(x)).clamp(max=1.0)
        x = F.relu(self.l2(x)).clamp(max=1.0)
        return self.l3(x).squeeze(-1)  # logit of win probability


def load(path):
    d = np.load(path)
    xb = np.unpackbits(d["xb"], axis=-1)[..., :64].reshape(len(d["z"]), -1)
    xv = np.unpackbits(d["xv"], axis=-1)[..., :64].reshape(len(d["z"]), -1)
    x = np.concatenate([xb, xv], 1).astype(np.float32)
    return x, d["z"].astype(np.float32), d["sf"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    parts = [load(p) for p in args.data]
    x = np.concatenate([p[0] for p in parts]); z = np.concatenate([p[1] for p in parts]); sf = np.concatenate([p[2] for p in parts])
    ok = ~np.isnan(sf)
    x, z, sf = x[ok], z[ok], sf[ok]
    t = args.lam * sf + (1 - args.lam) * (z + 1) / 2
    n = len(t); rng = np.random.default_rng(args.seed); perm = rng.permutation(n)
    nv = max(1000, n // 20); va, tr = perm[:nv], perm[nv:]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(x).to(dev); T = torch.from_numpy(t).to(dev); Z = torch.from_numpy(z).to(dev)
    net = ValueNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    tr_t = torch.from_numpy(tr).to(dev); va_t = torch.from_numpy(va).to(dev)

    def evaluate():
        with torch.no_grad():
            lo = net(X[va_t]); p = torch.sigmoid(lo)
            bce = F.binary_cross_entropy_with_logits(lo, T[va_t]).item()
            # outcome log-loss on decisive games only (how well we predict real results)
            dec = Z[va_t] != 0
            y = (Z[va_t][dec] + 1) / 2
            ll = F.binary_cross_entropy(p[dec].clamp(1e-4, 1 - 1e-4), y).item()
            acc = ((p[dec] > 0.5).float() == y).float().mean().item()
        return bce, ll, acc

    print(f"n={n} train={len(tr)} val={len(va)} lam={args.lam}")
    for ep in range(args.epochs):
        idx = tr_t[torch.randperm(len(tr_t), device=dev)]
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            loss = F.binary_cross_entropy_with_logits(net(X[b]), T[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        bce, ll, acc = evaluate()
        print(f"epoch {ep + 1}: val_bce={bce:.4f} outcome_logloss={ll:.4f} outcome_acc={acc:.3f}", flush=True)
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    np.savez(args.out, **sd)
    print("saved", args.out)


if __name__ == "__main__":
    main()
