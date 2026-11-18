"""Final evaluation + alignment-error (eps) measurement for the council RL test.

python analyze.py --runs ../../runs/council_rl --games 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

from fowenv import N_PLANES
from train import ARMS, MultiHeadQ, evaluate


def load(run_dir, dev):
    ck = torch.load(Path(run_dir) / "ckpt.pt", map_location=dev, weights_only=False)
    net = MultiHeadQ(len(ck["heads"])).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, list(ck["heads"])


def q_on_states(net, states, dev):
    """Return list of [n_legal, K] arrays, one per state."""
    out = []
    with torch.no_grad():
        for packed, masks in states:
            x = np.unpackbits(packed, axis=1).reshape(-1, N_PLANES, 8, 8)
            q = net(torch.from_numpy(x).to(dev, torch.float32)).cpu().numpy()
            for qi, mi in zip(q, masks):
                out.append(qi[:, mi].T)
    return out


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def eps_fit(per_state, adv_idx, win_idx, rng):
    n = len(per_state)
    perm = rng.permutation(n); tr, te = perm[: n // 2], perm[n // 2:]
    def stack(ids, center):
        X, y = [], []
        for i in ids:
            s = per_state[i]
            a, w = s[:, adv_idx], s[:, win_idx]
            if center:
                a = a - a.mean(0); w = w - w.mean()
            X.append(a); y.append(w)
        return np.concatenate(X), np.concatenate(y)
    res = {}
    for name, center in [("global", False), ("within_state", True)]:
        Xtr, ytr = stack(tr, center); Xte, yte = stack(te, center)
        if not center:  # NNLS with a free intercept: fit on [X, 1, -1]
            A = np.hstack([Xtr, np.ones((len(Xtr), 1)), -np.ones((len(Xtr), 1))])
            coef, _ = nnls(A, ytr); w, c = coef[:-2], coef[-2] - coef[-1]
        else:
            w, _ = nnls(Xtr, ytr); c = 0.0
        pred = Xte @ w + c
        resid = np.abs(yte - pred)
        res[name] = dict(weights=w.tolist(), intercept=float(c), r2_test=float(r2(yte, pred)),
                         resid_p95=float(np.quantile(resid, 0.95)), resid_max=float(resid.max()),
                         y_std=float(yte.std()))
    # how often the fitted council's argmax move equals the win head's argmax move (test states)
    w = np.array(res["within_state"]["weights"])
    agree = np.mean([np.argmax(per_state[i][:, adv_idx] @ w) == np.argmax(per_state[i][:, win_idx]) for i in te])
    spread = np.mean([per_state[i][:, win_idx].max() - per_state[i][:, win_idx].min() for i in te])
    res["argmax_agreement"] = float(agree)
    res["mean_win_q_spread_across_moves"] = float(spread)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../../runs/council_rl")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    runs = Path(args.runs)
    report = {}
    rng = np.random.default_rng(args.seed)

    def run_evals(tag, net, vote):
        r = {}
        for opp, seed in [("baseline", 20_000), ("random", 30_000)]:
            ev = evaluate(net, vote, args.lam, dev, args.games, opp, seed, collect=(tag == "E" and opp == "baseline"))
            st = ev.pop("_states", None)
            r[opp] = ev
            print(tag, opp, json.dumps(ev), flush=True)
            if st is not None:
                r["_states"] = st
        return r

    for arm in ("W", "G", "E"):
        net, heads = load(runs / f"{arm}_s0", dev)
        vote = torch.tensor([ARMS[arm]["vote"].get(h, 0.0) for h in heads], device=dev)
        report[arm] = run_evals(arm, net, vote)
        if arm == "E":
            states = report[arm].pop("_states")
            per_state = q_on_states(net, states, dev)
            adv = [heads.index(h) for h in ("material", "safety", "scout", "pressure")]
            fit = eps_fit(per_state, adv, heads.index("win"), rng)
            fit["n_states"] = len(per_state)
            report["eps"] = fit
            print("EPS", json.dumps(fit), flush=True)
            wfit = fit["within_state"]["weights"]
            vfit = torch.tensor([*wfit, 0.0], device=dev)  # fitted weights reproduce Q_win scale
            report["E-fit"] = run_evals("E-fit", net, vfit)
    (runs / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
