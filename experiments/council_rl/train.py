"""Small-scale test of the multi-advisor council with deep RL in FoW chess.

Arms (only the bootstrap target and the voting weights differ):
  W : single Win advisor, max (DQN) target, softmax behaviour on Q_win.
  G : four advisors (material, safety, scout, pressure), each with its own max
      target ("egocentric"), behaviour = softmax(lam * sum_k w_k Q_k).
  E : same four advisors + a Win head with vote weight 0, all trained with an
      expected-SARSA target under the council policy pi_w ("empathic"), so all
      heads evaluate the same policy and eps can be measured.

Usage: python train.py --arm E --steps 20000000 --out ../../runs/council_rl/E_s0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fowenv import HEADS, N_PLANES, NH, FowEnv

COUNCIL_W = {"material": 0.4, "safety": 0.3, "scout": 0.1, "pressure": 0.2, "win": 0.0}
ARMS = {
    "W": dict(heads=("win",), vote={"win": 1.0}, target="max"),
    "G": dict(heads=("material", "safety", "scout", "pressure"), vote=COUNCIL_W, target="max"),
    "E": dict(heads=("material", "safety", "scout", "pressure", "win"), vote=COUNCIL_W, target="expected"),
}


# ---------------------------------------------------------------- network
class MultiHeadQ(nn.Module):
    """K fully separate CNNs (grouped convolutions), one per advisor."""

    def __init__(self, k: int, ch: int = 64, depth: int = 4):
        super().__init__()
        self.k = k
        layers = [nn.Conv2d(N_PLANES * k, ch * k, 3, padding=1, groups=k), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(ch * k, ch * k, 3, padding=1, groups=k), nn.ReLU()]
        layers += [nn.Conv2d(ch * k, 64 * k, 1, groups=k)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, 28, 8, 8] float -> [B, K, 4096] (index to*64+from)
        b = x.shape[0]
        y = self.net(x.repeat(1, self.k, 1, 1))
        return y.view(b, self.k, 64, 64).reshape(b, self.k, 4096)


# ---------------------------------------------------------------- actors
def worker(conn, seeds, opponent):
    envs = [FowEnv(s, opponent) for s in seeds]
    obs = [e.reset() for e in envs]
    conn.send(([o[0] for o in obs], [o[1] for o in obs]))
    while True:
        cmd, actions = conn.recv()
        if cmd == "close":
            return
        xs, ms, rs, ds, infos = [], [], [], [], []
        for e, a in zip(envs, actions):
            x, m, r, d, info = e.step(int(a))
            if d:
                x, m = e.reset()
            xs.append(x); ms.append(m); rs.append(r); ds.append(d); infos.append(info)
        conn.send((xs, ms, rs, ds, infos))


class VecEnv:
    def __init__(self, n_workers, envs_per_worker, seed, opponent="baseline"):
        self.conns, self.procs = [], []
        ctx = mp.get_context("fork")
        for w in range(n_workers):
            a, b = ctx.Pipe()
            seeds = [seed * 1_000_003 + w * 1000 + i for i in range(envs_per_worker)]
            p = ctx.Process(target=worker, args=(b, seeds, opponent), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)
        self.epw = envs_per_worker
        first = [c.recv() for c in self.conns]
        self.x = np.stack([x for f in first for x in f[0]])
        self.m = np.stack([m for f in first for m in f[1]])

    def step(self, actions):
        for i, c in enumerate(self.conns):
            c.send(("step", actions[i * self.epw:(i + 1) * self.epw]))
        out = [c.recv() for c in self.conns]
        xs = np.stack([x for o in out for x in o[0]])
        ms = np.stack([m for o in out for m in o[1]])
        rs = np.stack([r for o in out for r in o[2]])
        ds = np.array([d for o in out for d in o[3]])
        infos = [i for o in out for i in o[4]]
        return xs, ms, rs, ds, infos

    def close(self):
        for c in self.conns:
            c.send(("close", None))


# ---------------------------------------------------------------- replay
class Replay:
    def __init__(self, cap, nh):
        self.cap, self.n, self.i = cap, 0, 0
        self.x = np.zeros((cap, N_PLANES * 8), np.uint8)    # packbits of 28*64
        self.x2 = np.zeros((cap, N_PLANES * 8), np.uint8)
        self.m2 = np.zeros((cap, 512), np.uint8)             # packbits of 4096
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros((cap, nh), np.float32)
        self.g = np.zeros(cap, np.float32)                   # gamma^k * (1 - done)

    def add(self, x, a, r, x2, m2, g):
        j = self.i
        self.x[j] = np.packbits(x); self.a[j] = a; self.r[j] = r; self.g[j] = g
        if g > 0:
            self.x2[j] = np.packbits(x2); self.m2[j] = np.packbits(m2)
        self.i = (j + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, bs, rng):
        idx = rng.integers(0, self.n, bs)
        unpack = lambda arr, shape: np.unpackbits(arr[idx], axis=1).reshape(bs, *shape)
        return (unpack(self.x, (N_PLANES, 8, 8)), self.a[idx], self.r[idx],
                unpack(self.x2, (N_PLANES, 8, 8)), unpack(self.m2, (4096,)).astype(bool), self.g[idx])


# ---------------------------------------------------------------- policy
def council_logits(q, mask, vote, lam):
    """q: [B,K,4096], vote: [K] tensor -> masked logits [B,4096]."""
    s = (q * vote.view(1, -1, 1)).sum(1) * lam
    return s.masked_fill(~mask, float("-inf"))


def act(net, x, mask, vote, lam, eps, dev, greedy=False):
    with torch.no_grad():
        xt = torch.from_numpy(x).to(dev, torch.float32)
        mt = torch.from_numpy(mask).to(dev)
        logits = council_logits(net(xt), mt, vote, lam)
        if greedy:
            a = logits.argmax(1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        if eps > 0:
            uni = torch.distributions.Categorical(logits=torch.zeros_like(logits).masked_fill(~mt, float("-inf"))).sample()
            pick = torch.rand(a.shape[0], device=dev) < eps
            a = torch.where(pick, uni, a)
    return a.cpu().numpy()


# ---------------------------------------------------------------- eval
def evaluate(net, vote, lam, dev, games, opponent, seed, greedy=False, collect=False, n_workers=8):
    epw = max(1, games // n_workers)
    ve = VecEnv(n_workers, epw, seed, opponent)
    total = n_workers * epw
    results, finished = [], np.zeros(total, bool)
    states = []
    x, m = ve.x, ve.m
    while not finished.all():
        if collect:
            states.append((np.packbits(x[~finished].reshape(-1, N_PLANES * 64), axis=1), m[~finished]))
        a = act(net, x, m, vote, lam, 0.0, dev, greedy)
        x, m, r, d, infos = ve.step(a)
        for i in np.flatnonzero(d & ~finished):
            finished[i] = True
            results.append(infos[i])
    ve.close()
    w = sum(r["result"] == "1-0" for r in results); l = sum(r["result"] == "0-1" for r in results)
    n = len(results); dr = n - w - l
    score = (w + 0.5 * dr) / n
    se = np.sqrt(score * (1 - score) / n)
    reasons = {}
    for r in results:
        k = f'{r["result"]} {r["reason"]}'; reasons[k] = reasons.get(k, 0) + 1
    out = dict(opponent=opponent, n=n, win=w, draw=dr, loss=l, score=score, score_ci95=1.96 * se,
               mean_plies=float(np.mean([r["plies"] for r in results])), reasons=reasons, greedy=greedy)
    if collect:
        out["_states"] = states
    return out


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--envs-per-worker", type=int, default=48)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--updates-per-iter", type=int, default=2)
    ap.add_argument("--target-every", type=int, default=2000)
    ap.add_argument("--replay", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=5_000_000)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    cfg = ARMS[args.arm]
    heads = cfg["heads"]; k = len(heads)
    head_idx = [HEADS.index(h) for h in heads]
    vote = torch.tensor([cfg["vote"].get(h, 0.0) for h in heads], device=dev)
    net = MultiHeadQ(k).to(dev); tgt = MultiHeadQ(k).to(dev); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rb = Replay(args.replay, k)
    ve = VecEnv(args.workers, args.envs_per_worker, args.seed)
    nenv = args.workers * args.envs_per_worker
    traj = [deque() for _ in range(nenv)]
    x, m = ve.x, ve.m
    steps = updates = 0; t0 = time.time(); next_eval = args.eval_every
    log = open(out / "log.jsonl", "a")
    ep_results = deque(maxlen=2000)
    stats = deque(maxlen=200)

    def flush(i, x2, m2, done):
        """Write the oldest transition of env i as an n-step sample."""
        q = traj[i]
        R = np.zeros(k, np.float32); g = 1.0
        for (_, _, r) in q:
            R += g * r; g *= args.gamma
        x0, a0, _ = q.popleft()
        rb.add(x0, a0, R, x2, m2, 0.0 if done else g)

    while steps < args.steps:
        a = act(net, x, m, vote, args.lam, args.eps, dev)
        x2, m2, r, d, infos = ve.step(a)
        for i in range(nenv):
            traj[i].append((x[i], a[i], r[i, head_idx]))
            if d[i]:
                while traj[i]:
                    flush(i, None, None, True)
                ep_results.append(infos[i]["result"])
            elif len(traj[i]) == args.nstep:
                flush(i, x2[i], m2[i], False)
        x, m = x2, m2
        steps += nenv

        if rb.n >= 20_000:
            for _ in range(args.updates_per_iter):
                bx, ba, br, bx2, bm2, bg = rb.sample(args.batch, rng)
                bx = torch.from_numpy(bx).to(dev, torch.float32)
                bx2 = torch.from_numpy(bx2).to(dev, torch.float32)
                bm2 = torch.from_numpy(bm2).to(dev)
                br = torch.from_numpy(br).to(dev); bg = torch.from_numpy(bg).to(dev)
                ba = torch.from_numpy(ba).to(dev)
                with torch.no_grad():
                    q2 = tgt(bx2)  # [B,K,4096]
                    valid = bg > 0
                    bm2 = bm2 | ~valid.view(-1, 1)  # avoid all -inf rows for terminal samples
                    if cfg["target"] == "max":
                        v2 = q2.masked_fill(~bm2.unsqueeze(1), float("-inf")).max(2).values
                    else:
                        pi = torch.softmax(council_logits(q2, bm2, vote, args.lam), 1)  # [B,4096]
                        v2 = (q2 * pi.unsqueeze(1)).sum(2)
                    v2 = torch.where(valid.view(-1, 1), v2, torch.zeros_like(v2))
                    y = br + bg.view(-1, 1) * v2
                q = net(bx).gather(2, ba.view(-1, 1, 1).expand(-1, k, 1)).squeeze(2)
                loss = F.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
                updates += 1
                if updates % args.target_every == 0:
                    tgt.load_state_dict(net.state_dict())
                if updates % 50 == 0:
                    stats.append((loss.item(), q.abs().mean(0).tolist()))

        if updates and updates % 1000 < args.updates_per_iter and stats:
            res = list(ep_results)
            rec = dict(steps=steps, updates=updates, sec=round(time.time() - t0),
                       sps=round(steps / (time.time() - t0)),
                       loss=float(np.mean([s[0] for s in stats])),
                       absq={h: float(np.mean([s[1][j] for s in stats])) for j, h in enumerate(heads)},
                       train_win=res.count("1-0") / max(1, len(res)), train_loss=res.count("0-1") / max(1, len(res)))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)

        if steps >= next_eval or steps >= args.steps:
            next_eval += args.eval_every
            torch.save({"arm": args.arm, "heads": heads, "state_dict": net.state_dict()}, out / "ckpt.pt")
            net.eval()
            ev = evaluate(net, vote, args.lam, dev, args.eval_games, "baseline", seed=10_000)
            net.train()
            ev["steps"] = steps
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(ev) + "\n")
            print("EVAL", json.dumps(ev), flush=True)
    ve.close()


if __name__ == "__main__":
    main()
