#!/usr/bin/env -S .venv/bin/python
"""Local evaluation harness: play a bot against a pool of opponents with the
official referee (subprocess bots, real protocol, real clocks).

Games are played in colour-swapped pairs with a shared seed. Results are
reported from the candidate's point of view with 95% intervals; an optional
SPRT stops early once the candidate is significantly better or worse.

Examples
  python3 scripts/arena_local.py baseline --vs random --pairs 50
  python3 scripts/arena_local.py v3 --vs baseline,random --pairs 200 --base 60 --inc 1
  python3 scripts/arena_local.py v3.1 --vs v3 --pairs 1000 --sprt 0,30

Bot names:
  random    starter/uniform_random.py   (seeded per game)
  baseline  bots/baseline/baseline.py   (seeded per game)
  <name>    bots/<name>/run.sh, run from bots/<name>; seed in env FOW_SEED
Bots run with .venv/bin first on PATH.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter"
VENV_BIN = ROOT / ".venv" / "bin"


def bot_argv(name: str, seed: int) -> list[str]:
    py = str(VENV_BIN / "python") if (VENV_BIN / "python").exists() else "python3"
    if name == "random":
        return [py, "-u", str(STARTER / "uniform_random.py"), "--seed", str(seed)]
    if name == "baseline":
        return [py, "-u", str(ROOT / "bots/baseline/baseline.py"), "--seed", str(seed)]
    d = ROOT / "bots" / name
    if not (d / "run.sh").exists():
        raise SystemExit(f"unknown bot {name!r}: no {d}/run.sh")
    return ["env", f"FOW_SEED={seed}", "bash", "-c", f"cd {str(d)!r} && exec bash run.sh"]


def play(task):
    """Run one game in a worker process. Returns a compact record."""
    white, black, seed, base, inc, max_plies, replay_dir, keep = task
    os.environ["PATH"] = f"{VENV_BIN}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(STARTER))
    from referee import run_match  # noqa: E402

    t0 = time.time()
    rep = run_match(bot_argv(white, seed), bot_argv(black, seed + 7919), base, inc,
                    match_id=f"{white}-{black}-{seed}", white_name=white, black_name=black,
                    max_plies=max_plies)
    frames = rep.get("frames", [])
    plies = len(frames) - 1
    used, worst = [0.0, 0.0], [0.0, 0.0]
    for prev, cur in zip(frames, frames[1:]):
        side = 0 if prev["ply"] % 2 == 0 else 1
        dt = prev["clocks"][side] - (cur["clocks"][side] - inc)
        used[side] += dt
        worst[side] = max(worst[side], dt)
    rec = dict(white=white, black=black, seed=seed, result=rep["result"], reason=rep["reason"],
               failed=rep.get("failedSide"), plies=plies, wall=round(time.time() - t0, 2),
               time_used=[round(u, 3) for u in used], max_move_time=[round(w, 3) for w in worst])
    if keep(rec):
        replay_dir.mkdir(parents=True, exist_ok=True)
        (replay_dir / f"{white}_vs_{black}_{seed}.json").write_text(json.dumps(rep))
    return rec


class KeepReplays:
    def __init__(self, mode, cand):
        self.mode, self.cand = mode, cand

    def __call__(self, rec):
        if self.mode == "all":
            return True
        if self.mode == "none":
            return False
        lost = (rec["result"] == "0-1" and rec["white"] == self.cand) or \
               (rec["result"] == "1-0" and rec["black"] == self.cand)
        return lost or rec["failed"] is not None


def cand_score(rec, cand):
    r = rec["result"]
    if r == "1/2-1/2":
        return 0.5
    if r == "*":
        return None
    white_won = r == "1-0"
    return 1.0 if white_won == (rec["white"] == cand) else 0.0


def elo(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / s - 1)


def llr(scores, elo0, elo1):
    """GSPRT log-likelihood ratio, normal approximation (Van den Bergh)."""
    n = len(scores)
    if n < 2:
        return 0.0
    m = sum(scores) / n
    v = sum(x * x for x in scores) / n - m * m
    if v <= 0:
        return 0.0
    s0, s1 = 1 / (1 + 10 ** (-elo0 / 400)), 1 / (1 + 10 ** (-elo1 / 400))
    return n * (s1 - s0) * (2 * m - s0 - s1) / (2 * v)


def summarize(recs, cand, opp):
    sc = [cand_score(r, cand) for r in recs if r["black"] == opp or r["white"] == opp]
    sc = [s for s in sc if s is not None]
    n = len(sc)
    if not n:
        return None
    w, d = sc.count(1.0), sc.count(0.5)
    l = n - w - d
    m = sum(sc) / n
    sd = math.sqrt(max(sum(x * x for x in sc) / n - m * m, 1e-12) / n)
    lo, hi = max(m - 1.96 * sd, 0), min(m + 1.96 * sd, 1)
    mine = [r for r in recs if opp in (r["white"], r["black"])]
    reasons = {}
    for r in mine:
        k = f'{r["result"]} {r["reason"]}'
        reasons[k] = reasons.get(k, 0) + 1
    fails = [r for r in mine if r["failed"] and (r["white"] if r["failed"] == "white" else r["black"]) == cand]
    as_white = [cand_score(r, cand) for r in mine if r["white"] == cand]
    as_black = [cand_score(r, cand) for r in mine if r["black"] == cand]
    side = lambda xs: round(sum(x for x in xs if x is not None) / max(1, len(xs)), 3)
    ci = [lambda r: r["time_used"][0 if r["white"] == cand else 1],
          lambda r: r["max_move_time"][0 if r["white"] == cand else 1]]
    return dict(opponent=opp, games=n, win=w, draw=d, loss=l, score=round(m, 4),
                score_ci95=[round(lo, 4), round(hi, 4)], elo=round(elo(m), 1),
                elo_ci95=[round(elo(lo), 1), round(elo(hi), 1)],
                score_white=side(as_white), score_black=side(as_black),
                mean_plies=round(sum(r["plies"] for r in mine) / len(mine), 1),
                cand_mean_time_used=round(sum(ci[0](r) for r in mine) / len(mine), 2),
                cand_max_move_time=round(max(ci[1](r) for r in mine), 3),
                cand_failures=len(fails), reasons=reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("--vs", default="baseline,random", help="comma-separated opponents")
    ap.add_argument("--pairs", type=int, default=50, help="colour-swapped game pairs per opponent")
    ap.add_argument("--base", type=float, default=180.0)
    ap.add_argument("--inc", type=float, default=2.0)
    ap.add_argument("--max-plies", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--seed", type=int, default=1, help="first seed; pair i uses seed+i")
    ap.add_argument("--sprt", default=None, help="elo0,elo1 (single opponent only), e.g. 0,30")
    ap.add_argument("--sprt-min", type=int, default=100, help="no SPRT decision before this many games")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--replays", choices=["losses", "all", "none"], default="losses")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    opps = [o for o in args.vs.split(",") if o]
    if args.sprt and len(opps) != 1:
        raise SystemExit("--sprt needs exactly one opponent")
    for b in [args.candidate, *opps]:
        bot_argv(b, 0)  # validate names early
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else ROOT / "runs" / "arena" / f"{stamp}_{args.candidate}_vs_{'+'.join(opps)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    keep = KeepReplays(args.replays, args.candidate)

    tasks = []
    for i in range(args.pairs):
        s = args.seed + i
        for opp in opps:
            for w, b in [(args.candidate, opp), (opp, args.candidate)]:
                tasks.append((w, b, s, args.base, args.inc, args.max_plies, out / "replays", keep))

    recs, sprt_state = [], None
    if args.sprt:
        e0, e1 = map(float, args.sprt.split(","))
        lo_b, hi_b = math.log(args.beta / (1 - args.alpha)), math.log((1 - args.beta) / args.alpha)
    t0 = time.time()
    print(f"{len(tasks)} games, {args.jobs} parallel, {args.base}+{args.inc}s -> {out}", flush=True)
    with open(out / "games.jsonl", "a") as gl, cf.ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(play, t) for t in tasks]
        try:
            for k, f in enumerate(cf.as_completed(futs), 1):
                rec = f.result()
                recs.append(rec)
                gl.write(json.dumps(rec) + "\n"); gl.flush()
                if rec["failed"]:
                    print(f"  FAILURE ({rec['failed']}): {rec['white']} vs {rec['black']} seed {rec['seed']}: {rec['reason']}", flush=True)
                if k % 20 == 0 or k == len(futs):
                    parts = []
                    for o in opps:
                        s = summarize(recs, args.candidate, o)
                        if s:
                            parts.append(f"{o}: {s['score']:.3f} [{s['score_ci95'][0]:.3f},{s['score_ci95'][1]:.3f}] n={s['games']}")
                    print(f"[{k}/{len(futs)} {time.time() - t0:.0f}s] " + " | ".join(parts), flush=True)
                if args.sprt:
                    sc = [x for x in (cand_score(r, args.candidate) for r in recs) if x is not None]
                    L = llr(sc, e0, e1)
                    if len(sc) >= args.sprt_min and (L >= hi_b or L <= lo_b):
                        sprt_state = dict(llr=round(L, 3), bounds=[round(lo_b, 3), round(hi_b, 3)],
                                          verdict="H1 accepted (candidate stronger)" if L >= hi_b else "H0 accepted (not stronger)")
                        print("SPRT:", sprt_state["verdict"], f"LLR={L:.2f}", flush=True)
                        for g in futs:
                            g.cancel()
                        break
        except KeyboardInterrupt:
            for g in futs:
                g.cancel()
            print("interrupted; summarizing finished games", flush=True)

    summary = dict(candidate=args.candidate, time_control=[args.base, args.inc], wall_sec=round(time.time() - t0),
                   per_opponent=[summarize(recs, args.candidate, o) for o in opps])
    if args.sprt:
        sc = [x for x in (cand_score(r, args.candidate) for r in recs) if x is not None]
        summary["sprt"] = sprt_state or dict(llr=round(llr(sc, e0, e1), 3), verdict="inconclusive",
                                             elo0=e0, elo1=e1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
