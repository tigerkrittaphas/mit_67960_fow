#!/usr/bin/env python3
"""v3: belief tracking + safety veto + shallow evaluation over sampled boards.

stdout carries only `ready` and `move ...`; all diagnostics go to stderr.
"""
from __future__ import annotations

import os
import random
import sys
import time

import chess

from belief import Belief
from obsmoves import OBSERVATION_CHARACTERS, ObservationTracker
import search
import voi

N_BOARDS = int(os.environ.get("V3_BOARDS", "48"))
CAP = int(os.environ.get("V3_CAP", "2000"))
DEBUG = os.environ.get("V3_DEBUG") == "1"
USE_VOI = os.environ.get("V3_VOI", "0") == "1"


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr, flush=True)


def budget(remaining: float, inc: float) -> float:
    """Seconds to spend on this move (belief update included)."""
    return max(0.05, min(remaining / 30.0 + 0.7 * inc, 4.0, remaining * 0.25))


def main():
    seed = int(os.environ.get("FOW_SEED", str(int(time.time() * 1000) % 2**31)))
    rng = random.Random(seed)
    if sys.stdin.readline().strip().lower() != "fow":
        return
    header = sys.stdin.readline().split()
    side = header[0].lower() == "w"
    inc = float(header[2]) if len(header) > 2 else 2.0
    tracker = ObservationTracker(side)
    belief = Belief(side, cap=CAP, seed=seed)
    print("ready", flush=True)
    pending = []          # observations not yet folded into the belief
    last_move = None
    seen_after = {}
    for raw in sys.stdin:
        line = raw.strip()
        if len(line) == 64 and set(line) <= OBSERVATION_CHARACTERS:
            tracker.observe(line)
            pending.append(line)
            continue
        try:
            remaining = float(line)
        except ValueError:
            continue
        t0 = time.perf_counter()
        total = budget(remaining, inc)
        move = None
        try:
            for i, obs in enumerate(pending):
                # belief update may use up to 40% of this move's budget
                belief.observe(obs, deadline=t0 + 0.45 * total)
            pending.clear()
            moves = tracker.legal_moves()
            if not moves:
                return
            boards = belief.sample(N_BOARDS)
            dl = t0 + 0.9 * total
            voi_fn = (lambda m: voi.voi(boards, m, side, tracker.rights, dl)) if USE_VOI else None
            move, info = search.choose(boards, moves, side, rng, deadline=dl,
                                       last_move=last_move, voi_fn=voi_fn, seen_after=seen_after,
                                       all_boards=belief.boards,
                                       safety_deadline=time.perf_counter() + 0.25 * total)
            log(f"ply {belief.ply} n={len(belief.boards)} capped={belief.capped} {info} "
                f"{move.uci()} {time.perf_counter() - t0:.2f}s/{total:.2f}s")
        except Exception as e:  # never die: fall back to any observation-legal move
            log("ERROR", repr(e))
            pending.clear()
            moves = tracker.legal_moves()
            if not moves:
                return
            move = move or rng.choice(moves)
        try:
            if belief.boards:
                k = search.own_key_after(belief.boards[0], move, side)
                seen_after[k] = seen_after.get(k, 0) + 1
            belief.apply_own(move)
        except Exception as e:
            log("ERROR apply_own", repr(e))
        tracker.record_move(move)
        last_move = move
        print("move " + move.uci(), flush=True)


if __name__ == "__main__":
    main()
