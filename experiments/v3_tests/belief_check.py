"""Offline belief-tracker validation on in-process games.

Movers: 'safe' = random move that does not leave own king capturable on the true
board (long games); 'base' = capture-greedy else safe. Checks, for each side,
whether the true board is in the belief set at every ply.
"""
import sys, random, time, json, statistics
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bots/v3"))
import chess
from referee import FowBoard, fow_moves, observation, outcome, position_key
from belief import Belief, key

VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}

def mover(board, rng, kind):
    ms = fow_moves(board)
    safe = []
    for m in ms:
        board.push(m)
        k = board.king(not board.turn)
        if not (k is not None and board.is_attacked_by(board.turn, k)):
            safe.append(m)
        board.pop()
    pool = safe or ms
    if kind == "base":
        caps = [m for m in pool if board.piece_at(m.to_square)]
        if caps:
            return max(caps, key=lambda m: VAL[board.piece_at(m.to_square).piece_type] + rng.random())
    return rng.choice(pool)

def play(seed, kind, max_plies=300):
    rng = random.Random(seed)
    b = FowBoard(); reps = Counter({position_key(b): 1}); frames = []
    while True:
        frames.append((b.copy(stack=False), observation(b, chess.WHITE), observation(b, chess.BLACK)))
        if outcome(b, reps) or len(b.move_stack) >= max_plies:
            return frames, [m for m in b.move_stack]
        m = mover(b, rng, kind)
        b.push(m); reps[position_key(b)] += 1

def check(frames, moves, side, cap):
    bel = Belief(side, cap=cap, seed=1)
    inc_unc = inc_cap = n_unc = n_cap = 0
    for ply, (truth, wobs, bobs) in enumerate(frames):
        bel.observe(wobs if side else bobs)
        ks = {key(x) for x in bel.boards}
        hit = key(truth) in ks
        if bel.capped: n_cap += 1; inc_cap += hit
        else: n_unc += 1; inc_unc += hit
        if not hit and not bel.capped:
            return dict(error=f"true board lost while uncapped at ply {ply}")
        if ply < len(moves) and truth.turn == side:
            bel.apply_own(moves[ply])
    return dict(rebuilds=sum(s["rebuilt"] for s in bel.stats), n_unc=n_unc, inc_unc=inc_unc, n_cap=n_cap, inc_cap=inc_cap,
                sizes=[s["n"] for s in bel.stats], secs=[s["sec"] for s in bel.stats])

if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    agg = Counter(); sizes = []; secs = []; errors = []; lengths = []
    for g in range(games):
        kind = "safe" if g % 2 == 0 else "base"
        frames, moves = play(1000 + g, kind)
        lengths.append(len(moves))
        for side in (chess.WHITE, chess.BLACK):
            r = check(frames, moves, side, cap)
            if "error" in r: errors.append((g, side, r["error"])); continue
            for k in ("rebuilds", "n_unc", "inc_unc", "n_cap", "inc_cap"): agg[k] += r[k]
            sizes += r["sizes"]; secs += r["secs"]
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    print(json.dumps(dict(cap=cap, games=games, mean_len=statistics.mean(lengths),
        uncapped_inclusion=f'{agg["inc_unc"]}/{agg["n_unc"]}', capped_inclusion=f'{agg["inc_cap"]}/{agg["n_cap"]}', rebuilds=agg["rebuilds"],
        size_p50=q(sizes, .5), size_p90=q(sizes, .9), size_max=max(sizes),
        sec_p50=round(q(secs, .5), 4), sec_p95=round(q(secs, .95), 4), sec_max=round(max(secs), 3), errors=errors[:5]), indent=1))
