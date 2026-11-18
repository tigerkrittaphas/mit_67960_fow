"""Move choice for v3: evaluate each legal move on a sample of belief boards.

Per (move, board):
  * our king capturable after the move            -> LOSS
  * enemy king attacked and every reply leaves it
    attacked (no reply can save it)               -> WIN
  * otherwise static eval after the move minus the opponent's best capture
    gain (a cheap one-reply exchange estimate), plus a bonus for attacking
    the enemy king.
Moves are then filtered by loss probability (hard safety veto) and ranked by
mean value; we mix uniformly among up to `topk` moves within `margin` of best.
"""
from __future__ import annotations

import os
import time

import numpy as np

import chess

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.2, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
WIN, LOSS = 50.0, -50.0


def popcount(x: int) -> int:
    return x.bit_count()


def material(b: chess.Board, side: chess.Color) -> float:
    s = 0.0
    for pt, v in VAL.items():
        if v:
            s += v * (popcount(b.pieces_mask(pt, side)) - popcount(b.pieces_mask(pt, not side)))
    return s


def king_capturable(b: chess.Board, victim: chess.Color) -> bool:
    k = b.king(victim)
    return k is None or b.is_attacked_by(not victim, k)


def pawn_advance(b: chess.Board, side: chess.Color) -> float:
    s = 0.0
    for sq in b.pieces(chess.PAWN, side):
        r = chess.square_rank(sq)
        s += (r - 1) if side == chess.WHITE else (6 - r)
    return s


def king_zone_pressure(b: chess.Board, attacker: chess.Color) -> int:
    k = b.king(not attacker)
    if k is None:
        return 0
    n = 0
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[k]):
        if b.is_attacked_by(attacker, sq):
            n += 1
    return n


def threat(b: chess.Board, victim: chess.Color) -> float:
    """Largest material the side to move (= not victim) can win with one capture,
    counting a recapture if the target square is defended."""
    best = 0.0
    attacker = not victim
    for sq in chess.SquareSet(b.occupied_co[victim] & ~b.kings):
        atk = b.attackers_mask(attacker, sq)
        if not atk:
            continue
        v = VAL[b.piece_type_at(sq)]
        cheapest = min(VAL[b.piece_type_at(a)] or 100.0 for a in chess.SquareSet(atk))
        gain = v - (cheapest if b.is_attacked_by(victim, sq) else 0.0)
        if gain > best:
            best = gain
    return best


def static_eval(b: chess.Board, side: chess.Color) -> float:
    return (material(b, side)
            + 0.05 * (pawn_advance(b, side) - pawn_advance(b, not side))
            + 0.15 * (king_zone_pressure(b, side) - king_zone_pressure(b, not side)))


def forced_king_win(b: chess.Board, side: chess.Color) -> bool:
    """b: after our move, opponent to move, their king attacked by us.
    True if every opponent reply still leaves their king capturable."""
    for r in list(b.generate_pseudo_legal_moves()):
        b.push(r)
        safe_for_them = not king_capturable(b, not side)
        b.pop()
        if safe_for_them:
            return False
    return True


def evaluate_move(b: chess.Board, m: chess.Move, side: chess.Color) -> float:
    b.push(m)
    try:
        if king_capturable(b, side):
            return LOSS
        if king_capturable(b, not side):
            if forced_king_win(b, side):
                return WIN
            bonus = 1.0  # opponent must see and answer the attack
        else:
            bonus = 0.0
        return static_eval(b, side) - threat(b, side) + bonus
    finally:
        b.pop()


def loss_rates(all_boards, moves, side, deadline):
    """Fraction of ALL belief boards on which each move leaves our king capturable.
    Rare boards matter here (an unseen queen check), so we do not subsample."""
    bad = {m: 0 for m in moves}
    n = 0
    for b in all_boards:
        if n and time.perf_counter() > deadline:
            break
        n += 1
        for m in moves:
            if not b.is_pseudo_legal(m):
                continue
            b.push(m)
            if king_capturable(b, side):
                bad[m] += 1
            b.pop()
    return {m: bad[m] / max(1, n) for m in moves}, n


def choose(boards, moves, side, rng, deadline, topk=3, margin=0.15, last_move=None,
           voi_fn=None, voi_k=5, voi_beta=0.5, all_boards=None, safety_deadline=None):
    """Returns (move, info). `moves` are the observation-legal moves."""
    full_loss, n_safety = (loss_rates(all_boards, moves, side, safety_deadline or deadline)
                           if all_boards is not None else (None, 0))
    stats = {m: [0.0, 0, 0] for m in moves}  # sum, n, losses
    leaves, owners = [], []
    processed = 0
    for b in boards:
        if processed and time.perf_counter() > deadline:
            break
        processed += 1
        for m in moves:
            if not b.is_pseudo_legal(m):
                continue
            st = stats[m]
            b.push(m)
            if king_capturable(b, side):
                v = LOSS
            elif king_capturable(b, not side) and forced_king_win(b, side):
                v = WIN
            else:
                v = None
                if VALUE_NET is not None:
                    leaves.append(b.copy(stack=False)); owners.append(m)
                    st[0] += -threat(b, side) + (1.0 if king_capturable(b, not side) else 0.0)
                else:
                    v = static_eval(b, side) - threat(b, side) + (1.0 if king_capturable(b, not side) else 0.0)
            b.pop()
            if v is not None:
                st[0] += v; st[1] += 1; st[2] += v == LOSS
    if leaves:
        p_opp = VALUE_NET.value(leaves)          # win prob for the side to move (opponent)
        p_opp = np.clip(p_opp, 1e-4, 1 - 1e-4)
        vals = np.log((1 - p_opp) / p_opp) * LOGIT_SCALE   # our value in ~pawn units
        for m, v in zip(owners, vals):
            st = stats[m]; st[0] += float(v); st[1] += 1
    scored = []
    for m, (s, n, losses) in stats.items():
        if n == 0:
            continue
        mean = s / n
        if last_move is not None and m.from_square == last_move.to_square and m.to_square == last_move.from_square:
            mean -= 0.05  # discourage shuffling back and forth
        pl = full_loss[m] if full_loss is not None else losses / n
        scored.append((pl, -mean, m))
    if not scored:
        return moves[0], {}
    min_loss = min(x[0] for x in scored)
    safe = [x for x in scored if x[0] <= min_loss + 1e-9]
    safe.sort(key=lambda x: x[1])
    info_extra = {}
    if voi_fn is not None and len(safe) > 1:
        bonus = {}
        for _, negmean, m in safe[:voi_k]:
            if time.perf_counter() > deadline:
                break
            bonus[m] = voi_fn(m)
        if bonus:
            safe = [(pl, nm - voi_beta * bonus.get(m, 0.0), m) for pl, nm, m in safe]
            safe.sort(key=lambda x: x[1])
            info_extra = dict(voi_max=round(max(bonus.values()), 3), voi_n=len(bonus))
    best = -safe[0][1]
    pool = [x[2] for x in safe if -x[1] >= best - margin][:topk]
    move = rng.choice(pool)
    return move, dict(n_boards=len(boards), n_safety=n_safety, best=round(best, 3), p_loss=round(min_loss, 3), pool=len(pool), **info_extra)


LOGIT_SCALE = 2.5   # map logit to roughly pawn units so margins/sentinels keep their meaning
try:
    from valuenet import ValueNet
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("V4_VALUE", "value.npz"))
    VALUE_NET = ValueNet(_p) if os.path.exists(_p) else None
except Exception:
    VALUE_NET = None
