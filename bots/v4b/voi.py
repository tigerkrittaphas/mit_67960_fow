"""Value of information for v3.1 (observation-aware one-step lookahead).

For a candidate move m and sampled boards B:
  after m, each board b produces an observation o(b) for us. Boards with the same
  observation are indistinguishable to us afterwards, so our next move can depend
  on o but not on b. With V2(b, m2) the static value of our follow-up m2 on b
  (opponent's intermediate move ignored: a "null-move" approximation),

    VOI(m) = sum_g P(g) * max_{m2 legal in g} mean_{b in g} V2(b, m2)
             - max_{m2} mean_{b in B} V2(b, m2)          (one shared choice)

  VOI >= 0 and is large only when what m reveals changes which follow-up is best.
No reward for "seeing more" per se.
"""
from __future__ import annotations

import time

import chess

from belief import visible_mask
from obsmoves import candidates
import search


def obs_string(b: chess.Board, vis: int) -> str:
    out = []
    for sq in range(64):
        if not (vis >> sq) & 1:
            out.append("?")
        else:
            p = b.piece_at(sq)
            out.append(p.symbol() if p else "-")
    return "".join(out)


def follow_up_values(boards, side, rights, obs, deadline):
    """mean V2 per observation-legal follow-up move over `boards` (all share `obs`)."""
    _, moves = candidates(obs, side, rights)
    best = None
    for m2 in moves:
        s = n = 0
        for b in boards:
            if not b.is_pseudo_legal(m2):
                continue
            s += search.evaluate_move(b, m2, side); n += 1
        if n:
            v = s / n
            if best is None or v > best:
                best = v
        if time.perf_counter() > deadline:
            break
    return best


def voi(boards, m, side, rights, deadline):
    """boards: sampled boards with us to move. Returns VOI(m) >= 0 (or 0 on timeout)."""
    after = []
    for b in boards:
        if not b.is_pseudo_legal(m):
            continue
        nb = b.copy(stack=False)
        nb.push(m)
        if search.king_capturable(nb, side):
            continue                     # those boards are losses for m anyway
        nb.turn = side                   # null move for the opponent
        nb.ep_square = None
        after.append(nb)
    if len(after) < 2:
        return 0.0
    groups = {}
    for nb in after:
        groups.setdefault(obs_string(nb, visible_mask(nb, side)), []).append(nb)
    if len(groups) == 1:
        return 0.0
    r2 = rights
    shared = follow_up_values(after, side, r2, _merge_obs(groups), deadline)
    if shared is None:
        return 0.0
    total = 0.0
    for o, g in groups.items():
        v = follow_up_values(g, side, r2, o, deadline)
        if v is None:
            return 0.0
        total += len(g) / len(after) * v
        if time.perf_counter() > deadline:
            return 0.0
    return min(3.0, max(0.0, total - shared))  # clip: null-move approx overstates big swings


def _merge_obs(groups):
    """Observation with every square that differs between groups hidden: the moves
    legal in it are those we could choose without knowing which group we are in."""
    keys = list(groups)
    base = list(keys[0])
    for o in keys[1:]:
        for i, c in enumerate(o):
            if base[i] != c:
                base[i] = "?"
    return "".join(base)
