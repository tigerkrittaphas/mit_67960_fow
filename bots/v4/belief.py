"""Belief tracking for Fog-of-War chess: the set of full boards consistent with
every observation received so far.

The referee sends one observation before every ply (both players' plies).
Usage per received observation at ply k (k = 0, 1, 2, ...):
    belief.observe(obs)          # expands opponent moves if the opponent moved last
and after choosing our own move:
    belief.apply_own(move)

The set is capped; when the cap binds we keep a random subset (so the true
board can be lost). If no board survives a filter we rebuild the set by
sampling hidden-piece placements consistent with the current observation.
"""
from __future__ import annotations

import random
import time

import chess

from referee import FowBoard, fow_moves

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
START_COUNTS = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}


def key(b: chess.Board):
    return (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
            b.occupied_co[chess.WHITE], b.turn, b.castling_rights, b.ep_square)


class ObsMasks:
    """Bitboard view of one 64-char observation."""

    __slots__ = ("visible", "empty", "masks", "obs")

    def __init__(self, obs: str):
        self.obs = obs
        self.visible = 0
        self.empty = 0
        m = {}
        for sq, c in enumerate(obs):
            if c == "?":
                continue
            bit = 1 << sq
            self.visible |= bit
            if c == "-":
                self.empty |= bit
            else:
                p = chess.Piece.from_symbol(c)
                m[(p.color, p.piece_type)] = m.get((p.color, p.piece_type), 0) | bit
        self.masks = tuple(m.get((c, pt), 0) for c in (chess.WHITE, chess.BLACK) for pt in PIECE_TYPES)

    def pieces_match(self, b: chess.Board) -> bool:
        v = self.visible
        if b.occupied & self.empty:
            return False
        i = 0
        for c in (chess.WHITE, chess.BLACK):
            occ = b.occupied_co[c]
            for pt in PIECE_TYPES:
                if (b.pieces_mask(pt, c) & v) != self.masks[i]:
                    return False
                i += 1
        return True


def visible_mask(b: FowBoard, side: chess.Color) -> int:
    """Exactly referee.observation()'s visible set, without copying the board."""
    turn, ep = b.turn, b.ep_square
    if side != turn:
        b.turn = side
        b.ep_square = None
    vis = b.occupied_co[side]
    for m in b.generate_pseudo_legal_moves():
        vis |= 1 << m.to_square
        if b.is_en_passant(m):
            vis |= 1 << (m.to_square - 8 if side else m.to_square + 8)
    b.turn, b.ep_square = turn, ep
    return vis


def consistent(b: FowBoard, om: ObsMasks, side: chess.Color) -> bool:
    return om.pieces_match(b) and visible_mask(b, side) == om.visible


class Belief:
    def __init__(self, side: chess.Color, cap: int = 3000, seed: int = 0):
        self.side = side
        self.cap = cap
        self.rng = random.Random(seed)
        self.boards: list[FowBoard] = [FowBoard()]
        self.ply = -1              # ply index of the last observation
        self.state_ply = 0         # ply index the boards in self.boards correspond to
        self.capped = False        # True once a cap or a rebuild has made the set approximate
        self.captured_by_us = {pt: 0 for pt in PIECE_TYPES}
        self.stats = []

    # -------------------------------------------------------------- updates
    def observe(self, obs: str, deadline: float | None = None):
        t0 = time.perf_counter()
        self.ply += 1
        om = ObsMasks(obs)
        expanded = False
        if self.state_ply < self.ply:
            # The previous ply was the opponent's move: expand all their moves.
            assert self.state_ply == self.ply - 1
            self._expand_filter(om, deadline)
            self.state_ply = self.ply
            expanded = True
        else:
            self.boards = [b for b in self.boards if consistent(b, om, self.side)]
        rebuilt = False
        if not self.boards:
            self._rebuild(om)
            rebuilt = True
        self.stats.append(dict(ply=self.ply, n=len(self.boards), expanded=expanded, rebuilt=rebuilt,
                               capped=self.capped, sec=time.perf_counter() - t0))

    def apply_own(self, move: chess.Move):
        """Push our move (legal on every consistent board) to every board."""
        out, seen = [], set()
        for b in self.boards:
            nb = b.copy(stack=False)
            nb.push(move)
            k = key(nb)
            if k not in seen:
                seen.add(k); out.append(nb)
        # record what we captured (identical on all boards: target square is visible)
        b0 = self.boards[0]
        if b0.is_en_passant(move):
            self.captured_by_us[chess.PAWN] += 1
        else:
            t = b0.piece_at(move.to_square)
            if t is not None and t.color != self.side:
                self.captured_by_us[t.piece_type] += 1
        self.boards = out
        self.state_ply = self.ply + 1

    def _expand_filter(self, om: ObsMasks, deadline):
        """Expand every parent by all opponent moves and keep consistent children.

        If the result would exceed the cap, keep at most `per` children per parent
        (random), so the kept set stays spread over many parents."""
        parents = self.boards
        self.rng.shuffle(parents)
        per = max(1, self.cap // max(1, len(parents)))
        out, seen = [], set()
        side = self.side
        for i, b in enumerate(parents):
            if deadline and out and time.perf_counter() > deadline:
                self.capped = True
                break
            kids = []
            for m in list(b.generate_pseudo_legal_moves()):
                b.push(m)
                if om.pieces_match(b):
                    k = key(b)
                    if k not in seen:
                        seen.add(k)
                        if visible_mask(b, side) == om.visible:
                            kids.append(b.copy(stack=False))
                b.pop()
            if len(kids) > per and len(parents) * len(kids) > self.cap:
                self.capped = True
                kids = self.rng.sample(kids, per)
            out += kids
        if len(out) > self.cap:
            self.capped = True
            out = self.rng.sample(out, self.cap)
        self.boards = out

    # -------------------------------------------------------------- recovery
    def _rebuild(self, om: ObsMasks, want: int = 300, tries: int = 30000):
        """Sample hidden enemy pieces onto unseen squares until consistent."""
        self.capped = True
        enemy = not self.side
        obs = om.obs
        visible_enemy = {pt: 0 for pt in PIECE_TYPES}
        for c in obs:
            if c not in "?-":
                p = chess.Piece.from_symbol(c)
                if p.color == enemy:
                    visible_enemy[p.piece_type] += 1
        hidden = []
        for pt in PIECE_TYPES:
            n = START_COUNTS[pt] - self.captured_by_us[pt] - visible_enemy[pt]
            hidden += [pt] * max(0, n)
        if visible_enemy[chess.KING] == 0 and chess.KING not in hidden:
            hidden.append(chess.KING)
        unseen = [sq for sq, c in enumerate(obs) if c == "?"]
        turn = chess.WHITE if self.ply % 2 == 0 else chess.BLACK
        out, seen = [], set()
        for _ in range(tries):
            if len(out) >= want:
                break
            b = FowBoard(None)
            for sq, c in enumerate(obs):
                if c not in "?-":
                    b.set_piece_at(sq, chess.Piece.from_symbol(c))
            free = unseen[:]
            self.rng.shuffle(free)
            pieces = sorted(hidden, key=lambda pt: pt != chess.KING)  # king first
            ok = True
            for pt in pieces:
                placed = False
                while free:
                    sq = free.pop()
                    if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                        continue
                    b.set_piece_at(sq, chess.Piece(pt, enemy))
                    placed = True
                    break
                if not placed and pt == chess.KING:
                    ok = False
                    break
            if not ok:
                continue
            b.turn = turn
            b.castling_rights = chess.BB_CORNERS
            b.castling_rights = b.clean_castling_rights()
            k = key(b)
            if k in seen:
                continue
            if om.pieces_match(b) and visible_mask(b, self.side) == om.visible:
                # side to move must not already be able to capture the other king
                ok_king = b.king(not b.turn)
                if ok_king is not None and b.is_attacked_by(b.turn, ok_king):
                    continue
                seen.add(k); out.append(b)
        if not out:  # last resort: no hidden pieces at all beyond the king
            raise RuntimeError("belief rebuild failed")
        self.boards = out

    # -------------------------------------------------------------- sampling
    def sample(self, n: int):
        if len(self.boards) <= n:
            return list(self.boards)
        return self.rng.sample(self.boards, n)
