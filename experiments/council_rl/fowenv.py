"""In-process Fog-of-War chess environment for RL experiments.

Rules come straight from starter/referee.py (FowBoard, fow_moves, observation,
outcome, position_key). The agent plays White; the opponent (Black) is the
course baseline (capture-highest-value, else random) or uniform random, both run
in-process on Black's own observation.

Per agent step we return a vector of advisor rewards computed from the TRUE board
(privileged information is allowed for training rewards only). The network input
is built only from observation strings.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np

STARTER = Path(__file__).resolve().parents[2] / "starter"
sys.path.insert(0, str(STARTER))
from referee import FowBoard, fow_moves, observation, outcome, position_key  # noqa: E402
from baseline import candidates as baseline_candidates  # noqa: E402

HEADS = ("material", "safety", "scout", "pressure", "win")
NH = len(HEADS)
SYMBOLS = "PNBRQKpnbrqk?-"
SYM_INDEX = {c: i for i, c in enumerate(SYMBOLS)}
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
BASE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
MAX_PLIES = 300
N_PLANES = 28  # current + previous own-turn observation, 14 planes each


def encode_obs(obs: str) -> np.ndarray:
    planes = np.zeros((14, 64), dtype=np.uint8)
    idx = np.fromiter((SYM_INDEX[c] for c in obs), dtype=np.int64, count=64)
    planes[idx, np.arange(64)] = 1
    return planes.reshape(14, 8, 8)


def action_index(move: chess.Move) -> int:
    # 4096 from-to encoding; index = to * 64 + from (network head is [to, from]).
    return move.to_square * 64 + move.from_square


def legal_actions(board: FowBoard):
    """Queen-only promotions, deduplicated from-to moves for the side to move."""
    out = {}
    for m in fow_moves(board):
        if m.promotion and m.promotion != chess.QUEEN:
            continue
        out[action_index(m)] = m
    return out


def material(board: FowBoard) -> int:
    s = 0
    for pt, v in VALUES.items():
        s += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return s


def zone(sq):
    return chess.SquareSet(chess.BB_KING_ATTACKS[sq]) | chess.SquareSet(chess.BB_SQUARES[sq])


def king_zone_attacks(board: FowBoard, attacker: chess.Color, king_color: chess.Color) -> int:
    k = board.king(king_color)
    if k is None:
        return 0
    return sum(len(board.attackers(attacker, sq)) for sq in zone(k))


class BaselineOpponent:
    """starter/baseline.py logic on Black's observation (no en passant)."""

    def __init__(self, rng: random.Random, uniform: bool = False):
        self.rng = rng
        self.uniform = uniform
        self.rights = chess.BB_SQUARES[chess.A8] | chess.BB_SQUARES[chess.H8]

    def act(self, obs: str):
        if obs[chess.E8] != "k":
            self.rights = 0
        for rook in (chess.A8, chess.H8):
            if obs[rook] != "r":
                self.rights &= ~chess.BB_SQUARES[rook]
        board, moves = baseline_candidates(obs, chess.BLACK, self.rights)
        if not moves:
            return None
        if self.uniform:
            move = self.rng.choice(moves)
        else:
            def score(m):
                t = board.piece_at(m.to_square)
                v = BASE_VALUES[t.piece_type] * 10 if t else 0
                if m.promotion:
                    v += BASE_VALUES[m.promotion] * 5
                return v + self.rng.random()
            move = max(moves, key=score)
        if board.piece_at(move.from_square).piece_type == chess.KING:
            self.rights = 0
        self.rights &= ~chess.BB_SQUARES[move.from_square]
        return move


class FowEnv:
    def __init__(self, seed: int, opponent: str = "baseline"):
        self.rng = random.Random(seed)
        self.opponent_kind = opponent

    # ---- helpers -------------------------------------------------------
    def _terminal(self):
        if len(self.board.move_stack) >= MAX_PLIES:
            return ("1/2-1/2", "Move limit")
        return outcome(self.board, self.reps)

    def _push(self, move):
        self.board.push(move)
        self.reps[position_key(self.board)] += 1

    def _features(self):
        obs = observation(self.board, chess.WHITE)
        cur = encode_obs(obs)
        prev = self.prev_planes if self.prev_planes is not None else cur
        self.prev_planes = cur
        self.last_obs = obs
        x = np.concatenate([cur, prev], axis=0)
        legal = legal_actions(self.board)
        mask = np.zeros(4096, dtype=bool)
        mask[list(legal)] = True
        self.legal = legal
        return x, mask

    # ---- API -------------------------------------------------------------
    def reset(self):
        self.board = FowBoard()
        self.reps = Counter({position_key(self.board): 1})
        self.opp = BaselineOpponent(self.rng, uniform=self.opponent_kind == "random")
        self.prev_planes = None
        return self._features()

    def step(self, a: int):
        """Returns (x, mask, rewards[NH], done, info). x/mask are of the next own turn
        (meaningless if done)."""
        b = self.board
        move = self.legal[a]
        mat0 = material(b)
        obs0 = self.last_obs
        r = np.zeros(NH, dtype=np.float32)
        self._push(move)
        term = self._terminal()
        if term is None:
            opp_obs = observation(b, chess.BLACK)
            om = self.opp.act(opp_obs)
            if om is None:  # cannot happen when outcome() said non-terminal, but be safe
                term = ("1-0", "No legal moves")
            else:
                self._push(om)
                term = self._terminal()
        # dense advisor rewards from the true board
        r[0] = (material(b) - mat0) / 10.0
        r[1] = -king_zone_attacks(b, chess.BLACK, chess.WHITE) / 10.0
        r[3] = king_zone_attacks(b, chess.WHITE, chess.BLACK) / 20.0
        if term is not None:
            res, reason = term
            z = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
            r[4] = z
            if z < 0 and reason == "King capturable":
                r[1] -= 1.0
            return None, None, r, True, {"result": res, "reason": reason, "plies": len(b.move_stack)}
        x, mask = self._features()
        new_obs = self.last_obs
        revealed = sum(1 for o, n in zip(obs0, new_obs) if o == "?" and n != "?")
        r[2] = revealed / 20.0
        return x, mask, r, False, {}
