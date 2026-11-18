"""Pure-numpy inference for the learned value net (see experiments/learned_eval).

value(boards) -> win probability for the side to move on each board, batched.
"""
from __future__ import annotations

import numpy as np
import chess

from belief import visible_mask

SYMS = "PNBRQK"
_MIRROR = np.array([chess.square_mirror(s) for s in range(64)])


class ValueNet:
    def __init__(self, path):
        w = np.load(path)
        self.W1, self.b1 = w["l1.weight"].T.copy(), w["l1.bias"]
        self.W2, self.b2 = w["l2.weight"].T.copy(), w["l2.bias"]
        self.W3, self.b3 = w["l3.weight"].T.copy(), w["l3.bias"]

    @staticmethod
    def features(b: chess.Board) -> np.ndarray:
        us = b.turn
        x = np.zeros(14 * 64, np.float32)
        flip = us == chess.BLACK
        for sq, p in b.piece_map().items():
            s = chess.square_mirror(sq) if flip else sq
            x[((0 if p.color == us else 6) + SYMS.index(p.symbol().upper())) * 64 + s] = 1.0
        for i, side in enumerate((us, not us)):
            v = visible_mask(b, side)
            while v:
                sq = (v & -v).bit_length() - 1
                v &= v - 1
                x[(12 + i) * 64 + (chess.square_mirror(sq) if flip else sq)] = 1.0
        return x

    def value(self, boards) -> np.ndarray:
        if not boards:
            return np.zeros(0, np.float32)
        X = np.stack([self.features(b) for b in boards])
        h = np.clip(X @ self.W1 + self.b1, 0.0, 1.0)
        h = np.clip(h @ self.W2 + self.b2, 0.0, 1.0)
        lo = (h @ self.W3 + self.b3)[:, 0]
        return 1.0 / (1.0 + np.exp(-lo))
