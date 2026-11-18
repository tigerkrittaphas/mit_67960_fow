"""Build a value-net dataset from harness replays (run with --replays all).

For every non-terminal position in every replay we store, from the side to
move's perspective (board flipped for Black so "us" is always White):
  x_board : 12x64 piece planes (us P..K, them P..K)
  x_vis   : 2x64 (squares we can see, squares they can see)
  z       : final game result for the side to move (+1/0/-1)
  sf      : Stockfish win probability for the side to move (depth D), 0..1
Stockfish labels are computed in parallel worker processes.

Usage:
  .venv/bin/python experiments/learned_eval/build_dataset.py runs/arena/<dir> [more dirs] \
      --out runs/learned_eval/data.npz --depth 8 --jobs 24
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bots/v3"))
import chess  # noqa: E402
import chess.engine  # noqa: E402
from belief import visible_mask  # noqa: E402
from referee import FowBoard  # noqa: E402

SF = str(ROOT / ".tools/stockfish")
SYMS = "PNBRQK"


def board_from(placement: str, turn: bool) -> FowBoard:
    b = FowBoard(None)
    for sq, c in enumerate(placement):
        if c != "-":
            b.set_piece_at(sq, chess.Piece.from_symbol(c))
    b.turn = turn
    return b


def features(b: FowBoard):
    """Planes from the side to move's perspective (mirror if Black to move)."""
    us = b.turn
    xb = np.zeros((12, 64), np.uint8)
    for sq, p in b.piece_map().items():
        s = sq if us == chess.WHITE else chess.square_mirror(sq)
        xb[(0 if p.color == us else 6) + SYMS.index(p.symbol().upper()), s] = 1
    xv = np.zeros((2, 64), np.uint8)
    for i, side in enumerate((us, not us)):
        v = visible_mask(b, side)
        for sq in chess.SquareSet(v):
            xv[i, sq if us == chess.WHITE else chess.square_mirror(sq)] = 1
    return xb, xv


def positions_from_replay(path):
    r = json.load(open(path))
    res = r.get("result")
    if res not in ("1-0", "0-1", "1/2-1/2"):
        return []
    zw = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
    out = []
    for f in r["frames"][:-1]:          # the last frame is terminal
        turn = f["ply"] % 2 == 0
        b = board_from(f["board"], turn)
        k = b.king(not turn)
        if k is None or b.king(turn) is None or b.is_attacked_by(turn, k):
            continue                      # terminal (side to move can capture the king)
        out.append((b.board_fen(), turn, zw if turn else -zw))
    return out


def sf_worker(args):
    chunk, depth = args
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})
    out = []
    for fen, turn, z in chunk:
        b = chess.Board(None)
        b.set_board_fen(fen)
        b.turn = turn
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=depth))
            sc = info["score"].pov(turn)
            if sc.is_mate():
                p = 1.0 if sc.mate() > 0 else 0.0
            else:
                p = 1 / (1 + math.exp(-sc.score() / 250.0))
        except Exception:
            p = float("nan")
            eng.quit()
            eng = chess.engine.SimpleEngine.popen_uci(SF)
        fb = FowBoard(None)
        fb.set_board_fen(fen)
        fb.turn = turn
        xb, xv = features(fb)
        out.append((xb, xv, z, p))
    eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--max-positions", type=int, default=5_000_000)
    args = ap.parse_args()
    files = [f for d in args.dirs for f in glob.glob(str(Path(d) / "replays" / "*.json"))]
    pos = []
    for f in files:
        pos += positions_from_replay(f)
        if len(pos) >= args.max_positions:
            break
    seen, uniq = set(), []
    for p in pos:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1])); uniq.append(p)
    print(f"{len(files)} replays, {len(pos)} positions, {len(uniq)} unique", flush=True)
    chunks = [(uniq[i:i + 2000], args.depth) for i in range(0, len(uniq), 2000)]
    xb, xv, z, sf = [], [], [], []
    with mp.Pool(args.jobs) as pool:
        for k, res in enumerate(pool.imap_unordered(sf_worker, chunks), 1):
            for a, b_, c, d in res:
                xb.append(a); xv.append(b_); z.append(c); sf.append(d)
            print(f"chunk {k}/{len(chunks)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, xb=np.packbits(np.array(xb), axis=-1), xv=np.packbits(np.array(xv), axis=-1),
                        z=np.array(z, np.float32), sf=np.array(sf, np.float32))
    print("saved", args.out, len(z))


if __name__ == "__main__":
    main()
