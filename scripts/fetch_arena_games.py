#!/usr/bin/env python3
"""Download arena game replays (public endpoints, no token needed).

Run this on a machine that can reach the arena (the lab server cannot):
    python3 fetch_arena_games.py --out arena_games
then copy the folder to the lab server, e.g.
    rsync -av arena_games/ <server>:/mnt/nvme2tb/krittaphas/mit_67960_fow/runs/arena_games/

Only the Python standard library is used. Already-downloaded games are skipped,
so re-running fetches just the new ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ARENA = "https://6s890.lids.mit.edu"


def get(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "fow-replay-fetch"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arena_games")
    ap.add_argument("--arena", default=ARENA)
    ap.add_argument("--max-pages", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out) / "replays"
    out.mkdir(parents=True, exist_ok=True)

    listing, cursor, pages = [], None, 0
    while pages < args.max_pages:
        url = f"{args.arena}/api/games" + (f"?cursor={urllib.parse.quote(str(cursor))}" if cursor else "")
        page = get(url)
        games = page.get("games", [])
        listing += games
        pages += 1
        cursor = page.get("nextCursor")
        if not cursor or not games:
            break
    (Path(args.out) / "listing.json").write_text(json.dumps(listing, indent=1))
    print(f"{len(listing)} games listed", file=sys.stderr)

    new = 0
    for g in listing:
        gid = g.get("id") or g.get("game") or g.get("gameId")
        if gid is None:
            continue
        path = out / f"{gid}.json"
        if path.exists():
            continue
        try:
            rep = get(f"{args.arena}/api/games/{urllib.parse.quote(str(gid))}")
        except Exception as e:  # unfinished or removed games
            print(f"skip {gid}: {e}", file=sys.stderr)
            continue
        path.write_text(json.dumps(rep))
        new += 1
        time.sleep(0.2)  # be polite to the course server
    print(f"downloaded {new} new replays into {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
