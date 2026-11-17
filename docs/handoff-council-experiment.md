# Handoff: multi-advisor ("council") bot experiment

Written 2026-11-17 for continuing this work on another machine. Read this file first, then `README.md`.

## 1. Project state

- **Challenge:** MIT 6.7980 Fog of War Chess. Team **Tiger**. Arena: `https://6s890.lids.mit.edu`. Site: https://www.mit.edu/~6.7980/fow/
- **Starter kit:** in `starter/` (from https://www.mit.edu/~6.7980/fow/starter/starter.zip, GPL-2.0). `starter/README.md` has the full protocol, rules summary and runtime limits.
- **Bots:** `bots/baseline/` is an exact copy of the course `baseline.py` (captures first, otherwise random), plus `referee.py` and `run.sh`.
- **Arena submissions:** v1 and v2 are both the baseline. Both **validated, neither activated.** Team Elo is 1500 with 0 rated games. Do not activate the baseline: it would pull Elo down toward the reference team's 1000.
- **Submit tooling:** `scripts/submit.py` with `make package` and `make submit` (see README, "Submitting a bot"). Each bot needs a `MANIFEST` allowlist and a `run.sh`. Uploads stay inactive unless `ACTIVATE=1` is given.
- **Git:** still not initialized. Run `git init -b main` before committing, then `make doctor`.

## 2. Setting up the new machine

1. Copy the project folder **without** `.venv/`, `runs/` and `artifacts/` (all regenerable).
2. **Transfer the team token separately** over a secure channel; never put it in this file or in Git:
   ```sh
   cp .env.example .env    # then set FOW_TEAM_TOKEN=...
   chmod 600 .env
   ```
3. Create the environment. The server runs Python 3.14.8; 3.14.6 worked locally.
   ```sh
   uv venv --python 3.14 .venv
   uv pip install --python .venv/bin/python -r starter/requirements.txt   # python-chess==1.999, chess==1.11.2
   ```
4. Smoke test: a local game, baseline vs the random bot.
   ```sh
   cd bots/baseline
   ../../.venv/bin/python ../../starter/referee.py --trusted-local \
     --white 'bash run.sh' --black '../../.venv/bin/python ../../starter/uniform_random.py --seed 2' \
     --white-name W --black-name B --base-time 180 --increment 2 --max-plies 1000 \
     --output ../../runs/smoke.json
   ```
   Expected output looks like `1-0: King capturable; N plies`. In testing, the baseline beat the random bot in both colors.
   `run.sh` calls `python3`, so put `.venv/bin` first on `PATH` when testing locally, or install `chess` for the system `python3`.
5. Check uploads: `make package BOT=baseline` (no upload). `make submit` waits for validation, which took about 1 minute.

## 3. Facts about the game

- The game is two-player and zero-sum, with imperfect information. You see only your own observation, never the opponent's moves. Good play is randomized.
- **Protocol:** read `fow`, then `w|b <base> <inc>`, then print `ready` within 15 s.
  - Before every ply you get a 64-character observation (a1…h8; uppercase is White, `-` is a visible empty square, `?` is unseen).
  - Only on your turn you then get a clock line. Reply `move <uci>`.
  - Any other line on stdout forfeits the game. Use stderr for logging.
- **Rules that differ from chess:**
  - There is no check. A move that leaves your own king capturable loses immediately, and capturing the enemy king wins.
  - Castling through attacked squares is allowed.
  - Having no legal moves is a loss. There is no insufficient-material draw.
  - Draws: threefold repetition, 100 plies without a capture or pawn move, or 1,000 plies.
- **Limits:**
  - Clock: 180 s + 2 s per move.
  - Hardware: 2 cores / 4 threads, 4 GiB, no network, no GPU. The opponent is paused during your turn.
  - Build: 5 minutes.
  - Libraries provided: numpy, python-chess, CPU builds of torch, jax and onnxruntime. g++ and rustc are also available.
- **Elo:** K=8, one rating across all versions. Matchmaking favors opponents with similar Elo.

## 4. The experiment idea

The playing agent (principal) gets move recommendations from several advisors, each with its own objective, and is optimized even though no single advisor is aligned with winning. The theory comes from:

**Collina, Goel, Roth, Ryu, Shi, "Emergent Alignment via Competition", arXiv:2509.15090** (v2, Feb 2026).

- **Approximate market alignment** (their Definition 1): there exist weights w ≥ 0 with Σw = 1, a constant c, and an error ε such that `sup_{a,y} |Σ_i w_i U_i(a,y) + c − u_A(a,y)| ≤ ε`, where y is the hidden state.
- **Theorems 1 and 6:** in a Nash equilibrium among strategic advisors, the principal gets at least the utility of having one perfectly aligned advisor, minus 2ε. Theorem 6 ("Best-AI selection") needs no other assumption.
- **Theorem 5:** with a softmax (quantal) principal and advisors that exchange beliefs, the bound is roughly OPT − 2ε − O(K^-1/6 estimation) − log|A|/λ. It needs a common prior and the "information substitutes" condition.
- **Their experiments** (Section 6) fit the weights with nonnegative least squares (NNLS) over 100 advisors. That beat the best single advisor; simple averaging did poorly.
- **The paper gives no online or regret guarantees** and is "highly stylized" in the authors' own words.

**How it applies to us:**
- **Not directly:** our advisors are fixed heuristics, not strategic players, and they share the same observation (no private information). The ε bound must hold for every move and hidden board, which fails at king-capture cliffs. So treat the theorems as motivation, not as guarantees.
- **What carries over:** win value ≈ a nonnegative weighted sum of advisor scores plus a constant, which amounts to a learned, interpretable linear evaluation. Fit the weights with NNLS or black-box search. Use the paper's softmax principal; randomization is also right for an imperfect-information game. Use "ensemble vs best single advisor" as an empirical check that alignment holds within ε.
- **Later extension:** give the advisors different beliefs about the hidden board (each sampling hidden boards consistent with our observations in its own way). That brings back the paper's "different information" setting.

## 5. Plan (agreed next steps)

1. **Evaluation harness** (`scripts/arena_local.py`): play N games against a pool of opponents (baseline, uniform random, earlier council versions) in both colors with fixed seeds, in parallel. Record result, termination reason, plies and time used. Report win/draw/loss with confidence intervals. Write replays to `runs/`.
2. **Council bot** (`bots/council/`): start from `starter/uniform_random.py`, which already does en passant inference and castling rights through `ObservationTracker` and `candidates()`.
   - **Override first:** if a visible king capture exists, take it.
   - **Advisors**, each giving a score per legal move:
     - Material: capture value, minus a penalty for leaving pieces exposed.
     - Safety: penalize king exposure on lines we can't see.
     - Scout: number of `?` squares the move would reveal.
     - Pressure: attacks toward squares where the enemy king may be.
     - Mobility or development.
   - **Principal:** sample a move from softmax(λ · Σ wᵢ · zscore(sᵢ)). Normalize scores within each position so the weights are comparable.
   - Give weights and λ as command-line flags or a JSON file so they can be tuned without code changes.
3. **Ablations:** each advisor alone (one weight = 1), uniform weights, and hand-set weights, all against the baseline and the random bot.
4. **Fit weights:** use black-box search (random search or CMA-ES) on win rate, or NNLS on (advisor scores of the chosen move → final game result). Keep a held-out set of seeds.
5. **Submit:** add a `MANIFEST`, run `make submit BOT=council`, and activate only if it clearly beats the baseline locally.

**Watch out for:** stdout hygiene (log only to stderr); the 15 s init and per-move time budget; staying CPU-only and single-threaded unless benchmarked; no third-party packages beyond what the runtime provides unless they are put in the ZIP.
