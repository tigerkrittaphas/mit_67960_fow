# Challenge work plan

## Initialization — 2026-11-05

Team name: Tiger. Registration was confirmed in the staff email supplied by the user. The token is stored only in the local `.env` file, covered by `.gitignore`.

Git ignore rules have been written, but repository initialization was blocked by the session's read-only `.git` policy. Run `git init -b main` in a normal terminal, then `make doctor` to verify exclusion of the credential file.

The [course project description](https://www.mit.edu/~6.7980/) describes a chess agent operating under partial information and lists a sandbox allocation of two CPU cores and 4 GiB of memory. Recheck these limits against the challenge instructions before choosing an approach.

The rules and submission pages have not yet been verified. Do not assume that standard chess legality, python-chess, ReconChess, or another fog-of-war implementation matches this tournament.

## 1. Verify the official contract

- Read the [rules](https://www.mit.edu/~6.7980/fow/?tab=rules) and [submission instructions](https://www.mit.edu/~6.7980/fow/?tab=submit).
- Obtain the official starter kit and record its source URL and version or checksum.
- Confirm observations, hidden information, move legality, invalid-move behavior, captures, promotions, castling, en passant, terminal conditions, and draws.
- Confirm process lifecycle, input/output schema, persistent state, randomness, time budget, CPU/memory limits, and permitted dependencies.
- Confirm container architecture/runtime, entry point, archive format, upload endpoint, authentication mechanism, validation flow, and submission limits.
- Record competition deadlines, scoring, and collaboration/AI-use rules.

## 2. Establish a runnable baseline

- Import the official starter without changing its protocol.
- Create an isolated environment matching its runtime and lock dependencies.
- Run the official example locally and preserve it as a comparison baseline.
- Add a simple agent that returns valid actions, plus a full-game smoke check using the official harness.
- Verify packaging excludes `.env`, credentials, development caches, and experiment logs.

## 3. Build repeatable evaluation

- Use fixed seed sets and play both colors when supported.
- Record agent version, opponent, seed, outcome, invalid actions, latency, and termination reason.
- Compare candidate changes against the baseline on held-out seeds.
- Save replays for losses and protocol failures in `runs/`.

## 4. Improve the agent

Candidate directions, subject to the actual rules: maintain beliefs about hidden pieces; prioritize king captures and threats; assess material, mobility, and information gain; then investigate sampled-state search or learning. First measure which failure modes dominate baseline losses.

## 5. Validate and submit

- Run the official local validator with production resource limits.
- Build and inspect the final artifact.
- Perform server-side validation and tournament submission when requested.
- Record the submission ID, artifact hash, code revision, and validation result.
