# Tiger — MIT 6.7980 Fog of War Challenge

Local workspace for developing and evaluating our tournament agent.

## Current setup

- Git ignore rules are ready. Repository initialization is pending because this session cannot write `.git`; no remote or commits were created.
- Team credentials are in `.env` (ignored by Git, owner-only permissions).
- The official starter kit is in `starter/`. The course baseline (`bots/baseline/`) has been uploaded and validated but is not active.

Run the local readiness check:

```sh
python3 scripts/doctor.py
# or
make doctor
```

Run `git init -b main` in your own terminal first. This session's filesystem policy makes `.git` read-only, so the readiness check reports a failure until the repository exists and credential exclusion can be verified.

Python, uv, and the Docker CLI are available on this machine. The official Python/runtime version and dependencies have not been verified, so no environment or dependency versions have been selected yet. The Docker daemon has not been checked.

## Official references

- [Rules](https://www.mit.edu/~6.7980/fow/?tab=rules)
- [Submission instructions](https://www.mit.edu/~6.7980/fow/?tab=submit)
- [Challenge website](https://www.mit.edu/~6.7980/fow/)
- [Public course repository](https://github.com/gabrfarina/MIT-6.7980-Topics-in-Multiagent-Learning)

The challenge pages could not be read during initialization. The public course repository was inspected but did not provide a starter kit. See [the work plan](docs/work-plan.md) for the next steps and information to verify.

## Layout

- `docs/`: verified facts, pending questions, and development plan.
- `scripts/`: local development helpers; these do not call the tournament server.
- `runs/`: ignored evaluation logs and replays.
- `artifacts/`: ignored models and submission bundles.

The starter kit will determine the agent and test layout once it is available.

## Submitting a bot

Each bot lives in `bots/<name>/` with a `run.sh` and a `MANIFEST` listing the files to ship (paths or globs, one per line). Only those files go into the ZIP.

```sh
make package BOT=baseline           # build artifacts/baseline.zip only
make submit BOT=baseline            # upload, wait for build + validation (stays inactive)
make submit BOT=baseline ACTIVATE=1 # auto-activate after validation: joins rated games
```

`scripts/submit.py` reads the token from `.env`, sends it only in the request header, and refuses to upload if the token appears in any packaged file, if a symlink or excluded file (`.env`, `.venv`, `__pycache__`, …) is listed, or if size limits are exceeded. If validation fails, it prints the end of the build log. The arena allows one upload per 30 seconds and two pending builds per team.

## Credential handling

The `.env` file is local configuration; it is not automatically loaded or executed. Future submission tooling should read the token without printing it, placing it in command-line arguments, or packaging it with the bot. Use an explicit allowlist when building submission archives. Keep this project private unless the course policy permits publishing it.

On another machine, copy `.env.example` to `.env`, populate the token locally, and run `chmod 600 .env`.
