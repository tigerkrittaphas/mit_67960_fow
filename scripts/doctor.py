"""Check local setup without displaying credentials or contacting a server."""

from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failed = False
    for tool in ("python3", "git", "uv", "docker"):
        available = shutil.which(tool) is not None
        print(f"{'OK' if available else 'INFO'}: {tool} {'available' if available else 'not installed'}")

    env_path = ROOT / ".env"
    if not env_path.is_file():
        print("FAIL: .env missing; copy .env.example and populate it locally")
        failed = True
    else:
        # Parse only the two plain KEY=value fields; never source or print the file.
        values = {}
        for line in env_path.read_text().splitlines():
            if line.startswith(("FOW_TEAM_NAME=", "FOW_TEAM_TOKEN=")):
                key, value = line.split("=", 1)
                values[key] = value.strip()
        valid = re.fullmatch(r"[0-9a-fA-F]{64}", values.get("FOW_TEAM_TOKEN", "")) is not None
        print(f"{'OK' if valid else 'FAIL'}: team token {'configured (redacted)' if valid else 'missing or unexpected format'}")
        failed |= not valid
        private = not (stat.S_IMODE(env_path.stat().st_mode) & 0o077)
        print(f"{'OK' if private else 'FAIL'}: .env {'has owner-only permissions' if private else 'permissions too broad; run chmod 600 .env'}")
        failed |= not private

    if not (ROOT / ".git").exists():
        print("FAIL: Git repository missing; run git init -b main in your terminal")
        print("INFO: .gitignore exists, but Git credential exclusion cannot yet be verified")
        failed = True
    elif shutil.which("git"):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", ".env"], cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"], cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        safe = ignored and not tracked
        print(f"{'OK' if safe else 'FAIL'}: .env {'ignored and untracked' if safe else 'is tracked or not ignored'}")
        failed |= not safe
    else:
        print("FAIL: cannot verify credential exclusion without Git")
        failed = True

    print("PENDING: verify official rules and import the starter kit before running or submitting a bot")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
