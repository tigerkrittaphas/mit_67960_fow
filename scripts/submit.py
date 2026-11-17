"""Package a bot from its MANIFEST allowlist and upload it to the FoW arena.

Usage:
    python3 scripts/submit.py BOT [--dry-run] [--activate] [--no-wait]

BOT is a folder under bots/ (or a path). Its MANIFEST lists the files to ship,
one relative path or glob per line (# comments allowed); run.sh is required.

The team token is read from FOW_TEAM_TOKEN in the environment or .env. It is
sent only in the Authorization header, never printed or put on a command line,
and the script refuses to upload if it appears in any packaged file.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARENA = "https://6s890.lids.mit.edu"
MAX_ZIP = 128 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_ENTRIES = 10_000
FORBIDDEN_PARTS = {".env", ".git", ".venv", "venv", "__pycache__", ".DS_Store", ".secrets"}
POLL_SECONDS = 10
POLL_LIMIT = 20 * 60


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def read_env() -> dict[str, str]:
    """Parse plain KEY=value lines from .env without executing it."""
    values = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    values.update({k: v for k, v in os.environ.items() if k.startswith("FOW_")})
    return values


def resolve_bot(name: str) -> Path:
    candidate = Path(name)
    bot = candidate if candidate.is_dir() else ROOT / "bots" / name
    if not bot.is_dir():
        fail(f"bot folder not found: {name} (looked in bots/)")
    return bot.resolve()


def collect_files(bot: Path) -> list[Path]:
    manifest = bot / "MANIFEST"
    if not manifest.is_file():
        fail(f"{manifest.relative_to(ROOT)} missing; list the files to ship, one per line")
    files: dict[str, Path] = {}
    for raw in manifest.read_text().splitlines():
        pattern = raw.split("#", 1)[0].strip()
        if not pattern:
            continue
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            fail(f"MANIFEST entry must be a relative path inside the bot: {pattern}")
        matches = sorted(bot.glob(pattern))
        if not matches:
            fail(f"MANIFEST entry matches nothing: {pattern}")
        for match in matches:
            paths = [match] if not match.is_dir() else sorted(p for p in match.rglob("*") if not p.is_dir())
            for path in paths:
                rel = path.relative_to(bot)
                if path.is_symlink() or any(p.is_symlink() for p in path.parents if bot in p.parents):
                    fail(f"symlinks are not accepted by the arena: {rel}")
                if FORBIDDEN_PARTS & set(rel.parts) or rel.name.startswith(".env") or rel.suffix == ".pyc":
                    fail(f"refusing to package excluded file: {rel}")
                files[rel.as_posix()] = path
    if "run.sh" not in files:
        fail("run.sh must be listed in MANIFEST at the bot root")
    return [files[k] for k in sorted(files)]


def build_zip(bot: Path, files: list[Path], token: str) -> bytes:
    if len(files) > MAX_ENTRIES:
        fail(f"{len(files)} files exceeds the {MAX_ENTRIES}-entry limit")
    secret = token.encode() if token else None
    expanded = 0
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            data = path.read_bytes()
            rel = path.relative_to(bot).as_posix()
            if secret and secret in data:
                fail(f"team token found in {rel}; remove it before submitting")
            expanded += len(data)
            info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel.endswith(".sh") else 0o644) << 16
            archive.writestr(info, data)
    if expanded > MAX_EXPANDED:
        fail(f"expanded size {expanded / 2**20:.1f} MiB exceeds 512 MiB")
    payload = buffer.getvalue()
    if len(payload) > MAX_ZIP:
        fail(f"ZIP size {len(payload) / 2**20:.1f} MiB exceeds 128 MiB")
    return payload


def api(arena: str, token: str, path: str, method: str = "GET", body: bytes | None = None,
        content_type: str | None = None) -> dict:
    request = urllib.request.Request(arena + path, data=body, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/json")
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        hint = {401: " (team token rejected)", 429: " (rate limited: wait 30 s between uploads,"
                " max two pending builds)"}.get(error.code, "")
        fail(f"{method} {path} returned HTTP {error.code}{hint}: {detail}")
    except urllib.error.URLError as error:
        fail(f"could not reach {arena}: {error.reason}")
    return {}


def upload(arena: str, token: str, name: str, payload: bytes, activate: bool) -> dict:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}.zip\"\r\n"
        "Content-Type: application/zip\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    flag = "true" if activate else "false"
    result = api(arena, token, f"/api/submissions?autoActivate={flag}", "POST", body,
                 f"multipart/form-data; boundary={boundary}")
    return result["submission"]


def wait(arena: str, token: str, submission_id: str) -> int:
    deadline = time.monotonic() + POLL_LIMIT
    last = None
    while time.monotonic() < deadline:
        listing = api(arena, token, "/api/submissions")
        current = next((s for s in listing.get("submissions", []) if s["id"] == submission_id), None)
        if current is None:
            fail("uploaded submission disappeared from the version list")
        status = current["status"]
        if status != last:
            message = current.get("statusMessage") or ""
            print(f"  {time.strftime('%H:%M:%S')} {status}{': ' + message if message else ''}")
            last = status
        if status not in ("queued", "building"):
            for warning in current.get("validationWarnings") or []:
                print(f"  warning ({warning.get('side')}): {warning.get('code')}")
            if status in ("validated", "active"):
                return 0
            logs = api(arena, token, f"/api/submissions/{submission_id}/logs").get("logs") or ""
            if logs:
                print("---- build & test log (tail) ----")
                print("\n".join(logs.splitlines()[-40:]))
            return 1
        time.sleep(POLL_SECONDS)
    print("still building after 20 minutes; check the Submit page")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bot", help="folder under bots/, e.g. baseline")
    parser.add_argument("--dry-run", action="store_true", help="build artifacts/<bot>.zip without uploading")
    parser.add_argument("--activate", action="store_true",
                        help="activate automatically after validation (joins rated games)")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for build and validation")
    args = parser.parse_args()

    env = read_env()
    token = env.get("FOW_TEAM_TOKEN", "")
    arena = env.get("FOW_ARENA", DEFAULT_ARENA).rstrip("/")
    bot = resolve_bot(args.bot)
    files = collect_files(bot)
    payload = build_zip(bot, files, token)

    out = ROOT / "artifacts" / f"{bot.name}.zip"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(payload)
    print(f"Packaged {len(files)} files -> {out.relative_to(ROOT)} ({len(payload) / 1024:.1f} KiB):")
    for path in files:
        print(f"  {path.relative_to(bot).as_posix()}")
    if args.dry_run:
        return 0
    if not token:
        fail("FOW_TEAM_TOKEN is not set in .env or the environment")

    submission = upload(arena, token, bot.name, payload, args.activate)
    mode = "auto-activate ON" if args.activate else "auto-activate off"
    print(f"Uploaded v{submission['version']} ({mode}) to {arena}")
    if args.no_wait:
        return 0
    code = wait(arena, token, submission["id"])
    if code == 0 and not args.activate:
        print(f"Validated but not active. Activate v{submission['version']} on the Submit page when ready.")
    return code


if __name__ == "__main__":
    sys.exit(main())
