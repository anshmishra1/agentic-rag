"""Run the local Compose app with persistent host logs and baked models.

This script uses only the Python standard library. Compose loads .env at
runtime; the script never opens or prints it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
OVERRIDE_FILE = Path(tempfile.gettempdir()) / "agentic-rag-local-runner.yml"
LOG_DIR = PROJECT_ROOT / "logs"


def _docker_executable() -> str:
    found = shutil.which("docker")
    if found:
        return found

    for candidate in (
        Path.home() / "AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe",
        Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe"),
    ):
        if candidate.is_file():
            return str(candidate)

    raise SystemExit("Docker CLI not found. Start Docker Desktop and try again.")


def _override_text(*, bounded: bool) -> str:
    environment = [
        '      HF_HUB_OFFLINE: "1"',
        '      TRANSFORMERS_OFFLINE: "1"',
    ]
    if bounded:
        environment += [
            '      MAX_RETRIES: "2"',
            '      LLM_MAX_RETRIES_PER_PROVIDER: "1"',
        ]

    mounts = [f"{LOG_DIR.resolve().as_posix()}:/app/logs"]
    return "\n".join(
        ["services:", "  api:", "    environment:", *environment, "    volumes:"]
        + [f"      - {json.dumps(mount)}" for mount in mounts]
        + [""]
    )


def _compose_command(docker: str, *args: str, override: bool = False) -> list[str]:
    command = [docker, "compose", "-f", str(COMPOSE_FILE)]
    if override:
        command += ["-f", str(OVERRIDE_FILE)]
    return command + list(args)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("start", "stop", "status", "logs"), default="start")
    parser.add_argument("--bounded", action="store_true", help="Use one LLM attempt per call; keep graph retries at two.")
    parser.add_argument("--build", choices=("api", "frontend", "all"), help="Explicitly rebuild selected image(s) before starting.")
    parser.add_argument("--plan", action="store_true", help="Print the start plan without writing files or calling Docker.")
    args = parser.parse_args()

    if args.action != "start" and (args.bounded or args.build or args.plan):
        parser.error("--bounded, --build, and --plan apply only to start")

    if args.plan:
        print(_override_text(bounded=args.bounded), end="")
        print(f"Compose override: {OVERRIDE_FILE}")
        print(f"Build: {args.build or 'none'}; then up -d --no-build")
        return 0

    docker = _docker_executable()
    if args.action == "stop":
        _run(_compose_command(docker, "stop"))
        return 0
    if args.action == "status":
        _run(_compose_command(docker, "ps"))
        return 0
    if args.action == "logs":
        _run(_compose_command(docker, "logs", "-f", "--tail", "100", "api"))
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    override_text = _override_text(bounded=args.bounded)
    if not OVERRIDE_FILE.is_file() or OVERRIDE_FILE.read_text(encoding="utf-8") != override_text:
        OVERRIDE_FILE.write_text(override_text, encoding="utf-8")

    _run(_compose_command(docker, "config", "--quiet", override=True))
    if args.build:
        services = ["api"]
        _run(_compose_command(docker, "build", *services, override=True))
    _run(_compose_command(docker, "up", "-d", "--no-build", override=True))
    _run(_compose_command(docker, "ps", override=True))
    print(f"Application logs: {LOG_DIR / 'runs'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
