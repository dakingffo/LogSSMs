"""Launch vLLM in its own Python environment without importing it here."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def make_command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--vllm-python", required=True)
    known, server_argv = parser.parse_known_args(argv)

    # Resolving a venv executable symlink would select its base environment.
    python = Path(os.path.abspath(Path(known.vllm_python).expanduser()))
    if not python.is_file():
        parser.error(f"vLLM Python does not exist: {python}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return [str(python), "-m", "vllm.entrypoints.cli.main", "serve", *server_argv], env


def main(argv: list[str] | None = None) -> None:
    command, env = make_command(sys.argv[1:] if argv is None else argv)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
