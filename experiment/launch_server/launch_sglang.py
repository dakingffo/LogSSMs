"""Launch SGLang in its own Python environment without importing it here."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def make_command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sglang-python", required=True)
    parser.add_argument("--sglang-root", type=Path)
    known, server_argv = parser.parse_known_args(argv)

    # Keep the venv executable path: resolving its symlink can select the base
    # interpreter and silently drop the venv's installed dependencies.
    python = Path(os.path.abspath(Path(known.sglang_python).expanduser()))
    if not python.is_file():
        parser.error(f"SGLang Python does not exist: {python}")
    env = os.environ.copy()
    # Never import LogSSMs or another checkout into the SGLang child process.
    env.pop("PYTHONPATH", None)
    if known.sglang_root is not None:
        source = known.sglang_root.expanduser().resolve() / "python"
        if not (source / "sglang").is_dir():
            parser.error(f"SGLang source tree does not exist: {source}")
        env["PYTHONPATH"] = str(source)
    return [str(python), "-m", "sglang.launch_server", *server_argv], env


def main(argv: list[str] | None = None) -> None:
    command, env = make_command(sys.argv[1:] if argv is None else argv)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
