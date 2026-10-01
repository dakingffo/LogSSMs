"""Launch LogSSMs; all CLI flags belong to the LogSSMs server parser."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    repo = Path(__file__).resolve().parents[2]
    args = sys.argv[1:] if argv is None else argv
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo / "python"), env.get("PYTHONPATH", ""))
    )
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "logssms", *args],
        env,
    )


if __name__ == "__main__":
    main()
