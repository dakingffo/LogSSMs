from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def generate_clangd() -> None:
    from logssms.tvm import DEFAULT_INCLUDE
    from logssms.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
    )
    compute_cap = status.stdout.strip().splitlines()[0]
    major, minor = compute_cap.split(".")
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-std=c++20",
            "-Wall",
            "-Wextra",
        ]
        + [f"-isystem{path}" for path in include_paths]
    )
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    clangd_path = Path.cwd() / ".clangd"
    if clangd_path.exists():
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        clangd_path.write_text(clangd_content)
        logger.info(f"{clangd_path} generated.")


if __name__ == "__main__":
    generate_clangd()
