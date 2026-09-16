from pathlib import Path
from shutil import copytree

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class BuildPyWithCSrc(_build_py):
    """Bundle the repository-level C++ sources with the Python package."""

    def run(self) -> None:
        super().run()
        repository_root = Path(__file__).resolve().parent
        copytree(
            repository_root / "csrc",
            Path(self.build_lib) / "logssms" / "csrc",
            dirs_exist_ok=True,
        )


setup(cmdclass={"build_py": BuildPyWithCSrc})
