from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeAlias, TypeVar, Union

if TYPE_CHECKING:
    from tvm_ffi import Module


_TVM_FILE = pathlib.Path(__file__).resolve()
_CSRC_CANDIDATES = (
    _TVM_FILE.parents[3] / "csrc",  # source checkout
    _TVM_FILE.parents[1] / "csrc",  # installed wheel
)
CPPIMPL_PATH = next((path for path in _CSRC_CANDIDATES if path.is_dir()), _CSRC_CANDIDATES[0])
DEFAULT_INCLUDE = [str(CPPIMPL_PATH / "include")]
DEFAULT_SRC = CPPIMPL_PATH / "src"
DEFAULT_JIT = CPPIMPL_PATH / "jit"

DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = DEFAULT_CFLAGS + ["--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []

CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]
P = ParamSpec("P")
R = TypeVar("R")


def call_cuda_ffi(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Launch FFI CUDA work on the current PyTorch stream."""
    from tvm_ffi import use_torch_stream

    with use_torch_stream():
        return func(*args, **kwargs)


class CppArglist(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


def _make_name(*args: str) -> str:
    return "logssms__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, (logssms::{kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArglist:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArglist(_convert(arg) for arg in args)


def load_aot(
    *args              : str,
    cpp_files          : list[str]  | None = None,
    cuda_files         : list[str]  | None = None,
    extra_cflags       : list[str]  | None = None,
    extra_cuda_cflags  : list[str]  | None = None,
    extra_ldflags      : list[str]  | None = None,
    extra_include_paths: list[str]  | None = None,
    build_directory    : str | None = None,
) -> Module:
    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((DEFAULT_SRC / f).resolve()) for f in cpp_files]
    cuda_files = [str((DEFAULT_SRC / f).resolve()) for f in cuda_files]

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args              : str,
    cpp_files          : list[str]             | None = None,
    cuda_files         : list[str]             | None = None,
    cpp_wrappers       : list[tuple[str, str]] | None = None,
    cuda_wrappers      : list[tuple[str, str]] | None = None,
    extra_cflags       : list[str]             | None = None,
    extra_cuda_cflags  : list[str]             | None = None,
    extra_ldflags      : list[str]             | None = None,
    extra_include_paths: list[str]             | None = None,
    build_directory    : str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_paths = [(DEFAULT_JIT / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    cuda_paths = [(DEFAULT_JIT / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
