from __future__ import annotations

import functools
import hashlib
from typing import TYPE_CHECKING

from logssms.tvm import DEFAULT_SRC, load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def load_logstate_core() -> Module:
    src = DEFAULT_SRC / "logstate.cpp"
    headers = sorted((DEFAULT_SRC / "logstate").glob("*.hpp"))
    digest = hashlib.sha256(
        b"".join(p.read_bytes() for p in (src, *headers))
    ).hexdigest()[:12]
    return load_aot(
        "logstate_core",
        digest,
        cpp_files=["logstate.cpp"],
        extra_include_paths=[str(DEFAULT_SRC)],
    )
