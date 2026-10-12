from logssms import core


def clear_global_ctx(key: str | None = None) -> None:
    if key is None:
        core._GLOBAL_CTX = None
    else:
        assert isinstance(core._GLOBAL_CTX, dict)
        assert key in core._GLOBAL_CTX
        core._GLOBAL_CTX[key] = None
