from contextlib import contextmanager


@contextmanager
def record_kernel_variants(*, strict=False):
    import triton
    from triton.runtime.jit import JITFunction

    target, attribute = JITFunction, "cache_hook"
    if not hasattr(target, attribute):
        target, attribute = triton.knobs.runtime, "jit_cache_hook"
    previous = getattr(target, attribute)
    variants = []

    def hook(*args, **kwargs):
        name = kwargs.get("repr", kwargs.get("key", "unknown"))
        variants.append(str(name))
        if strict:
            raise RuntimeError(f"Kernel variant was not warmed: {name}")
        return previous(*args, **kwargs) if previous is not None else False

    setattr(target, attribute, hook)
    try:
        yield variants
    finally:
        setattr(target, attribute, previous)
