PREFILL_CHUNK_BOUNDS = (4, 12, 16)


def prefill_config(stage, chunks):
    output, kkt, solve = PREFILL_CHUNK_BOUNDS
    if stage == "output":
        return dict(BK=32, BV=32 if chunks < output else 64,
                    num_warps=2 if chunks < output else 1, num_stages=3)
    if stage == "kkt":
        return dict(BK=32, num_warps=4 if chunks < kkt else 1, num_stages=3)
    if stage == "solve":
        return dict(DOT_PRECISION="ieee", num_warps=4 if chunks < solve else 1,
                    num_stages=2 if chunks < solve else 3)
    if stage == "wy":
        return dict(num_warps=4 if chunks < solve else 1, num_stages=3)
    raise ValueError(f"Unknown prefill stage: {stage}")
