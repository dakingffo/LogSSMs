from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import torch

from logssms.core import Batch, Request, get_global_ctx
from logssms.kernel.metadata import make_batch_mapping
from logssms.utils import init_logger
from logssms.utils.batch import BatchIndices


logger = init_logger(__name__)


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


@contextmanager
def linear_warmup_storage(engine, count, pages):
    from logssms.cache.logstate_pool.mha_pool import MHALogStatePool

    ctx = get_global_ctx()
    pool = ctx.linear_state
    original = ctx.page_table, ctx.state_table, ctx.work_state_table
    ctx.page_table = original[0].clone()
    ctx.state_table = original[1].clone()
    ctx.work_state_table = original[2].clone() if original[2] is not None else None
    slots = kv = works = None
    try:
        if isinstance(pool, MHALogStatePool):
            allocation = pool.reserve(logs=count * pages, works=count)
            if allocation is None:
                raise RuntimeError("Insufficient cache capacity for mixed kernel warmup")
            kv, slots, works = allocation
            frames = []
            for slot in (*slots, *works):
                record = pool.allocator.slot(slot)
                frames.extend(record.payload)
                frames.extend(frame for group in record.conv for frame in group)
                if record.decay is not None:
                    start = record.decay.index * pool.layout.log_decay_bytes
                    pool.storage[record.decay.frame, start:start + pool.layout.log_decay_bytes].zero_()
            pool.storage.index_fill_(0, torch.tensor(frames, device=pool.device), 0)
            ctx.work_state_table[:count, 0] = torch.tensor(works, device=pool.device)
        else:
            slots = engine.cache_manager.free_states[:count * pages].tolist()
            if len(slots) != count * pages:
                raise RuntimeError("Insufficient state capacity for mixed kernel warmup")
            index = torch.tensor(slots, device=pool.device)
            pool.conv_state.index_fill_(0, index, 0)
            pool.recurrent_state.index_fill_(0, index, 0)
        for row in range(count):
            size = min(pages * engine.config.page_size, ctx.page_table.shape[1])
            ctx.page_table[row, :size] = torch.arange(size, device=engine.device)
            states = torch.tensor(slots[row * pages:(row + 1) * pages], device=engine.device)
            ctx.state_table[row, :size] = states.repeat_interleave(engine.config.page_size)[:size]
        yield
    finally:
        torch.cuda.current_stream(engine.device).synchronize()
        ctx.page_table, ctx.state_table, ctx.work_state_table = original
        if isinstance(pool, MHALogStatePool) and works is not None:
            pool.release_slots((*slots, *works))


@torch.inference_mode()
def warmup_mixed_linear(engine):
    config = engine.config
    if not config.model_config.is_hybrid or config.max_running_req < 2:
        return
    layers = [getattr(layer, "linear_attn", None) or getattr(layer, "mamba", None)
              for layer in engine.model.model.layers.op_list]
    layers = [layer for layer in layers if layer is not None]
    if not layers:
        return
    backend = engine.linear_attn_backend
    granularity = config.cache_config.prefill_alignment
    budget = min(config.mixed_prefill_tokens, engine.max_seq_len)
    maximum = config.max_running_req - 1
    threshold = getattr(backend, "tile_min_batch_size", maximum)
    decode_counts = sorted({1, min(2, maximum), maximum,
                            max(1, min(threshold - 1, maximum)), min(threshold, maximum)})
    lengths = sorted({1, min(2, budget), min(granularity - 1, budget),
                      min(granularity, budget), min(granularity + 1, budget), budget})
    cases = [([budget], n) for n in decode_counts]
    cases += [([n], min(2, maximum)) for n in lengths]
    if config.max_running_req >= 3 and budget >= 2:
        cases.append(([1, budget - 1], 1))
    pages = max(1, (max(budget, 64) + config.page_size - 1) // config.page_size)
    started = perf_counter()
    logger.info_rank0("Preparing mixed linear-attention kernels")
    profile_enabled = backend._profile_enabled if hasattr(backend, "_profile_enabled") else None
    if profile_enabled is not None:
        backend._profile_enabled = False
    try:
        with linear_warmup_storage(engine, config.max_running_req, pages), record_kernel_variants() as variants:
            for case_index, (prefill_lengths, num_decode) in enumerate(cases):
                reqs = []
                for row, length in enumerate([*prefill_lengths, *([1] * num_decode)]):
                    cached = 0 if row < len(prefill_lengths) else (row * 7) % 32
                    reqs.append(Request(torch.zeros(cached + length, dtype=torch.int32), row,
                                        cached, 1, -row - 1, None, None, cached + length + 1))
                batch = Batch(reqs, "mixed", num_prefill_reqs=len(prefill_lengths))
                batch.padded_reqs = reqs
                batch.logstate_replay_counts = [0] * len(prefill_lengths) + [r.cached_len % 16 for r in reqs[len(prefill_lengths):]]
                p, d = Batch(reqs[:len(prefill_lengths)], "prefill"), Batch(reqs[len(prefill_lengths):], "decode")
                batch.mixed_parts = p, d
                page_size = getattr(engine.kv_cache, "kv_page_size", 1)
                packet, pp, dp = BatchIndices.make_mixed(batch, engine.device, page_size)
                batch.positions, mapping, batch.out_loc = make_batch_mapping(packet, engine.ctx.page_table, decode=True)
                offset = 0
                for part, part_packet in ((p, pp), (d, dp)):
                    part.padded_reqs = part.reqs
                    part.prepared_indices = part_packet
                    end = offset + part_packet.num_tokens
                    part.positions, part.out_loc = batch.positions[offset:end], batch.out_loc[offset:end]
                    part_packet.token_table_ids = mapping[0][offset:end]
                    part.logstate_first_offsets = [0] * part.size
                    backend.prepare_metadata(part)
                    offset = end
                batch.input_ids = torch.zeros(packet.num_tokens, dtype=torch.int32, device=engine.device)
                with engine.ctx.forward_batch(batch):
                    for layer in layers:
                        c = layer._config
                        x = torch.zeros(packet.num_tokens, c.conv_dim, dtype=engine.dtype, device=engine.device)
                        heads = c.num_v_heads if hasattr(c, "num_v_heads") else c.num_heads
                        gate = torch.zeros(packet.num_tokens, heads, dtype=engine.dtype, device=engine.device)
                        if hasattr(c, "num_v_heads"):
                            layer.attend(x, gate, gate)
                        else:
                            layer.attend(x, gate)
                logger.info_rank0(f"Mixed linear warmup: {case_index + 1}/{len(cases)}")
        logger.info_rank0(f"Mixed linear kernels ready: {len(variants)} loaded variants, {perf_counter() - started:.2f}s")
    finally:
        if profile_enabled is not None:
            backend._profile_enabled = profile_enabled
