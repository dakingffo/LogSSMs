from contextlib import contextmanager

import torch

from logssms.core import Batch, Request
from logssms.kernel.metadata import make_batch_mapping
from logssms.utils import init_logger
from logssms.utils.batch import BatchIndices

logger = init_logger(__name__)


def prefill_warmup_lengths(granularity, token_budget):
    from logssms.kernel.linear_attention.vendor.fla.config import PREFILL_CHUNK_BOUNDS

    chunks = {1, *(n - 1 for n in PREFILL_CHUNK_BOUNDS), *PREFILL_CHUNK_BOUNDS}
    return sorted({min(token_budget, max(1, n * granularity - offset))
                   for n in chunks for offset in (0, 1)})


def prefill_warmup_cases(granularity, token_budget, max_requests):
    cases = set()
    for count in range(1, min(4, max_requests, token_budget) + 1):
        for total in prefill_warmup_lengths(granularity, token_budget):
            total = max(total, count)
            length, remainder = divmod(total, count)
            cases.add(tuple(length + (i < remainder) for i in range(count)))
    return sorted(cases)


@contextmanager
def prefill_storage(engine, lengths):
    from logssms.cache.logstate_pool.mha_pool import MHALogStatePool

    ctx, pool = engine.ctx, engine.ctx.linear_state
    original = ctx.page_table, ctx.state_table, ctx.work_state_table
    ctx.page_table = original[0].clone()
    ctx.state_table = original[1].clone() if original[1] is not None else None
    ctx.work_state_table = original[2].clone() if original[2] is not None else None
    kv_size = getattr(engine.kv_cache, "kv_page_size", engine.config.page_size)
    kv_counts = [(n + kv_size - 1) // kv_size for n in lengths]
    state_counts = [(n + engine.config.page_size - 1) // engine.config.page_size for n in lengths]
    allocation = None
    try:
        if isinstance(pool, MHALogStatePool):
            allocation = pool.reserve(kv=sum(kv_counts), logs=sum(state_counts), works=len(lengths))
            if allocation is None:
                raise RuntimeError("Insufficient cache capacity for prefill graph warmup")
            kv, states, works = allocation
            kv = torch.tensor(kv, device=engine.device) * kv_size
            states = torch.tensor(states, device=engine.device)
            ctx.work_state_table[:len(lengths), 0] = torch.tensor(works, device=engine.device)
            frames = []
            for slot in (*allocation[1], *works):
                record = pool.allocator.slot(slot)
                frames.extend(record.payload)
                frames.extend(frame for group in record.conv for frame in group)
                if record.decay is not None:
                    start = record.decay.index * pool.layout.log_decay_bytes
                    pool.storage[record.decay.frame, start:start + pool.layout.log_decay_bytes].zero_()
            pool.storage.index_fill_(0, torch.tensor(frames, device=engine.device), 0)
        else:
            kv = engine.cache_manager.free_pages[:sum(kv_counts)]
            if len(kv) != sum(kv_counts):
                raise RuntimeError("Insufficient KV capacity for prefill graph warmup")
            if pool is not None:
                states = engine.cache_manager.free_states[:sum(state_counts)]
                if len(states) != sum(state_counts):
                    raise RuntimeError("Insufficient state capacity for prefill graph warmup")
                pool.conv_state.index_fill_(0, states, 0)
                pool.recurrent_state.index_fill_(0, states, 0)
        kv_offset = state_offset = 0
        for row, length in enumerate(lengths):
            pages = kv[kv_offset:kv_offset + kv_counts[row]]
            mapping = (pages[:, None] + torch.arange(kv_size, device=engine.device)).flatten()
            ctx.page_table[row, :length] = mapping[:length]
            if pool is not None:
                slots = states[state_offset:state_offset + state_counts[row]]
                ctx.state_table[row, :length] = slots.repeat_interleave(engine.config.page_size)[:length]
            kv_offset += kv_counts[row]
            state_offset += state_counts[row]
        yield
    finally:
        torch.cuda.current_stream(engine.device).synchronize()
        ctx.page_table, ctx.state_table, ctx.work_state_table = original
        if allocation is not None:
            pool.release_kv(allocation[0])
            pool.release_slots((*allocation[1], *allocation[2]))


@torch.inference_mode()
def warmup_prefill(engine):
    runner = engine.piecewise_graph_runner
    limit = min(runner.buckets[-1], engine.max_seq_len)
    cases = {(n,) for n in runner.buckets if n <= limit}
    if engine.config.model_config.is_hybrid:
        cases.update(prefill_warmup_cases(engine.config.cache_config.prefill_alignment,
                                          limit, engine.config.max_running_req))
    cases = sorted(cases, key=lambda lengths: (sum(lengths), lengths), reverse=True)
    logger.info_rank0(f"Preparing complete prefill forwards: {len(cases)} cases")
    replays, padded = runner.replays, runner.padded_tokens
    try:
        with torch.cuda.stream(engine.stream):
            for index, lengths in enumerate(cases):
                with prefill_storage(engine, lengths):
                    reqs = [Request(torch.zeros(n, dtype=torch.int32), row, 0, 1,
                                    -row - 1, None, None, n + 1)
                            for row, n in enumerate(lengths)]
                    batch = Batch(reqs, "prefill")
                    batch.padded_reqs = reqs
                    batch.logstate_first_offsets = [0] * len(reqs)
                    page_size = getattr(engine.kv_cache, "kv_page_size", 1)
                    packet = batch.prepared_indices = BatchIndices.make(batch, engine.device, page_size)
                    batch.positions, mapping, batch.out_loc = make_batch_mapping(
                        packet, engine.ctx.page_table, decode=engine.attn_backend.uses_paged_indices,
                    )
                    packet.token_table_ids = mapping[0]
                    batch.sample_indices, batch.sample_req_indices = packet.sample_tuple
                    batch.input_ids = torch.zeros(sum(lengths), dtype=torch.int32, device=engine.device)
                    engine.attn_backend.prepare_metadata(batch)
                    if engine.config.model_config.is_hybrid:
                        engine.linear_attn_backend.prepare_metadata(batch)
                    with engine.ctx.forward_batch(batch):
                        engine.model.forward()
                if (index + 1) % 8 == 0:
                    logger.info_rank0(f"Complete prefill forwards: {index + 1}/{len(cases)}")
    finally:
        runner.replays, runner.padded_tokens = replays, padded
    logger.info_rank0("Complete prefill forwards ready")
