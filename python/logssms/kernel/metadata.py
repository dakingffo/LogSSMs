from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cache_mapping_kernel(
    regions, page_table, state_table,
    page_stride: tl.constexpr, state_stride: tl.constexpr, BLOCK: tl.constexpr,
):
    region = tl.program_id(0) * 5
    row = tl.load(regions + region).to(tl.int64)
    begin = tl.load(regions + region + 1)
    end = tl.load(regions + region + 2)
    page = tl.load(regions + region + 3)
    state = tl.load(regions + region + 4)
    offset = tl.arange(0, BLOCK)
    mask = begin + offset < end
    tl.store(page_table + row * page_stride + begin + offset, page + offset, mask=mask)
    tl.store(state_table + row * state_stride + begin + offset, state, mask=mask)


def write_cache_mapping(regions, page_table, state_table, page_size):
    if not regions:
        return
    if page_table.is_cuda:
        host = torch.tensor(regions, dtype=torch.int32, pin_memory=True)
        packet = host.to(page_table.device, non_blocking=True)
        _cache_mapping_kernel[(len(regions),)](
            packet, page_table, state_table,
            page_stride=page_table.stride(0), state_stride=state_table.stride(0),
            BLOCK=triton.next_power_of_2(page_size),
        )
    else:
        for row, begin, end, page, state in regions:
            page_table[row, begin:end] = torch.arange(page, page + end - begin, dtype=page_table.dtype)
            state_table[row, begin:end] = state


@triton.jit
def _slot_descriptors_kernel(
    indices, payload, conv, decay, payload_table, conv_table, decay_table,
    PAYLOAD: tl.constexpr, CONV: tl.constexpr, CLEAR: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(indices + row).to(tl.int64)
    column = tl.arange(0, BLOCK)
    if CLEAR:
        p, c, d = -1, -1, -1
    else:
        p = tl.load(payload + row * PAYLOAD + column, mask=column < PAYLOAD, other=-1)
        c = tl.load(conv + row * CONV + column, mask=column < CONV, other=-1)
        d = tl.load(decay + row * 2 + column, mask=column < 2, other=-1)
    tl.store(payload_table + slot * PAYLOAD + column, p, mask=column < PAYLOAD)
    tl.store(conv_table + slot * CONV + column, c, mask=column < CONV)
    tl.store(decay_table + slot * 2 + column, d, mask=column < 2)


def write_slot_descriptors(indices, payload, conv, decay, payload_table, conv_table, decay_table):
    if not indices.numel():
        return
    width = conv_table.shape[1] * conv_table.shape[2]
    _slot_descriptors_kernel[(indices.numel(),)](
        indices, payload, conv, decay, payload_table, conv_table, decay_table,
        PAYLOAD=payload_table.shape[1], CONV=width, CLEAR=payload is None,
        BLOCK=triton.next_power_of_2(max(payload_table.shape[1], width, 2)),
        num_warps=1,
    )


@triton.jit
def _copy_metadata_kernel(sources, outputs, counts: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(len(sources)):
        count = counts.value[i]
        value = tl.load(sources[i] + offset, mask=offset < count, other=0)
        tl.store(outputs[i] + offset, value, mask=offset < count)


def copy_metadata(sources, outputs):
    if not outputs or len(sources) != len(outputs):
        raise ValueError("metadata copies require equally sized nonempty source/output groups")
    counts = tuple(output.numel() for output in outputs)
    if any(source.numel() != count for source, count in zip(sources, counts)):
        raise ValueError("metadata copies require matching source/output tensor lengths")
    count = max(counts)
    if count == 0:
        return
    if outputs[0].is_cuda:
        _copy_metadata_kernel[(triton.cdiv(count, 128),)](
            sources, outputs, counts, BLOCK=128, num_warps=1,
        )
    else:
        for source, output in zip(sources, outputs, strict=True):
            output.copy_(source)


@triton.jit
def _batch_mapping_kernel(
    tables, cached, cu_q, page_table,
    positions, input_rows, input_cols, out_loc,
    cu_k, page_indices, table_stride: tl.constexpr, page_size: tl.constexpr,
    padding, padding_after, dummy_row, PAGED: tl.constexpr, BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu_q + req)
    end = tl.load(cu_q + req + 1)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < end - start
    output_start = start + tl.where(req >= padding_after, padding, 0)
    row = tl.load(tables + req).to(tl.int64)
    column = tl.load(cached + req) + offsets
    loc = tl.load(page_table + row * table_stride + column, mask=mask, other=0)
    tl.store(positions + output_start + offsets, column, mask=mask)
    tl.store(input_rows + output_start + offsets, row, mask=mask)
    tl.store(input_cols + output_start + offsets, column, mask=mask)
    tl.store(out_loc + output_start + offsets, loc, mask=mask)
    if req + 1 == padding_after:
        pad_mask = offsets < padding
        tl.store(positions + end + offsets, 0, mask=pad_mask)
        tl.store(input_rows + end + offsets, dummy_row, mask=pad_mask)
        tl.store(input_cols + end + offsets, 0, mask=pad_mask)
        tl.store(out_loc + end + offsets, -1, mask=pad_mask)
    if PAGED:
        page_start = tl.load(cu_k + req)
        page_end = tl.load(cu_k + req + 1)
        page_mask = offsets < page_end - page_start
        values = tl.load(page_table + row * table_stride + offsets * page_size,
                         mask=page_mask, other=0)
        tl.store(page_indices + page_start + offsets, values // page_size, mask=page_mask)


@triton.jit
def _paged_indices_kernel(
    tables, cu_k, page_table, indices,
    table_stride: tl.constexpr, page_size: tl.constexpr, BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu_k + req)
    end = tl.load(cu_k + req + 1)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(tables + req).to(tl.int64)
    mask = offsets < end - start
    values = tl.load(page_table + row * table_stride + offsets * page_size, mask=mask, other=0)
    tl.store(indices + start + offsets, values // page_size, mask=mask)


def make_batch_mapping(indices, page_table, *, decode=False, dummy_row=0):
    device = page_table.device
    rows = indices.num_tokens + indices.prefill_padding
    positions = torch.empty(rows, dtype=torch.int32, device=device)
    mapping = torch.empty((2, triton.cdiv(rows, 2) * 2), dtype=torch.int64, device=device)[:, :rows]
    out_loc = torch.empty(rows, dtype=page_table.dtype, device=device)
    page_indices = torch.empty(indices.num_pages, dtype=page_table.dtype, device=device) if decode else None
    if rows:
        width = max(indices.max_extend_len, indices.max_pages) if decode else indices.max_extend_len
        width = max(width, indices.prefill_padding)
        _batch_mapping_kernel[(len(indices.table_ids), triton.cdiv(width, 256))](
            indices.table_ids, indices.cached_lens, indices.cu_seqlens_q_gpu,
            page_table, positions, mapping[0], mapping[1], out_loc,
            indices.cu_seqlens_k_gpu, page_indices,
            table_stride=page_table.stride(0), page_size=indices.page_size,
            padding=indices.prefill_padding, padding_after=indices.padding_after,
            dummy_row=dummy_row, PAGED=decode, BLOCK=256,
        )
    indices.page_indices = page_indices
    return positions, (mapping[0], mapping[1]), out_loc


def gather_paged_indices(indices, page_table):
    if indices.page_indices is not None:
        return indices.page_indices
    output = torch.empty(indices.num_pages, dtype=page_table.dtype, device=page_table.device)
    if indices.num_pages:
        _paged_indices_kernel[(len(indices.table_ids), triton.cdiv(indices.max_pages, 256))](
            indices.table_ids, indices.cu_seqlens_k_gpu, page_table, output,
            table_stride=page_table.stride(0), page_size=indices.page_size, BLOCK=256,
        )
    return output


@triton.jit(do_not_specialize=["count"], do_not_specialize_on_alignment=["out_loc"])
def _linear_slots_kernel(
    tables, cached, state_table, work_table, out_loc,
    output, count, output_stride: tl.constexpr, table_stride: tl.constexpr,
    table_width: tl.constexpr,
    work_stride: tl.constexpr, page_size: tl.constexpr,
    LOGSTATE: tl.constexpr, BLOCK: tl.constexpr,
):
    req = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = req < count
    row = tl.load(tables + req, mask=mask, other=0).to(tl.int64)
    pos = tl.load(cached + req, mask=mask, other=1)
    if LOGSTATE:
        work = tl.load(work_table + row * work_stride, mask=mask, other=0)
        log = tl.load(state_table + row * table_stride + pos, mask=mask, other=0)
        offset = tl.load(out_loc + req, mask=mask, other=0) % page_size
        tl.store(output + req, work, mask=mask)
        tl.store(output + output_stride + req, log, mask=mask)
        tl.store(output + 2 * output_stride + req, offset, mask=mask)
        tl.store(output + 3 * output_stride + req, 0, mask=mask)
    else:


        previous = tl.where(pos > 0, pos - 1, table_width - 1)
        read = tl.load(state_table + row * table_stride + previous, mask=mask, other=0)
        write = tl.load(state_table + row * table_stride + pos, mask=mask, other=0)
        tl.store(output + 2 * req, read, mask=mask)
        tl.store(output + 2 * req + 1, write, mask=mask)


def gather_linear_slots(indices, state_table, *, work_table=None, out_loc=None, page_size=1):
    count = len(indices.table_ids)
    logstate = work_table is not None
    stride = triton.cdiv(state_table.shape[0], 4) * 4
    output = torch.empty((4, stride) if logstate else (count, 2),
                         dtype=torch.int32 if logstate else torch.int64,
                         device=state_table.device)
    _linear_slots_kernel[(triton.cdiv(count, 128),)](
        indices.table_ids, indices.cached_lens, state_table, work_table, out_loc, output,
        count=count, output_stride=stride, table_stride=state_table.stride(0),
        table_width=state_table.shape[1],
        work_stride=work_table.stride(0) if logstate else 0,
        page_size=page_size, LOGSTATE=logstate, BLOCK=128,
    )
    return output[:, :count] if logstate else output
