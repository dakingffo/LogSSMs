from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_mapping_kernel(
    tables, cached, cu_q, page_table,
    positions, input_rows, input_cols, out_loc,
    table_stride: tl.constexpr, BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu_q + req)
    end = tl.load(cu_q + req + 1)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < end - start
    row = tl.load(tables + req).to(tl.int64)
    column = tl.load(cached + req) + offsets
    loc = tl.load(page_table + row * table_stride + column, mask=mask, other=0)
    tl.store(positions + start + offsets, column, mask=mask)
    tl.store(input_rows + start + offsets, row, mask=mask)
    tl.store(input_cols + start + offsets, column, mask=mask)
    tl.store(out_loc + start + offsets, loc, mask=mask)


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


def make_batch_mapping(indices, page_table):
    device = page_table.device
    rows = indices.num_tokens
    positions = torch.empty(rows, dtype=torch.int32, device=device)
    mapping = torch.empty((2, rows), dtype=torch.int64, device=device)
    out_loc = torch.empty(rows, dtype=page_table.dtype, device=device)
    if rows:
        _batch_mapping_kernel[(len(indices.table_ids), triton.cdiv(indices.max_extend_len, 256))](
            indices.table_ids, indices.cached_lens, indices.cu_seqlens_q_gpu,
            page_table, positions, mapping[0], mapping[1], out_loc,
            table_stride=page_table.stride(0), BLOCK=256,
        )
    return positions, (mapping[0], mapping[1]), out_loc


def gather_paged_indices(indices, page_table):
    output = torch.empty(indices.num_pages, dtype=page_table.dtype, device=page_table.device)
    if indices.num_pages:
        _paged_indices_kernel[(len(indices.table_ids), triton.cdiv(indices.max_pages, 256))](
            indices.table_ids, indices.cu_seqlens_k_gpu, page_table, output,
            table_stride=page_table.stride(0), page_size=indices.page_size, BLOCK=256,
        )
    return output


@triton.jit
def _linear_slots_kernel(
    tables, cached, state_table, work_table, out_loc,
    output, count: tl.constexpr, table_stride: tl.constexpr,
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
        tl.store(output + count + req, log, mask=mask)
        tl.store(output + 2 * count + req, offset, mask=mask)
        tl.store(output + 3 * count + req, 0, mask=mask)
    else:


        previous = tl.where(pos > 0, pos - 1, table_width - 1)
        read = tl.load(state_table + row * table_stride + previous, mask=mask, other=0)
        write = tl.load(state_table + row * table_stride + pos, mask=mask, other=0)
        tl.store(output + 2 * req, read, mask=mask)
        tl.store(output + 2 * req + 1, write, mask=mask)


def gather_linear_slots(indices, state_table, *, work_table=None, out_loc=None, page_size=1):
    count = len(indices.table_ids)
    logstate = work_table is not None
    output = torch.empty((4, count) if logstate else (count, 2),
                         dtype=torch.int32 if logstate else torch.int64,
                         device=state_table.device)
    _linear_slots_kernel[(triton.cdiv(count, 128),)](
        indices.table_ids, indices.cached_lens, state_table, work_table, out_loc, output,
        count=count, table_stride=state_table.stride(0),
        table_width=state_table.shape[1],
        work_stride=work_table.stride(0) if logstate else 0,
        page_size=page_size, LOGSTATE=logstate, BLOCK=128,
    )
    return output
