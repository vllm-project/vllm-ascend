"""AICPU scheduling with ASC's fixed int32[1024] LI/LD boundary ABI.

LI: 36 records x 8 fields: enable, start B/M/S2, end B/M/S2,
first LD workspace index. LD: 72 records x 8 fields: enable, B/M,
workspace index/count, M start/count, reserved. Remaining words are zero.
M is the operator's query tile index; S2 is a 256-token tile index.
All query groups schedule indivisible 256-token blocks and write those
indices directly. Odd lengths use a final tail.
"""

from functools import lru_cache
import tempfile
import torch
from cannbotdsl.aicpu import (
    GmIn,
    GmOut,
    I32,
    I64,
    U32,
    aicpu_kernel,
    current_raw_stream,
)


class Args:
    cu: GmIn(I32)
    used_q: GmIn(I32)
    used_k: GmIn(I32)
    residual: GmIn(I32)
    candidate_length: GmIn(I32)
    output_offset: GmIn(I32)
    output: GmOut(I32)
    scratch: GmOut(I64)
    has_cu: U32
    has_q: U32
    has_k: U32
    has_residual: U32
    sparse: U32
    mask: U32
    ratio: U32
    total_q: I64
    batch: I64
    max_tasks: I64
    capacity: I64
    splits: I64
    workers: I64
    groups: I64
    query_rows: I64
    heads: I64
    ld: U32
    has_offset: U32


def _group_field(a, g, total_q, field):
    if g >= a.scratch[a.batch]:
        if field == 6:
            return a.batch
        return 0
    # Prefix boundaries map a compact query tile index to its batch.
    lo = 0
    hi = a.batch
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if a.scratch[mid] <= g:
            lo = mid
        else:
            hi = mid
    b = lo
    qoff = (g - a.scratch[b]) * a.query_rows
    begin = 0
    end = total_q
    if a.has_cu != 0:
        begin = a.cu[b]
        end = a.cu[b + 1]
    used = end - begin
    if a.has_q != 0:
        used = a.used_q[b]
    span = min(a.query_rows, max(0, end - begin - qoff))
    active = min(span, max(0, used - qoff))
    if field == 6:
        return b
    if field == 7:
        return qoff
    if field == 2:
        return active
    if field == 9:
        return 6 * ((active * a.heads + 15) // 16) + 20
    actual = a.capacity
    if a.has_k != 0:
        actual = a.used_k[b]
    rem = 0
    if a.has_residual != 0:
        rem = a.residual[b]
    visible = actual
    if a.mask == 3:
        visible = min(
            actual, max(0, (actual * a.ratio + rem - used + qoff + active) // a.ratio)
        )
    if active == 0:
        visible = 0
    return (visible + 255) // 256


@aicpu_kernel
def metadata_kernel(a: Args):
    total_q = a.total_q
    if a.has_cu != 0:
        total_q = a.cu[a.batch]
    elif total_q < 0 and a.has_q != 0:
        total_q = a.used_q[0]
    # ASC GetS1SeqSize: seqused_q takes precedence over the TND span.
    # Only B+1 prefix entries are needed; workspace never depends on GM values.
    groups = 0
    a.scratch[0] = 0
    for b in range(a.batch):
        length = total_q
        if a.has_cu != 0:
            length = a.cu[b + 1] - a.cu[b]
        if a.has_q != 0:
            length = a.used_q[b]
        groups += (length + a.query_rows - 1) // a.query_rows
        a.scratch[b + 1] = groups
    workers = a.workers
    for i in range(1024):
        a.output[i] = 0
    total_cost = 0
    for g in range(groups):
        n = _group_field(a, g, total_q, 8)
        total_cost += n * _group_field(a, g, total_q, 9)
    # ASC AssignByBatch -> AssignByRow -> AssignByBlock -> ForceAssign.
    # The cursor is a lexicographic (B, M, S2) boundary. No per-query split cap.
    first_group = zeros(I64, workers)
    last_group = zeros(I64, workers)
    first_tile = zeros(I64, workers)
    last_tile = zeros(I64, workers)
    max_row_cost = 0
    for g in range(groups):
        n = _group_field(a, g, total_q, 8)
        if n > 0:
            max_row_cost = max(
                max_row_cost,
                (n - 1) * _group_field(a, g, total_q, 9)
                + _group_field(a, g, total_q, 9),
            )
    g = 0
    at = 0
    remaining = total_cost
    used_cores = 0
    for c in range(workers):
        if g >= groups or remaining <= 0:
            break
        first_group[c] = g
        first_tile[c] = at
        limit = remaining // (workers - c)
        if a.ld == 0:
            limit = max(limit, max_row_cost)
        used = 0
        blocks = 0
        # Assign remaining complete batches with ASC half-last-block tolerance.
        while g < groups:
            b = _group_field(a, g, total_q, 6)
            end = g
            batch_cost = 0
            batch_blocks = 0
            batch_last = 0
            while end < groups and _group_field(a, end, total_q, 6) == b:
                n = _group_field(a, end, total_q, 8)
                start = 0
                if end == g:
                    start = at
                if n > start:
                    batch_cost += (n - start - 1) * _group_field(
                        a, end, total_q, 9
                    ) + _group_field(a, end, total_q, 9)
                    batch_blocks += n - start
                    batch_last = _group_field(a, end, total_q, 9)
                end += 1
            if batch_cost != 0 and used + batch_cost > limit + batch_last // 2:
                break
            used += batch_cost
            blocks += batch_blocks
            g = end
            at = 0
        # Assign complete M rows.
        while g < groups:
            n = _group_field(a, g, total_q, 8)
            row_cost = 0
            if n > at:
                row_cost = (n - at - 1) * _group_field(a, g, total_q, 9) + _group_field(
                    a, g, total_q, 9
                )
            if used + row_cost > limit + _group_field(a, g, total_q, 9) // 2:
                break
            used += row_cost
            blocks += n - at
            g += 1
            at = 0
        # Assign S2 blocks only when LD is enabled.
        if a.ld != 0 and g < groups:
            n = _group_field(a, g, total_q, 8)
            while at < n:
                cost = _group_field(a, g, total_q, 9)
                if at == n - 1:
                    cost = _group_field(a, g, total_q, 9)
                if used + cost > limit + cost // 2:
                    break
                used += cost
                blocks += 1
                at += 1
            if blocks == 0 and at < n:
                cost = _group_field(a, g, total_q, 9)
                if at == n - 1:
                    cost = _group_field(a, g, total_q, 9)
                used += cost
                at += 1
            if at >= n:
                g += 1
                at = 0
        remaining -= used
        if remaining <= 0:
            g = groups
            at = 0
        last_group[c] = g
        last_tile[c] = at
        used_cores = c + 1
    if used_cores == 0:
        used_cores = 1
        first_group[0] = 0
        last_group[0] = groups
    for c in range(used_cores):
        a.output[c * 8] = 1
        start = first_group[c]
        end = last_group[c]
        if start < groups:
            a.output[c * 8 + 1] = _group_field(a, start, total_q, 6)
            m = _group_field(a, start, total_q, 7)
            if a.sparse == 0:
                m = m // a.query_rows
            a.output[c * 8 + 2] = m
            a.output[c * 8 + 3] = first_tile[c]
        if end < groups:
            a.output[c * 8 + 4] = _group_field(a, end, total_q, 6)
            m = _group_field(a, end, total_q, 7)
            if a.sparse == 0:
                m = m // a.query_rows
            a.output[c * 8 + 5] = m
            a.output[c * 8 + 6] = last_tile[c]
        else:
            a.output[c * 8 + 4] = a.batch
    # Record only cross-core rows. Compact workspace slots follow ASC FD order.
    fd_group = zeros(I64, workers)
    fd_parts = zeros(I64, workers)
    fd_base = zeros(I64, workers)
    fd_rows = zeros(I64, workers)
    fd_num = 0
    workspace = 0
    total_fd_load = 0
    c = 0
    while c < used_cores - 1:
        if last_tile[c] > 0:
            group = last_group[c]
            first = c
            end = c + 1
            while (
                end < used_cores - 1 and last_group[end] == group and last_tile[end] > 0
            ):
                end += 1
            parts = end - first + 1
            fd_group[fd_num] = group
            fd_parts[fd_num] = parts
            fd_base[fd_num] = workspace
            fd_rows[fd_num] = _group_field(a, group, total_q, 2)
            total_fd_load += parts * fd_rows[fd_num]
            workspace += parts
            fd_num += 1
            c = end
        else:
            c += 1
    # ASC firstFdDataWorkspaceIdx is the running partial-slot cursor, including
    # cores whose work is entirely unsplit.
    for owner in range(used_cores):
        cursor = workspace
        for f in range(fd_num):
            if first_group[owner] <= fd_group[f]:
                cursor = fd_base[f]
                if first_group[owner] == fd_group[f]:
                    for previous in range(owner):
                        if (
                            last_group[previous] == fd_group[f]
                            and last_tile[previous] > 0
                        ):
                            cursor += 1
                break
        a.output[owner * 8 + 7] = cursor
    # ASC SplitFD balances parts * M rows over the available AIVs.
    vectors = workers * 2
    average = (total_fd_load + vectors - 1) // vectors
    spare = vectors - fd_num
    vector = 0
    for f in range(fd_num):
        count = 1
        row_chunk = fd_rows[f]
        if spare > 0:
            count = max(1, fd_parts[f] * fd_rows[f] // average)
            row_chunk = (fd_rows[f] + count - 1) // count
            count = (fd_rows[f] + row_chunk - 1) // row_chunk
            count = min(count, spare + 1)
        group = fd_group[f]
        m = _group_field(a, group, total_q, 7)
        if a.sparse == 0:
            m = m // a.query_rows
        for v in range(count):
            dst = 288 + vector * 8
            a.output[dst] = 1
            a.output[dst + 1] = _group_field(a, group, total_q, 6)
            a.output[dst + 2] = m
            a.output[dst + 3] = fd_base[f]
            a.output[dst + 4] = fd_parts[f]
            a.output[dst + 5] = v * row_chunk
            rows = row_chunk
            if v == count - 1:
                rows = fd_rows[f] - v * row_chunk
            a.output[dst + 6] = rows
            vector += 1
        spare -= count - 1

    return 0


@lru_cache(None)
def compiled_metadata():
    from pathlib import Path
    from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_aicpu_kernel

    binary = Path(__file__).parent / "_aicpu" / "metadata_kernel.so"
    if binary.is_file():
        return None, CompiledAicpuKernel(
            str(binary),
            metadata_kernel.spec,
            name="metadata_kernel",
            launch_mode="interface",
        )
    directory = tempfile.TemporaryDirectory(prefix="qli_metadata_aicpu_")
    return directory, compile_aicpu_kernel(
        metadata_kernel, workdir=directory.name, launch_mode="interface"
    )


def choose_splits(tasks, tokens, topk):
    # UB capacity for one streaming merge pass, not a limit on LD partitions.
    # AICPU computes the actual partition count and per-core ownership.
    return max(1, min(8, 16384 // topk))


def build_metadata(
    q,
    cu,
    used_q,
    used_k,
    residual,
    candidate_length,
    *,
    batch,
    max_tasks,
    capacity,
    splits,
    mask,
    ratio,
    sparse=False,
    query_rows=6,
    ld=True,
    output_idx_offset=None,
):
    groups = q.shape[0] if sparse else batch * max_tasks
    workers = 32
    output = torch.empty((1024,), dtype=torch.int32, device=q.device)
    scratch = torch.empty((batch + 1,), dtype=torch.int64, device=q.device)
    _, compiled = compiled_metadata()
    tensors = dict(
        cu=cu,
        used_q=used_q,
        used_k=used_k,
        residual=residual,
        candidate_length=candidate_length,
        output_offset=output_idx_offset,
    )
    compiled.launch(
        current_raw_stream(q.device.index),
        **{k: 0 if v is None else v.data_ptr() for k, v in tensors.items()},
        output=output.data_ptr(),
        scratch=scratch.data_ptr(),
        has_cu=int(cu is not None),
        has_q=int(used_q is not None),
        has_k=int(used_k is not None),
        has_residual=int(mask == 3 and ratio != 1 and residual is not None),
        sparse=int(sparse),
        mask=mask,
        ratio=ratio,
        total_q=q.shape[0],
        batch=batch,
        max_tasks=max_tasks,
        capacity=capacity,
        splits=splits,
        workers=workers,
        groups=groups,
        query_rows=query_rows,
        heads=q.shape[1],
        ld=int(ld),
        has_offset=int(output_idx_offset is not None),
    )
    stream = torch.npu.current_stream(q.device)
    scratch.record_stream(stream)
    for v in tensors.values():
        if v is not None:
            v.record_stream(stream)
    return output


def metadata_geometry(
    cu_seqlens_q,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    *,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode,
    cmp_ratio,
    layout_q,
    layout_k,
):
    from types import SimpleNamespace

    if layout_q != "TND" or layout_k != "PA_BBND":
        raise ValueError("metadata supports TND queries and PA_BBND keys")
    if num_heads_q not in (32, 64) or num_heads_k != 1 or head_dim != 128:
        raise ValueError(
            "MX4 metadata requires Nq=32/64, Nk=1 and logical head_dim=128"
        )
    if mask_mode not in (0, 3) or not 1 <= cmp_ratio <= 128 or topk <= 0:
        raise ValueError("invalid mask_mode, cmp_ratio or topk")
    tensors = [
        x for x in (cu_seqlens_q, seqused_q, seqused_k, cmp_residual_k) if x is not None
    ]
    device = (
        tensors[0].device
        if tensors
        else torch.device("npu", torch.npu.current_device())
    )
    if device.type != "npu":
        raise ValueError("metadata sequence inputs must be on NPU")
    if any(
        x.dtype != torch.int32 or x.ndim != 1 or x.device != device for x in tensors
    ):
        raise ValueError(
            "sequence inputs must be one-dimensional int32 tensors on one device"
        )
    batch = (
        cu_seqlens_q.numel() - 1
        if cu_seqlens_q is not None
        else (seqused_q.numel() if seqused_q is not None else (batch_size or 1))
    )
    if batch <= 0 or (batch_size is not None and batch_size != batch):
        raise ValueError("batch_size disagrees with sequence input shapes")
    if any(
        x is not None and x.numel() != batch
        for x in (seqused_q, seqused_k, cmp_residual_k)
    ):
        raise ValueError("sequence length tensors must have B elements")
    if cu_seqlens_q is None and batch != 1:
        raise ValueError("TND metadata requires cu_seqlens_q for multiple batches")
    # Sequence values are consumed by AICPU, including during graph replay.
    max_q = int(max_seqlen_q)
    max_k = int(max_seqlen_k)
    if max_q == 0 or max_q < -1 or max_k < -1:
        raise ValueError("invalid maximum sequence length")
    if max_q == -1 and cu_seqlens_q is None and seqused_q is None:
        raise ValueError("max_seqlen_q or query sequence lengths are required")
    if max_k == -1 and seqused_k is None:
        raise ValueError("PA metadata requires max_seqlen_k or seqused_k")

    geometry = SimpleNamespace(
        shape=(batch * max_q if max_q >= 0 else -1, num_heads_q), device=device
    )
    return geometry, batch, max_q, max_k


def quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Build flat AICPU metadata without requiring Q/K data tensors."""
    if (num_heads_q, num_heads_k, head_dim) != (32, 1, 128):
        raise ValueError("QLI metadata requires N1=32, N2=1 and logical D=128")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError("disabled candidate requires block size -1")
    elif candidate_topk_blocks <= 0 or candidate_block_size != 8:
        raise ValueError("candidate requires positive capacity and block size 8")
    if layout_k == 'TND':
        return _tnd_metadata(
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
            batch_size=batch_size, max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k, num_heads_q=num_heads_q,
            num_heads_k=num_heads_k, head_dim=head_dim, topk=topk,
            mask_mode=mask_mode, cmp_ratio=cmp_ratio, layout_q=layout_q,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size)
    q, batch, max_q, capacity = metadata_geometry(
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
    )
    rows = min(6, 256 // num_heads_q)
    tasks = (max_q + rows - 1) // rows if max_q >= 0 else -1
    return build_metadata(
        q,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        None,
        batch=batch,
        max_tasks=tasks,
        capacity=capacity,
        splits=choose_splits(batch * tasks, capacity, topk),
        mask=mask_mode,
        ratio=cmp_ratio,
        query_rows=rows,
    )


__all__ = ["quant_lightning_indexer_metadata"]


def _tnd_metadata(cu_q, cu_k, used_q, used_k, residual, *, batch_size,
                  max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k,
                  head_dim, topk, mask_mode, cmp_ratio, layout_q,
                  candidate_topk_blocks, candidate_block_size):
    if (num_heads_q, num_heads_k, head_dim) != (32, 1, 128):
        raise ValueError("TND QLI requires Nq=32, Nk=1 and logical D=128")
    if batch_size is not None:
        raise ValueError("TND metadata requires batch_size=None")
    if layout_q != 'TND' or max_seqlen_q < -1 or max_seqlen_k < -1:
        raise ValueError("invalid layout or sequence maximum")
    if topk <= 0 or mask_mode not in (0, 3) or not 1 <= cmp_ratio <= 128:
        raise ValueError("invalid topk, mask_mode or cmp_ratio")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError("disabled candidate requires block size -1")
    elif candidate_topk_blocks <= 0 or candidate_block_size != 8:
        raise ValueError("candidate requires positive capacity and block size 8")
    tensors = [tensor for tensor in (cu_q, cu_k, used_q, used_k, residual) if tensor is not None]
    device = tensors[0].device if tensors else torch.device('npu', torch.npu.current_device())
    if device.type != 'npu':
        raise ValueError("sequence inputs must be on NPU")
    for tensor in tensors:
        if tensor.device != device or tensor.dtype != torch.int32 or tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError("sequence inputs must be contiguous int32 vectors on one NPU")
    batch = cu_q.numel() - 1 if cu_q is not None else 1
    if batch <= 0 or (cu_k is not None and cu_k.numel() != batch + 1):
        raise ValueError("cu_q and cu_k must describe the same positive batch count")
    if any(tensor is not None and tensor.numel() != batch for tensor in (used_q, used_k, residual)):
        raise ValueError("sequence lengths must have B elements")
    if (mask_mode == 3 and cmp_ratio != 1) != (residual is not None):
        raise ValueError("cmp_residual_k is required only for causal compressed K")
    if cu_k is None and batch != 1:
        raise ValueError("multiple TND batches require cu_seqlens_k")
    effective_k = used_k
    if effective_k is None:
        if cu_k is not None:
            effective_k = cu_k[1:] - cu_k[:-1]
        elif max_seqlen_k >= 0:
            effective_k = torch.full((batch,), max_seqlen_k, dtype=torch.int32, device=device)
        else:
            raise ValueError("K boundaries, used lengths or max_seqlen_k are required")
    query, batch, max_query, capacity = metadata_geometry(
        cu_q, used_q, effective_k, residual, batch_size=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=0, num_heads_q=num_heads_q, num_heads_k=num_heads_k,
        head_dim=head_dim, topk=topk, mask_mode=mask_mode, cmp_ratio=cmp_ratio,
        layout_q=layout_q, layout_k='PA_BBND')
    tasks = (max_query + 5) // 6 if max_query >= 0 else -1
    return build_metadata(
        query, cu_q, used_q, effective_k, residual, None,
        batch=batch, max_tasks=tasks, capacity=capacity,
        splits=choose_splits(batch * tasks, capacity, topk),
        mask=mask_mode, ratio=cmp_ratio, query_rows=6).flatten()
