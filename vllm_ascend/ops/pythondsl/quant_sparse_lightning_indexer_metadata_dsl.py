"""AICPU scheduling with ASC's fixed int32[1024] LI/LD boundary ABI for QSLI.

LI: 36 records x 8 fields: enable, start B/M/S2, end B/M/S2,
first LD workspace index. LD: 72 records x 8 fields: enable, B/M,
workspace index/count, M start/count, reserved. Remaining words are zero.
M is the operator's query tile index; S2 is a 512-token tile index.
"""

import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
from cannbotdsl.aicpu import (
    I32,
    I64,
    U32,
    GmIn,
    GmOut,
    aicpu_kernel,
    current_raw_stream,
)

from vllm_ascend.ops.pythondsl.utils import get_indexer_worker_count

if TYPE_CHECKING:
    # Intrinsic resolved by the AICPU compiler inside decorated kernels.
    zeros: Any


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


@aicpu_kernel
def metadata_kernel(a: Args):
    total_q = a.total_q
    if a.has_cu != 0:
        total_q = a.cu[a.batch]
    groups = a.groups
    if a.sparse != 0:
        groups = total_q
    workers = a.workers
    # ASC cost formula, with the active M extent for tail groups.
    for i in range(1024):
        a.output[i] = 0
    # Variable-size scheduler state belongs in GM, not the AICPU local arena.
    for i in range(0, groups * (12 + 3 * a.splits)):
        a.scratch[i] = 0
    total_cost = 0
    b = 0
    for g in range(0, groups):
        qoff = 0
        begin = 0
        end = total_q
        group_size = a.query_rows
        if a.sparse != 0:
            group_size = 1
            if a.has_cu != 0:
                while b + 1 < a.batch and g >= a.cu[b + 1]:
                    b += 1
                begin = a.cu[b]
                end = a.cu[b + 1]
            qoff = g - begin
        else:
            b = g // a.max_tasks
            qoff = (g % a.max_tasks) * a.query_rows
            if a.has_cu != 0:
                begin = a.cu[b]
                end = a.cu[b + 1]
        used = end - begin
        if a.has_q != 0:
            used = a.used_q[b]
        span = min(group_size, max(0, end - begin - qoff))
        active = min(span, max(0, used - qoff))
        actual = a.capacity
        if a.has_k != 0:
            actual = a.used_k[b]
        rem = 0
        if a.has_residual != 0:
            rem = a.residual[b]
        visible = actual
        if a.mask == 3:
            visible = min(
                actual,
                max(0, (actual * a.ratio + rem - used + qoff + active) // a.ratio),
            )
        if active == 0:
            visible = 0
        length = visible
        if a.sparse != 0:
            length = a.candidate_length[g] * 8
            if visible == 0:
                length = 0
        n = (length + 511) // 512
        a.scratch[0 + (g)] = n
        a.scratch[groups + (g)] = 4 * (6 * ((active * a.heads + 15) // 16) + 20)
        tail = length % 512
        a.scratch[2 * groups + (g)] = a.scratch[groups + (g)]
        if tail != 0:
            a.scratch[2 * groups + (g)] = ((tail + 127) // 128) * 6 * ((active * a.heads + 15) // 16) + 10 * (
                (tail + 63) // 64
            )
        if n > 0:
            row_cost = (n - 1) * a.scratch[groups + (g)] + a.scratch[2 * groups + (g)]
            total_cost += row_cost
        row = begin + qoff
        field1 = span
        if a.sparse != 0:
            row = b
            field1 = qoff
        a.scratch[3 * groups + (g * 8)] = row
        a.scratch[3 * groups + (g * 8 + 1)] = field1
        a.scratch[3 * groups + (g * 8 + 2)] = active
        a.scratch[3 * groups + (g * 8 + 3)] = used
        a.scratch[3 * groups + (g * 8 + 4)] = actual
        a.scratch[3 * groups + (g * 8 + 5)] = rem
        a.scratch[3 * groups + (g * 8 + 6)] = b
        a.scratch[3 * groups + (g * 8 + 7)] = qoff
    # ASC AssignByBatch -> AssignByRow -> AssignByBlock -> ForceAssign.
    # The cursor is a lexicographic (B, M, S2) boundary. No per-query split cap.
    first_group = zeros(I64, workers)  # noqa: F821
    last_group = zeros(I64, workers)  # noqa: F821
    first_tile = zeros(I64, workers)  # noqa: F821
    last_tile = zeros(I64, workers)  # noqa: F821
    max_row_cost = 0
    for g in range(groups):
        n = a.scratch[g]
        if n > 0:
            max_row_cost = max(
                max_row_cost,
                (n - 1) * a.scratch[groups + g] + a.scratch[2 * groups + g],
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
            b = a.scratch[3 * groups + g * 8 + 6]
            end = g
            batch_cost = 0
            batch_blocks = 0
            batch_last = 0
            while end < groups and a.scratch[3 * groups + end * 8 + 6] == b:
                n = a.scratch[end]
                start = 0
                if end == g:
                    start = at
                if n > start:
                    batch_cost += (n - start - 1) * a.scratch[groups + end] + a.scratch[2 * groups + end]
                    batch_blocks += n - start
                    batch_last = a.scratch[2 * groups + end]
                end += 1
            if batch_cost != 0 and used + batch_cost > limit + batch_last // 2:
                break
            used += batch_cost
            blocks += batch_blocks
            g = end
            at = 0
        # Assign complete M rows.
        while g < groups:
            n = a.scratch[g]
            row_cost = 0
            if n > at:
                row_cost = (n - at - 1) * a.scratch[groups + g] + a.scratch[2 * groups + g]
            if used + row_cost > limit + a.scratch[2 * groups + g] // 2:
                break
            used += row_cost
            blocks += n - at
            g += 1
            at = 0
        # Assign S2 blocks only when LD is enabled.
        if a.ld != 0 and g < groups:
            n = a.scratch[g]
            while at < n:
                cost = a.scratch[groups + g]
                if at == n - 1:
                    cost = a.scratch[2 * groups + g]
                if used + cost > limit + cost // 2:
                    break
                used += cost
                blocks += 1
                at += 1
            if blocks == 0 and at < n:
                cost = a.scratch[groups + g]
                if at == n - 1:
                    cost = a.scratch[2 * groups + g]
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
            a.output[c * 8 + 1] = a.scratch[3 * groups + start * 8 + 6]
            m = a.scratch[3 * groups + start * 8 + 7]
            if a.sparse == 0:
                m = m // a.query_rows
            a.output[c * 8 + 2] = m
            a.output[c * 8 + 3] = first_tile[c]
        if end < groups:
            a.output[c * 8 + 4] = a.scratch[3 * groups + end * 8 + 6]
            m = a.scratch[3 * groups + end * 8 + 7]
            if a.sparse == 0:
                m = m // a.query_rows
            a.output[c * 8 + 5] = m
            a.output[c * 8 + 6] = last_tile[c]
        else:
            a.output[c * 8 + 4] = a.batch
    # Record only cross-core rows. Compact workspace slots follow ASC FD order.
    fd_group = zeros(I64, workers)  # noqa: F821
    fd_parts = zeros(I64, workers)  # noqa: F821
    fd_base = zeros(I64, workers)  # noqa: F821
    fd_rows = zeros(I64, workers)  # noqa: F821
    fd_num = 0
    workspace = 0
    total_fd_load = 0
    c = 0
    while c < used_cores - 1:
        if last_tile[c] > 0:
            group = last_group[c]
            first = c
            end = c + 1
            while end < used_cores - 1 and last_group[end] == group and last_tile[end] > 0:
                end += 1
            parts = end - first + 1
            fd_group[fd_num] = group
            fd_parts[fd_num] = parts
            fd_base[fd_num] = workspace
            fd_rows[fd_num] = a.scratch[3 * groups + group * 8 + 2]
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
                        if last_group[previous] == fd_group[f] and last_tile[previous] > 0:
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
        m = a.scratch[3 * groups + group * 8 + 7]
        if a.sparse == 0:
            m = m // a.query_rows
        for v in range(count):
            dst = 288 + vector * 8
            a.output[dst] = 1
            a.output[dst + 1] = a.scratch[3 * groups + group * 8 + 6]
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

    binary = Path(__file__).parent / "_aicpu" / "qsli_metadata_kernel.so"
    if binary.is_file():
        return None, CompiledAicpuKernel(
            str(binary),
            metadata_kernel.spec,
            name="metadata_kernel",
            launch_mode="interface",
        )
    directory = tempfile.TemporaryDirectory(prefix="qsli_metadata_aicpu_")
    return directory, compile_aicpu_kernel(metadata_kernel, workdir=directory.name, launch_mode="interface")


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
    workers = get_indexer_worker_count(q.device)
    output = torch.empty((1024,), dtype=torch.int32, device=q.device)
    scratch = torch.empty((groups * (12 + 3 * splits),), dtype=torch.int64, device=q.device)
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
    total_query_rows,
):
    from types import SimpleNamespace

    if layout_q != "TND" or layout_k != "PA_BBND":
        raise ValueError("metadata supports TND queries and PA_BBND keys")
    if num_heads_q not in (32, 64) or num_heads_k != 1 or head_dim != 128:
        raise ValueError("MX4 metadata requires Nq=32/64, Nk=1 and logical head_dim=128")
    if mask_mode not in (0, 3) or not 1 <= cmp_ratio <= 128 or topk <= 0:
        raise ValueError("invalid mask_mode, cmp_ratio or topk")
    tensors = [x for x in (cu_seqlens_q, seqused_q, seqused_k, cmp_residual_k) if x is not None]
    device = tensors[0].device if tensors else torch.device("npu", torch.npu.current_device())
    if device.type != "npu":
        raise ValueError("metadata sequence inputs must be on NPU")
    if any(x.dtype != torch.int32 or x.ndim != 1 or x.device != device for x in tensors):
        raise ValueError("sequence inputs must be one-dimensional int32 tensors on one device")
    batch = (
        cu_seqlens_q.numel() - 1
        if cu_seqlens_q is not None
        else (seqused_q.numel() if seqused_q is not None else (batch_size or 1))
    )
    if batch <= 0 or (batch_size is not None and batch_size != batch):
        raise ValueError("batch_size disagrees with sequence input shapes")
    if any(x is not None and x.numel() != batch for x in (seqused_q, seqused_k, cmp_residual_k)):
        raise ValueError("sequence length tensors must have B elements")
    if cu_seqlens_q is None and batch != 1:
        raise ValueError("TND metadata requires cu_seqlens_q for multiple batches")
    # Packed query capacity comes from candidate lengths, not sequence values.
    # AICPU reads cu/used_q/used_k on every invocation, including graph replay.
    max_q = int(max_seqlen_q)
    max_k = int(max_seqlen_k)
    if max_q == 0 or max_q < -1 or max_k < -1:
        raise ValueError("invalid maximum sequence length")
    if max_k == -1 and seqused_k is None:
        raise ValueError("PA metadata requires max_seqlen_k or seqused_k")
    geometry = SimpleNamespace(shape=(total_query_rows, num_heads_q), device=device)
    return geometry, batch, max_q, max_k


def quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
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
    quant_mode,
    candidate_block_size,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
):
    if layout_k == "TND":
        if batch_size is not None:
            raise ValueError("TND metadata infers batch from cu arrays; batch_size must be None")
        if (num_heads_q, num_heads_k, head_dim, quant_mode, candidate_block_size) != (32, 1, 128, 1, 8):
            raise ValueError("TND QSLI requires Nq=32, Nk=1, D=128, quant_mode=1, block size=8")
        if layout_q != "TND" or int(max_seqlen_q) < -1 or int(max_seqlen_k) < -1:
            raise ValueError("invalid TND layout or maximum sequence length")
        batch, effective_k = _tnd_geometry(
            candidate_block_length,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            topk=topk,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )
        return _build_tnd_metadata(
            candidate_block_length,
            cu_seqlens_q,
            seqused_q,
            effective_k,
            cmp_residual_k,
            batch=batch,
            topk=topk,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )

    if quant_mode != 1 or candidate_block_size != 8:
        raise ValueError("QSLI MX4 requires quant_mode=1, candidate block size8")
    if num_heads_q != 32:
        raise ValueError("QSLI currently supports 32 query heads")
    q, batch, _, capacity = metadata_geometry(
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
        total_query_rows=candidate_block_length.numel(),
    )
    if candidate_block_length.dtype != torch.int32 or candidate_block_length.ndim not in (1, 2):
        raise ValueError("candidate_block_length must be int32 [T] or [T,1]")
    if candidate_block_length.ndim == 2 and candidate_block_length.shape[1] != 1:
        raise ValueError("candidate_block_length must have one KV head")
    if candidate_block_length.device != q.device:
        raise ValueError("candidate lengths and sequence lengths must share a device")
    tokens = 0
    return build_metadata(
        q,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        candidate_block_length.reshape(-1, 1),
        batch=batch,
        max_tasks=1,
        capacity=capacity,
        splits=choose_splits(q.shape[0], tokens, topk),
        mask=mask_mode,
        ratio=cmp_ratio,
        sparse=True,
    )


__all__ = ["quant_sparse_lightning_indexer_metadata"]


def _sequence(tensor, name, device, count=None):
    if tensor is None or tensor.dtype != torch.int32 or tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int32 tensor")
    if tensor.device != device or device.type != "npu" or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the input NPU device")
    if count is not None and tensor.numel() != count:
        raise ValueError(f"{name} must have {count} entries")


def _tnd_geometry(lengths, cu_q, cu_k, used_q, used_k, residual, *, topk, mask_mode, cmp_ratio):
    if (
        lengths.ndim != 2
        or lengths.shape[1] != 1
        or lengths.shape[0] <= 0
        or lengths.dtype != torch.int32
        or lengths.device.type != "npu"
        or not lengths.is_contiguous()
    ):
        raise ValueError("candidate_block_length must be contiguous NPU int32 (T1,1), T1>0")
    device = lengths.device
    _sequence(cu_q, "cu_seqlens_q", device)
    if cu_q.numel() < 2:
        raise ValueError("cu_seqlens_q must contain at least two boundaries")
    batch = cu_q.numel() - 1
    _sequence(cu_k, "cu_seqlens_k", device, batch + 1)
    for name, tensor in (("seqused_q", used_q), ("seqused_k", used_k)):
        if tensor is not None:
            _sequence(tensor, name, device, batch)
    if int(topk) <= 0 or int(mask_mode) not in (0, 3) or not 1 <= int(cmp_ratio) <= 128:
        raise ValueError("invalid topk, mask_mode or cmp_ratio")
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        _sequence(residual, "cmp_residual_k", device, batch)
    elif residual is not None:
        raise ValueError("cmp_residual_k requires mask_mode=3 and cmp_ratio!=1")
    # Boundary values remain a device-side input contract; no CPU read/sync.
    effective_k = used_k if used_k is not None else cu_k[1:] - cu_k[:-1]
    return batch, effective_k


def _build_tnd_metadata(
    lengths, cu_q, used_q, effective_k, residual, *, batch, topk, mask_mode, cmp_ratio, output_idx_offset=None
):
    from types import SimpleNamespace

    geometry = SimpleNamespace(shape=(lengths.shape[0], 32), device=lengths.device)
    return build_metadata(
        geometry,
        cu_q,
        used_q,
        effective_k,
        residual,
        lengths,
        batch=batch,
        max_tasks=1,
        capacity=0,
        splits=choose_splits(lengths.shape[0], 16384, int(topk)),
        mask=int(mask_mode),
        ratio=int(cmp_ratio),
        sparse=True,
        output_idx_offset=output_idx_offset,
    )
