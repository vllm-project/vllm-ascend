# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash engram ops."""

import ctypes
import math
from functools import lru_cache
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm.logger import logger
from vllm.triton_utils import tl, triton

_RESIDUAL_BLOCK = 1024


@triton.jit
def _residual_kernel(
    hidden,
    value,
    gamma,
    inverse,
    gate,
    scales,
    hashes,
    output,
    rows: tl.constexpr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hash_width: tl.constexpr,
    hash_block: tl.constexpr,
    block: tl.constexpr,
):
    row_channel = tl.program_id(0)
    row = row_channel // channels
    channel = row_channel % channels
    column = tl.program_id(1) * block + tl.arange(0, block)
    live = column < width
    source = tl.load(value + row * width + column, live, 0).to(tl.float32)
    gain = tl.load(gamma + column, live, 0).to(tl.float32)
    rstd = tl.load(inverse + row).to(tl.float32)
    # Preserve gamma * value * inverse ordering, with contraction disabled.
    signal = (gain * source) * rstd
    probability = tl.load(gate + row_channel).to(tl.float32)
    offset = tl.load(scales + channel * width + column, live, 0).to(tl.float32)
    residual = ((probability + offset) * signal).to(tl.bfloat16).to(tl.float32)
    hash_index = tl.arange(0, hash_block)
    index = tl.load(hashes + row * hash_width + hash_index, hash_index < hash_width, -1)
    any_valid = tl.sum((index >= 0).to(tl.int32), 0) > 0
    residual = tl.where(any_valid, residual, 0.0)
    previous = tl.load(hidden + row_channel * width + column, live, 0).to(tl.float32)
    result = (previous + residual).to(tl.bfloat16)
    tl.store(output + row_channel * width + column, result, live)


def fused_residual(hidden, value, gamma, inverse, gate, scales, hashes):
    rows, channels, width = hidden.shape
    output = torch.empty_like(hidden)
    _residual_kernel[(rows * channels, triton.cdiv(width, _RESIDUAL_BLOCK))](
        hidden,
        value,
        gamma,
        inverse,
        gate,
        scales,
        hashes,
        output,
        rows,
        channels,
        width,
        hashes.shape[1],
        triton.next_power_of_2(hashes.shape[1]),
        _RESIDUAL_BLOCK,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _lookup_norm_kernel(
    source,
    hashes,
    gamma,
    output,
    row_stride: tl.constexpr,
    head_stride: tl.constexpr,
    col_stride: tl.constexpr,
    hash_row_stride: tl.constexpr,
    hash_head_stride: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, block)
    valid = tl.load(hashes + row * hash_row_stride + head * hash_head_stride) >= 0
    value = tl.load(
        source + row * row_stride + head * head_stride + col * col_stride,
        col < width,
        0,
    ).to(tl.float32)
    value = tl.where(valid, value, 0.0)
    inverse = tl.rsqrt(tl.sum(value * value, 0) / width + epsilon)
    weight = tl.load(gamma + head * width + col, col < width, 0).to(tl.float32)
    normalized = (value * inverse) * weight
    tl.store(output + (row * heads + head) * width + col, normalized, col < width)


def fused_lookup_norm(source, hashes, gamma, epsilon, output_dtype):
    rows, heads, width = source.shape
    output = torch.empty(source.shape, device=source.device, dtype=output_dtype)
    _lookup_norm_kernel[(rows, heads)](
        source,
        hashes,
        gamma,
        output,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        hashes.stride(0),
        hashes.stride(1),
        heads,
        width,
        epsilon,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
        enable_graph_optimize=False,
    )
    return output


@triton.jit
def _norm_dot_gate_kernel(
    key,
    query,
    gamma_key,
    gamma_query,
    output,
    key_row_stride: tl.constexpr,
    key_col_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_col_stride: tl.constexpr,
    width: tl.constexpr,
    eps_key: tl.constexpr,
    eps_query: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    live = col < width
    k = tl.load(key + row * key_row_stride + col * key_col_stride, live, 0).to(tl.float32)
    q = tl.load(query + row * query_row_stride + col * query_col_stride, live, 0).to(tl.float32)
    wk = tl.load(gamma_key + col, live, 0).to(tl.float32)
    wq = tl.load(gamma_query + col, live, 0).to(tl.float32)
    ik = tl.rsqrt(tl.sum(k * k, 0) / width + eps_key)
    iq = tl.rsqrt(tl.sum(q * q, 0) / width + eps_query)
    nk = ((k * ik) * wk).to(key.dtype.element_ty).to(tl.float32)
    nq = ((q * iq) * wq).to(query.dtype.element_ty).to(tl.float32)
    product = (nk * nq).to(output.dtype.element_ty).to(tl.float32)
    dot = tl.sum(product, 0).to(output.dtype.element_ty).to(tl.float32)
    scaled = (dot * scale).to(output.dtype.element_ty).to(tl.float32)
    floor = tl.full((), 1.0e-6, tl.float32).to(output.dtype.element_ty).to(tl.float32)
    magnitude = tl.maximum(tl.abs(scaled), floor, propagate_nan=tl.PropagateNan.ALL)
    magnitude = magnitude.to(output.dtype.element_ty).to(tl.float32)
    root = tl.sqrt(magnitude).to(output.dtype.element_ty).to(tl.float32)
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    logit = (root * sign).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + row, tl.sigmoid(logit))


def fused_norm_dot_gate(key, query, key_norm, query_norm):
    rows, width = key.shape
    output = torch.empty((rows,), device=key.device, dtype=torch.promote_types(key.dtype, query.dtype))
    eps_key = key_norm.eps if key_norm.eps is not None else torch.finfo(torch.float32).eps
    eps_query = query_norm.eps if query_norm.eps is not None else torch.finfo(torch.float32).eps
    _norm_dot_gate_kernel[(rows,)](
        key,
        query,
        key_norm.weight,
        query_norm.weight,
        output,
        key.stride(0),
        key.stride(1),
        query.stride(0),
        query.stride(1),
        width,
        eps_key,
        eps_query,
        width**-0.5,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output


class PrefetchedEngramRows:
    """Wait only for the table read by the consuming layer."""

    def __init__(self, outputs):
        self.outputs = outputs

    def __getitem__(self, index):
        rows, ready = self.outputs[index]
        stream = torch.npu.current_stream(device=rows.device)
        stream.wait_event(ready)
        rows.record_stream(stream)
        return rows


@triton.jit
def _flash_engram_mapped_rows_kernel(
    hashes,
    weight_ptrs,
    scale_ptrs,
    output,
    vocab_start,
    vocab_end,
    HASH_STRIDE: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // LOCAL_HEADS
    head = HEAD_START + row % LOCAL_HEADS
    index = tl.load(hashes + token * HASH_STRIDE + head).to(tl.int64)
    local_row = tl.minimum(tl.maximum(index - vocab_start, 0), vocab_end - vocab_start - 1)
    chunk = local_row // CHUNK
    offset = local_row % CHUNK
    cols = tl.arange(0, BLOCK)
    addr = tl.load(weight_ptrs + chunk)
    if FP8:
        weight = addr.to(tl.pointer_type(tl.uint8))
        bits = tl.load(weight + offset * WIDTH + cols, cols < WIDTH, other=0)
        value = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        value = tl.where((bits & 127) == 127, float("nan"), value)
        scales = tl.load(scale_ptrs + chunk).to(tl.pointer_type(tl.bfloat16))
        value = value * tl.load(scales + offset).to(tl.float32)
    else:
        weight = addr.to(tl.pointer_type(tl.bfloat16))
        value = tl.load(weight + offset * WIDTH + cols, cols < WIDTH, other=0).to(tl.float32)
    # A negative hash is a padded row. Write the mask with the gathered values.
    tl.store(output + row * WIDTH + cols, tl.where(index >= 0, value, 0.0), cols < WIDTH)


def gather_mapped_rows(hashes, weight, scale, head_start, head_end, vocab_start, vocab_end):
    from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    tokens, local_heads = hashes.size(0), head_end - head_start
    width = weight.tensor.size(1)
    output = torch.empty((tokens, local_heads, width), dtype=torch.float32, device=hashes.device)
    if output.numel():
        init_device_properties_triton()
        _flash_engram_mapped_rows_kernel[(tokens * local_heads,)](
            hashes,
            weight.ptrs,
            weight.ptrs if scale is None else scale.ptrs,
            output,
            vocab_start,
            vocab_end,
            hashes.stride(0),
            head_start,
            local_heads,
            width,
            triton.next_power_of_2(width),
            CHUNK_ROWS,
            scale is not None,
        )
    return output


@triton.jit
def _scatter_engram_rows(
    source,
    output0,
    output1,
    output2,
    output3,
    strides: tl.constexpr,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SOURCE_LAYER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    value = tl.load(source + layer * SOURCE_LAYER + i, valid, other=0).to(tl.float32)
    token = i // (HEADS * DIM)
    head = (i // DIM) % HEADS
    channel = i % DIM
    if layer == 0:
        offset = token * strides[0][0] + head * strides[0][1] + channel * strides[0][2]
        tl.store(output0 + offset, value, valid)
    elif layer == 1:
        offset = token * strides[1][0] + head * strides[1][1] + channel * strides[1][2]
        tl.store(output1 + offset, value, valid)
    elif layer == 2:
        offset = token * strides[2][0] + head * strides[2][1] + channel * strides[2][2]
        tl.store(output2 + offset, value, valid)
    else:
        offset = token * strides[3][0] + head * strides[3][1] + channel * strides[3][2]
        tl.store(output3 + offset, value, valid)


def scatter_engram_rows(source, outputs):
    """Convert stacked BF16 NPU rows into existing FP32 graph buffers once.

    Returns False for unsupported layouts so callers retain their old copies.
    Performs no allocation, address-table upload, or device-to-host conversion.
    """
    # Ascend launch ABI requires explicit output pointer arguments. The current
    # HY4 model has four Engram layers; other layouts retain the original copies.
    if len(outputs) != 4 or source.ndim != 4 or not source.is_contiguous():
        return False
    if source.dtype != torch.bfloat16 or source.device.type != "npu":
        return False
    if source.shape[0] != len(outputs):
        return False
    shape = tuple(source.shape[1:])
    if any(tuple(out.shape) != shape or out.dtype != torch.float32 or out.device != source.device for out in outputs):
        return False
    # Outputs are distinct, persistent layer buffers owned by model_state.
    size = source[0].numel()
    if size:
        _scatter_engram_rows[(triton.cdiv(size, 256), len(outputs))](
            source,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            tuple(tuple(out.stride()) for out in outputs),
            size,
            shape[1],
            shape[2],
            source.stride(0),
            256,
        )
    return True


class PageableEngramBuffer:
    def __init__(self, shape, dtype, group):
        self.shm = None
        self.tensor = None
        self.group = group
        count = math.prod(shape)
        size = count * torch.empty((), dtype=dtype).element_size()
        leader = dist.get_global_rank(group.cpu_group, 0)
        payload = [None]
        if group.rank_in_group == 0:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                payload[0] = self.shm.name
            except Exception as exc:
                payload[0] = "ERROR: " + repr(exc)
        dist.broadcast_object_list(payload, src=leader, group=group.cpu_group)
        if payload[0] is None or payload[0].startswith("ERROR:"):
            raise RuntimeError(f"CPU shared Engram creation failed: {payload[0]}")
        error = None
        try:
            if self.shm is None:
                with patch("multiprocessing.resource_tracker.register", lambda *a, **k: None):
                    self.shm = shared_memory.SharedMemory(name=payload[0])
            self.tensor = torch.frombuffer(self.shm.buf, dtype=dtype, count=count).reshape(shape)
        except Exception as exc:
            error = repr(exc)
        errors = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
        dist.barrier(group=group.cpu_group)
        if group.rank_in_group == 0:
            self.shm.unlink()
        if any(errors):
            raise RuntimeError(f"CPU shared Engram mapping failed: {errors}")

    def close(self):
        self.tensor = None
        if self.shm is not None:
            try:
                self.shm.close()
            except BufferError:
                return
            self.shm = None


NUM_LAYERS = 4
METADATA_FIELDS = 4
PACK_BLOCK = 256


@triton.jit
def _pack_engram_row_hashes(
    src,
    dst,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + (i // (T * H)) * S0 + ((i // H) % T) * S1 + (i % H) * S2, i < N, other=0)
    tl.store(dst + i, value, i < N)


@lru_cache(maxsize=1)
def _lookup_library():
    """Retain one library handle for this worker process; CDLL calls release the GIL."""
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name("engram_row_lookup_cpu.so")))
        fn = library.hy4_engram_rows
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_void_p,
            *([ctypes.c_int64] * 5),
        ]
        fn.restype = ctypes.c_int
        return library, fn
    except (OSError, AttributeError) as exc:
        logger.warning_once("HYV4 row preparation native library unavailable; retaining original path: %s", exc)
        return None


class _RowWorkspace:
    def __init__(self, hashes, rows, embeddings, capacity, signature):
        self.signature = signature
        self.capacity = capacity
        layers, _, heads = hashes.shape
        _, local_heads, dim = rows[0].shape
        self.hash_host = torch.empty(layers * capacity * heads, dtype=torch.int32, pin_memory=True)
        self.hash_device = torch.empty_like(self.hash_host, device=hashes.device)
        size = layers * capacity * local_heads * dim
        self.row_host = torch.empty(size, dtype=torch.bfloat16, pin_memory=True)
        self.row_device = torch.empty(size, dtype=torch.bfloat16, device=hashes.device)
        self.weights = tuple(e.embedding.weight for e in embeddings)
        self.pointers = (ctypes.c_void_p * layers)(*(w.data_ptr() for w in self.weights))
        self.metadata = (ctypes.c_int64 * (layers * METADATA_FIELDS))(
            *(value for e in embeddings for value in (e.head_start, e.head_end, e.vocab_start, e.vocab_end))
        )
        self.download = torch.npu.Event()
        self.consumed = torch.npu.Event()
        self.pending = False


def try_prepare_engram_rows(state, hashes, rows, packed_hashes=None):
    """Return True only when all existing graph output buffers were filled."""
    embeddings = state._engram_embeddings
    if (
        len(rows) != NUM_LAYERS
        or len(embeddings) != NUM_LAYERS
        or hashes.ndim != 3
        or hashes.shape[0] != NUM_LAYERS
        or hashes.dtype != torch.int32
        or hashes.device.type != "npu"
    ):
        return False
    tokens, heads = hashes.shape[1:]
    shape = tuple(rows[0].shape)
    if len(shape) != 3 or shape[0] != tokens:
        return False
    for embedding, out in zip(embeddings, rows, strict=True):
        weight = embedding.embedding.weight
        if (
            tuple(out.shape) != shape
            or out.dtype != torch.float32
            or out.device != hashes.device
            or weight.device.type != "cpu"
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.ndim != 2
            or embedding.fp8
            or embedding.dim != shape[2]
            or weight.shape[1] != shape[2]
            or embedding.head_start < 0
            or embedding.head_end > heads
            or embedding.head_end - embedding.head_start != shape[1]
            or embedding.vocab_end <= embedding.vocab_start
            or weight.shape[0] < embedding.vocab_end - embedding.vocab_start
        ):
            return False
    if packed_hashes is not None and (
        packed_hashes.shape != hashes.shape
        or packed_hashes.dtype != hashes.dtype
        or packed_hashes.device != hashes.device
        or not packed_hashes.is_contiguous()
    ):
        raise ValueError("Invalid packed Engram hash staging view")
    library = _lookup_library()
    if library is None:
        return False
    fn = library[1]
    if tokens == 0:
        return True
    signature = (
        heads,
        shape[1:],
        hashes.device,
        tuple(
            (
                e.embedding.weight.data_ptr(),
                tuple(e.embedding.weight.shape),
                e.head_start,
                e.head_end,
                e.vocab_start,
                e.vocab_end,
                e.dim,
            )
            for e in embeddings
        ),
    )
    workspace = getattr(state, "_engram_row_workspace", None)
    if workspace is None or workspace.signature != signature or workspace.capacity < tokens:
        if workspace is not None and workspace.pending:
            workspace.consumed.synchronize()
        capacity = max(tokens, state._graph_hashes.shape[1])
        workspace = state._engram_row_workspace = _RowWorkspace(hashes, rows, embeddings, capacity, signature)
    if workspace.pending:
        # CPU pinned buffers cannot be overwritten while the previous H2D reads them.
        workspace.consumed.synchronize()
        workspace.pending = False
    stream = torch.npu.current_stream(hashes.device)
    count = hashes.numel()
    host_hashes = workspace.hash_host[:count].view(hashes.shape)
    source = packed_hashes
    if source is None:
        source = workspace.hash_device[:count].view(hashes.shape)
        _pack_engram_row_hashes[(triton.cdiv(count, PACK_BLOCK),)](
            hashes, source, *hashes.stride(), tokens, heads, count, PACK_BLOCK
        )
    host_hashes.copy_(source, non_blocking=True)
    workspace.download.record(stream)
    workspace.download.synchronize()  # CPU lookup consumes downloaded hashes.
    row_count = NUM_LAYERS * tokens * shape[1] * shape[2]
    host_rows = workspace.row_host[:row_count]
    result = fn(
        host_hashes.data_ptr(),
        workspace.pointers,
        workspace.metadata,
        host_rows.data_ptr(),
        NUM_LAYERS,
        tokens,
        heads,
        shape[1],
        shape[2],
    )
    if result:
        raise RuntimeError(f"Engram native row lookup contract failed: {result}")
    device_rows = workspace.row_device[:row_count].view(NUM_LAYERS, *shape)
    device_rows.copy_(host_rows.view(NUM_LAYERS, *shape), non_blocking=True)
    if not scatter_engram_rows(device_rows, rows):
        raise RuntimeError("Validated Engram row scatter unexpectedly declined")
    workspace.consumed.record(stream)
    workspace.pending = True
    logger.info_once("HYV4 Engram row preparation: pinned buffers, native CPU batch lookup and fused scatter enabled")
    return True
