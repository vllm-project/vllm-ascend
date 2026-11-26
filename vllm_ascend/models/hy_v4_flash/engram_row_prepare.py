# SPDX-License-Identifier: Apache-2.0
"""Graph-external Engram CPU lookup with reusable pinned staging.

Selected automatically for supported Decode inputs. The original FP8/unsupported-layout path is preserved.
The native CPU library is built before service startup; no runtime compilation.
"""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton

from .engram_row_scatter import scatter_engram_rows

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
