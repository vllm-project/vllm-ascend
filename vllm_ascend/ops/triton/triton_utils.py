from typing import Any

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_NUM_AICORE = -1
_NUM_VECTORCORE = -1
_UB_SIZE_BYTES = -1
_extension_module = None

# Safe default for Ascend 910B (A2) and 910C (A3), both have 192 KB UB.
# Used when device properties do not expose UB size and no env override is set.
DEFAULT_UB_SIZE_BYTES = 192 * 1024

# Minimum plausible UB size in bytes. Some Ascend driver versions return
# invalid values (e.g., 1) for UB-related device properties when the real
# information is unavailable. Any detected value below this threshold is
# treated as invalid and falls back to DEFAULT_UB_SIZE_BYTES. The smallest
# real UB on any shipping Ascend NPU is 128 KB (older 310P), so 4 KB is a
# conservative floor that rejects sentinel/error values while accepting any
# genuine UB report.
_MIN_UB_SIZE_BYTES = 4 * 1024

# Candidate keys that Triton Ascend driver may use to report UB size.
_UB_SIZE_PROPERTY_KEYS = (
    "ub_size",
    "max_ub_size",
    "unified_buffer_size",
    "max_shared_memory",
    "max_shared_mem",
)

if HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module  # type: ignore
    except ImportError:
        _extension_module = None


def _resolve_triton_ascend_op(op_name: str):
    if not HAS_TRITON:
        raise RuntimeError(f"Triton op '{op_name}' cannot be resolved because HAS_TRITON is False")

    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


if HAS_TRITON:
    insert_slice = _resolve_triton_ascend_op("insert_slice")
    extract_slice = _resolve_triton_ascend_op("extract_slice")
    get_element = _resolve_triton_ascend_op("get_element")
else:
    insert_slice = None
    extract_slice = None
    get_element = None


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE, _UB_SIZE_BYTES
    if _NUM_AICORE == -1 and HAS_TRITON:
        device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        _NUM_AICORE = device_properties.get("num_aicore", -1)
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."

        # Detect UB size: try each candidate key from device properties.
        # Reject values below _MIN_UB_SIZE_BYTES: some driver versions return
        # sentinels (e.g., 1) when UB info is unavailable, which would produce
        # absurdly small tiles and kernel compilation failures.
        for key in _UB_SIZE_PROPERTY_KEYS:
            value = device_properties.get(key)
            if isinstance(value, (int, float)) and value >= _MIN_UB_SIZE_BYTES:
                _UB_SIZE_BYTES = int(value)
                break
        # Fall back to safe default if not found or invalid.
        if _UB_SIZE_BYTES <= 0:
            _UB_SIZE_BYTES = DEFAULT_UB_SIZE_BYTES


def get_aicore_num():
    global _NUM_AICORE
    assert _NUM_AICORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE


def get_ub_size_bytes():
    """Return the Unified Buffer (UB) size in bytes for the current NPU.

    Used by Triton kernels to dynamically size tiles so they fit within UB.
    The value is auto-detected from device properties during
    ``init_device_properties_triton()``, with a safe fallback of 192 KB
    (covering Ascend 910B/A3).
    """
    global _UB_SIZE_BYTES
    assert _UB_SIZE_BYTES > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _UB_SIZE_BYTES


def prepare_chunk_indices_dev(
    cu_seqlens: torch.LongTensor,
    total_tokens: int,
    chunk_size: int,
) -> torch.LongTensor:
    """Device-side equivalent of :func:`prepare_chunk_indices`.

    ``prepare_chunk_indices`` ends with ``.tolist()``, i.e. a device->host synchronisation. On a
    serving hot path that stalls the NPU pipeline once per kernel launch, and it makes the
    surrounding op impossible to capture into an ACLGraph (host synchronisation is forbidden
    during capture). This variant never copies anything back to the host.

    The row count must be known on the host, so it is the shape-derived upper bound
    ``total_tokens // chunk_size + num_seqs`` instead of the exact
    ``sum(ceil(len_s / chunk_size))``. Rows past the real chunk count are phantom rows: their
    ``seq_id`` clamps to 0 while ``chunk_in_seq`` runs past the end of the first sequence, so
    every lane they address is fully masked and they store nothing on the paths this helper is
    used from - the ``head_first`` layouts are excluded on purpose, because their block pointers
    mask only the head axis (see the ``head_first`` guard in the FLA/KDA ``cumsum`` wrappers).

    Zero-length segments are not supported by either helper: ``prepare_chunk_indices`` numbers
    its ``seq_id`` over the non-empty sequences only, which does not match ``cu_seqlens``
    indexing, so callers have to drop empty segments first (the prefill path already does,
    see ``_compact_empty_segments``).

    Args:
        cu_seqlens: ``[num_seqs + 1]`` cumulative sequence lengths, on device.
        total_tokens: number of tokens in the packed batch, i.e. ``cu_seqlens[-1]``.
        chunk_size: chunk size the indices are built for.

    Returns:
        ``[total_chunks, 2]`` int tensor of ``(seq_id, chunk_in_seq)`` with the same dtype as
        ``cu_seqlens``. Its first ``sum(ceil(len_s / chunk_size))`` rows are element-wise
        identical to ``fla``/``kda``'s host-side ``prepare_chunk_indices``.
    """
    num_seqs = cu_seqlens.numel() - 1
    if num_seqs <= 0:
        return torch.zeros((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    total_chunks = total_tokens // chunk_size + num_seqs
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    chunks_per_seq = (lens + (chunk_size - 1)) // chunk_size
    chunk_ends = torch.cumsum(chunks_per_seq, 0)
    row_ids = torch.arange(total_chunks, dtype=torch.int32, device=cu_seqlens.device)
    seq_ids = torch.searchsorted(chunk_ends, row_ids, right=True, out_int32=True)
    seq_ids = torch.where(seq_ids >= num_seqs, torch.zeros_like(seq_ids), seq_ids)
    chunk_in_seq = row_ids - (chunk_ends - chunks_per_seq)[seq_ids]
    return torch.stack((seq_ids, chunk_in_seq), dim=1).to(cu_seqlens)
