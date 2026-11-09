# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Suffix Q5 FULL-graph attention using aliasing BSH query/output views.

K/V remain in their allocated NHD or HND physical layout. Metadata snapshots
belong to one immutable metadata build and are shared by that build's layers.
"""

from dataclasses import dataclass
from typing import Any

import torch

_BLOCK_SIZE = 128


_WIDTH = 5


_SWA_INT_MAX = 2147483647


_METADATA_CACHE = "_suffix_bsh_metadata_cache"


def _metadata_geometry(metadata, capacity: int, width: int, max_model_len: int):
    # Exact Python integers exclude bool and numeric cache-key aliases.
    if type(capacity) is not int or capacity <= 0 or type(width) is not int or width != _WIDTH:  # noqa: E721
        raise ValueError("suffix BSH requires positive capacity and query width 5")
    if type(max_model_len) is not int or max_model_len < width:  # noqa: E721
        raise ValueError("suffix BSH max_model_len must be an integer at least query width")
    num_tokens = metadata.num_actual_tokens
    if type(num_tokens) is not int or num_tokens <= 0 or num_tokens % width:  # noqa: E721
        raise ValueError("suffix BSH actual tokens must be a positive multiple of query width")
    real_rows = num_tokens // width
    if real_rows > capacity:
        raise ValueError("suffix BSH actual requests exceed captured capacity")

    boundaries = metadata.actual_seq_lengths_q
    if not isinstance(boundaries, (list, tuple)) or not real_rows <= len(boundaries) <= capacity:
        raise ValueError("suffix BSH query boundaries must cover real rows within captured capacity")
    for row in range(real_rows):
        boundary = boundaries[row]
        if type(boundary) is not int or boundary != (row + 1) * width:  # noqa: E721
            raise ValueError("suffix BSH real query boundaries must have uniform width 5")
    previous_boundary = num_tokens
    for boundary in boundaries[real_rows:]:
        if type(boundary) is not int or not previous_boundary <= boundary <= capacity * width:  # noqa: E721
            raise ValueError("suffix BSH padded query boundaries must be monotone and bounded")
        previous_boundary = boundary
    if len(boundaries) > real_rows and previous_boundary != capacity * width:
        raise ValueError("suffix BSH padded tail must end at captured token capacity")

    lengths = metadata.seq_lens_list
    if not isinstance(lengths, (list, tuple)) or len(lengths) < real_rows:
        raise ValueError("suffix BSH requires the original CPU KV-length mirror for every real row")
    table = metadata.block_tables
    if (
        not isinstance(table, torch.Tensor)
        or table.ndim != 2
        or table.shape[0] < real_rows
        or table.shape[1] <= 0
        or table.dtype != torch.int32
        or not table.is_contiguous()
    ):
        raise ValueError("suffix BSH requires a contiguous int32 block table covering all real rows")
    kv_limit = min(max_model_len, table.shape[1] * _BLOCK_SIZE)
    real_lengths = list(lengths[:real_rows])
    if any(type(length) is not int or not width <= length <= kv_limit for length in real_lengths):  # noqa: E721
        raise ValueError("suffix BSH KV lengths exceed query/model/block-table bounds")
    return real_rows, real_lengths, table


def prepare_metadata(metadata, capacity: int, width: int, *, max_model_len: int) -> tuple[list[int], torch.Tensor]:
    """Return one immutable CPU-length/device-table snapshot per metadata build.

    A TND synthetic tail can cover several BSH rows. Only num_actual_tokens /
    width rows are real: ignore the synthetic request's KV length and table row,
    and give *each* remaining BSH row KV length width and a zero block row.
    No source list or tensor is modified and no device-to-host read is performed.
    Callers must not mutate this result or reuse a metadata object across steps.
    """
    # bool/float keys compare equal to integers in a dict; do not let a cached
    # valid shape turn an invalid subsequent API argument into a cache hit.
    if type(capacity) is not int or type(width) is not int or type(max_model_len) is not int:  # noqa: E721
        raise ValueError("suffix BSH capacity, width, and max_model_len must be integers")
    cache_key = (capacity, width, max_model_len)
    cache = getattr(metadata, _METADATA_CACHE, None)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    real_rows, lengths, table = _metadata_geometry(metadata, capacity, width, max_model_len)
    lengths.extend([width] * (capacity - real_rows))
    if real_rows == capacity:
        padded_table = table[:capacity]
    else:
        # These device operations precede task registration during capture, and
        # run before the manager's update-stream wait during replay resolution.
        padded_table = table.new_zeros((capacity, table.shape[1]))
        padded_table[:real_rows].copy_(table[:real_rows])
    result = (lengths, padded_table)
    if cache is None:
        cache = {}
        setattr(metadata, _METADATA_CACHE, cache)
    cache[cache_key] = result
    return result


@dataclass(frozen=True, slots=True)
class BSHParamProvider:
    layer_name: str | None
    capacity: int
    width: int
    max_model_len: int
    captured_columns: int

    def resolve(self, attn_metadata_dict) -> dict[str, Any]:
        metadata = attn_metadata_dict[self.layer_name]
        if metadata.block_tables.ndim != 2 or metadata.block_tables.shape[1] != self.captured_columns:
            raise ValueError("suffix BSH block-table columns changed from capture")
        lengths, table = prepare_metadata(
            metadata,
            self.capacity,
            self.width,
            max_model_len=self.max_model_len,
        )
        return {
            "actual_seq_lengths": None,
            "actual_seq_lengths_kv": lengths,
            "block_table": table,
        }


def eligible(impl, query, metadata, width: int = _WIDTH) -> bool:
    """Check live tensor/metadata geometry after the shared configuration gate."""
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import is_suffix_fia_cache_enabled

    config = impl.vllm_config
    if not is_suffix_fia_cache_enabled(config, impl):
        return False
    if (
        width != _WIDTH
        or getattr(metadata, "uses_pcp_sharded_layout", False)
        or metadata.attn_state
        not in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
            AscendAttentionState.ChunkedPrefill,
        )
        or metadata.num_prefills != 0
        or not metadata.causal
        or metadata.attn_mask is None
        or query.ndim != 3
        or query.shape[0] == 0
        or query.shape[0] % width
        or tuple(query.shape[1:]) != (32, 128)
        or query.dtype != torch.bfloat16
        or query.device.type != "npu"
    ):
        return False
    cache_shape = (8, _BLOCK_SIZE, 128) if impl.use_bnsd_kv_cache else (_BLOCK_SIZE, 8, 128)
    for cache in (impl.key_cache, impl.value_cache):
        if (
            not isinstance(cache, torch.Tensor)
            or cache.ndim != 4
            or cache.shape[0] == 0
            or tuple(cache.shape[1:]) != cache_shape
            or cache.dtype != query.dtype
            or cache.device != query.device
            or not cache.is_contiguous()
        ):
            return False
    if impl.key_cache.shape != impl.value_cache.shape:
        return False
    try:
        _, _, table = _metadata_geometry(metadata, query.shape[0] // width, width, config.model_config.max_model_len)
    except (AttributeError, TypeError, ValueError):
        return False
    return table.device == query.device and metadata.attn_mask.device == query.device


def capture_bsh(impl, query, key, value, metadata, output, width: int = _WIDTH, *, layer_name: str | None = None):
    """Capture real FIA on aliasing BSH query/output views without transpose or KV copies.

    The native backend calls this only for eligible FULL capture. Standalone
    operator tests may call it directly with explicit geometry and layer identity.
    """
    import torch_npu

    from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task

    if (
        width != _WIDTH
        or query.ndim != 3
        or query.shape[0] == 0
        or query.shape[0] % width
        or (impl.num_heads, impl.num_kv_heads, impl.head_size) != (32, 8, 128)
        or tuple(query.shape[1:]) != (impl.num_heads, impl.head_size)
        or query.dtype != torch.bfloat16
        or query.device.type != "npu"
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
        or not metadata.causal
        or metadata.attn_mask is None
    ):
        raise ValueError("suffix BSH capture requires BF16 NPU Q5 causal token-major query/output")
    num_tokens = query.shape[0]
    capacity = num_tokens // width
    max_model_len = impl.vllm_config.model_config.max_model_len
    key, value, block_size, original_table, _ = impl._get_fia_params(key, value, metadata)
    if block_size != _BLOCK_SIZE or original_table is not metadata.block_tables:
        raise ValueError("suffix BSH requires the original paged block table and 128-token blocks")
    cache_shape = (
        (impl.num_kv_heads, block_size, impl.head_size)
        if impl.use_bnsd_kv_cache
        else (block_size, impl.num_kv_heads * impl.head_size)
    )
    for cache in (key, value):
        if (
            cache.ndim != len(cache_shape) + 1
            or cache.shape[0] == 0
            or tuple(cache.shape[1:]) != cache_shape
            or cache.dtype != query.dtype
            or cache.device != query.device
            or not cache.is_contiguous()
        ):
            raise ValueError("suffix BSH requires contiguous BF16 views matching the physical KV layout")
    if key.shape != value.shape:
        raise ValueError("suffix BSH key/value cache geometries differ")
    lengths, table = prepare_metadata(metadata, capacity, width, max_model_len=max_model_len)
    if table.device != query.device or metadata.attn_mask.device != query.device:
        raise ValueError("suffix BSH mask and block table must be on the query device")

    packed_query = query.view(capacity, width, impl.num_heads * impl.head_size)
    packed_output = output.view(capacity, width, impl.num_heads * impl.head_size)
    softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
    kwargs = {
        "query": packed_query,
        "key": key,
        "value": value,
        "atten_mask": metadata.attn_mask,
        "block_table": table,
        "input_layout": "BSH",
        "block_size": block_size,
        "actual_seq_lengths": None,
        "actual_seq_lengths_kv": lengths,
        "num_key_value_heads": impl.num_kv_heads,
        "num_heads": impl.num_heads,
        "pre_tokens": _SWA_INT_MAX,
        "next_tokens": _SWA_INT_MAX,
        "scale": impl.scale,
        "sparse_mode": 3,
    }
    # Never size from capture's usually tiny dummy KV lengths. Runtime table
    # columns are fixed by the configured KV-cache spec; metadata only pads rows.
    workspace_limit = min(max_model_len, table.shape[1] * block_size)
    workspace_key = (
        "suffix_bsh_attention.workspace",
        capacity,
        width,
        impl.num_heads,
        impl.num_kv_heads,
        impl.head_size,
        query.dtype,
        query.device,
        tuple(key.shape),
        tuple(value.shape),
        tuple(metadata.attn_mask.shape),
        table.shape[1],
        workspace_limit,
    )
    workspace = get_capture_resource(
        workspace_key,
        lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            **{**kwargs, "actual_seq_lengths_kv": [workspace_limit] * capacity}
        ),
    )
    register_task(
        torch_npu.npu_fused_infer_attention_score.out,
        {**kwargs, "workspace": workspace, "out": [packed_output, softmax_lse]},
        BSHParamProvider(
            layer_name if layer_name is not None else impl._layer_name, capacity, width, max_model_len, table.shape[1]
        ),
    )
    return output, num_tokens
