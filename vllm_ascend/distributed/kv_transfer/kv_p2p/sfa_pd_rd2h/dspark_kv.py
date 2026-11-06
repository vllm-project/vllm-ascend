# SPDX-License-Identifier: Apache-2.0
"""Direct transfer of P-computed DSpark MLA KV into D's resident pages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec

MAX_DRAFT_KV_READ_DESCRIPTORS = 1024


def get_resident_dspark_layer_names(
    vllm_config: VllmConfig,
    speculator: Any,
    *,
    sparse_offload_enabled: bool,
    is_last_pp_rank: bool,
    shared_kv_cache_layers: Mapping[str, str] | None = None,
) -> set[str]:
    """Resolve and validate draft residency without inspecting inactive paths."""
    speculative = getattr(vllm_config, "speculative_config", None)
    if not sparse_offload_enabled or not is_last_pp_rank or speculative is None or speculative.method != "dspark":
        return set()
    names = getattr(speculator, "draft_attn_layer_names", None)
    if not names:
        raise ValueError("Sparse KV offload requires DSpark cache ownership from the loaded draft model.")
    forward_context = vllm_config.compilation_config.static_forward_context
    shared_kv_cache_layers = shared_kv_cache_layers or {}
    for name in names:
        source = shared_kv_cache_layers.get(name) or getattr(
            forward_context.get(name), "kv_sharing_target_layer_name", None
        )
        if source is not None and source not in names:
            raise ValueError("Resident DSpark draft layers cannot share target-model KV caches.")
    return set(names)


def apply_dspark_resident_kv_specs(
    specs: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    speculator: Any,
    *,
    sparse_offload_enabled: bool,
    is_last_pp_rank: bool,
    shared_kv_cache_layers: Mapping[str, str] | None = None,
) -> dict[str, KVCacheSpec]:
    """Keep draft MLA pages in HBM without changing attention semantics."""
    names = get_resident_dspark_layer_names(
        vllm_config,
        speculator,
        sparse_offload_enabled=sparse_offload_enabled,
        is_last_pp_rank=is_last_pp_rank,
        shared_kv_cache_layers=shared_kv_cache_layers,
    )
    for name in names & specs.keys():
        spec = specs[name]
        if not isinstance(spec, AscendMLAAttentionSpec):
            raise ValueError("Sparse KV offload currently requires an MLA DSpark draft checkpoint.")
        # Preserve dtype, non-causal attention and CP layout; change placement only.
        specs[name] = replace(spec, store_on_host=False)
    return specs


@dataclass(frozen=True)
class DraftKVCacheMetadata:
    """Component strides distinguish actual KV bytes from padded page gaps."""

    group_id: int
    block_size: int
    num_blocks: int
    base_addrs: tuple[int, ...]
    block_strides: tuple[int, ...]
    block_lens: tuple[int, ...]
    block_scales: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.base_addrs)
        if count == 0 or any(
            len(values) != count
            for values in (self.block_strides, self.block_lens, self.block_scales, self.shapes, self.dtypes)
        ):
            raise ValueError("DSpark KV component metadata is incomplete")
        if min(self.block_size, self.num_blocks) <= 0 or self.group_id < 0:
            raise ValueError("DSpark KV cache bounds must be positive")
        if any(
            base <= 0 or length <= 0 or stride < length or scale <= 0
            for base, stride, length, scale in zip(
                self.base_addrs, self.block_strides, self.block_lens, self.block_scales
            )
        ):
            raise ValueError("DSpark KV component address/stride is invalid")


def build_draft_kv_metadata(kv_cache_config: Any, kv_caches: Mapping[str, Any]) -> dict[str, DraftKVCacheMetadata]:
    """Use loader-owned names; never infer draft ownership from layer numbers."""
    names = tuple(getattr(kv_cache_config, "dspark_draft_layer_names", ()))
    groups = {
        name: (group_id, group.kv_cache_spec.block_size)
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        for name in group.layer_names
    }
    result = {}
    for name in names:
        if name not in kv_caches or name not in groups:
            raise ValueError(f"Loaded DSpark cache {name} is missing from connector registration")
        group_id, block_size = groups[name]
        cache = kv_caches[name]
        tensors = cache if isinstance(cache, (tuple, list)) else (cache,)
        tensors = tuple(tensor for tensor in tensors if tensor is not None and tensor.numel())
        strides, lengths, scales, shapes, dtypes, bases = [], [], [], [], [], []
        for tensor in tensors:
            if tensor.ndim < 2 or tensor.shape[0] % kv_cache_config.num_blocks or not tensor[0].is_contiguous():
                raise ValueError(f"DSpark cache {name} must expose contiguous inner kernel pages")
            scale = tensor.shape[0] // kv_cache_config.num_blocks
            if scale * tensor.shape[1] != block_size:
                raise ValueError(f"DSpark cache {name} does not cover one logical token block")
            bases.append(tensor.data_ptr())
            strides.append(tensor.stride(0) * tensor.element_size())
            lengths.append(math.prod(tensor.shape[1:]) * tensor.element_size())
            scales.append(scale)
            shapes.append(tuple(tensor.shape[1:]))
            dtypes.append(str(tensor.dtype))
        result[name] = DraftKVCacheMetadata(
            group_id,
            block_size,
            kv_cache_config.num_blocks,
            tuple(bases),
            tuple(strides),
            tuple(lengths),
            tuple(scales),
            tuple(shapes),
            tuple(dtypes),
        )
    return result


def build_draft_kv_read_batches(
    remote: Mapping[str, DraftKVCacheMetadata],
    local: Mapping[str, DraftKVCacheMetadata],
    source_blocks: Mapping[str, Sequence[int]],
    dest_blocks: Mapping[int, Sequence[int]],
    prompt_tokens: int,
):
    """Yield bounded direct-HBM reads; copy valid prompt rows, never lookahead."""
    if set(source_blocks) != set(local) or not local or not set(local).issubset(remote):
        raise ValueError("DSpark KV transfer must cover every loaded draft layer exactly")
    peer_ptrs, local_ptrs, lengths = [], [], []
    for name, dst in local.items():
        src = remote[name]
        if (src.block_size, src.shapes, src.dtypes, src.block_lens, src.block_scales) != (
            dst.block_size,
            dst.shapes,
            dst.dtypes,
            dst.block_lens,
            dst.block_scales,
        ):
            raise ValueError(f"DSpark KV P/D layout mismatch for {name}")
        src_ids = source_blocks[name]
        dst_ids = dest_blocks.get(dst.group_id, ())
        needed = (prompt_tokens + dst.block_size - 1) // dst.block_size
        if len(src_ids) != needed or len(dst_ids) < needed:
            raise ValueError(f"DSpark KV block tables do not cover the prompt for {name}")
        if len(set(src_ids)) != len(src_ids) or len(set(dst_ids[:needed])) != needed:
            raise ValueError("DSpark KV prompt block tables must not alias")
        for logical_index, (src_id, dst_id) in enumerate(zip(src_ids, dst_ids)):
            if type(src_id) is not int or type(dst_id) is not int:
                raise ValueError("DSpark KV block IDs must be integers")
            if not 0 <= src_id < src.num_blocks or not 0 <= dst_id < dst.num_blocks:
                raise ValueError("DSpark KV block ID exceeds its owned cache")
            remaining = min(dst.block_size, prompt_tokens - logical_index * dst.block_size)
            for component, scale in enumerate(dst.block_scales):
                kernel_rows = dst.shapes[component][0]
                row_bytes = dst.block_lens[component] // kernel_rows
                for kernel_index in range(scale):
                    rows = min(kernel_rows, max(remaining - kernel_index * kernel_rows, 0))
                    if rows == 0:
                        continue
                    peer_ptrs.append(
                        src.base_addrs[component] + (src_id * scale + kernel_index) * src.block_strides[component]
                    )
                    local_ptrs.append(
                        dst.base_addrs[component] + (dst_id * scale + kernel_index) * dst.block_strides[component]
                    )
                    lengths.append(rows * row_bytes)
                    if len(lengths) == MAX_DRAFT_KV_READ_DESCRIPTORS:
                        yield peer_ptrs, local_ptrs, lengths
                        peer_ptrs, local_ptrs, lengths = [], [], []
    if lengths:
        yield peer_ptrs, local_ptrs, lengths
