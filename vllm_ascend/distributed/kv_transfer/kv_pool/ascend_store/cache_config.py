# SPDX-License-Identifier: Apache-2.0
"""Resolve AscendStore topology and cache-group configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable


def is_kv_save_role(kv_role: str, consumer_is_to_put: bool) -> bool:
    return kv_role in ("kv_producer", "kv_both") or consumer_is_to_put


@dataclass(frozen=True)
class TPMismatchInfo:
    enabled: bool
    peer_tp_size: int
    effective_tp_size: int
    local_heads_per_rank: int
    effective_heads_per_rank: int
    num_sub_keys: int


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def infer_dcp_mismatch_info(
    kv_role: str,
    extra_config: Mapping[str, Any] | object,
    local_dcp_size: int | object,
    local_pcp_size: int | object = 1,
) -> bool:
    """Whether the peer P/D stage disagrees with this stage on CP layout.

    Both the layerwise GVA shard stride and the per-shard save-leader rule
    derive from the local dcp size/rank. In PD-disaggregation the producer
    and consumer are separate worker groups, so when they are started with
    unequal decode-context-parallel sizes they compute different shard
    layouts for the SAME pool region and silently corrupt the KV pool.

    The peer topology is carried through the flat peer keys used by the
    store connector path (``prefill_dcp_size`` / ``decode_dcp_size``),
    mirroring the existing ``prefill_tp_size`` / ``decode_tp_size``
    convention. When the key is absent the single-group path is assumed and
    the local layout is authoritative.
    """
    local_dcp_size = _as_positive_int(local_dcp_size, 1)
    local_pcp_size = _as_positive_int(local_pcp_size, 1)
    if not isinstance(extra_config, Mapping):
        return False
    if kv_role == "kv_consumer":
        peer_dcp_key = "prefill_dcp_size"
        peer_pcp_key = "prefill_pcp_size"
    elif kv_role == "kv_producer":
        peer_dcp_key = "decode_dcp_size"
        peer_pcp_key = "decode_pcp_size"
    else:
        return False
    peer_dcp_size = _as_positive_int(extra_config.get(peer_dcp_key, local_dcp_size), local_dcp_size)
    peer_pcp_size = _as_positive_int(extra_config.get(peer_pcp_key, local_pcp_size), local_pcp_size)
    return peer_dcp_size != local_dcp_size or peer_pcp_size != local_pcp_size


def infer_tp_mismatch_info(
    kv_role: str,
    extra_config: Mapping[str, Any] | object,
    local_tp_size: int | object,
    num_kv_heads: int | object,
    use_mla: bool,
    use_hybrid: bool = False,
) -> TPMismatchInfo:
    local_tp_size = _as_positive_int(local_tp_size, 1)
    num_kv_heads = _as_positive_int(num_kv_heads, 1)
    peer_tp_size = local_tp_size
    if isinstance(extra_config, Mapping):
        peer_key = "prefill_tp_size" if kv_role == "kv_consumer" else "decode_tp_size"
        peer_tp_size = _as_positive_int(extra_config.get(peer_key, local_tp_size), local_tp_size)

    effective_tp_size = max(local_tp_size, peer_tp_size)
    enabled = (
        peer_tp_size != local_tp_size
        and not use_mla
        and not use_hybrid
        and num_kv_heads >= effective_tp_size
        and num_kv_heads % effective_tp_size == 0
    )
    local_heads_per_rank = num_kv_heads // local_tp_size if local_tp_size <= num_kv_heads else 1
    effective_heads_per_rank = num_kv_heads // effective_tp_size if enabled else local_heads_per_rank
    num_sub_keys = local_heads_per_rank // effective_heads_per_rank if enabled else 1
    return TPMismatchInfo(
        enabled=enabled,
        peer_tp_size=peer_tp_size,
        effective_tp_size=effective_tp_size,
        local_heads_per_rank=local_heads_per_rank,
        effective_heads_per_rank=effective_heads_per_rank,
        num_sub_keys=num_sub_keys,
    )


def infer_cache_family_from_ratio(compress_ratio: int | None) -> str:
    if compress_ratio is None:
        return "default"
    if compress_ratio <= 1:
        return "c1"
    return f"c{compress_ratio}"


def _get_layer_compress_ratio(
    layer_name: str,
    compress_ratios: Sequence[int] | None,
    hf_config: Any | None = None,
) -> int | None:
    if compress_ratios is None:
        return None
    if getattr(hf_config, "model_type", None) == "deepseek_v4":
        from vllm_ascend.utils import extract_dsv4_layer_index, get_dsv4_compress_ratio

        return get_dsv4_compress_ratio(hf_config, extract_dsv4_layer_index(hf_config, layer_name))
    from vllm.model_executor.models.utils import extract_layer_index

    return compress_ratios[extract_layer_index(layer_name)]


def _get_group_spec_ratios(group: object) -> set[int | None]:
    kv_cache_spec = getattr(group, "kv_cache_spec", None)
    if kv_cache_spec is None:
        return set()
    kv_cache_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if kv_cache_specs is not None:
        return {getattr(spec, "compress_ratio", None) for spec in kv_cache_specs.values()}
    return {getattr(kv_cache_spec, "compress_ratio", None)}


def infer_group_cache_families(
    kv_cache_groups: Sequence[object] | None,
    compress_ratios: Sequence[int] | None,
    hf_config: Any | None = None,
) -> list[str]:
    if kv_cache_groups is None:
        return ["default"]

    families: list[str] = []
    for group in kv_cache_groups:
        spec_ratios = _get_group_spec_ratios(group)
        if len(spec_ratios) == 1:
            families.append(infer_cache_family_from_ratio(next(iter(spec_ratios))))
            continue
        if len(spec_ratios) > 1:
            families.append("mixed")
            continue

        layer_names = list(getattr(group, "layer_names", []))
        if compress_ratios is None or not layer_names:
            families.append("default")
            continue

        group_ratios = {_get_layer_compress_ratio(layer_name, compress_ratios, hf_config) for layer_name in layer_names}
        if len(group_ratios) == 1:
            families.append(infer_cache_family_from_ratio(next(iter(group_ratios))))
        else:
            logger.debug(
                "KV cache group has mixed layer compress ratios %s for layers %s; using mixed cache family.",
                sorted(group_ratios, key=lambda ratio: -1 if ratio is None else ratio),
                layer_names,
            )
            families.append("mixed")
    return families


def uses_hybrid_kv_cache(scheduler_config: Any, kv_cache_groups: Sequence[Any] | None) -> bool:
    return bool(
        kv_cache_groups
        and not getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False)
        and len(kv_cache_groups) > 1
        and any(not isinstance(group.kv_cache_spec, FullAttentionSpec) for group in kv_cache_groups)
    )


def infer_group_block_sizes(
    cache_block_size: int,
    kv_cache_groups: Sequence[Any] | None,
) -> list[int]:
    if not kv_cache_groups:
        return [cache_block_size]

    block_sizes: list[int] = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        block_sizes.append(spec.block_size)
    return block_sizes


def infer_cacheable_group_ids(kv_cache_groups: Sequence[Any] | None) -> list[int]:
    if not kv_cache_groups:
        return [0]
    group_ids = [i for i, group in enumerate(kv_cache_groups) if is_prefix_cacheable(group.kv_cache_spec)]
    assert group_ids, "AscendStore requires at least one prefix-cacheable KV cache group"
    return group_ids


def get_group_block_size(group_block_sizes: Sequence[int], group_id: int) -> int:
    return group_block_sizes[group_id] if group_id < len(group_block_sizes) else group_block_sizes[0]


def get_group_cache_family(group_cache_families: Sequence[str], group_id: int) -> str:
    return group_cache_families[group_id] if group_id < len(group_cache_families) else "default"


def infer_cache_transfer_granularity(
    group_block_sizes: Sequence[int],
    lcm_block_size: int,
    kv_cache_group_ids: Sequence[int],
) -> int:
    granularities = [lcm_block_size]
    for group_id in kv_cache_group_ids:
        granularities.append(get_group_block_size(group_block_sizes, group_id))
    return math.lcm(*granularities)
