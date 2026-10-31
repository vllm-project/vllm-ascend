# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pool adapters for V2 state-transfer rules; no separate MTP conversion."""

import json
from math import prod

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolReadRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import CacheComponentBinding
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import (
    CacheSlice,
    SourceCacheSlice,
    TargetCacheSlice,
)
from vllm_ascend.distributed.kv_transfer.utils.state_layout import mamba_component_copy_slices

MAMBA_FORMAT_PREFIX = "mamba-v2:"


def _raw_page(binding: CacheComponentBinding, block: int, token_end: int, signature: str) -> TargetCacheSlice:
    if type(block) is not int or not 0 < block < binding.shape[0] or binding.block_size_scale != 1:
        raise ValueError("State snapshot requires a valid non-null scheduler page")
    if type(token_end) is not int or token_end <= 0:
        raise ValueError("State snapshot requires a positive checkpoint boundary")
    expected = binding.element_size
    for size, stride in reversed(list(zip(binding.shape[1:], binding.byte_strides[1:], strict=True))):
        if size > 1 and stride != expected:
            raise ValueError("V2 raw-state transfer requires contiguous component contents")
        expected *= size
    span = binding.page_span
    cache = CacheSlice(
        binding.layer, str(binding.component), signature, token_end - 1, token_end, 0, 1, span, span, span
    )
    return TargetCacheSlice(cache, binding.address + block * binding.byte_strides[0])


def bind_mamba_checkpoint(
    components: tuple[CacheComponentBinding, ...],
    spec: MambaSpec,
    block_ids: list[int],
    token_end: int,
    *,
    rank: int,
    size: int,
) -> list[TargetCacheSlice]:
    """Bind the current page including exactly the components V2 transfers.

    The column follows mamba_get_block_table_tensor. Async D allocation has
    zero new query tokens, so its column is the imported checkpoint boundary.
    """
    if spec.mamba_cache_mode not in ("none", "align") or token_end <= 0:
        raise ValueError("Mamba transfer requires a valid none/align state checkpoint")
    if not 0 <= rank < size or len(components) != len(spec.dtypes):
        raise ValueError("Mamba state topology or component count is invalid")
    index = 0 if spec.mamba_cache_mode == "none" else (token_end - 1) // spec.block_size
    if index >= len(block_ids):
        raise ValueError("Mamba checkpoint block table is incomplete")
    signature = MAMBA_FORMAT_PREFIX + json.dumps(
        {
            "mamba_type": spec.mamba_type.name,
            "shapes": [list(component.shape[1:]) for component in components],
            "lengths": [component.page_span for component in components],
            "dtypes": [component.dtype for component in components],
            "element_sizes": [component.element_size for component in components],
            "tp_replicated": spec.tp_replicated,
            "rank": rank,
            "size": size,
            "checkpoint": token_end,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return [_raw_page(component, block_ids[index], token_end, signature) for component in components]


def bind_circular_snapshot(binding: CacheComponentBinding, block_ids: list[int], token_end: int) -> TargetCacheSlice:
    """V2 circular/tail rule: identical layout, one complete request page."""
    if len(block_ids) != 1:
        raise ValueError("Circular state requires exactly one request-lifetime page")
    signature = json.dumps(
        {
            "family": "circular-v2",
            "dtype": binding.dtype,
            "shape": binding.shape[1:],
            "checkpoint": token_end,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _raw_page(binding, block_ids[0], token_end, signature)


def _mamba_description(cache: CacheSlice) -> dict:
    try:
        value = json.loads(cache.format_id[len(MAMBA_FORMAT_PREFIX) :])
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid Mamba state description") from exc
    required = {
        "mamba_type",
        "shapes",
        "lengths",
        "dtypes",
        "element_sizes",
        "tp_replicated",
        "rank",
        "size",
        "checkpoint",
    }
    if not isinstance(value, dict) or value.keys() != required:
        raise ValueError("Invalid Mamba state description fields")
    if any(type(value[name]) is not int for name in ("rank", "size", "checkpoint")):
        raise ValueError("Invalid Mamba state coordinates")
    if not 0 <= value["rank"] < value["size"] or value["checkpoint"] != cache.token_end:
        raise ValueError("Mamba state checkpoint/topology mismatch")
    if type(value["tp_replicated"]) is not bool or value["mamba_type"] not in MambaAttentionBackendEnum.__members__:
        raise ValueError("Invalid Mamba state type")
    arrays = [value[name] for name in ("shapes", "lengths", "dtypes", "element_sizes")]
    if not all(isinstance(array, list) and array for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("Mamba state component metadata count mismatch")
    for shape, length, dtype, width in zip(*arrays, strict=True):
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or type(length) is not int
            or type(width) is not int
            or width <= 0
            or prod(shape) * width != length
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise ValueError("Invalid Mamba component geometry")
    component = int(cache.component)
    if not 0 <= component < len(arrays[0]) or value["lengths"][component] != cache.vector_bytes:
        raise ValueError("Mamba component byte size mismatch")
    return value


def plan_mamba_restore(sources: list[SourceCacheSlice], targets: list[TargetCacheSlice]) -> list[PoolReadRange]:
    """Translate V2's byte offsets into fully covered pool key/offset reads."""
    plan = []
    for target in targets:
        local = _mamba_description(target.cache)
        component = int(target.cache.component)
        coverage = []
        selected_ranks = set()
        for source in sources:
            if (source.cache.layer, source.cache.component) != (target.cache.layer, target.cache.component):
                continue
            remote = _mamba_description(source.cache)
            if (
                remote["mamba_type"] != local["mamba_type"]
                or remote["dtypes"] != local["dtypes"]
                or remote["element_sizes"] != local["element_sizes"]
                or remote["checkpoint"] != local["checkpoint"]
                or remote["tp_replicated"] != local["tp_replicated"]
            ):
                raise ValueError("Incompatible V2 Mamba state metadata")
            local_size, remote_size = local["size"], remote["size"]
            local_rank, remote_rank = local["rank"], remote["rank"]
            ratio, remainder = divmod(max(local_size, remote_size), min(local_size, remote_size))
            if remainder and not local["tp_replicated"]:
                raise ValueError("Mooncake Mamba head shard sizes must have an integer ratio")
            if local["tp_replicated"]:
                if selected_ranks:
                    continue
                rank_offset = 0
            elif remote_size >= local_size:
                if remote_rank // ratio != local_rank:
                    continue
                rank_offset = remote_rank % ratio
            else:
                if local_rank // ratio != remote_rank:
                    continue
                rank_offset = local_rank % ratio
            if (remote_size, remote_rank) in selected_ranks:
                continue
            selected_ranks.add((remote_size, remote_rank))
            offsets = mamba_component_copy_slices(
                mamba_type=MambaAttentionBackendEnum[local["mamba_type"]],
                dtype_size=local["element_sizes"][component],
                component=component,
                local_shapes=[tuple(shape) for shape in local["shapes"]],
                remote_shapes=[tuple(shape) for shape in remote["shapes"]],
                local_block_len=local["lengths"][component],
                remote_block_len=remote["lengths"][component],
                local_size=local_size,
                remote_size=remote_size,
                rank_offset=rank_offset,
                tp_replicated=local["tp_replicated"],
            )
            for local_offset, remote_offset, length in offsets:
                if not 0 <= local_offset <= target.cache.vector_bytes - length:
                    raise ValueError("Mamba restore exceeds local state component")
                plan.append(
                    PoolReadRange(
                        source.key,
                        source.object_size,
                        source.object_offset + remote_offset,
                        target.address + local_offset,
                        length,
                    )
                )
                coverage.append((local_offset, local_offset + length))
        end = 0
        for start, stop in sorted(coverage):
            if start != end:
                raise ValueError("Incomplete or overlapping V2 Mamba restore coverage")
            end = stop
        if end != target.cache.vector_bytes:
            raise ValueError("Incomplete V2 Mamba restore coverage")
    return plan
