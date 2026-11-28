# SPDX-License-Identifier: Apache-2.0
"""Block selection and prompt truncation helpers for the mooncake v1 schedulers.

The pull scheduler (``mooncake_connector``) and the layerwise scheduler
(``mooncake_layerwise_connector``) used to keep these helpers as private
methods. They only read scheduler state and never touch threads/engines, so they
are extracted here as explicit-parameter functions; the scheduler classes keep
thin same-name delegates for compatibility with existing callers and tests.

Dependency note: unlike the other ``mooncake_v1.common`` modules this one
imports a few vLLM *data types* (``BlockIds``, the KV-cache spec classes) and
``cdiv``. Those are needed to keep ``isinstance`` checks and the numeric
semantics byte-identical; no vLLM worker/scheduler/engine/thread objects are
imported, and there is no dependency back on the old V1 connector modules.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.utils import get_pp_indices
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.utils.utils import truncate_block_hashes_fallback

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass(frozen=True)
class GroupTransferInfo:
    tokens_per_block: int
    blocks_per_window: int
    is_state_group: bool
    dcp_sharded: bool = True


def group_unique_specs(group: Any) -> list[Any]:
    if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return [group.kv_cache_spec]

    specs = []
    for layer_name in group.layer_names:
        layer_spec = group.kv_cache_spec.kv_cache_specs[layer_name]
        if layer_spec not in specs:
            specs.append(layer_spec)
    return specs


def build_group_transfer_info(group: Any, *, default_block_size: int) -> GroupTransferInfo:
    specs = group_unique_specs(group)
    first_spec = specs[0] if specs else group.kv_cache_spec
    block_size = getattr(group.kv_cache_spec, "block_size", getattr(first_spec, "block_size", default_block_size))
    is_state_group = any(isinstance(spec, MambaSpec) for spec in specs)
    sliding_window = 0
    compress_ratio = 1
    for spec in specs:
        if isinstance(spec, SlidingWindowSpec):
            sliding_window = spec.sliding_window
        elif hasattr(spec, "compress_ratio"):
            compress_ratio = spec.compress_ratio

    return GroupTransferInfo(
        tokens_per_block=block_size * max(1, int(compress_ratio)),
        blocks_per_window=cdiv(sliding_window, block_size) + 1 if sliding_window else 0,
        is_state_group=is_state_group,
        dcp_sharded=first_spec.dcp_sharded,
    )


def transfer_block_ids(
    block_ids: BlockIds,
    prompt_len: int,
    *,
    group_transfer_info: list[GroupTransferInfo],
    mamba_cache_mode: str | None,
    dcp_size: int,
) -> BlockIds:
    """Return blocks that contain prompt KV, dropping MTP extra blocks.

    In aligned Mamba mode, normalize each state group to the single block
    containing the final prompt state. This prevents the receiver from
    inferring a block index from a speculative *token* count. Non-aligned
    state groups keep their existing behavior. SWA tail clipping is handled
    as a separate step after this.
    """
    if len(block_ids) == 0:
        return block_ids

    assert len(block_ids) == len(group_transfer_info), "Number of KV cache groups must match"

    transfer_block_ids = []
    for blocks, group_info in zip(block_ids, group_transfer_info):
        is_aligned_state_group = group_info.is_state_group and mamba_cache_mode == "align"
        if group_info.is_state_group and not is_aligned_state_group:
            transfer_block_ids.append(blocks)
        elif is_aligned_state_group:
            # Mamba state is not DCP-sharded like attention KV. Its aligned
            # block index is derived from the actual (already truncated)
            # prompt length, without multiplying by the DCP size.
            num_prompt_state_blocks = cdiv(prompt_len, group_info.tokens_per_block)
            if num_prompt_state_blocks <= 0 or num_prompt_state_blocks > len(blocks):
                raise RuntimeError(
                    "Invalid aligned Mamba state block metadata: "
                    f"prompt_len={prompt_len}, tokens_per_block={group_info.tokens_per_block}, "
                    f"required_block_count={num_prompt_state_blocks}, available_block_count={len(blocks)}."
                )
            transfer_block_ids.append(blocks[num_prompt_state_blocks - 1 : num_prompt_state_blocks])
        else:
            # Each scheduler-visible block id is a DCP-grouped virtual
            # block shared by all DCP ranks.
            group_dcp_size = dcp_size if group_info.dcp_sharded else 1
            num_prompt_blocks = cdiv(prompt_len, group_info.tokens_per_block * group_dcp_size)
            transfer_block_ids.append(blocks[:num_prompt_blocks])
    return tuple(transfer_block_ids)


def swa_transfer_block_ids(block_ids: BlockIds, *, group_transfer_info: list[GroupTransferInfo]) -> BlockIds:
    """Clip SWA groups to their window tail and drop placeholder block 0."""
    if len(block_ids) == 0:
        return block_ids

    assert len(block_ids) == len(group_transfer_info), "Number of KV cache groups must match"

    transfer_block_ids = []
    for blocks, group_info in zip(block_ids, group_transfer_info):
        if group_info.is_state_group or group_info.blocks_per_window == 0:
            transfer_block_ids.append(blocks)
        else:
            window_blocks = blocks[-group_info.blocks_per_window :]
            transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
    return tuple(transfer_block_ids)


def state_prefill_token_count(num_prompt_tokens: int, *, need_truncate: bool) -> int:
    """D-side only. Returns N-1 for Mamba models since the decoder
    always recomputes the last token and must start from h(N-1)."""
    if need_truncate and num_prompt_tokens > 1:
        return num_prompt_tokens - 1
    return num_prompt_tokens


def truncate_request_for_prefill(
    request: "Request",
    *,
    hash_block_size: int | None = None,
    num_hash_lookahead_tokens: int = 0,
) -> None:
    """P-side only: drop the last prompt token so the prefiller computes
    h(N-1) instead of h(N). The decoder recomputes the last token to
    derive h(N) correctly.

    Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
    request is preempted and rescheduled. The hash tail is trimmed in place so
    the prefix is not rehashed (see the matching scheduler in
    ``kv_p2p/mooncake/base_scheduler.py``)."""
    params = request.kv_transfer_params
    if (
        params is not None
        # Guard against repeated truncation after preemption/reschedule.
        and not params.get("_p_side_truncated")
        and request.num_prompt_tokens > 1
    ):
        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return

        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        # The request already hashed the full prompt; trim the stale tail
        # instead of rebuilding the whole prefix.
        assert hash_block_size is not None
        truncate_hashes = getattr(request, "truncate_block_hashes", None)
        if callable(truncate_hashes):
            truncate_hashes(hash_block_size, num_hash_lookahead_tokens)
        else:
            truncate_block_hashes_fallback(request, hash_block_size, num_hash_lookahead_tokens)
        params["_p_side_truncated"] = True


def iter_kv_cache_specs(kv_cache_config: "KVCacheConfig") -> Iterator[Any]:
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            yield from kv_cache_spec.kv_cache_specs.values()
        else:
            yield kv_cache_spec


def has_attn_mamba_hybrid_cache(kv_cache_config: "KVCacheConfig") -> bool:
    has_attn = False
    has_mamba = False
    for kv_cache_spec in iter_kv_cache_specs(kv_cache_config):
        has_attn = has_attn or isinstance(kv_cache_spec, AttentionSpec)
        has_mamba = has_mamba or isinstance(kv_cache_spec, MambaSpec)
    return has_attn and has_mamba


def hybrid_prefill_token_count(num_prompt_tokens: int, *, need_truncate: bool) -> int:
    if need_truncate and num_prompt_tokens > 1:
        return num_prompt_tokens - 1
    return num_prompt_tokens


def truncate_request_for_hybrid_prefill(request: "Request", *, need_truncate: bool) -> None:
    params = request.kv_transfer_params
    if (
        params is None
        or not need_truncate
        or params.get("_p_side_truncated")
        or getattr(request, "num_prompt_tokens", len(request.prompt_token_ids or [])) <= 1
    ):
        return

    if request.prompt_token_ids is not None:
        request.prompt_token_ids.pop()
    elif request.prompt_embeds is not None:
        request.prompt_embeds = request.prompt_embeds[:-1]
    else:
        return

    request._all_token_ids.pop()
    request.num_prompt_tokens -= 1
    request.max_tokens = 1
    params["_p_side_truncated"] = True


def trim_hybrid_remote_block_ids(
    block_ids: tuple[list[int], ...],
    prompt_len: int,
    *,
    need_truncate: bool,
    block_sizes: list[int],
) -> tuple[list[int], ...]:
    if not need_truncate or prompt_len <= 1:
        return block_ids

    trimmed_block_ids: list[list[int]] = []
    for group_block_ids, block_size in zip(block_ids, block_sizes):
        if prompt_len % block_size == 1:
            trimmed_block_ids.append(list(group_block_ids[:-1]))
        else:
            trimmed_block_ids.append(list(group_block_ids))
    return tuple(trimmed_block_ids)


def is_dcp_replicated_attention_group(group_spec: dict[str, Any]) -> bool:
    """Attention groups with complete sequences use TP routing without DCP slicing."""
    return group_spec.get("kv_cache_spec_type") != "MambaSpec" and not group_spec.get("dcp_sharded", True)


__all__ = [
    "GroupTransferInfo",
    "build_group_transfer_info",
    "group_unique_specs",
    "has_attn_mamba_hybrid_cache",
    "hybrid_prefill_token_count",
    "is_dcp_replicated_attention_group",
    "iter_kv_cache_specs",
    "state_prefill_token_count",
    "swa_transfer_block_ids",
    "transfer_block_ids",
    "trim_hybrid_remote_block_ids",
    "truncate_request_for_hybrid_prefill",
    "truncate_request_for_prefill",
]


def transfer_groups_need_independent_block_ids(
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
    block_size_scale: list[list[int]],
) -> bool:
    """Whether split transfer groups need separately expanded block IDs.

    Transfer groups that share one KV cache manager group normally share the
    same logical block table. They only need independent block-id lists when
    their tensor layouts use different logical-to-kernel block scales.
    """
    group_scales: dict[int, int] = {}
    for group_idx, (group_spec, layer_indices) in kv_group2layeridx.items():
        if is_dcp_replicated_attention_group(group_spec):
            return True
        if group_spec.get("kv_cache_spec_type") == "MambaSpec":
            continue
        kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
        scale = 1
        if layer_indices and layer_indices[0] < len(block_size_scale) and block_size_scale[layer_indices[0]]:
            scale = block_size_scale[layer_indices[0]][0]
        previous_scale = group_scales.setdefault(kv_cache_group_id, scale)
        if previous_scale != scale:
            return True
    return False


# decode node should know pp_partition_layer in prefill node,
# it is configured in kv_transfer_config by partition_list_str,
# default using vllm layer split algorithm.
def build_layer_name_to_metadata_idx(
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
) -> dict[str, int]:
    layer_name_to_idx: dict[str, int] = {}
    for group_spec, layer_indices in kv_group2layeridx.values():
        layer_names = group_spec.get("layer_names", [])
        for layer_name, layer_idx in zip(layer_names, layer_indices):
            layer_name_to_idx[layer_name] = layer_idx
    return layer_name_to_idx


def resolve_remote_layer_idx(
    local_layer_idx: int,
    group_spec: dict[str, Any],
    local_layer_indices: list[int],
    remote_layer_name_to_idx: dict[str, int],
) -> int:
    """Resolve a remote metadata index by the stable cache-layer name."""
    layer_names = group_spec.get("layer_names", [])
    try:
        position = local_layer_indices.index(local_layer_idx)
        layer_name = layer_names[position]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Cannot resolve local KV metadata index {local_layer_idx} to a layer name.") from error
    try:
        return remote_layer_name_to_idx[layer_name]
    except KeyError as error:
        raise RuntimeError(f"Remote KV metadata does not contain layer {layer_name!r}.") from error


def get_prefill_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int, partition_list_str: str | None = None
) -> tuple[int, int]:
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    else:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
        return (start_layer, end_layer)
