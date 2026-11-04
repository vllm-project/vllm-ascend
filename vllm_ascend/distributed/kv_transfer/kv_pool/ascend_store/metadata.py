from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorWorkerMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    KeyMetadata as KeyMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    PoolKey as PoolKey,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    block_hash_to_bytes as block_hash_to_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    block_hash_to_str as block_hash_to_str,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    get_block_hashes as get_block_hashes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.token_database import (
    ChunkedTokenDatabase as ChunkedTokenDatabase,
)


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


def normalize_block_ids_by_group(block_ids: tuple[list[int], ...] | list[int] | list[list[int]]) -> list[list[int]]:
    if isinstance(block_ids, tuple):
        return [group.copy() for group in block_ids]
    if isinstance(block_ids, list):
        if not block_ids:
            return [[]]
        if isinstance(block_ids[0], list):
            grouped_block_ids = cast(list[list[int]], block_ids)
            return [group.copy() for group in grouped_block_ids]
        flat_block_ids = cast(list[int], block_ids)
        return [flat_block_ids.copy()]
    raise ValueError(f"Unsupported block_ids type {type(block_ids)}")


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool
    # Raw KVPool hit length used to avoid storing an already pooled prefix.
    kvpool_store_skip_tokens: int | None = None

    token_len: int = 0

    # Exact request-scoped snapshot identity, separate from the logical length
    # which can roll back after speculative-token rejection.
    layerwise_snapshot_end_token: int | None = None
    layerwise_snapshot_block_index: int | None = None
    # Distinguish the preceding step of this request from external prefix hits.
    layerwise_previous_step: bool = False
    # First block stored under request-scoped keys, including unhashed full blocks.
    layerwise_snapshot_start_block: int | None = None


@dataclass(init=False)
class RequestTracker:
    # Request id
    req_id: str

    token_len: int

    # The block ids that has been allocated so far, grouped by KV cache group.
    # NOTE: allocated blocks could be more than the number of tokens.
    allocated_block_ids_by_group: list[list[int]]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    # The token ids that has been scheduled so far
    # NOTE: This field will only be used when you enable kv-event
    token_ids: list[int] | None = None

    # Full prompt length before chunk truncation, used by sparse retention masks.
    num_prompt_tokens: int | None = None

    # End of the range being (re)prefilled. For a resumed request this also
    # includes previously generated tokens, so decode-only offload does not
    # publish intermediate recomputation state.
    prefill_end_tokens: int | None = None
    block_gvas: list[int] = field(default_factory=list)
    block_gvas_by_group: list[list[int]] = field(default_factory=list)
    gva_block_offset: int = 0
    last_block_gva: int | None = None

    # Identity of the latest request-scoped layerwise partial snapshot.
    last_layerwise_snapshot_end_token: int | None = None
    last_layerwise_snapshot_block_index: int | None = None
    last_layerwise_snapshot_start_block: int | None = None

    # Number of speculative scratch blocks for each Mamba cache group.
    num_speculative_blocks_by_group: dict[int, int] | None = None

    block_sizes: list[int] | None = None

    def __init__(
        self,
        req_id: str,
        token_len: int,
        allocated_block_ids_by_group: list[list[int]] | None = None,
        allocated_block_ids: list[int] | list[list[int]] | None = None,
        num_saved_tokens: int = 0,
        token_ids: list[int] | None = None,
        num_prompt_tokens: int | None = None,
        prefill_end_tokens: int | None = None,
        block_gvas: list[int] | None = None,
        block_gvas_by_group: list[list[int]] | None = None,
        gva_block_offset: int = 0,
        last_block_gva: int | None = None,
        num_speculative_blocks_by_group: dict[int, int] | None = None,
        block_sizes: list[int] | None = None,
        last_layerwise_snapshot_end_token: int | None = None,
        last_layerwise_snapshot_block_index: int | None = None,
        last_layerwise_snapshot_start_block: int | None = None,
    ) -> None:
        self.req_id = req_id
        self.token_len = token_len
        self.num_speculative_blocks_by_group = num_speculative_blocks_by_group
        block_ids = allocated_block_ids_by_group
        if block_ids is None:
            block_ids = normalize_block_ids_by_group(allocated_block_ids or [])
        self.allocated_block_ids_by_group = block_ids
        self.num_saved_tokens = num_saved_tokens
        self.token_ids = token_ids
        self.num_prompt_tokens = num_prompt_tokens
        self.prefill_end_tokens = prefill_end_tokens
        self.block_gvas = [] if block_gvas is None else block_gvas
        self.block_gvas_by_group = block_gvas_by_group if block_gvas_by_group is not None else []
        self.gva_block_offset = gva_block_offset
        self.last_block_gva = last_block_gva
        self.block_sizes = block_sizes
        self.last_layerwise_snapshot_end_token = last_layerwise_snapshot_end_token
        self.last_layerwise_snapshot_block_index = last_layerwise_snapshot_block_index
        self.last_layerwise_snapshot_start_block = last_layerwise_snapshot_start_block

    @property
    def allocated_block_ids(self) -> list[int]:
        return self.allocated_block_ids_by_group[0] if self.allocated_block_ids_by_group else []

    @allocated_block_ids.setter
    def allocated_block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.allocated_block_ids_by_group = normalize_block_ids_by_group(block_ids)

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
        num_computed_tokens: int = 0,
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        normalized = normalize_block_ids_by_group(new_block_ids)
        if len(normalized) > len(self.allocated_block_ids_by_group):
            self.allocated_block_ids_by_group.extend(
                [[] for _ in range(len(normalized) - len(self.allocated_block_ids_by_group))]
            )
        for group_id, ids in enumerate(normalized):
            self.update_mamba_spec_blocks(ids, group_id, num_computed_tokens)
            self.allocated_block_ids_by_group[group_id].extend(ids)

    def update_mamba_spec_blocks(self, block_ids: list[int], kv_cache_group_id: int, num_computed_tokens: int):
        """
        for mamba align groups, each step will:
            - Firstly, remove some previous blocks and append some necessary null blocks
            - Secondly, move the speculative blocks(maybe all or partially) to the last position for reuse
            - Finally, allocate a new block
        so, if a speculative block is moved to last position and replaced with null block,
        we also need to update the previous allocated_block_ids to 0.
        """
        if (
            self.num_speculative_blocks_by_group is not None
            and (num_speculative_blocks := self.num_speculative_blocks_by_group.get(kv_cache_group_id)) is not None
        ):
            assert self.block_sizes is not None and len(self.block_sizes) > kv_cache_group_id
            num_skipped_blocks = (
                max(num_computed_tokens - num_speculative_blocks - 1, 0) // self.block_sizes[kv_cache_group_id]
            )
            num_skipped_blocks = min(len(self.allocated_block_ids_by_group[kv_cache_group_id]), num_skipped_blocks)
            if num_skipped_blocks > 0:
                self.allocated_block_ids_by_group[kv_cache_group_id][:num_skipped_blocks] = [0] * num_skipped_blocks
            if not block_ids or num_speculative_blocks <= 0:
                return
            mask_spec_count = min(len(block_ids) - 1, num_speculative_blocks)
            group_block_ids = self.allocated_block_ids_by_group[kv_cache_group_id]
            if mask_spec_count >= num_speculative_blocks:
                group_block_ids[-num_speculative_blocks:] = [0] * num_speculative_blocks
            else:
                group_block_ids[-num_speculative_blocks : mask_spec_count - num_speculative_blocks] = [0] * (
                    mask_spec_count
                )


@dataclass(init=False)
class ReqMeta:
    # Request id
    req_id: str
    token_len_chunk: int
    # End token for full-block KV save.
    save_end_token: int
    # Token length after this scheduled step finishes.
    target_token_len: int
    block_ids_by_group: list[list[int]]

    block_hashes: list[BlockHash]

    # First token that has not been saved before this metadata was built.
    save_start_token: int = 0

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    current_event: torch.npu.Event | None = None
    kv_cache_group_ids: list[int] | None = None
    skip_null_blocks_by_group: list[bool] | None = None
    num_prompt_tokens: int | None = None

    # The following parameters are only used for kv event generation
    # TODO: add lora_request which used for gen lora_id/lora_name in kv event
    token_ids: list[int] | None = None
    original_block_size: list[int] | int | None = None

    event_id: int | None = None
    # Monotonic identity of one queued store operation. Unlike req_id, this
    # value is never reused while the engine is alive, so an old asynchronous
    # operation cannot update the save progress of a new request generation
    # that happens to reuse the same req_id.
    store_job_id: int | None = None

    def __init__(
        self,
        req_id: str,
        token_len_chunk: int | None = None,
        block_ids_by_group: list[list[int]] | None = None,
        block_hashes: list[BlockHash] | None = None,
        can_save: bool | None = None,
        load_spec: LoadSpec | None = None,
        is_last_chunk: bool | None = None,
        current_event: torch.npu.Event | None = None,
        kv_cache_group_ids: list[int] | None = None,
        skip_null_blocks_by_group: list[bool] | None = None,
        num_prompt_tokens: int | None = None,
        token_ids: list[int] | None = None,
        original_block_size: list[int] | int | None = None,
        block_ids: list[int] | list[list[int]] | None = None,
        event_id: int | None = None,
        store_job_id: int | None = None,
        save_end_token: int | None = None,
        target_token_len: int | None = None,
        save_start_token: int = 0,
        last_block_gva: int | None = None,
        partial_block_index: int | None = None,
        block_ids_np: np.ndarray | None = None,
        block_ids_by_group_np: list[np.ndarray] | None = None,
        block_gvas_np: np.ndarray | None = None,
        block_gvas_by_group_np: list[np.ndarray] | None = None,
        gva_block_offset: int = 0,
        load_block_gvas_np: np.ndarray | None = None,
        load_block_gvas_by_group_np: list[np.ndarray] | None = None,
        load_gva_block_offset: int = 0,
        partial_save_gva_per_group: list[int] | None = None,
        partial_load_gva_per_group: list[int] | None = None,
        save_block_keys: list[str | None] | None = None,
        save_key_block_offset: int = 0,
        save_last_block_key: str | None = None,
        load_block_keys: list[str | None] | None = None,
        load_key_block_offset: int = 0,
        load_last_block_key: str | None = None,
        load_keys: list[str] | None = None,
    ) -> None:
        if token_len_chunk is None:
            token_len_chunk = 0 if save_end_token is None else save_end_token
        self.req_id = req_id
        self.token_len_chunk = token_len_chunk
        self.save_end_token = token_len_chunk if save_end_token is None else save_end_token
        self.target_token_len = token_len_chunk if target_token_len is None else target_token_len
        self.save_start_token = save_start_token
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.block_hashes = [] if block_hashes is None else block_hashes
        self.can_save = can_save
        self.load_spec = load_spec
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event
        self.kv_cache_group_ids = kv_cache_group_ids
        self.skip_null_blocks_by_group = skip_null_blocks_by_group
        self.num_prompt_tokens = num_prompt_tokens
        self.token_ids = token_ids
        self.original_block_size = original_block_size
        self.event_id = event_id
        self.store_job_id = store_job_id
        self.last_block_gva = last_block_gva
        self.partial_block_index = partial_block_index
        self.block_ids_np = block_ids_np
        self.block_ids_by_group_np = block_ids_by_group_np
        self.block_gvas_np = block_gvas_np
        self.block_gvas_by_group_np = block_gvas_by_group_np
        self.gva_block_offset = gva_block_offset
        self.load_block_gvas_np = load_block_gvas_np
        self.load_block_gvas_by_group_np = load_block_gvas_by_group_np
        self.load_gva_block_offset = load_gva_block_offset
        self.partial_save_gva_per_group = partial_save_gva_per_group or []
        self.partial_load_gva_per_group = partial_load_gva_per_group or []
        self.save_block_keys = [] if save_block_keys is None else list(save_block_keys)
        self.save_key_block_offset = save_key_block_offset
        self.save_last_block_key = save_last_block_key
        self.load_block_keys = [] if load_block_keys is None else list(load_block_keys)
        self.load_key_block_offset = load_key_block_offset
        self.load_last_block_key = load_last_block_key
        self.load_keys = [] if load_keys is None else list(load_keys)

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)

    def get_event_token_ids(self, start: int, end: int) -> list[int] | None:
        """Return token ids for an absolute request range.

        Metadata carries only the newly savable suffix to avoid repeatedly
        serializing the full request history during long decode sequences.
        An empty list means this metadata does not cover the requested range.
        """
        if self.token_ids is None:
            return None
        token_ids_end = self.save_start_token + len(self.token_ids)
        if start < self.save_start_token or end > token_ids_end:
            return []
        return self.token_ids[start - self.save_start_token : end - self.save_start_token]

    def get_block_ids_array(self, group_id: int) -> np.ndarray | None:
        if self.block_ids_by_group_np is not None and group_id < len(self.block_ids_by_group_np):
            return self.block_ids_by_group_np[group_id]
        return self.block_ids_np

    def get_transfer_arrays(self, group_id: int, is_save: bool = True) -> tuple[np.ndarray | None, np.ndarray | None]:
        grouped_ids = self.block_ids_by_group_np
        grouped_gvas = self.block_gvas_by_group_np if is_save else self.load_block_gvas_by_group_np
        if (
            grouped_ids is not None
            and grouped_gvas is not None
            and group_id < len(grouped_ids)
            and group_id < len(grouped_gvas)
        ):
            return grouped_ids[group_id], grouped_gvas[group_id]
        return self.block_ids_np, self.block_gvas_np if is_save else self.load_block_gvas_np

    last_block_gva: int | None = None
    partial_block_index: int | None = None
    save_keys: list[str] | None = None
    load_keys: list[str] = field(default_factory=list)
    save_block_keys: list[str | None] = field(default_factory=list)
    save_key_block_offset: int = 0
    save_last_block_key: str | None = None
    load_block_keys: list[str | None] = field(default_factory=list)
    load_key_block_offset: int = 0
    load_last_block_key: str | None = None

    block_ids_np: np.ndarray | None = None
    block_ids_by_group_np: list[np.ndarray] | None = None
    block_gvas_np: np.ndarray | None = None
    block_gvas_by_group_np: list[np.ndarray] | None = None
    gva_block_offset: int = 0
    load_block_gvas_np: np.ndarray | None = None
    load_block_gvas_by_group_np: list[np.ndarray] | None = None
    load_gva_block_offset: int = 0
    partial_save_gva_per_group: list[int] = field(default_factory=list)
    partial_load_gva_per_group: list[int] = field(default_factory=list)
    # Per-group reachable masks for the layerwise transfer, computed once per
    # scheduler step (None = no filtering, e.g. full-attention groups or when
    # the coordinator is unavailable).
    store_masks: tuple[Sequence[bool] | None, ...] | None = None
    load_masks: tuple[Sequence[bool] | None, ...] | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        cache_transfer_granularity: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: list[int] | int | None = None,
        save_partial_block: bool = False,
        hash_block_size: int | None = None,
    ) -> ReqMeta | None:
        """Create the request metadata from a request tracker."""
        if block_hashes is None:
            block_hashes = []
        target_token_len = tracker.token_len
        previous_saved_tokens = tracker.num_saved_tokens

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, cache_transfer_granularity) * cache_transfer_granularity
            if discard_partial_chunks
            else 0
        )
        num_tokens_to_save = (
            (target_token_len // cache_transfer_granularity * cache_transfer_granularity)
            if discard_partial_chunks
            else target_token_len
        )
        hash_block_size = hash_block_size or cache_transfer_granularity
        assert cache_transfer_granularity % hash_block_size == 0
        # Request hashes use hash_block_size, which may be finer than the
        # transfer granularity used to advance num_saved_tokens.
        hashes_per_transfer_block = cache_transfer_granularity // hash_block_size
        full_block_count = target_token_len // cache_transfer_granularity
        available_full_block_count = len(block_hashes) // hashes_per_transfer_block
        boundary_without_hash = (
            target_token_len > 0
            and target_token_len % cache_transfer_granularity == 0
            and full_block_count > available_full_block_count
        )
        # Scheduled draft tokens can cross a page before that page has a
        # committed request hash. Do not mark an unsent page as saved.
        if full_block_count > available_full_block_count:
            num_tokens_to_save = available_full_block_count * cache_transfer_granularity
        if tracker.last_block_gva is not None and (
            target_token_len % cache_transfer_granularity != 0 or boundary_without_hash
        ):
            partial_block_index = (
                full_block_count if target_token_len % cache_transfer_granularity != 0 else full_block_count - 1
            )
        else:
            partial_block_index = None

        should_save_partial_block = save_partial_block and (
            target_token_len % cache_transfer_granularity != 0 or full_block_count > available_full_block_count
        )
        skip_save = skip_save or (
            num_tokens_to_save < chunk_boundary and partial_block_index is None and not should_save_partial_block
        )
        # A ReqMeta must never carry both a save AND a load.
        # The save would also be wasted work — the bytes are being looked up
        # in the store right now. Later cached_reqs steps save new tokens
        # normally.
        if load_spec is not None and load_spec.can_load and not save_partial_block:
            skip_save = True
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = max(
                tracker.num_saved_tokens,
                num_tokens_to_save,
            )
            if should_save_partial_block:
                tracker.last_layerwise_snapshot_end_token = target_token_len
                tracker.last_layerwise_snapshot_start_block = min(full_block_count, available_full_block_count)
                tracker.last_layerwise_snapshot_block_index = (
                    full_block_count if target_token_len % cache_transfer_granularity != 0 else full_block_count - 1
                )
            elif save_partial_block:
                tracker.last_layerwise_snapshot_end_token = None
                tracker.last_layerwise_snapshot_block_index = None
                tracker.last_layerwise_snapshot_start_block = None

        token_ids = None
        if tracker.token_ids and not skip_save:
            # The worker consumes metadata asynchronously. Own a stable,
            # bounded snapshot rather than sharing the scheduler's growing
            # full-history list.
            token_ids_end = target_token_len if should_save_partial_block else num_tokens_to_save
            token_ids = tracker.token_ids[previous_saved_tokens:token_ids_end].copy()

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None
        logger.debug("request:%s, meta save spec:%s, meta load spec:%s", tracker.req_id, not skip_save, load_spec)
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            save_end_token=num_tokens_to_save,
            target_token_len=target_token_len,
            save_start_token=previous_saved_tokens,
            block_ids_by_group=tracker.allocated_block_ids_by_group,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            # A resumed MRV2 request re-prefills prompt + previously generated
            # tokens. Reachability masks must treat that complete range as the
            # prefill boundary for this generation.
            num_prompt_tokens=tracker.prefill_end_tokens or tracker.num_prompt_tokens or target_token_len,
            original_block_size=original_block_size,
            last_block_gva=tracker.last_block_gva,
            partial_block_index=partial_block_index,
            block_ids_np=np.asarray(tracker.allocated_block_ids, dtype=np.int64),
            block_ids_by_group_np=[np.asarray(ids, dtype=np.int64) for ids in tracker.allocated_block_ids_by_group]
            if tracker.allocated_block_ids_by_group
            else None,
            block_gvas_np=np.asarray(tracker.block_gvas, dtype=np.int64),
            block_gvas_by_group_np=[np.asarray(gvas, dtype=np.int64) for gvas in tracker.block_gvas_by_group]
            if hasattr(tracker, "block_gvas_by_group") and tracker.block_gvas_by_group
            else None,
            gva_block_offset=tracker.gva_block_offset,
            kv_cache_group_ids=list(range(len(tracker.allocated_block_ids_by_group))),
        )


class AscendConnectorMetadata(KVConnectorMetadata):
    def __init__(
        self,
        preempted_req_ids,
        loading_req_ids: set[str] | None = None,
    ):
        self.requests: list[ReqMeta] = []
        self.preempted_req_ids = preempted_req_ids
        self.loading_req_ids = loading_req_ids or set()
        # Requests whose blocks became reusable since the previous step
        # (finished + preempted); the worker fences their in-flight saves.
        self.released_req_ids: set[str] = set(preempted_req_ids or ())

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata."""
        self.requests.append(req_meta)


@dataclass
class AscendStoreKVConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    completed_events: dict[int, int] = field(default_factory=dict)
    """key: event_id, value: completed worker count"""

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, AscendStoreKVConnectorWorkerMetadata), (
            "aggregate worker metadata must be type of AscendStoreKVConnectorWorkerMetadata"
        )

        merged: dict[int, int] = dict(self.completed_events)
        for event_id in other.completed_events:
            if event_id not in merged:
                merged[event_id] = other.completed_events[event_id]
            else:
                merged[event_id] = merged[event_id] + other.completed_events[event_id]
        return AscendStoreKVConnectorWorkerMetadata(merged)
