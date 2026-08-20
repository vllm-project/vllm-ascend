from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class AscendSFAIndexerBackend(AttentionBackend):
    """Placeholder backend for split SFA indexer cache layers.

    The SFA indexer cache is represented as its own AttentionLayerBase so the
    KV-cache planner can assign an independent physical tensor while sharing
    block ids with the main MLA cache group. The current SFA forward path still
    consumes metadata from the real ``*.attn`` layer and recomposes the legacy
    cache tuple before calling the kernel, so this backend only needs to make
    the indexer cache visible to cache initialization.

    Do not reuse AscendSFAMetadataBuilder here. It inherits vLLM's
    MLACommonMetadataBuilder, whose initializer assumes layer_names[0] points to
    a real MLAAttention object with ``prefill_backend`` in static_forward_context.
    The indexer cache layer points to DeepseekV32IndexerCache instead, which has
    no ``prefill_backend``. Keeping a cache-only builder avoids that false
    attention-layer assumption and avoids building unused indexer metadata.
    """

    accept_output_buffer: bool = True

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_INDEXER"

    @staticmethod
    def get_builder_cls():
        return AscendSFAIndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendSFAIndexerMetadataBuilder(AttentionMetadataBuilder[Any]):
    """Cache-only metadata builder for split SFA indexer cache layers."""

    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> None:
        return None


def validate_indexer_pp_stage(
    config: Any,
    start_layer: int,
    end_layer: int,
    pp_rank: int,
    pp_size: int,
) -> None:
    """Validate Indexer dependencies against a PP stage boundary.

    This covers explicit IndexShare configurations, where ``shared`` layers
    reuse Top-K indices from a preceding ``full`` layer, and runtime IndexCache
    configurations, where a layer skips Top-K computation and reads previously
    cached indices. Since Top-K indices are worker-local and are not propagated
    in PP intermediate tensors, each PP stage must begin with a layer that
    recomputes the indices.

    Models without either ``indexer_types`` or ``use_index_cache`` are left
    unchanged.
    """
    if pp_size <= 1 or start_layer >= end_layer:
        return

    indexer_types = getattr(config, "indexer_types", None)
    use_index_cache = getattr(config, "use_index_cache", False)
    if indexer_types is None and not use_index_cache:
        return

    if use_index_cache and _get_index_cache_skip_topk(config, start_layer):
        raise ValueError(
            "Index cache dependency crosses a pipeline-parallel stage boundary: "
            f"PP rank {pp_rank}/{pp_size} owns layers [{start_layer}, {end_layer}), "
            f"but layer {start_layer} skips Top-K computation without a preceding "
            "Top-K recomputation in the same PP stage. "
            "Cross-PP Top-K index propagation is not supported. "
            "Please choose a pipeline-parallel partition whose first layer "
            "recomputes the Top-K index."
        )

    if indexer_types is None:
        return

    has_full_indexer = False
    for layer_id in range(start_layer, end_layer):
        indexer_type = indexer_types[layer_id] if layer_id < len(indexer_types) else None
        normalized_type = indexer_type.lower() if isinstance(indexer_type, str) else None
        if normalized_type == "full":
            has_full_indexer = True
        elif normalized_type == "shared" and not has_full_indexer:
            raise ValueError(
                "IndexShare group crosses a pipeline-parallel stage boundary: "
                f"PP rank {pp_rank}/{pp_size} owns layers [{start_layer}, {end_layer}), "
                f"but layer {layer_id} uses a shared Indexer without a preceding "
                "full Indexer in the same PP stage. "
                "Cross-PP Top-K index propagation is not supported. "
                "Please choose a pipeline-parallel partition aligned to an IndexShare group."
            )


def validate_indexer_pp_partition(
    config: Any,
    num_hidden_layers: int,
    pp_size: int,
) -> None:
    """Validate all PP stages locally without cross-rank communication."""
    if pp_size <= 1:
        return

    from vllm.distributed.utils import get_pp_indices

    for pp_rank in range(pp_size):
        start_layer, end_layer = get_pp_indices(
            num_hidden_layers,
            pp_rank,
            pp_size,
        )
        try:
            validate_indexer_pp_stage(
                config,
                start_layer,
                end_layer,
                pp_rank,
                pp_size,
            )
        except ValueError as error:
            nearest_partitions = _get_nearest_valid_indexer_pp_partitions(
                config,
                num_hidden_layers,
                pp_size,
                invalid_pp_rank=pp_rank,
            )
            if not nearest_partitions:
                raise

            suggestions = [
                f'VLLM_PP_LAYER_PARTITION="{",".join(map(str, partition))}"' for partition in nearest_partitions
            ]
            partition_label = "partition is" if len(suggestions) == 1 else "partitions are"
            raise ValueError(f"{error} The nearest valid layer {partition_label} {' or '.join(suggestions)}.") from None


def _get_nearest_valid_indexer_pp_partitions(
    config: Any,
    num_hidden_layers: int,
    pp_size: int,
    invalid_pp_rank: int,
) -> list[list[int]]:
    if invalid_pp_rank <= 0 or invalid_pp_rank >= pp_size:
        return []

    from vllm.distributed.utils import get_pp_indices

    try:
        validate_indexer_pp_stage(
            config,
            start_layer=0,
            end_layer=num_hidden_layers,
            pp_rank=0,
            pp_size=pp_size,
        )
    except ValueError:
        return []

    current_boundaries = [get_pp_indices(num_hidden_layers, pp_rank, pp_size)[0] for pp_rank in range(1, pp_size)]
    valid_boundaries = []
    for boundary in range(1, num_hidden_layers):
        try:
            validate_indexer_pp_stage(
                config,
                boundary,
                num_hidden_layers,
                pp_rank=1,
                pp_size=pp_size,
            )
        except ValueError:
            continue
        valid_boundaries.append(boundary)

    if len(valid_boundaries) < pp_size - 1:
        return []

    invalid_boundary_index = invalid_pp_rank - 1

    def find_nearest_partition(move_boundary_higher: bool) -> list[int] | None:
        states: dict[int, tuple[int, tuple[int, ...]]] = {
            0: (0, ()),
        }
        for boundary_index, current_boundary in enumerate(current_boundaries):
            next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
            for boundary in valid_boundaries:
                if boundary_index == invalid_boundary_index:
                    if move_boundary_higher and boundary <= current_boundary:
                        continue
                    if not move_boundary_higher and boundary >= current_boundary:
                        continue

                candidates = [
                    (
                        cost + abs(boundary - current_boundary),
                        path + (boundary,),
                    )
                    for previous_boundary, (cost, path) in states.items()
                    if previous_boundary < boundary
                ]
                if candidates:
                    next_states[boundary] = min(candidates)
            states = next_states

        if not states:
            return None

        _, nearest_boundaries = min(states.values())
        boundaries = (0, *nearest_boundaries, num_hidden_layers)
        return [boundaries[index + 1] - boundaries[index] for index in range(pp_size)]

    return [
        partition
        for partition in (
            find_nearest_partition(move_boundary_higher=False),
            find_nearest_partition(move_boundary_higher=True),
        )
        if partition is not None
    ]


def _get_index_cache_skip_topk(config: Any, layer_id: int) -> bool:
    """Mirror the existing IndexCache skip decision for PP validation only."""
    index_topk_freq = getattr(config, "index_topk_freq", 1)
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)

    if index_topk_pattern is None:
        return max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
    if 0 <= layer_id < len(index_topk_pattern):
        return index_topk_pattern[layer_id] == "S"
    return False
