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


def validate_indexshare_pp_partition(
    config: Any,
    start_layer: int,
    end_layer: int,
    pp_rank: int,
    pp_size: int,
) -> None:
    """Validate an IndexShare layer range against a PP stage boundary.

    IndexShare configurations describe layers as ``full`` or ``shared``.
    A ``shared`` layer reuses Top-K indices produced by an earlier ``full``
    layer. Since those indices are worker-local and are not propagated in PP
    intermediate tensors, each PP stage must encounter a local ``full`` layer
    before it encounters a ``shared`` layer.

    Models that do not expose ``indexer_types`` are not IndexShare models from
    this validator's perspective and are left unchanged.
    """
    if pp_size <= 1:
        return

    indexer_types = getattr(config, "indexer_types", None)
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
                f"but layer {layer_id} is shared before a full Indexer exists in this stage. "
                "Cross-PP Top-K index propagation is not supported. "
                "Please choose a pipeline-parallel partition aligned to an IndexShare group."
            )
