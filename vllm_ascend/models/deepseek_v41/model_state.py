# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 metadata and indexer FULL graph selection."""

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.dsa_v41_indexer_shard import (
    V41_FULL_INDEXER_SHARD,
    should_shard_v41_full_decode,
    uses_v41_full_indexer_variants,
    v41_full_indexer_sharding,
)
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class DeepSeekV41ModelState(EngramModelState):
    """Choose the indexer variant once, before its metadata is prepared."""

    v41_full_indexer_sharded: bool = False

    @staticmethod
    def get_graph_manager_cls():
        # Import after the worker and model state have been initialized.
        from vllm_ascend.models.deepseek_v41.aclgraph import DeepSeekV41AclGraphManager

        return DeepSeekV41AclGraphManager

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata=None,
    ) -> dict[str, Any]:
        shard_indexer = V41_FULL_INDEXER_SHARD.get()
        if (
            shard_indexer is None
            and cudagraph_mode == CUDAGraphMode.FULL
            and uses_v41_full_indexer_variants(self.vllm_config)
        ):
            shard_indexer = False
            if not input_batch.is_dummy:
                batch = input_batch
                parallel = self.vllm_config.parallel_config
                min_rows = None
                if parallel.prefill_context_parallel_size > 1:
                    min_rows = 0
                    if self.pcp_manager is not None and self.pcp_manager.is_target_attn_groups(attn_groups):
                        # partition_batch has already computed this step's global
                        # history and real row counts on every PCP rank.
                        batch = self.pcp_manager.global_batch
                        min_rows = self.pcp_manager.min_local_num_tokens
                model_config = self.vllm_config.model_config
                assert model_config is not None
                shard_indexer = should_shard_v41_full_decode(
                    input_batch.num_tokens_after_padding,
                    parallel.tensor_parallel_size,
                    batch.num_computed_tokens_np[: batch.num_reqs],
                    int(model_config.hf_text_config.index_topk),
                    pcp_min_query_rows=min_rows,
                )

        # Replay consumes this exact decision; it does not repeat the heuristic.
        self.v41_full_indexer_sharded = shard_indexer is True
        with v41_full_indexer_sharding(shard_indexer):
            return super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=ubatch_idx,
                model_specific_attn_metadata=model_specific_attn_metadata,
            )
