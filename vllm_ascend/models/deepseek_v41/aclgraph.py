# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned FULL graph variants for the V4.1 indexer."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from functools import partial
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.dsa_v41_indexer_shard import (
    MIN_PCP_INDEXER_ROWS_PER_TP_RANK,
    uses_v41_full_indexer_variants,
    v41_full_indexer_sharding,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


@dataclass(frozen=True)
class _ShardedV41DecodeDescriptor(BatchExecutionDescriptor):
    """Distinct cache key for the sharded variant of the same FULL bucket."""


class DeepSeekV41AclGraphManager(ModelAclGraphManager):
    _capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int | Mapping[int, int],
        model_runner: Any,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: UBatchRunner | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
        )
        self._v41_sharded_descs: dict[BatchExecutionDescriptor, BatchExecutionDescriptor] = {}
        if cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY and uses_v41_full_indexer_variants(vllm_config):
            parallel = vllm_config.parallel_config
            min_rows = parallel.tensor_parallel_size
            if parallel.prefill_context_parallel_size > 1:
                min_rows *= MIN_PCP_INDEXER_ROWS_PER_TP_RANK
            full_descs = self._capture_descs.get(CUDAGraphMode.FULL, [])
            self._v41_sharded_descs = {
                desc: _ShardedV41DecodeDescriptor(**{field.name: getattr(desc, field.name) for field in fields(desc)})
                for desc in full_descs
                if desc.num_tokens >= min_rows
            }
            # Both variants participate in the inherited graph memory estimate.
            # Token sizes are identical, so shared attention buffers need no resize.
            full_descs.extend(self._v41_sharded_descs.values())

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if self.model_runner.model_state.v41_full_indexer_sharded:
            desc = self._v41_sharded_descs[desc]
        return super().run_fullgraph(desc)

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        pcp_manager: Any = None,
    ) -> None:
        capture = partial(
            super().capture,
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
            has_lora=has_lora,
            use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
            lora_capture_hook=lora_capture_hook,
            progress_bar_desc=progress_bar_desc,
            pcp_manager=pcp_manager,
        )
        if not self._v41_sharded_descs:
            capture()
            return
        capture_descs = self._capture_descs
        try:
            self._capture_descs = {
                **capture_descs,
                CUDAGraphMode.FULL: [
                    desc
                    for desc in capture_descs[CUDAGraphMode.FULL]
                    if not isinstance(desc, _ShardedV41DecodeDescriptor)
                ],
            }
            with v41_full_indexer_sharding(False):
                capture()
            self._capture_descs = {CUDAGraphMode.FULL: list(self._v41_sharded_descs.values())}
            with v41_full_indexer_sharding(True):
                capture()
        finally:
            self._capture_descs = capture_descs
