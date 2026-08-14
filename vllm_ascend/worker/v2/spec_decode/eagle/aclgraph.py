# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.aclgraph_utils import model_capture_wrapper
from vllm_ascend.worker.v2.updatable_graph import (
    SharedSource,
    UpdatableGraph,
)
from vllm_ascend.worker.v2.utils import communicator_switch


class EagleAclGraphManager(SpeculatorCudaGraphManager):
    """AclGraphManager for Eagle speculative decoding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ) -> None:
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )

        # set speculator attribute, so we can access attributes speculator
        # when call `run_fullgraph` method in CudaGraphManager,
        # then we don't need to # copy `propose` method in `AscendEagleSpeculator` class.
        self.speculator: Any = None
        self.is_draft_model_prefill = decode_query_len > 1

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for Eagle."""

        with communicator_switch(), model_capture_wrapper(
            self.speculator,
            self.is_draft_model_prefill,
        ):
            super().capture(
                forward_fn,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                progress_bar_desc=progress_bar_desc,
            )

    def run_fullgraph(
        self,
        desc: BatchExecutionDescriptor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        fia_params = self.speculator.build_fia_params(
            desc.num_reqs,
            self.is_draft_model_prefill,
        )
        resolved_tasks = graph.resolve_tasks(SharedSource(fia_params))

        assert self.update_stream is not None
        self.update_stream.wait_stream(torch.npu.current_stream())
        output = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return output
