# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.utils import vllm_version_is
if not vllm_version_is("0.20.1"):
    from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import DecodeEagleCudaGraphManager, PrefillEagleCudaGraphManager
else:
    from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import EagleCudaGraphManager

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.utils import communicator_switch

if not vllm_version_is("0.20.1"):
    class PrefillEagleAclGraphManager(PrefillEagleCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)

            # set speculator attribute, so we can access attributes speculator
            # when call `run_fullgraph` method in CudaGraphManager,
            # then we don't need to # copy `propose` method in `AscendEagleSpeculator` class.
            self.speculator = speculator
            # capture_sizes sorts in ascending order.
            self.capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
            # vllm-ascend need to update draft graph params of attention backend.
            # so we need to set draft graph params before capture full graph.
            # `prefill` graph and `decodes` graph are different, `decode_query_len` can be used to distinguish them
            self.is_draft_model_prefill = decode_query_len > 1
            if super().needs_capture():
                if self.is_draft_model_prefill:
                    set_draft_graph_prefill_params(self.capture_sizes)
                else:
                    set_draft_graph_params(self.capture_sizes)

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
            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
                super().capture(
                    forward_fn,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    progress_bar_desc,
                )

        def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Override run_fullgraph to update full graph params in run_fullgraph."""
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(f"draft prefill run_fullgraph with num_tokens={num_tokens}")
            else:
                logger.info_once(f"draft run_fullgraph with num_tokens={num_tokens}")

            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(desc.num_reqs, self.is_draft_model_prefill)

            ret = super().run_fullgraph(desc)

            positions = self.speculator.input_buffers.positions[:num_tokens]
            # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
            # calculate num_tokens_across_dp.
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens, device=self.device)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ):
                # decide to update draft graph params
                _EXTRA_CTX.is_draft_model = True

                # decide to run `prefill` graph or `decodes` graph
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill

                forward_context = get_forward_context()
                update_full_graph_params(
                    # FIXME(Ronald1995): support hybrid attn backend
                    list(self.speculator.attn_backends.values())[0],
                    self.speculator.update_stream,
                    forward_context,
                    num_tokens,
                    self.vllm_config,
                    self.speculator.speculative_config,
                    positions.shape[0],
                    draft_attn_metadatas=draft_attn_metadatas,
                )
            return ret


    class DecodeEagleAclGraphManager(DecodeEagleCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)

            # set speculator attribute, so we can access attributes speculator
            # when call `run_fullgraph` method in CudaGraphManager,
            # then we don't need to # copy `propose` method in `AscendEagleSpeculator` class.
            self.speculator = speculator
            # capture_sizes sorts in ascending order.
            self.capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
            # vllm-ascend need to update draft graph params of attention backend.
            # so we need to set draft graph params before capture full graph.
            # `prefill` graph and `decodes` graph are different, `decode_query_len` can be used to distinguish them
            self.is_draft_model_prefill = decode_query_len > 1
            if super().needs_capture():
                if self.is_draft_model_prefill:
                    set_draft_graph_prefill_params(self.capture_sizes)
                else:
                    set_draft_graph_params(self.capture_sizes)

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
            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
                super().capture(
                    forward_fn,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    progress_bar_desc,
                )

        def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Override run_fullgraph to update full graph params in run_fullgraph."""
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(f"draft prefill run_fullgraph with num_tokens={num_tokens}")
            else:
                logger.info_once(f"draft run_fullgraph with num_tokens={num_tokens}")

            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(desc.num_reqs, self.is_draft_model_prefill)

            ret = super().run_fullgraph(desc)

            positions = self.speculator.input_buffers.positions[:num_tokens]
            # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
            # calculate num_tokens_across_dp.
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens, device=self.device)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ):
                # decide to update draft graph params
                _EXTRA_CTX.is_draft_model = True

                # decide to run `prefill` graph or `decodes` graph
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill

                forward_context = get_forward_context()
                update_full_graph_params(
                    # FIXME(Ronald1995): support hybrid attn backend
                    list(self.speculator.attn_backends.values())[0],
                    self.speculator.update_stream,
                    forward_context,
                    num_tokens,
                    self.vllm_config,
                    self.speculator.speculative_config,
                    positions.shape[0],
                    draft_attn_metadatas=draft_attn_metadatas,
                )
            return ret
        
else:
    class EagleAclGraphManager(EagleCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)

            # set speculator attribute, so we can access attributes speculator
            # when call `run_fullgraph` method in CudaGraphManager,
            # then we don't need to # copy `propose` method in `AscendEagleSpeculator` class.
            self.speculator = speculator
            # capture_sizes sorts in ascending order.
            self.capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
            # vllm-ascend need to update draft graph params of attention backend.
            # so we need to set draft graph params before capture full graph.
            # `prefill` graph and `decodes` graph are different, `decode_query_len` can be used to distinguish them
            self.is_draft_model_prefill = decode_query_len > 1
            if super().needs_capture():
                if self.is_draft_model_prefill:
                    set_draft_graph_prefill_params(self.capture_sizes)
                else:
                    set_draft_graph_params(self.capture_sizes)

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
            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
                super().capture(
                    forward_fn,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    progress_bar_desc,
                )

        def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Override run_fullgraph to update full graph params in run_fullgraph."""
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(f"draft prefill run_fullgraph with num_tokens={num_tokens}")
            else:
                logger.info_once(f"draft run_fullgraph with num_tokens={num_tokens}")

            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(desc.num_reqs, self.is_draft_model_prefill)

            ret = super().run_fullgraph(desc)

            positions = self.speculator.input_buffers.positions[:num_tokens]
            # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
            # calculate num_tokens_across_dp.
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens, device=self.device)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ):
                # decide to update draft graph params
                _EXTRA_CTX.is_draft_model = True

                # decide to run `prefill` graph or `decodes` graph
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill

                forward_context = get_forward_context()
                update_full_graph_params(
                    # FIXME(Ronald1995): support hybrid attn backend
                    list(self.speculator.attn_backends.values())[0],
                    self.speculator.update_stream,
                    forward_context,
                    num_tokens,
                    self.vllm_config,
                    self.speculator.speculative_config,
                    positions.shape[0],
                    draft_attn_metadatas=draft_attn_metadatas,
                )
            return ret


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
