# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, prepare_inputs_to_capture

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.24.0"):
    from vllm.v1.worker.gpu.cudagraph_utils import AttentionStatePair  # type: ignore[import-not-found]
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState

if vllm_version_is("0.24.0"):
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (  # type: ignore[import-not-found]
        DecodeSpeculatorCudaGraphManager,
        PrefillSpeculatorCudaGraphManager,
    )
else:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.utils import communicator_switch

if vllm_version_is("0.24.0"):

    class PrefillEagleAclGraphManager(PrefillSpeculatorCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding (0.24.0)."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any = None,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)
            self.speculator = speculator
            self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
            self.is_draft_model_prefill = decode_query_len > 1
            if super().needs_capture():
                if self.is_draft_model_prefill:
                    set_draft_graph_prefill_params(self.capture_sizes)
                else:
                    set_draft_graph_params(self.capture_sizes)

        def capture(
            self,
            forward_fn: Callable,
            attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
            progress_bar_desc: str = "Capturing CUDA graphs",
        ) -> None:
            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
                super().capture(forward_fn, attn_states, progress_bar_desc=progress_bar_desc)

        def run_fullgraph(
            self, desc: BatchExecutionDescriptor
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(
                    "PrefillEagleAclGraphManager: draft prefill run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            else:
                logger.info_once(
                    "DecodeEagleAclGraphManager: draft run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
                desc.num_reqs, self.is_draft_model_prefill
            )
            ret = super().run_fullgraph(desc)
            positions = self.speculator.input_buffers.positions[:num_tokens]
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,
                slot_mapping=None,
            ):
                _EXTRA_CTX.is_draft_model = True
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill
                forward_context = get_forward_context()
                update_full_graph_params(
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

    class PrefillEagleAclGraphManager(SpeculatorCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any = None,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)
            self.speculator = speculator
            self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
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
            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
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
            self, desc: BatchExecutionDescriptor
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(
                    "PrefillEagleAclGraphManager: draft prefill run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            else:
                logger.info_once(
                    "DecodeEagleAclGraphManager: draft run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
                desc.num_reqs, self.is_draft_model_prefill
            )
            ret = super().run_fullgraph(desc)
            positions = self.speculator.input_buffers.positions[:num_tokens]
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,
                slot_mapping=None,
            ):
                _EXTRA_CTX.is_draft_model = True
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill
                forward_context = get_forward_context()
                update_full_graph_params(
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


if vllm_version_is("0.24.0"):

    class DecodeEagleAclGraphManager(DecodeSpeculatorCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding (0.24.0)."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any = None,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)
            self.speculator = speculator
            self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
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
            def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
                num_tokens = desc.num_tokens
                num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
                num_tokens_across_dp = (
                    torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                    if self.dp_size > 1
                    else None
                )
                attn_state = prepare_inputs_to_capture(
                    num_reqs,
                    num_tokens,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                )
                fwd = lambda cg_mode: forward_fn(
                    num_reqs,
                    cg_mode == CUDAGraphMode.PIECEWISE,
                    BatchExecutionDescriptor(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs),
                    num_tokens_across_dp,
                )
                return fwd, attn_state

            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
                super(DecodeSpeculatorCudaGraphManager, self).capture(create_forward_fn, progress_bar_desc)

        def run_fullgraph(
            self, desc: BatchExecutionDescriptor
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(
                    "PrefillEagleAclGraphManager: draft prefill run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            else:
                logger.info_once(
                    "DecodeEagleAclGraphManager: draft run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
                desc.num_reqs, self.is_draft_model_prefill
            )
            ret = super().run_fullgraph(desc)
            positions = self.speculator.input_buffers.positions[:num_tokens]
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,
                slot_mapping=None,
            ):
                _EXTRA_CTX.is_draft_model = True
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill
                forward_context = get_forward_context()
                update_full_graph_params(
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

    class DecodeEagleAclGraphManager(SpeculatorCudaGraphManager):
        """AclGraphManager for Eagle speculative decoding."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            speculator: Any = None,
        ):
            super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)
            self.speculator = speculator
            self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
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
            with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
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
            self, desc: BatchExecutionDescriptor
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            num_tokens = desc.num_tokens
            if self.is_draft_model_prefill:
                logger.info_once(
                    "PrefillEagleAclGraphManager: draft prefill run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            else:
                logger.info_once(
                    "DecodeEagleAclGraphManager: draft run_fullgraph with num_tokens=%s",
                    num_tokens,
                )
            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
                desc.num_reqs, self.is_draft_model_prefill
            )
            ret = super().run_fullgraph(desc)
            positions = self.speculator.input_buffers.positions[:num_tokens]
            num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,
                slot_mapping=None,
            ):
                _EXTRA_CTX.is_draft_model = True
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill
                forward_context = get_forward_context()
                update_full_graph_params(
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
