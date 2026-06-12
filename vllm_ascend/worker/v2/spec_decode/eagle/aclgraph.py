# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from collections.abc import Callable
from contextlib import contextmanager
from copy import copy
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import (
    CapturedAttentionState,
    DecodeEagleCudaGraphManager,
    PrefillEagleCudaGraphManager,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.utils import communicator_switch


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
        full_cg_attn_states: dict[BatchExecutionDescriptor, CapturedAttentionState],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for Eagle."""
        with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
            super().capture(
                forward_fn,
                full_cg_attn_states,
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
    """Captures all (num_speculative_steps - 1) decode steps in one ACL graph.

    Slot-mapping strategy
    ---------------------
    Each decode step writes KV to a *different* cache slot, so N-1 separate
    static slot-mapping buffers are pre-allocated (one per step).  Before every
    graph replay, ``_prefill_draft_slot_mappings`` computes the slot mappings
    for all steps and copies them into the corresponding buffers.

    NOTE: only single-KV-group models are fully supported.  Multi-group support
    requires per-group buffer indexing (TODO).
    """

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
        num_steps = self.speculator.num_speculative_steps - 1

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> tuple[Callable[[CUDAGraphMode], None], CapturedAttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            skip_attn = desc.cg_mode == CUDAGraphMode.PIECEWISE

            # Build the base attn_state once; its slot_mapping tensor comes from
            # block_tables.slot_mappings[0, :num_tokens].
            base_attn_state = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                skip_attn=skip_attn,
            )
            base_attn_metadata, base_slot_mappings = base_attn_state

            # Build per-step attn_metadata, each pointing to a dedicated
            # slot_mapping buffer so the graph references N-1 distinct addresses.
            all_attn_states: list[tuple] = []
            draft_buffers = self.speculator.draft_slot_mappings_buffers
            for step_idx in range(num_steps):
                step_attn_metadata: dict = {}
                step_slot_map = draft_buffers[step_idx][:num_tokens]
                for layer_name, layer_meta in base_attn_metadata.items():
                    layer_meta_copy = copy(layer_meta)
                    layer_meta_copy.slot_mapping = step_slot_map
                    step_attn_metadata[layer_name] = layer_meta_copy
                # Build a per-step slot_mappings dict so set_forward_context
                # also receives the correct buffer address for this step.
                step_slot_mappings = {layer: step_slot_map for layer in base_slot_mappings}
                all_attn_states.append((step_attn_metadata, step_slot_mappings))

            def fwd(cg_mode: CUDAGraphMode) -> None:
                for step_idx, (attn_metadata, slot_mappings) in enumerate(all_attn_states):
                    self.speculator.current_draft_step.fill_(step_idx + 1)
                    forward_fn(
                        num_reqs,
                        num_tokens,
                        attn_metadata,
                        slot_mappings,
                        num_tokens_across_dp,
                        cg_mode,
                    )

            return fwd, base_attn_state

        with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
            # Bypass DecodeEagleCudaGraphManager.capture (which builds a single-
            # step forward_fn) and call CudaGraphManager.capture directly so that
            # our multi-step fwd is what gets captured.
            CudaGraphManager.capture(self, create_forward_fn, progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> None:
        """Pre-compute all N-1 slot mappings, then replay the all-steps graph once."""
        num_tokens = desc.num_tokens
        num_reqs = desc.num_reqs or num_tokens
        speculator = self.speculator

        logger.info_once(f"all-steps decode run_fullgraph with num_tokens={num_tokens}")

        # Prefer the actual (unpadded) num_reqs written by multi_step_decode so
        # _prefill_draft_slot_mappings doesn't touch stale padded entries.
        actual_num_reqs = getattr(speculator, '_current_decode_num_reqs', num_reqs)

        # 1. Fill draft slot-mapping buffers for every decode step.
        self._prefill_draft_slot_mappings(actual_num_reqs, num_tokens)

        # 2. Replay the captured all-steps graph (single shot).
        CudaGraphManager.run_fullgraph(self, desc)

        # 3. Update graph params for all N-1 steps so the attention backend can
        #    patch dynamic parameters (seq_lens, etc.) for the next forward pass.
        draft_attn_metadatas = speculator.build_draft_attn_metadatas(num_reqs, False)
        positions = speculator.input_buffers.positions[:num_tokens]
        num_tokens_across_dp = torch.full(
            [speculator.dp_size], num_tokens, device=self.device
        )
        with set_forward_context(
            speculator.model_state.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,
            slot_mapping=None,
        ):
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = False
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                list(speculator.attn_backends.values())[0],
                speculator.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                speculator.speculative_config,
                positions.shape[0],
                draft_attn_metadatas=draft_attn_metadatas,
            )

    def _prefill_draft_slot_mappings(self, actual_num_reqs: int, num_tokens: int) -> None:
        """Compute slot_mappings for every decode step and write to static buffers.

        Step k (0-indexed) corresponds to draft positions = initial_positions + k,
        where initial_positions are the positions set by prepare_eagle_decode
        (already incremented once from the target model's positions).

        Only the first ``actual_num_reqs`` entries are computed from live data;
        padded slots are filled with PAD_SLOT_ID so the captured graph never
        writes to an unintended KV-cache location.
        """
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID
        speculator = self.speculator
        block_tables = speculator.block_tables
        idx_mapping = speculator.idx_mapping[:actual_num_reqs]
        query_start_loc = speculator.input_buffers.query_start_loc[:actual_num_reqs + 1]
        positions = speculator.input_buffers.positions[:actual_num_reqs]
        draft_buffers = speculator.draft_slot_mappings_buffers
        num_steps = speculator.num_speculative_steps - 1

        for k in range(num_steps):
            pos_k = positions if k == 0 else positions + k
            slot_maps = block_tables.compute_slot_mappings(
                idx_mapping, query_start_loc, pos_k, num_tokens
            )
            # slot_maps shape: [num_kv_groups, num_tokens]; copy group-0 values.
            draft_buffers[k][:actual_num_reqs].copy_(slot_maps[0][:actual_num_reqs])
            if actual_num_reqs < num_tokens:
                draft_buffers[k][actual_num_reqs:num_tokens].fill_(PAD_SLOT_ID)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
