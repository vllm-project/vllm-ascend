# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from contextlib import contextmanager
from copy import copy
from typing import Any, cast

import torch
import vllm
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import AttentionBackend
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as vllm_speculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.utils import vllm_version_is
if not vllm_version_is("0.20.1"):
    from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import PrefillEagleCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator, update_eagle_draft_inputs
    from vllm_ascend.worker.v2.spec_decode.eagle.aclgraph import PrefillEagleAclGraphManager
else:
    from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import EagleCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator, gumbel_sample, update_eagle_inputs
    from vllm_ascend.worker.v2.spec_decode.eagle.aclgraph import EagleAclGraphManager

class AscendEagleSpeculator(EagleSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """Override GPU EagleSpeculator.__init__ for Ascend NPUs.
        attnention metadata building in Ascend backend needs more information,
        such as seq_lens_cpu from input_batch, so we need to override __init__.
        """
        super().__init__(vllm_config, device)

        del self.input_buffers
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )

        # add more attributes for `input_buffers` in graph mode
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            self.input_buffers.draft_seq_lens_cpus = [
                torch.zeros(self.max_num_reqs, dtype=torch.int32, device="cpu")
                for _ in range(self.num_speculative_steps - 1)
            ]

        # we need to update full graph params in run_fullgraph,
        # so create a stream to update full graph params.
        if cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()

        # when in decode phase of eagle speculator, we need some value in
        # draft model's input_batch. so we keep a reference here.
        self.input_batch: InputBatch | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        with graph_manager_wrapper(self):
            super().init_cudagraph_manager(cudagraph_mode)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: Any = None,
    ):
        """Override GPU EagleSpeculator.propose for Ascend NPUs,
        because npu attention metadata needs more information,
        we need to cache input_batch, so we can use it later in
        generate_draft.
        """
        self.input_batch = input_batch
        # wrap build_attn_metadata to use Ascend attention metadata building.
        # so we can call super().propose() directly.
        with build_attn_metadata_wrapper(), torch_gather_wrapper():
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        # npu needs attn_backends to update graph params
        attn_backends: dict[str, type[AttentionBackend]] = {}

        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_id, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backend = attn_layers[layer_name].get_attn_backend()
                attn_backends[layer_name] = attn_backend

        self.attn_backends = attn_backends

    if not vllm_version_is("0.20.1"):
        def generate_draft(
            self,
            num_reqs: int,
            num_tokens_padded: int,
            attn_metadata: dict[str, Any],
            slot_mappings: dict[str, torch.Tensor],
            num_tokens_across_dp: torch.Tensor | None,
            cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        ):
            """Override GPU EagleSpeculator.generate_draft for Ascend NPUs, because
            attn_metadata is created in super propose method, it does not have some
            attribute that Ascend attention backend needs, so we update it.
            """
            self._init_decode_attn_metadata(attn_metadata, num_reqs)
            self._increment_decode_attn_metadata(attn_metadata)
            idx_mapping = self.idx_mapping[:num_reqs]
            positions = self.input_buffers.positions[:num_reqs]
            # Run the eagle model forward pass.
            last_hidden_states, hidden_states = self.run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            last_hidden_states = last_hidden_states[:num_reqs]

            # Sample the draft tokens.
            logits = self.model.compute_logits(last_hidden_states)
            draft_tokens = self._sample_draft(
                logits,
                idx_mapping,
                positions,
                self.current_draft_step,
                self.draft_logits,
            )

            # Update the inputs for the next step.
            update_eagle_draft_inputs(
                draft_tokens,
                self.current_draft_step,
                hidden_states,
                self.draft_tokens,
                self.hidden_states,
                self.input_buffers,
                num_reqs,
                self.max_model_len,
                self.num_speculative_steps,
            )
            # npu's own update logic
            self._increment_decode_attn_metadata(attn_metadata)
    else:
        def generate_draft(
            self,
            num_reqs: int,
            num_tokens_padded: int,
            attn_metadata: dict[str, Any],
            slot_mappings: dict[str, torch.Tensor],
            num_tokens_across_dp: torch.Tensor | None,
            cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        ):
            """Override GPU EagleSpeculator.generate_draft for Ascend NPUs, because
            attn_metadata is created in super propose method, it does not have some
            attribute that Ascend attention backend needs, so we update it.
            """
            self._init_decode_attn_metadata(attn_metadata, num_reqs)
            self._increment_decode_attn_metadata(attn_metadata)

            # NOTE(drslark): following lines (from 145 to 184) come from raw gpu's generate_draft logic
            pos = self.input_buffers.positions[:num_reqs]
            query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
            idx_mapping = self.idx_mapping[:num_reqs]
            for step in range(1, self.num_speculative_steps):
                # Run the eagle model.
                last_hidden_states, hidden_states = self.run_model(
                    num_tokens_padded,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode,
                )
                last_hidden_states = last_hidden_states[:num_reqs]
                hidden_states = hidden_states[:num_reqs]
                logits = self.model.compute_logits(last_hidden_states)

                # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
                # used for draft and target sampling.
                draft_tokens = gumbel_sample(
                    logits,
                    idx_mapping,
                    self.temperature,
                    self.seeds,
                    pos + 1,
                    apply_temperature=True,
                    processed_logits_out=self.draft_logits[:, step] if self.draft_logits is not None else None,
                )
                self.draft_tokens[:num_reqs, step] = draft_tokens

                if step < self.num_speculative_steps - 1:
                    # Update the inputs for the next step.
                    update_eagle_inputs(
                        draft_tokens,
                        hidden_states,
                        self.input_buffers,
                        self.hidden_states,
                        self.max_model_len,
                    )
                    if attn_metadata is not None:
                        self.block_tables.compute_slot_mappings(idx_mapping, query_start_loc, pos, num_tokens_padded)

                        # npu's own update logic
                        self._increment_decode_attn_metadata(attn_metadata)

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override GPU EagleSpeculator.run_model for Ascend NPUs, because
        in decode phase, we need to update seq_lens_cpu in attn_metadata after
        run model.
        """
        last_hidden_states, hidden_states = super().run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

        # attn_metadata is None in profile_run and dummy_run.
        if attn_metadata is not None:
            for attn_meta in attn_metadata.values():
                # seq_lens in AscendMetadata is a cpu tensor.
                attn_meta.seq_lens = attn_meta.seq_lens + 1
                attn_meta.seq_len_list = attn_meta.seq_lens.tolist()
        return last_hidden_states, hidden_states

    def build_draft_attn_metadatas(self, num_reqs_padded, is_draft_model_prefill):
        """Build draft_attn_metadatas for partial-merged draft graph."""
        attn_metadata = self.model_state.attn_metadata
        attn_metadata = {
            name: metadata for name, metadata in attn_metadata.items() if name in self.draft_attn_layer_names
        }

        if is_draft_model_prefill:
            return [attn_metadata]

        draft_attn_metadatas = self._init_decode_draft_attn_metadatas(attn_metadata, num_reqs_padded)

        for i, per_step_attn_metadata in enumerate(draft_attn_metadatas):
            step = i + 1
            assert self.input_batch is not None
            self._update_decode_attn_metadata(per_step_attn_metadata, step, self.input_batch.num_reqs)

        return draft_attn_metadatas

    def _init_decode_attn_metadata(self, attn_metadata: dict[str, Any], num_reqs: int):
        """Initialize attention metadata for decode phase on Ascend NPUs."""
        if attn_metadata is None:
            return

        attn_state = AscendAttentionState.DecodeOnly
        seq_lens_cpu = self._get_seq_lens_cpu()[:num_reqs]

        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for metadata in attn_metadata.values():
            metadata.attn_state = attn_state
            metadata.seq_lens_cpu = seq_lens_cpu

    def _init_decode_draft_attn_metadatas(self, attn_metadata: dict[str, Any], num_reqs_padded: int):
        """Initialize attention metadata for decode phase in graph mode on Ascend NPUs."""
        if attn_metadata is None:
            return

        attn_state = AscendAttentionState.DecodeOnly

        draft_attn_metadatas = []
        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for seq_lens_cpu in self.input_buffers.draft_seq_lens_cpus:
            per_step_attn_metadata = {k: copy(v) for k, v in attn_metadata.items()}

            seq_lens_cpu = seq_lens_cpu[:num_reqs_padded]
            for metadata in per_step_attn_metadata.values():
                metadata.attn_state = attn_state
                metadata.seq_lens_cpu = seq_lens_cpu
            draft_attn_metadatas.append(per_step_attn_metadata)

        return draft_attn_metadatas

    def _increment_decode_attn_metadata(self, attn_metadata: dict[str, Any]):
        """Increment attention metadata for decode phase on Ascend NPUs."""
        # in eager mode, attn_metadata's seq_lens_cpu and input_buffers's seq_lens_cpu shares the memory
        self._update_decode_attn_metadata(attn_metadata, 1)

    def _update_decode_attn_metadata(self, attn_metadata: dict[str, Any], step: int, num_reqs: int | None = None):
        """Update attention metadata for decode phase on Ascend NPUs."""
        if attn_metadata is None:
            return

        num_reqs_padded = next(iter(attn_metadata.values())).seq_lens_cpu.shape[0]
        seq_lens_cpu = self._get_seq_lens_cpu()[:num_reqs_padded]
        if num_reqs is None:
            num_reqs = num_reqs_padded
        next_seq_lens_cpu = self._calc_next_seq_lens_cpu(seq_lens_cpu, num_reqs, num_reqs_padded, step)

        query_lens_list = [i for i in range(1, num_reqs_padded + 1)]
        seq_lens_list = next_seq_lens_cpu.tolist()
        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list
            metadata.seq_lens_cpu.copy_(next_seq_lens_cpu)
            metadata.seq_lens_list = seq_lens_list

    def _calc_next_seq_lens_cpu(self, seq_lens_cpu, num_reqs, num_reqs_padded, step):
        # NOTE(drslark) to achieve fully alignment with vllm, `num_rejected` should be subtracted from `seq_lens`
        # to avoid extra sync overhead, `v2` is currently aligned with NPU `v1` only

        # follows the logic in `prepare_eagle_decode` and `update_eagle_inputs`
        next_seqs_cpu = torch.clamp(seq_lens_cpu[:num_reqs_padded] + step, max=self.max_model_len)
        next_seqs_cpu[num_reqs:].fill_(0)
        return next_seqs_cpu

    def _get_seq_lens_cpu(self) -> torch.Tensor:
        """Get seq_lens_cpu from input_batch."""
        assert self.input_batch is not None
        seq_lens_cpu = torch.from_numpy(self.input_batch.seq_lens_np)
        return seq_lens_cpu


@contextmanager
def build_attn_metadata_wrapper():
    """Context manager to override attention metadata building for Ascend NPUs."""
    original_func = vllm.v1.worker.gpu.spec_decode.eagle.speculator.build_attn_metadata
    try:
        vllm.v1.worker.gpu.spec_decode.eagle.speculator.build_attn_metadata = build_attn_metadata
        yield
    finally:
        vllm.v1.worker.gpu.spec_decode.eagle.speculator.build_attn_metadata = original_func


# TODO Remove this patch when cann fix the gather bug.
# NOTE(Ronald1995): torch.gather will pollute the cache such as self.input_buffers.positions
# the bug is reported to huawei CANN team, but not fixed yet.
# NOTE(drslark): make a temporary patch only for `torch.gather`
_original_gather = torch.gather


def gather(input, dim, index, *, sparse_grad=False, out=None):
    if out is None:
        return _original_gather(input, dim, index, sparse_grad=sparse_grad)
    out[:] = _original_gather(input, dim, index, sparse_grad=sparse_grad)
    return out


@contextmanager
def torch_gather_wrapper():
    """Context manager to override torch.gather for Ascend NPUs."""
    original_gather = torch.gather
    try:
        torch.gather = gather
        yield
    finally:
        torch.gather = original_gather

if not vllm_version_is("0.20.1"):
    @contextmanager
    def graph_manager_wrapper(speculator):
        """Context manager to override graph manager."""
        original_graph_manager = PrefillEagleCudaGraphManager

        def factory(vllm_config: VllmConfig, device: torch.device, cudagraph_mode: CUDAGraphMode, decode_query_len: int):
            return PrefillEagleAclGraphManager(vllm_config, device, cudagraph_mode, decode_query_len, speculator)

        try:
            vllm_speculator.PrefillEagleCudaGraphManager = factory
            yield
        finally:
            vllm_speculator.PrefillEagleCudaGraphManager = original_graph_manager
else:
    @contextmanager
    def graph_manager_wrapper(speculator):
        """Context manager to override graph manager."""
        original_graph_manager = EagleCudaGraphManager

        def factory(vllm_config: VllmConfig, device: torch.device, cudagraph_mode: CUDAGraphMode, decode_query_len: int):
            return EagleAclGraphManager(vllm_config, device, cudagraph_mode, decode_query_len, speculator)

        try:
            vllm_speculator.EagleCudaGraphManager = factory
            yield
        finally:
            vllm_speculator.EagleCudaGraphManager = original_graph_manager
