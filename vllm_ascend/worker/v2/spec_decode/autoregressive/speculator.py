# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/autoregressive/speculator.py
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
import logging
from contextlib import contextmanager
from copy import copy
from typing import Any, cast

import torch
import vllm.v1.worker.gpu.spec_decode.speculator as vllm_speculator
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers

logger = logging.getLogger(__name__)


@contextmanager
def build_draft_attn_metadata_factory(positions, pad):
    """Wrap build_attn_metadata to forward MLA rotary positions for the block.

    MLA reads positions inside build_decode_metadata for cos/sin; the flat
    super() path doesn't forward them. Must run inside build_attn_metadata_wrapper().
    TODO:This field is removed when the external cos/sin solution is removed from the MLA.
    """
    raw = vllm_speculator.build_attn_metadata  # cache

    def build_attn_metadata(*args, **kwargs):
        kwargs["positions"] = positions[:pad]
        return raw(*args, **kwargs)

    try:
        vllm_speculator.build_attn_metadata = build_attn_metadata
        yield
    finally:
        vllm_speculator.build_attn_metadata = raw  # restore


class AscendAutoRegressiveSpeculator(AutoRegressiveSpeculator):
    """Shared Ascend spec-decode loop for AscendEagle/AscendMTPSpeculator.

    GQA, MLA, and DSA draft decode state share one path. The current MTP path
    uses the draft attention backend recorded by ``set_attn``.

    MLA's per-step state lives in ``.decode`` (cloned per step, written via an
    alias), GQA's is top-level. MLA also rebuilds the base (live ``.decode`` is
    None/wrong-batch) and forwards rotary ``positions`` into
    build_attn_metadata. DSA manages its draft state in its metadata builder
    and skips the generic MLA/GQA init and update logic.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """Override the upstream __init__ for Ascend NPUs.

        Ascend attention-metadata building needs more information (e.g.
        seq_lens_cpu from input_batch), so we replace input_buffers with
        AscendInputBuffers after super().__init__.
        """
        super().__init__(vllm_config, device)

        self.attn_architecture: str | None = None

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

        # when in decode phase of eagle speculator, we need some value in
        # draft model's input_batch. so we keep a reference here.
        self.input_batch: InputBatch | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph managers are patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # They need this speculator to update full-graph params, so set it here.
        self.prefill_cudagraph_manager.speculator = self
        self.decode_cudagraph_manager.speculator = self
        self.prefill_cudagraph_manager.update_stream = self.update_stream
        self.decode_cudagraph_manager.update_stream = self.update_stream

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
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

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
        first_attn_backend = list(self.attn_backends.values())[0]
        if issubclass(first_attn_backend, AscendDSABackend):
            self.attn_architecture = "DSA"
        elif issubclass(first_attn_backend, AscendMLABackend):
            self.attn_architecture = "MLA"
        elif issubclass(first_attn_backend, AscendAttentionBackend):
            self.attn_architecture = "GQA"
        else:
            raise ValueError(f"Unsupported attention backend: {first_attn_backend}")

    def capture(self) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.model_state,
            self.target_input_buffers,
            self.block_tables,
            self.target_attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return

        # Capture all decode draft generation steps as a single graph.
        assert self.decode_cudagraph_manager is not None
        with build_attn_metadata_wrapper():
            self.decode_cudagraph_manager.capture(
                self._multi_step_decode,
                self.model_state,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing decode CUDA graphs",
            )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override AutoRegressiveSpeculator._run_model for Ascend NPUs."""
        last_hidden_states, hidden_states = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )
        self._ascend_update_seq_lens(attn_metadata)
        return last_hidden_states, hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Thin override: delegate to upstream single-step ``_generate_draft``,
        then apply Ascend-specific attention-metadata updates required by the
        FIA operator."""
        super()._generate_draft(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if attn_metadata is not None:
            self._update_decode_attn_metadata(attn_metadata, 1, num_reqs)

    def _multi_step_decode(  # type: ignore[misc]
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    ) -> None:
        """Minimal override to handle the merged multi-step graph in FULL mode.

        In FULL mode the captured graph already contains all speculative
        steps, so ``run_fullgraph`` is called once instead of once per
        step.  For PIECEWISE / NONE modes we delegate to the upstream
        ``_multi_step_decode`` which iterates over steps and calls
        ``_generate_draft`` per step.
        """
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return
        super()._multi_step_decode(num_reqs, skip_attn, batch_desc, num_tokens_across_dp, seq_lens_cpu_upper_bound)

    def _build_draft_attn_metadata(  # type: ignore[misc]
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int = 1,
        causal: bool = True,
    ) -> dict[str, Any] | None:
        with build_draft_attn_metadata_factory(self.input_buffers.positions, num_tokens_padded):
            attn_metadata = super()._build_draft_attn_metadata(
                num_reqs,
                num_reqs_padded,
                num_tokens_padded,
                seq_lens_cpu_upper_bound,
                step,
                num_query_per_req,
                causal,
            )
        if attn_metadata is not None:
            # Ascend-specific: force DecodeOnly attention state for the draft model.
            for metadata in attn_metadata.values():
                metadata.attn_state = AscendAttentionState.DecodeOnly
        return attn_metadata

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

    def _ascend_update_seq_lens(self, attn_metadata: dict[str, Any] | None) -> None:
        if attn_metadata is not None:
            for attn_meta in attn_metadata.values():
                attn_meta.seq_lens = attn_meta.seq_lens + 1
                attn_meta.seq_len_list = attn_meta.seq_lens.tolist()

    def _init_decode_draft_attn_metadatas(self, attn_metadata: dict[str, Any] | None, num_reqs_padded: int):
        """Initialize per-step decode attention metadata for graph mode."""
        if attn_metadata is None:
            return

        # DSA owns its per-step slot mapping, SAS, and QLI state in the DSA
        # metadata builder and does not use draft graph metadata updates.
        if self.attn_architecture == "DSA":
            return []

        # TODO: _build_draft_attn_metadata pulls data (seq_lens, block_table,
        # ...) from input_buffers internally; future may pass these as CPU
        # params directly to build_attn_metadata, decoupling from input_buffers.
        if self.attn_architecture == "MLA":
            assert self.input_batch is not None
            attn_metadata = self._build_draft_attn_metadata(  # type: ignore[call-arg]
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_reqs_padded,  # decode: 1 token/req
                seq_lens_cpu_upper_bound=self.input_batch.seq_lens_cpu_upper_bound,
                step=1,
            )
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
                if self.attn_architecture == "MLA":
                    # clone .decode so per-step seq_lens_list writes don't alias.
                    metadata.decode = copy(metadata.decode)
            draft_attn_metadatas.append(per_step_attn_metadata)

        return draft_attn_metadatas

    def _update_decode_attn_metadata(
        self, attn_metadata: dict[str, Any] | None, step: int, num_reqs: int | None = None
    ):
        """Update per-step decode attention metadata on Ascend."""
        if attn_metadata is None:
            return

        if self.attn_architecture == "DSA":
            return

        attn_meta = next(iter(attn_metadata.values()))
        num_reqs_padded = attn_meta.seq_lens_cpu.shape[0]
        seq_lens_cpu = self._get_seq_lens_cpu()[:num_reqs_padded]
        if num_reqs is None:
            num_reqs = num_reqs_padded
        next_seq_lens_cpu = self._calc_next_seq_lens_cpu(seq_lens_cpu, num_reqs, num_reqs_padded, step)

        query_lens_list = [i for i in range(1, num_reqs_padded + 1)]
        seq_lens_list = next_seq_lens_cpu.tolist()
        for metadata in attn_metadata.values():
            if self.attn_architecture == "MLA":
                decode_metadata = metadata.decode
            else:
                decode_metadata = metadata
            decode_metadata.seq_lens_list = seq_lens_list
            decode_metadata.actual_seq_lengths_q = query_lens_list
            metadata.seq_lens_cpu.copy_(next_seq_lens_cpu)

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
