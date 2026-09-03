#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.physical_k import (
    initialize_physical_k_buffers,
    physical_k_scope,
)

logger = logging.getLogger(__name__)


class _RowwiseConfidenceBuffer:
    """Expose a fixed confidence buffer with safe active-width assignment."""

    def __init__(self, buffer: torch.Tensor, active_k: int):
        self._buffer = buffer
        self._active_k = active_k

    @property
    def ndim(self) -> int:
        return self._buffer.ndim

    @property
    def shape(self):
        return (self._buffer.shape[0], self._active_k)

    def __setitem__(self, key, value) -> None:
        # Upstream DSpark assigns ``buffer[:num_reqs] = [B, K]``.  Assigning
        # that 2-D slice is a strided copy when K < max_K.  Copy each request
        # row instead; every row is contiguous and remains graph-safe.
        if not isinstance(key, slice) or key.start not in (None, 0) or key.step not in (None, 1):
            raise TypeError("unsupported confidence buffer index")
        num_reqs = self._buffer.shape[0] if key.stop is None else key.stop
        values = value.reshape(num_reqs, self._active_k)
        for req_idx in range(num_reqs):
            self._buffer[req_idx, : self._active_k].copy_(values[req_idx])


class _RowwiseDraftTokenBuffer:
    """Route active-K column writes through contiguous scalar locations."""

    def __init__(self, buffer: torch.Tensor):
        self._buffer = buffer

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("unsupported draft token buffer index")
        reqs, col = key
        if not isinstance(reqs, slice) or reqs.start not in (None, 0) or reqs.step not in (None, 1):
            raise TypeError("unsupported draft token request index")
        num_reqs = self._buffer.shape[0] if reqs.stop is None else reqs.stop
        values = value.reshape(num_reqs)
        for req_idx in range(num_reqs):
            self._buffer[req_idx, col] = values[req_idx]


class AscendDSparkSpeculator(DSparkSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None
        self._vllm_ascend_max_speculative_steps = self.num_speculative_steps
        self._dynamic_update_log_count = 0
        # DSpark changes ``sample_from_anchor`` after DFlash initialization,
        # so initialize the width-dependent anchor indices only now.
        initialize_physical_k_buffers(self)

        # V2 uses the upstream DSpark implementation for the draft forward,
        # so it does not inherit the V1 Ascend proposer that creates the
        # hardware-aware scheduler.  Reuse the confidence probabilities that
        # DSpark already computes instead of running the confidence head twice.
        dynamic_config = get_ascend_config().dynamic_spec_config
        self.dynamic_spec: DynamicSpecScheduler | None = None
        if (
            dynamic_config.method == "dspark"
            and dynamic_config.policy == "hardware_aware"
        ):
            if not self.enable_adaptive_verification:
                logger.warning(
                    "V2 hardware-aware DSpark scheduling requires "
                    "enable_adaptive_verification=true; keeping fixed K."
                )
            else:
                self.dynamic_spec = DynamicSpecScheduler(
                    method="dspark",
                    policy=dynamic_config.policy,
                    method_params=dynamic_config.method_params,
                    max_batch_size=self.max_num_reqs,
                    num_speculative_tokens=self.num_speculative_steps,
                    device=device,
                )

    def _update_dynamic_spec(self, num_reqs: int, active_k: int) -> None:
        """Publish DSpark confidence output to the Ascend K policy."""

        if self.dynamic_spec is None or active_k <= 0:
            return
        confidence = self.draft_token_confidence_probs[:num_reqs, :active_k]
        request_ids = None
        if self.input_batch is not None:
            request_ids = self.input_batch.req_ids[:num_reqs]
        selected = self.dynamic_spec.update_from_token_probs(
            confidence,
            request_ids=request_ids,
        )
        if self._dynamic_update_log_count < 8:
            logger.warning(
                "V2 hardware-aware K decision #%d: reqs=%d active_k=%d selected_shape=%s",
                self._dynamic_update_log_count + 1,
                num_reqs,
                active_k,
                tuple(selected.shape),
            )
            self._dynamic_update_log_count += 1

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)  # type: ignore[has-type]
        # npu needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Keep DSpark confidence writes compatible with active physical K.

        Upstream DSpark assigns the confidence result to the whole fixed-width
        request buffer.  During V2 graph capture the physical-K scope exposes a
        smaller ``num_speculative_steps``, so the result has shape ``[B, K]``
        while the buffer is still ``[B, max_K]``.  A narrow view preserves the
        fixed backing allocation and makes both the dense and top-k sampling
        paths shape-safe; the full view is restored before the caller records
        confidences for the next scheduler step.
        """
        active_k = int(self.num_speculative_steps)
        max_k = int(
            getattr(self, "_vllm_ascend_max_speculative_steps", active_k)
        )
        if active_k >= max_k:
            # Keep the fixed-K path unchanged; the row-wise adapter is only
            # needed by a smaller physical-K graph.
            super()._sample_sequential(num_reqs, head_hidden)
            self._update_dynamic_spec(num_reqs, active_k)
            return

        confidence_probs = getattr(self, "draft_token_confidence_probs", None)
        old_draft_tokens = self.draft_tokens
        self.draft_tokens = _RowwiseDraftTokenBuffer(old_draft_tokens)
        if confidence_probs is not None and confidence_probs.ndim >= 2:
            self.draft_token_confidence_probs = _RowwiseConfidenceBuffer(
                confidence_probs, active_k
            )
        try:
            super()._sample_sequential(num_reqs, head_hidden)
        finally:
            self.draft_tokens = old_draft_tokens
            if confidence_probs is not None and confidence_probs.ndim >= 2:
                self.draft_token_confidence_probs = confidence_probs
        self._update_dynamic_spec(num_reqs, active_k)

    def build_draft_attn_metadatas(
        self,
        num_reqs_padded,
        seq_lens_cpu_upper_bound,
        num_tokens_padded=None,
    ):
        num_tokens_padded = num_tokens_padded or (
            num_reqs_padded * self.num_query_per_req
        )
        assert self.input_batch is not None
        # The draft attention metadata is built through the generic
        # (Ascend) build_attn_metadata path; the factory forwards the draft
        # query positions that the DSA metadata builder needs for RoPE.
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                torch.from_numpy(self.input_batch.is_prefilling_np),
            ),
        ):
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        return [self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        DSpark inherits DFlash's full-graph path, and upstream
        ``Speculator._build_draft_attn_metadata`` clamps ``query_start_loc`` at
        the real ``num_reqs`` to keep the cumulative series non-decreasing, so
        when a batch is padded to a capture size (``num_reqs_padded >
        num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list
        return attn_metadata

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        with physical_k_scope(self, input_batch):
            with (
                build_attn_metadata_wrapper(),
                build_draft_attn_metadata_factory(
                    self.input_buffers.positions,
                    self.max_num_tokens,
                    torch.from_numpy(self.input_batch.is_prefilling_np),
                ),
            ):
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
                    dp_sync,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                )
