# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend V2 model-runner state for Qwen4Exp."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_ascend.models.qwen4_exp.input_adapter import (
    Qwen4ExpPreparedInputAdapter,
)
from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)


class AscendQwen4ExpModelState(Qwen4ExpModelState, AscendMambaHybridModelState):
    """Qwen4Exp state with Ascend attention metadata construction.

    Method resolution order keeps the upstream PLE n-gram buffers and input
    preparation from :class:`Qwen4ExpModelState`, while ``prepare_attn``
    resolves to :class:`AscendMambaHybridModelState`, which builds Ascend
    attention metadata and stores it on the state for FULL aclgraph replay
    (``update_full_graph_params`` reads ``model_state.attn_metadata``).
    """

    _skip_spec_cpu_length_sync: bool
    ngram_context: torch.Tensor
    ple_query_start_loc: torch.Tensor

    def __init__(
        self, vllm_config: VllmConfig, model: nn.Module, encoder_cache: EncoderCache | None, device: torch.device
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.qsa_prepared_input_adapter = Qwen4ExpPreparedInputAdapter(model)
        if self.uses_ngram_embedding and vllm_config.parallel_config.prefill_context_parallel_size > 1:
            # DualChunkSwap can put two segments of each request on one rank.
            capacity = 2 * self.max_num_reqs
            self.ngram_context = self.ngram_context.new_full(
                (capacity, self.ngram_context_len), self.ngram_eos_token_id
            )
            self.ple_query_start_loc = self.ple_query_start_loc.new_zeros(capacity + 1)
        self._remote_ple_layers = []
        from .ple_embedding import AscendPLEHostEmbedding, shared_ple_config

        shared = shared_ple_config(vllm_config)
        if shared is not None and shared.backend != "a3":
            for layer in model.modules():
                embedding = getattr(layer, "ngram_embedding", None)
                if isinstance(embedding, AscendPLEHostEmbedding):
                    embedding.bind_preparation()
                    self._remote_ple_layers.append(layer)

    def prepare_inputs(self, input_batch: Any, req_states: Any) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        # Qwen4Exp expands text positions to three MRoPE axes. Prepare the
        # same positions passed to forward to avoid a replay-unsafe QSA gather.
        positions = model_inputs.get("positions", input_batch.positions)
        self.qsa_prepared_input_adapter.prepare_positions(positions, input_batch.num_tokens_after_padding)
        if getattr(self, "_remote_ple_layers", ()):
            from vllm.models.qwen4_exp.common.ops.ple import ple_ngram_ids

            for layer in self._remote_ple_layers:
                if input_batch.num_reqs == 0:
                    layer.ngram_embedding.clear_prepared_rows()
                    continue
                ids = ple_ngram_ids(
                    input_ids=input_batch.input_ids,
                    query_start_loc=model_inputs["query_start_loc"],
                    ngram_context=model_inputs["ngram_context"],
                    layer_multipliers=layer.layer_multipliers,
                    ngram_heads_vocab_sizes=layer.ngram_heads_vocab_sizes,
                    ngram_heads_offsets=layer.ngram_heads_offsets,
                    eos_token_id=layer.eos_token_id,
                    heads_per_ngram=layer.heads_per_ngram,
                )
                layer.ngram_embedding.prepare_rows(ids)
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        positions = model_inputs.get("positions")
        if positions is not None:
            self.qsa_prepared_input_adapter.prepare_positions(positions, num_tokens)
        for layer in self._remote_ple_layers:
            layer.ngram_embedding.clear_prepared_rows()
        return model_inputs

    def prepare_attn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        metadata = super().prepare_attn(*args, **kwargs)
        cudagraph_mode = kwargs.get("cudagraph_mode")
        if cudagraph_mode is None and len(args) > 1:
            cudagraph_mode = args[1]
        self.qsa_prepared_input_adapter.prepare_attention_metadata(
            metadata,
            full_graph=cudagraph_mode == CUDAGraphMode.FULL,
        )
        return metadata

    def _uses_global_mamba_pcp(self) -> bool:
        return True

    def _prepare_ngram_context(self, input_batch, req_states):
        if input_batch.pcp is None or getattr(input_batch, "is_dummy", False):
            return super()._prepare_ngram_context(input_batch, req_states)
        num_reqs = input_batch.num_reqs
        context = self.ngram_context[: input_batch.num_reqs_after_padding]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs:
            # Each local segment starts at its own position in the request.
            starts = input_batch.seq_lens[:num_reqs] - input_batch.query_start_loc[: num_reqs + 1].diff()
            indices = starts.long().unsqueeze(1) + self.ngram_context_offsets
            requests = input_batch.idx_mapping[:num_reqs].long().unsqueeze(1)
            history = req_states.all_token_ids.gpu[requests, indices.clamp_min(0)]
            context[:num_reqs].copy_(torch.where(indices >= 0, history, self.ngram_eos_token_id))
        return context

    def initialize_spec_cpu_length_sync(self, runner: Any) -> None:
        """Bind length consumers once, after target and draft KV initialization."""
        from vllm_ascend.models.qwen4_exp.cpu_length_sync import can_skip_qwen_cpu_length_sync

        self._skip_spec_cpu_length_sync = False
        if not can_skip_qwen_cpu_length_sync(runner):
            return
        stream = getattr(runner, "num_computed_tokens_stream", None)
        if stream is None or not callable(getattr(stream, "synchronize", None)):
            return
        # Retire any profiling/previous-initialization copy before dropping
        # both its producer and consumer. This runs once, never per token.
        stream.synchronize()
        self._skip_spec_cpu_length_sync = True

    def can_skip_spec_cpu_length_sync(self) -> bool:
        """Keep the D2H producer and consumer on the same stable decision."""
        return getattr(self, "_skip_spec_cpu_length_sync", False) is True


__all__ = ["AscendQwen4ExpModelState"]
