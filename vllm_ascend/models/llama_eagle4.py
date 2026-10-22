# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dense EAGLE-4 draft: vLLM EAGLE3 plus the external low-rank candidate head."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.spec_decode.eagle4 import validate_draft_config
from vllm_ascend.spec_decode.eagle4_head import (
    candidates_to_logits,
    local_candidates,
    pack_candidates,
    unpack_candidates,
)


class AscendEagle4ForCausalLM(Eagle3LlamaForCausalLM):
    # EAGLE-4 trains its own full head. Only the input embedding is shared with
    # the target by upstream load_eagle_model.
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.speculative_config.method != "eagle4":
            raise ValueError("Ascend EAGLE-4 checkpoints require speculative method='eagle4'.")
        window = validate_draft_config(vllm_config.speculative_config.draft_model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if window is not None:
            attention = self.model.layers[0].self_attn.attn
            if type(attention.impl) is not AscendAttentionBackendImpl:
                raise NotImplementedError("EAGLE-4 sliding-window drafts require the Ascend FIA attention backend.")
            # The checkpoint's W includes the query token. FIA pre_tokens is
            # the number of preceding keys; retain W on the KV cache spec.
            attention.impl.sliding_window = window - 1
        head = self.config.efficient_lm_head_config
        self.num_candidates = head["num_candidates"]
        self.efficient_lm_head = ParallelLMHead(
            self.config.vocab_size, head["rank"], prefix=f"{prefix}.efficient_lm_head"
        )
        self.efficient_lm_head_down_proj = nn.Linear(self.config.hidden_size, head["rank"], bias=False)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.register_buffer("candidate_weight", None, persistent=False)
        shard = self.lm_head.shard_indices
        self.vocab_start = shard.org_vocab_start_index
        self.vocab_rows = shard.org_vocab_end_index - self.vocab_start
        if self.num_candidates > self.vocab_rows:
            raise ValueError("EAGLE-4 num_candidates exceeds the number of real vocabulary rows on this TP rank.")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.candidate_weight is None:
            raise RuntimeError("EAGLE-4 candidate weight must be prepared after loading and target-head sharing.")
        logits, ids = local_candidates(
            hidden_states,
            self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight,
            self.candidate_weight,
            self.num_candidates,
            self.vocab_start,
            self.vocab_rows,
        )
        if self.tp_size > 1:
            packed = tensor_model_parallel_all_gather(pack_candidates(logits, ids), dim=1)
            logits, ids = unpack_candidates(packed, self.tp_size, self.num_candidates)
        return candidates_to_logits(logits, ids, self.config.vocab_size)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        required = {"lm_head.weight", "efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"}

        def checked_weights():
            for name, tensor in weights:
                required.discard(name)
                yield name, tensor

        result = super().load_weights(checked_weights())
        if required:
            raise ValueError(f"EAGLE-4 checkpoint is missing trained candidate/full head weights: {sorted(required)}")
        return result
