# SPDX-License-Identifier: Apache-2.0
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
# Adapted from vllm-project/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py
#

from typing import Optional

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding

from vllm_ascend.distributed.parallel_state import get_lmhead_group
from vllm_ascend.distributed.utils import is_lmhead_tp


class CustomLogitsProcessor(LogitsProcessor):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.

        if is_lmhead_tp():
            hidden_states = get_lmhead_group().all_gather(hidden_states, 0)

        logits = lm_head.quant_method.apply(lm_head,
                                            hidden_states,
                                            bias=embedding_bias)

        if is_lmhead_tp():
            logits = get_lmhead_group().all_to_all(logits)

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits
