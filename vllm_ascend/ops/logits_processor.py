#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import TYPE_CHECKING, Optional

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_ascend.distributed.parallel_state import get_lmhead_tp_group
from vllm_ascend.utils import lmhead_tp_enable

if TYPE_CHECKING:
    from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead


class AscendLogitsProcessor(LogitsProcessor):
    """
    Register LogitsProcessor as a custom op for Ascend.
    Added the feature of lmheadTP in pure dp scenario
    """

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: "AscendParallelLMHead",
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if lmhead_tp_enable():
            return self._get_logits_lmheadtp(hidden_states, lm_head,
                                             embedding_bias)
        else:
            return self._get_logits_normal(hidden_states, lm_head,
                                           embedding_bias)

    def _get_logits_lmheadtp(
        self,
        hidden_states: torch.Tensor,
        lm_head: "AscendParallelLMHead",
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Gather hidden states from all devices in tensor parallel group
        gathered_hidden_states = get_lmhead_tp_group().all_gather(
            hidden_states, dim=0)
        local_logits = lm_head.quant_method.apply(lm_head,
                                                  gathered_hidden_states,
                                                  bias=embedding_bias)
        # Gather logits for tensor parallel
        logits = get_lmhead_tp_group().all_to_all(local_logits)
        # Remove paddings in vocab (if any)
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def _get_logits_normal(
        self,
        hidden_states: torch.Tensor,
        lm_head: "AscendParallelLMHead",
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        local_logits = lm_head.quant_method.apply(lm_head,
                                                  hidden_states,
                                                  bias=embedding_bias)
        # Gather logits for tensor parallel
        logits = self._gather_logits(local_logits)

        # Remove paddings in vocab (if any)
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]

        return logits
