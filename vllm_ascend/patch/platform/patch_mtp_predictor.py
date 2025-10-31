# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import vllm
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP


def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_index: int = 0,
) -> torch.Tensor:
    assert inputs_embeds is not None
    # masking inputs at position 0, as not needed by MTP
    # Patch this for aclgraph support, as the original operation introduced d2h sync,
    # which breaks aclgraph
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = self.hnorm(previous_hidden_states)

    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

    hidden_states, residual = self.mtp_block(positions=positions,
                                             hidden_states=hidden_states,
                                             residual=None)
    hidden_states = residual + hidden_states
    return hidden_states


# Patch this only for aclgraph support, as this is not support in vLLM 0.11.0
@support_torch_compile
class AscendDeepSeekMTP(DeepSeekMTP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)


vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer.forward = forward
