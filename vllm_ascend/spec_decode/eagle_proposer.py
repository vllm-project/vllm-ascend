# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)


class AscendQwen4ExpMTPProposer(
    Qwen4ExpMTPProposer,
    AscendSpecDecodeBaseProposer,
):
    """Ascend proposer for Qwen4Exp's multi-stream MTP draft."""

    hidden_size: int

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            True,
            runner=runner,
        )
        qwen_hidden_size = self._get_hidden_size()
        if self.hidden_size != qwen_hidden_size:
            self.hidden_size = qwen_hidden_size
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

    def _get_hidden_size(self) -> int:
        text_config = self.draft_model_config.hf_text_config
        return int(text_config.hidden_size * text_config.hc_count)

    def maybe_pad_and_reduce(self, hidden_states, positions):
        return hidden_states, positions

    def maybe_all_gather_and_unpad(
        self,
        last_hidden_states,
        positions,
        hidden_states=None,
    ):
        return last_hidden_states, positions, hidden_states
