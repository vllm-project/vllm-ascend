# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)


class AscendQwen4ExpMTPProposer(Qwen4ExpMTPProposer, AscendSpecDecodeBaseProposer):
    """Ascend proposer for Qwen4Exp multi-stream MTP."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

    def _get_hidden_size(self) -> int:
        # ModelConfig.get_hidden_size() resolves the multimodal outer width for
        # Qwen3.8-Flash-Next. The MTP feedback is the text backbone's HC stream.
        text_config = self.draft_model_config.hf_text_config
        return int(text_config.hidden_size * text_config.hc_count)
