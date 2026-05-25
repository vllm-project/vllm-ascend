# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(
            self, vllm_config, device, pass_hidden_states_to_model=True, runner=runner
        )
