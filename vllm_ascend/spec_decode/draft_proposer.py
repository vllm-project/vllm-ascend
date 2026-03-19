import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.eagle_proposer import AscendSpecDecodeBaseProposer


class AscendDraftModelProposer(AscendSpecDecodeBaseProposer, DraftModelProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
