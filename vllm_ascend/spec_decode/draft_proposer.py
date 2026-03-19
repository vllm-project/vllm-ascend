import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.eagle_proposer import AscendSpecDecodeBaseProposer


class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        DraftModelProposer.__init__(self, vllm_config, device, runner=runner)
 
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
