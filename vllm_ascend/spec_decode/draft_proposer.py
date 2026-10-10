import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _maybe_share_lm_head(self, model: nn.Module) -> None:
        # Upstream DraftModelProposer._maybe_share_lm_head is a no-op ("draft
        # models don't share lm_head with the target model"). Because it comes
        # first in the MRO, it shadows the Ascend implementation. The Ascend
        # version is still required for the draft_model method: it leaves
        # lm_head untouched (the eagle/dflash and mtp branches don't apply),
        # but it sets up ACL full-graph support -- self.update_stream and the
        # ACLGraphWrapper-wrapped self._runnable. Without this delegation,
        # cudagraph mode crashes with "AttributeError: 'AscendDraftModelProposer'
        # object has no attribute 'update_stream'" and the draft model
        # silently runs in eager mode (no graph capture).
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, model)
