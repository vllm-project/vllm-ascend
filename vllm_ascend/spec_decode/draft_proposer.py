import logging

import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

logger = logging.getLogger(__name__)


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

    def _get_model(self):
        draft_vllm_config = self._create_draft_vllm_config()
        draft_load_config = draft_vllm_config.load_config
        draft_model_config = self.speculative_config.draft_model_config
        logger.info(
            "AscendDraftModelProposer._get_model(): loading draft model with "
            "load_format=%s, model=%s",
            getattr(draft_load_config, "load_format", None),
            getattr(draft_model_config, "model", None),
        )
        return super()._get_model()
