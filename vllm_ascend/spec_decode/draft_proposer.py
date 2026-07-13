import torch
from vllm.config import VllmConfig
from vllm.tokenizers.registry import get_tokenizer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.vocab_mapping import VocabMapping

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
        self._raise_if_draft_tp_mismatch()

        # TLI (vllm#38174): when target and draft use different tokenizers,
        # build a VocabMapping to translate token IDs between the two vocabs
        # and constrain draft logits to the intersection. NOTE: the ascend
        # base __init__ above does NOT run core DraftModelProposer.__init__,
        # so we must build vocab_mapping here (mirror of draft_model.py).
        self.use_heterogeneous_vocab = self.speculative_config.use_heterogeneous_vocab
        spec = self.speculative_config
        if self.use_heterogeneous_vocab:
            target_tokenizer = get_tokenizer(
                spec.target_model_config.tokenizer,
                trust_remote_code=spec.target_model_config.trust_remote_code,
            )
            draft_tokenizer = get_tokenizer(
                spec.draft_model_config.model,
                trust_remote_code=spec.draft_model_config.trust_remote_code,
            )
            self.vocab_mapping: VocabMapping | None = VocabMapping(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
                target_vocab_size=spec.target_model_config.get_vocab_size(),
                draft_vocab_size=spec.draft_model_config.get_vocab_size(),
                device=device,
            )
        else:
            self._raise_if_vocab_size_mismatch()
            self.vocab_mapping = None
