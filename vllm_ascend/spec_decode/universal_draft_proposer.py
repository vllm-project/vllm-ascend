# SPDX-License-Identifier: Apache-2.0
"""
Ascend adaptation for universal speculative decoding with heterogeneous
vocabularies (Token-Level Intersection / TLI).

Depends on upstream vllm PR: https://github.com/vllm-project/vllm/pull/38174
which provides ``VocabMapping`` and the ``universal_draft`` config method.
"""

import torch
from vllm.config import VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.vocab_mapping import VocabMapping

from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class AscendUniversalDraftProposer(AscendDraftModelProposer):
    """Draft-model proposer that supports heterogeneous vocabularies.

    The draft and target models may use different tokenizers.  A
    ``VocabMapping`` translates token IDs between the two vocabularies,
    and draft logits are constrained to the vocabulary intersection so
    that rejection sampling remains valid (provably lossless).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # Intentionally skip AscendDraftModelProposer.__init__ to avoid
        # the vocab-size-mismatch check.  We still need the base class
        # initialisation and the TP-mismatch check.
        SpecDecodeBaseProposer.__init__(
            self,
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_draft_tp_mismatch()
        self.vocab_mapping: VocabMapping | None = None

    # -- model loading -------------------------------------------------------

    def load_model(self, model) -> None:
        super().load_model(model)

        target_tokenizer = get_tokenizer(
            self.vllm_config.model_config.tokenizer,
            trust_remote_code=self.vllm_config.model_config.trust_remote_code,
        )
        draft_tokenizer = get_tokenizer(
            self.speculative_config.draft_model_config.tokenizer,
            trust_remote_code=(
                self.speculative_config.draft_model_config.trust_remote_code
            ),
        )
        self.vocab_mapping = VocabMapping(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=(
                self.vllm_config.model_config.get_vocab_size()
            ),
            draft_vocab_size=(
                self.speculative_config.draft_model_config.get_vocab_size()
            ),
            device=self.device,
        )

    # -- propose with vocabulary translation ---------------------------------

    def _propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        **kwargs,
    ) -> torch.Tensor:
        assert self.vocab_mapping is not None

        # 1) Map incoming target-vocab IDs → draft-vocab IDs so the draft
        #    model receives tokens it understands.
        draft_input_ids = self.vocab_mapping.map_target_to_draft_ids(
            target_token_ids
        )
        draft_next_ids = self.vocab_mapping.map_target_to_draft_ids(
            next_token_ids
        )

        # 2) Temporarily wrap compute_logits on the raw draft model so that
        #    logits are constrained to the vocabulary intersection before
        #    argmax.  This avoids duplicating the complex multi-step loop
        #    in SpecDecodeBaseProposer._run_merged_draft.
        raw_model = self.get_model()
        orig_compute_logits = raw_model.compute_logits

        def _constrained_compute_logits(hidden_states):
            logits = orig_compute_logits(hidden_states)
            return self.vocab_mapping.constrain_draft_logits(logits)

        raw_model.compute_logits = _constrained_compute_logits

        try:
            draft_token_ids = super()._propose(
                target_token_ids=draft_input_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=draft_next_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                **kwargs,
            )
        finally:
            raw_model.compute_logits = orig_compute_logits

        # 3) Map draft-vocab IDs → target-vocab IDs before returning to the
        #    verification stage.
        return self.vocab_mapping.map_draft_to_target_ids(draft_token_ids)
