# SPDX-License-Identifier: Apache-2.0
#
# This file is a part of the vllm-ascend project.
#
# Domino speculative decoding proposer.
#
# Domino extends DFlash (parallel drafting) with a recurrent GRU correction head.
# It reuses the entire DFlash path (parallel-draft input preparation, context-KV
# precompute, draft forward, dummy_run/aclgraph) unchanged and only:
#   1. shifts ``token_indices_to_sample`` by one so the bonus (root) position is
#      sampled too (``shift_label=true`` semantics: the root position contributes
#      the first draft token via next-token prediction), and
#   2. overrides ``compute_draft_token_ids`` to refine the parallel drafts with
#      the GRU correction head before emitting the final proposal.
#
# See the Domino section in the speculative decoding user guide.

from typing import Any

import torch
from torch import nn
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.domino_correction import domino_correct, domino_correct_fused


class AscendDominoProposer(AscendDflashProposer):
    """DFlash proposer + Domino recurrent GRU correction.

    Keeps ``self.method == "dflash"`` (inherited), so ``parallel_drafting`` is
    ``True`` and every DFlash code path (input prep, context-KV precompute,
    draft forward, aclgraph capture) is reused as-is. Only the two behaviours
    above are layered on top.
    """

    # Domino's correction hook is semantic model logic, not an optional reduce-
    # sampling optimization. The base proposer must call it even when
    # enable_reduce_sample is disabled.
    requires_custom_draft_sampling = True
    _SUPPORTED_CANDIDATE_TOPK = (0, 64)

    def __init__(self, vllm_config, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner=runner)

        parallel_config = getattr(vllm_config, "parallel_config", None)
        target_tp = getattr(parallel_config, "tensor_parallel_size", 1) or 1
        draft_tp = getattr(self.speculative_config, "draft_tensor_parallel_size", 1) or 1
        if target_tp != 1 or draft_tp != 1:
            raise ValueError(
                "Domino currently requires target and draft tensor_parallel_size=1; "
                f"got target_tp={target_tp}, draft_tp={draft_tp}. The correction "
                "head is not vocab-sharded and cannot index a TP-local embedding safely."
            )

        self._domino_shift_label = None
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        candidate_topk = additional_config.get("domino_candidate_topk", 0)
        if (
            isinstance(candidate_topk, bool)
            or not isinstance(candidate_topk, int)
            or candidate_topk not in self._SUPPORTED_CANDIDATE_TOPK
        ):
            raise ValueError(
                "additional_config.domino_candidate_topk must be one of "
                f"{self._SUPPORTED_CANDIDATE_TOPK}; got {candidate_topk!r}."
            )
        # Fixed at engine initialization, so top-k output shapes stay stable
        # across ACL graph capture and replay.
        self.domino_candidate_topk = candidate_topk
        # Bonus (last accepted) token buffer for aclgraph capture. Must be a
        # FIXED tensor (not None) so the captured graph's control flow matches
        # between dummy_run (zeros) and real propose (actual bonus copied in by
        # set_inputs_first_pass). A None-or-tensor branch would diverge between
        # capture and replay and silently corrupt the GRU seed.
        max_batch = vllm_config.scheduler_config.max_num_seqs
        self._domino_bonus_buf = torch.zeros(max_batch, dtype=torch.long, device=device)
        # NOTE: ``self.model`` (the draft) is not loaded yet at __init__ time -
        # it is populated later by ``load_model()``. Do not touch it here.
        logger.info(
            "[spec_decode/domino] AscendDominoProposer initialized: "
            "num_speculative_tokens=%s, candidate_topk=%s.",
            self.num_speculative_tokens,
            self.domino_candidate_topk,
        )

    def load_model(self, model: nn.Module) -> None:
        super().load_model(model)
        self._domino_shift_label = getattr(self.model, "shift_label", None)
        if self._domino_shift_label is not True:
            raise ValueError(
                "Domino requires dflash_config.shift_label=true because its "
                "proposal positions are shifted to include the bonus token."
            )
        # Precompute the gi lookup table (and split fc1 into w_z/w_s) BEFORE
        # aclgraph capture (dummy_run), so these are cached on the draft model
        # and not re-recorded on every graph replay. super().load_model has
        # already shared the target's embed_tokens/lm_head onto the draft.
        from vllm_ascend.spec_decode.domino_correction import (
            _get_domino_correction_weights,
        )

        _get_domino_correction_weights(self.model)
        if self.domino_candidate_topk:
            vocab_size = self.model.lm_head.weight.shape[0]
            if self.domino_candidate_topk > vocab_size:
                raise ValueError(
                    "additional_config.domino_candidate_topk must not exceed "
                    f"the draft vocabulary size; got {self.domino_candidate_topk} "
                    f"for vocab_size={vocab_size}."
                )

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        num_query_total, token_indices_to_sample, cad, long_seq_args = super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )

        if self._domino_shift_label is not True:
            raise RuntimeError(
                "Domino proposal requested without validated shift_label=true."
            )

        # shift_label=true: the bonus (root) position (q_idx=0) contributes the
        # first draft token via next-token prediction. The DFlash kernel wrote
        # token_indices_to_sample for the k mask positions (q_idx=1..k); shift
        # them by -1 so we sample [bonus, mask_1, ..., mask_{k-1}] instead of
        # [mask_1, ..., mask_k]. The draft-token positions (C+1..C+k) are
        # unchanged, so the verifier alignment is identical to DFlash.
        token_indices_to_sample = token_indices_to_sample - 1

        # Copy the bonus (last accepted) token into the fixed buffer; the
        # aclgraph-replayed compute_draft_token_ids reads from this buffer.
        self._domino_bonus_buf[: next_token_ids.shape[0]].copy_(next_token_ids)

        return num_query_total, token_indices_to_sample, cad, long_seq_args

    def compute_draft_token_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Produce draft tokens via DFlash parallel logits + GRU correction.

        ``hidden_states`` is ``sample_hidden_states`` = ``[B*k, H]`` where, after
        the shift_label shift, position 0 is the bonus (root) hidden and
        positions 1..k-1 are the first k-1 mask hiddens. With the Domino
        checkpoint (``shift_label=true``, next-token prediction), ``base_logits``
        at position i predicts draft token i+1.

        Uses the fused triton path (``domino_correct_fused``) by default; falls
        back to the eager pure-torch path (``domino_correct``, unit-tested on
        CPU) if triton is unavailable. The bonus is read from the fixed
        ``_domino_bonus_buf`` (aclgraph-safe).
        """
        # Prefer fused (triton kernels + precompute); fall back to eager.
        try:
            return domino_correct_fused(
                hidden_states=hidden_states,
                num_speculative_tokens=self.num_speculative_tokens,
                model=self.model,
                bonus_token_ids=self._domino_bonus_buf,
                candidate_topk=self.domino_candidate_topk,
            )
        except ImportError:
            return domino_correct(
                hidden_states=hidden_states,
                num_speculative_tokens=self.num_speculative_tokens,
                model=self.model,
                bonus_token_ids=self._domino_bonus_buf,
                candidate_topk=self.domino_candidate_topk,
            )
