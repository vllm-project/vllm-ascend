from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ops.triton.spec_decode.utils import dflash2_greedy_selector_walk_kernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


def is_dflash2_draft(speculative_config: Any) -> bool:
    """True when method is dflash and the draft architecture is DFlash2DraftModel."""
    if speculative_config.method != "dflash":
        return False
    return "DFlash2DraftModel" in (speculative_config.draft_model_config.architectures or [])


def greedy_select_path(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Walk the best path from the verified anchor (T=0).

    ``candidate_ids``: ``[num_reqs, num_steps, top_k]``
    ``scores``: ``[num_reqs, num_steps, top_k, top_k]`` (prev → cand)
    Returns ``[num_reqs, num_steps]`` token ids.
    """
    num_reqs, num_steps, top_k = candidate_ids.shape
    assert candidate_ids.is_contiguous()
    assert scores.is_contiguous()
    output_tokens = torch.empty(
        (num_reqs, num_steps),
        dtype=candidate_ids.dtype,
        device=candidate_ids.device,
    )
    init_device_properties_triton()
    num_programs = min(num_reqs, get_vectorcore_num())
    dflash2_greedy_selector_walk_kernel[(num_programs,)](
        scores,
        candidate_ids,
        output_tokens,
        num_reqs,
        num_steps=num_steps,
        top_k=top_k,
    )
    return output_tokens


class AscendDflash2Proposer(AscendDflashProposer):
    """DFlash2 v1 proposer: same input layout as DFlash, selector instead of argmax."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DFlash2 probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self.use_dflash2_selector = True

        num_query_per_req = 1 + self.num_speculative_tokens
        self._anchor_indices = (
            torch.arange(self.max_batch_size, device=self.device, dtype=torch.int64) * num_query_per_req
        )

    def _maybe_share_lm_head(self, model: nn.Module) -> None:
        if getattr(self.model, "draft_id_to_target_id", None) is not None:
            raise ValueError(
                "DFlash2 does not support a reduced draft vocabulary; "
                "the selector top-k needs the unquantized target LM head."
            )
        self.model.has_own_lm_head = False
        super()._maybe_share_lm_head(model)

    def compute_draft_token_ids(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # DFlash2 always drafts greedily: probabilistic sampling is rejected in
        # __init__, so draft probs are never produced here.
        num_sample = hidden_states.shape[0]
        num_steps = self.num_speculative_tokens
        if num_sample % num_steps != 0:
            raise ValueError(
                f"DFlash2 expected hidden states divisible by num_speculative_tokens={num_steps}, got {num_sample}."
            )
        num_reqs = num_sample // num_steps
        hidden = hidden_states.view(num_reqs, num_steps, -1)
        candidate_ids, unary_logits = self.model.compute_candidates(hidden.flatten(0, 1))
        candidate_ids = candidate_ids.view(num_reqs, num_steps, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )
        return greedy_select_path(candidate_ids, scores).reshape(-1), None
