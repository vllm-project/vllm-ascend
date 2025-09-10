import torch
import torch_npu
from vllm.config import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.utils import is_310p

DEFAULT_LOGPROBS_MODE = LogprobsMode.RAW_LOGPROBS


class AscendSampler(Sampler):

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler()


class AscendTopKTopPSampler(TopKTopPSampler):

    def _apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        k: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        # npu_top_k_top_p uses the operator aclnnApplyTopKTopP, but aclnnApplyTopKTopP currently does not support 310P
        if not is_310p() and p is not None and k is not None:
            # npu_top_k_top_p's parameter order is (logits, p, k), not (logits, k, p)
            return torch_npu.npu_top_k_top_p(logits, p, k)

        if p is None and k is None:
            return logits

        probs = logits.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)

        if k is not None:
            top_k_count = probs_sort.size(1) - k.to(
                torch.long)  # shape: (batch, )
            top_k_count = top_k_count.unsqueeze(dim=1)
            top_k_cutoff = probs_sort.gather(-1, top_k_count)

            # Make sure the no top-k rows are no-op.
            no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

            elements_to_discard = probs < top_k_cutoff
            logits.masked_fill_(elements_to_discard, -float("inf"))

        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False  # at least one

            top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            elements_to_discard = probs < top_p_cutoff
            logits.masked_fill_(elements_to_discard, -float("inf"))

        return logits

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        logits = self._apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
            logits_to_return = logits
        elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return
