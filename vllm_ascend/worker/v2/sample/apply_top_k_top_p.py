import torch

from vllm_ascend.utils import vllm_version_is


# Canonical sort-based top-k/top-p masking implementation.
# It lives under the MRv2 worker because vllm_ascend.sample is MRv1-only and is
# scheduled for deletion: MRv1 imports it from here, so the dependency only ever
# points from the dying code to the surviving code.
def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
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


# NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
# As a result, its workspace is not reserved during warmup, which may lead to an OOM
# when the op is executed later.
# TODO: Fix the workspace reservation issue in the future.
if vllm_version_is("0.32.0"):

    def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
        # use pytorch ops
        return _apply_top_k_top_p_pytorch(logits, k, p)

else:
    # vLLM main added `has_topp_only_rows` to select the split top-p Triton
    # pipeline. This override delegates to the PyTorch reference, which handles
    # top-p-only rows either way, so the value is accepted but unused.
    def apply_top_k_top_p_npu(  # type: ignore[misc]
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        has_topp_only_rows: bool = True,
    ) -> torch.Tensor:
        # use pytorch ops
        return _apply_top_k_top_p_pytorch(logits, k, p)
