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

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = logits_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = logits_sort.gather(1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        logits_sort.masked_fill_(logits_sort < top_k_cutoff, -float("inf"))

    if p is not None:
        # Re-normalize after the top-k cut: top-p must be applied to the
        # distribution truncated by top-k, not to the full-vocabulary one.
        # Reusing the pre-cut distribution silently keeps extra tokens, and
        # degrades to top-k only whenever the retained mass is <= p.
        probs_sort = logits_sort.softmax(dim=-1)
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


# NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
# As a result, its workspace is not reserved during warmup, which may lead to an OOM
# when the op is executed later.
# TODO: Fix the workspace reservation issue in the future.
if vllm_version_is("0.31.0"):

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
