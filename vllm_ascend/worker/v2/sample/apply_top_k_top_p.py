import torch

from vllm_ascend.utils import vllm_version_is


# Canonical sort-based top-k/top-p masking implementation.
# It lives under the MRv2 worker because vllm_ascend.sample is MRv1-only and is
# scheduled for deletion: MRv1 imports it from here, so the dependency only ever
# points from the dying code to the surviving code.
#
# The previous implementation softmax+sort+cumsum the full [B, V] logits at
# once. On NPU the sort path allocates extra scratch (sorted values + int64
# indices + native sort workspace) that is not reserved at warmup (k/p=None),
# which can OOM under high-batch verify. Chunk rows so the *input tile* stays
# under a small
# budget (TOP_K_TOP_P_INPUT_TILE_BYTES); scratch still scales with tile rows,
# and a single row carries an irreducible O(V) requirement. Each chunk runs the
# identical original filter logic — no change to
# softmax/sort/cumsum/cutoff/ties/RNG.
TOP_K_TOP_P_INPUT_TILE_BYTES = 32 * 1024 * 1024  # 32 MiB input tile limit


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    V = logits.shape[1]
    # Per-chunk row count: bytes per row of the input tile.
    bytes_per_row = V * max(logits.element_size(), 4)
    rows_per_chunk = max(1, TOP_K_TOP_P_INPUT_TILE_BYTES // bytes_per_row)

    B = logits.shape[0]
    for lo in range(0, B, rows_per_chunk):
        hi = min(lo + rows_per_chunk, B)
        logits_chunk = logits[lo:hi]
        k_chunk = k[lo:hi] if k is not None else None
        p_chunk = p[lo:hi] if p is not None else None
        _apply_top_k_top_p_chunk(logits_chunk, k_chunk, p_chunk)
    return logits


def _apply_top_k_top_p_chunk(
    logits: torch.Tensor,  # [b, V_local]
    k: torch.Tensor,  # [b] or None
    p: torch.Tensor,  # [b] or None
) -> torch.Tensor:
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


# NOTE: The input tile is bounded by TOP_K_TOP_P_INPUT_TILE_BYTES row-chunking;
# this bounds (but does not hard-cap) the sort scratch, which still scales with
# rows-per-chunk. A single row exceeding the budget is processed alone with its
# irreducible O(V) requirement.
if vllm_version_is("0.32.0"):

    def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
        # use pytorch ops (row-chunked)
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
        # use pytorch ops (row-chunked)
        return _apply_top_k_top_p_pytorch(logits, k, p)
