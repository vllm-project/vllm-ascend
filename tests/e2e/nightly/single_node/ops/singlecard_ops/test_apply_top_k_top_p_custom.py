import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

LARGE_TOP_K = 128 * 1024


def cpu_apply_top_k_top_p(logits: torch.Tensor, p: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)

    if k is not None:
        k = torch.minimum(k, torch.full_like(k, logits.size(-1)))
        top_k_idx = logits_sort.size(1) - k.to(torch.long)
        top_k_threshold = logits_sort.gather(1, top_k_idx.unsqueeze(1))
        logits_sort.masked_fill_(logits_sort < top_k_threshold, -float("inf"))

    if p is not None:
        probs_sort = logits_sort.to(torch.float32).softmax(dim=-1)
        top_p_mask = probs_sort.cumsum(dim=-1) <= 1 - p.to(torch.float32).unsqueeze(1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)


def custom_apply_top_k_top_p(logits: torch.Tensor, p: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
    logits_npu = logits.npu()
    p_npu = p.npu() if p is not None else None
    k_npu = k.npu() if k is not None else None
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits_npu, p=p_npu, k=k_npu).cpu()


def assert_output_close(expected: torch.Tensor, actual: torch.Tensor, rtol: float, atol: float) -> None:
    expected_mask = torch.isinf(expected) & (expected < 0)
    actual_mask = torch.isinf(actual) & (actual < 0)

    mismatch = expected_mask ^ actual_mask
    mismatch_ratio = mismatch.sum().item() / expected.numel()
    if mismatch_ratio > 0.001:
        pytest.fail(f"mask mismatch ratio too high: {mismatch_ratio:.6f}")

    valid_mask = (~expected_mask) & (~actual_mask)
    if valid_mask.any():
        torch.testing.assert_close(expected[valid_mask], actual[valid_mask], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("batch_size", "vocab_size", "dtype", "p_value", "k_value", "rtol", "atol"),
    [
        (4, 15206, torch.float32, 0.9, 4096, 1e-4, 1e-4),
        (2, 152064, torch.float32, 0.9, LARGE_TOP_K, 1e-4, 1e-4),
        (2, 152064, torch.float16, 0.95, LARGE_TOP_K, 1e-3, 1e-3),
        (1, LARGE_TOP_K, torch.float32, 0.95, None, 1e-4, 1e-4),
        (1, LARGE_TOP_K, torch.float16, None, LARGE_TOP_K, 1e-3, 1e-3),
    ],
)
def test_apply_top_k_top_p_custom(batch_size, vocab_size, dtype, p_value, k_value, rtol, atol):
    torch.manual_seed(batch_size + vocab_size)
    logits = (torch.rand((batch_size, vocab_size), dtype=torch.float32) * 10 - 5).to(dtype)
    p = torch.full((batch_size,), p_value, dtype=dtype) if p_value is not None else None
    k = torch.full((batch_size,), k_value, dtype=torch.int32) if k_value is not None else None

    expected = cpu_apply_top_k_top_p(logits.clone(), p, k)
    actual = custom_apply_top_k_top_p(logits, p, k)

    assert_output_close(expected, actual, rtol=rtol, atol=atol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
