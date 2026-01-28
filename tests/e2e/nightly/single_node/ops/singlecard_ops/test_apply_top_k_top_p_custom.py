import numpy as np
import pytest
import torch
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def cpu_op_exec(logits, p, k):
    """
    Apply top-k and top-p sampling filtering.
    """
    # Sort logits in ascending order
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)

    # 1. Apply top-k filtering
    if k is not None:
        # Ensure k does not exceed vocab_size
        k = torch.minimum(k, torch.tensor(logits.size(-1), device=k.device))
        top_k_mask_idx = logits_sort.size(1) - k.to(torch.long)
        top_k_threshold = logits_sort.gather(1, top_k_mask_idx.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_threshold
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # 2. Apply top-p (nucleus) filtering
    if p is not None:
        probs_sort = logits_sort.to(torch.float32).softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # 3. Restore original order
    logits = torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def cpu_op_exec_top_k(logits, p, k):
    return cpu_op_exec(logits, None, k)


def cpu_op_exec_top_p(logits, p, k):
    return cpu_op_exec(logits, p, None)


def ascendc_op_exec(logits, p, k):
    """
    Execute the custom Ascend NPU operator.
    """
    logits_npu = logits.npu()
    p_npu = p.npu() if p is not None else None
    k_npu = k.npu() if k is not None else None

    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits_npu, k=k_npu, p=p_npu).cpu()


def assert_output_close(out_cpu, out_npu, rtol=1e-4, atol=1e-4):
    """
    Custom assertion to handle Top-P boundary precision issues.
    """
    # 1. Check mask consistency (inf vs finite)
    mask_cpu = torch.isinf(out_cpu) & (out_cpu < 0)
    mask_npu = torch.isinf(out_npu) & (out_npu < 0)

    mismatch_mask = mask_cpu ^ mask_npu
    mismatch_count = mismatch_mask.sum().item()
    total_elements = out_cpu.numel()

    # Allow 0.1% mismatch for boundary floating point precision differences
    mismatch_ratio = mismatch_count / total_elements
    if mismatch_ratio > 0.001:
        pytest.fail(f"Mask mismatch ratio too high: {mismatch_ratio:.6f} ({mismatch_count}/{total_elements})")

    # 2. Check value consistency for valid elements
    valid_mask = (~mask_cpu) & (~mask_npu)
    if valid_mask.any():
        torch.testing.assert_close(
            out_cpu[valid_mask],
            out_npu[valid_mask],
            rtol=rtol,
            atol=atol
        )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('vocab_size', [15206, 152064])
@pytest.mark.parametrize('batch_size', [4, 8, 16, 32, 64, 96, 128, 256])
@pytest.mark.parametrize('p_val', [0.5, 0.9, 0.99])
@pytest.mark.parametrize('k_val', [50, 200, 1024, 4096, 8192])
def test_npu_apply_top_k_top_p(vocab_size, batch_size, p_val, k_val):
    shape = [batch_size, vocab_size]
    dtype = torch.float32

    logits = torch.from_numpy(np.random.uniform(-5, 5, shape)).to(dtype)
    p = torch.full((batch_size,), p_val, dtype=dtype)
    k = torch.full((batch_size,), k_val, dtype=torch.int32)

    out_cpu = cpu_op_exec(logits.clone(), p, k)
    out_npu = ascendc_op_exec(logits, p, k)

    assert_output_close(out_cpu, out_npu)


@pytest.mark.parametrize('vocab_size', [15206, 152064])
@pytest.mark.parametrize('batch_size', [4, 8, 16, 32, 64, 96, 128, 256])
@pytest.mark.parametrize('k_val', [50, 200, 1024, 4096, 8192])
def test_npu_apply_top_k(vocab_size, batch_size, k_val):
    shape = [batch_size, vocab_size]
    dtype = torch.float32

    logits = torch.from_numpy(np.random.uniform(-5, 5, shape)).to(dtype)
    p = None
    k = torch.full((batch_size,), k_val, dtype=torch.int32)

    out_cpu = cpu_op_exec_top_k(logits.clone(), p, k)
    out_npu = ascendc_op_exec(logits, p, k)

    assert_output_close(out_cpu, out_npu)


@pytest.mark.parametrize('vocab_size', [15206, 152064])
@pytest.mark.parametrize('batch_size', [4, 8, 16, 32, 64, 96, 128, 256])
@pytest.mark.parametrize('p_val', [0.5, 0.9, 0.99])
def test_npu_apply_top_p(vocab_size, batch_size, p_val):
    shape = [batch_size, vocab_size]
    dtype = torch.float32

    logits = torch.from_numpy(np.random.uniform(-5, 5, shape)).to(dtype)
    p = torch.full((batch_size,), p_val, dtype=dtype)
    k = None

    out_cpu = cpu_op_exec_top_p(logits.clone(), p, k)
    out_npu = ascendc_op_exec(logits, p, k)

    assert_output_close(out_cpu, out_npu)
