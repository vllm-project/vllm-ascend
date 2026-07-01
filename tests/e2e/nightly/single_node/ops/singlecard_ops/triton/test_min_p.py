import torch
import pytest
from vllm_ascend.worker.v2.sample.min_p import apply_min_p
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
@pytest.mark.parametrize("num_reqs,vocab_size", [
    (48, 102400),
    (96, 102400),
    (24, 151936),
    (1, 32000),
])
def test_apply_min_p_kernel(num_reqs, vocab_size):

    """Test apply_min_p for computing Min-P sampling mask
    Args:
        num_reqs: Number of sequences in the batch
        vocab_size: Size of the vocabulary
    """
    
    init_device_properties_triton()
    # ========== Setup test data ==========
    torch.manual_seed(42)

    # Generate random logits (using float32 as specified in your kernel)
    device = 'npu'

    original_logits = torch.randn(num_reqs, vocab_size, device=device, dtype=torch.float32)

    triton_logits = original_logits.clone()
    ref_logits = original_logits.clone()

    # Generate random min_p values (valid range is typically (0, 1.0])
    min_p = torch.empty(num_reqs, device=device, dtype=torch.float32).uniform_(0.01, 0.5)

    # ========== Execute test ==========
    # 1. Invoke your Triton kernel wrapper
    apply_min_p(triton_logits, min_p)
    torch.npu.synchronize()

    # 2. Compute reference values using PyTorch
    max_vals = ref_logits.max(dim=-1, keepdim=True).values

    thresholds = max_vals + torch.log(min_p.unsqueeze(1))

    ref_logits[ref_logits < thresholds] = float("-inf")

    # ========== Verify results ==========
    triton_inf_mask = torch.isinf(triton_logits)
    ref_inf_mask = torch.isinf(ref_logits)

    assert torch.equal(triton_inf_mask, ref_inf_mask), \
        "Masked positions (where logits == -inf) do not match between Triton and PyTorch."

    valid_triton_logits = triton_logits[~triton_inf_mask]
    valid_ref_logits = ref_logits[~ref_inf_mask]

    assert torch.allclose(valid_triton_logits, valid_ref_logits, rtol=1e-4, atol=1e-4), \
        f"Logits values differ between Triton and PyTorch reference.\n" \
        f"Max diff: {torch.max(torch.abs(valid_triton_logits - valid_ref_logits))}"
