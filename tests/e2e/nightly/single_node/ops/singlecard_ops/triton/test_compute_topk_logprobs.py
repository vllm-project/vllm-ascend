import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs, compute_topk_logprobs, compute_topk_scores

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)


@pytest.mark.parametrize(
    "batch_size,vocab_size,num_logprobs",
    [
        (48, 1024, 5),
        (96, 1024, 0),
        (24, 1519, 1),
        (1, 320, 10),
    ],
)
def test_compute_topk_logprobs(batch_size, vocab_size, num_logprobs):
    """Test compute_topk_logprobs for correctness of IDs, logprobs, and ranks.
    Args:
        batch_size: Number of sequences in the batch
        vocab_size: Size of the vocabulary
        num_logprobs: Number of top-k logprobs to return (excluding the sampled token)
    """
    init_device_properties_triton()
    # ========== 1. Setup test data ==========
    torch.manual_seed(42)
    device = "npu"

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    sampled_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

    # ========== 2. Execute Triton implementation ==========
    triton_output = compute_topk_logprobs(logits, num_logprobs, sampled_token_ids)
    torch.npu.synchronize()

    # ========== 3. Compute reference values using PyTorch ==========
    if num_logprobs == 0:
        ref_token_ids = sampled_token_ids.unsqueeze(-1)
    else:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        ref_token_ids = torch.cat((sampled_token_ids.unsqueeze(-1), topk_indices), dim=1)

    ref_all_logprobs = torch.log_softmax(logits, dim=-1)
    ref_logprobs = torch.gather(ref_all_logprobs, dim=1, index=ref_token_ids)

    sampled_logits = torch.gather(logits, 1, sampled_token_ids.unsqueeze(-1))
    # Match upstream rank semantics: count logits >= the selected logit.
    ref_ranks = (logits >= sampled_logits).sum(dim=1).to(torch.int64)

    # ========== 4. Verify results ==========
    assert torch.equal(triton_output.logprob_token_ids, ref_token_ids), (
        "Token IDs (Sampled + TopK) do not match between Triton and PyTorch."
    )

    assert torch.equal(triton_output.selected_token_ranks, ref_ranks), (
        f"Token Ranks do not match.\nTriton: {triton_output.selected_token_ranks}\nPyTorch: {ref_ranks}"
    )

    assert torch.allclose(triton_output.logprobs, ref_logprobs, rtol=1e-4, atol=1e-4), (
        f"Logprobs values differ between Triton and PyTorch.\n"
        f"Max diff: {torch.max(torch.abs(triton_output.logprobs - ref_logprobs))}"
    )


def test_selected_token_rank_is_one_based():
    """Check ranks for a unique maximum, runner-up, and top tie."""
    init_device_properties_triton()

    vocab_size = 128
    torch.manual_seed(0)
    logits = torch.randn(3, vocab_size, device="npu", dtype=torch.float32)

    # Force a two-way top tie.
    logits[2].clamp_(max=6.0)
    logits[2, 5] = 7.0
    logits[2, 9] = 7.0

    sampled_token_ids = torch.stack(
        (
            torch.argmax(logits[0]),  # unique argmax -> rank 1
            torch.argsort(logits[1], descending=True)[1],  # runner-up -> rank 2
            torch.tensor(5, device="npu", dtype=torch.int64),  # one of the tie
        )
    ).to(torch.int64)

    output = compute_topk_logprobs(logits, 0, sampled_token_ids)
    torch.npu.synchronize()

    ranks = output.selected_token_ranks.cpu().tolist()
    assert ranks[0] == 1, f"the argmax must rank 1, got {ranks[0]}"
    assert ranks[1] == 2, f"the runner-up must rank 2, got {ranks[1]}"
    assert ranks[2] == 2, f"each of a two-way top tie must report 2, got {ranks[2]}"
    assert all(rank >= 1 for rank in ranks), f"ranks must be 1-based, got {ranks}"


@pytest.mark.parametrize("vocab_size", [65535, 65536, 151937])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wide_vocab_with_a_fully_masked_tile(vocab_size, dtype):
    """Compare sampled logprobs and tied ranks at the vocabulary dispatch boundary."""
    init_device_properties_triton()
    logits = torch.full((2, vocab_size), float("-inf"), dtype=dtype, device="npu")
    logits[0, 0] = 4
    logits[0, -1] = 4
    logits[1, 0] = 4
    logits[1, -1] = 3
    sampled_ids = torch.tensor([vocab_size - 1, 0], dtype=torch.int64, device="npu")

    result = compute_topk_logprobs(logits, 1, sampled_ids)

    expected_logprobs = torch.log_softmax(logits.float(), dim=1).gather(1, result.logprob_token_ids)
    torch.testing.assert_close(result.logprobs, expected_logprobs, atol=1e-5, rtol=0)
    assert result.selected_token_ranks.tolist() == [2, 1]

    scores = compute_topk_scores(logits, 0, sampled_ids, cu_num_logits=[0, 1, 2])
    torch.testing.assert_close(scores.logprobs, expected_logprobs[:, :1], atol=1e-5, rtol=0)
    assert scores.selected_token_ranks.tolist() == [2, 1]
    assert scores.cu_num_generated_tokens == [0, 1, 2]


def test_online_rank_matches_upstream_padding():
    vocab_size = 151937
    logits = torch.zeros((1, vocab_size), device="npu", dtype=torch.float32)
    logits[0, -1] = float("-inf")
    sampled_ids = torch.tensor([vocab_size - 1], device="npu", dtype=torch.int64)

    result = compute_topk_logprobs(logits, 0, sampled_ids)
    assert result.logprobs.tolist() == [[float("-inf")]]
    assert result.selected_token_ranks.tolist() == [((vocab_size + 8191) // 8192) * 8192]


@pytest.mark.parametrize("vocab_size,num_logprobs", [(2049, 1024), (151937, 1023), (151937, 1024), (151937, 151937)])
def test_compute_topk_scores_wide_output(vocab_size, num_logprobs):
    init_device_properties_triton()
    logits = torch.linspace(-2, 2, vocab_size, device="npu", dtype=torch.float32).unsqueeze(0)
    sampled_ids = torch.tensor([vocab_size // 2], device="npu", dtype=torch.int64)

    result = compute_topk_scores(logits, num_logprobs, sampled_ids)

    expected_ids = torch.cat((sampled_ids[:, None], logits.topk(num_logprobs, dim=-1).indices), dim=1)
    expected_logprobs = logits.log_softmax(dim=-1).gather(1, expected_ids)
    expected_ranks = (logits >= logits.gather(1, sampled_ids[:, None])).sum(dim=1)
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(result.logprobs, expected_logprobs, atol=1e-5, rtol=0)
    torch.testing.assert_close(result.selected_token_ranks, expected_ranks, atol=0, rtol=0)


def test_empty_logprob_requests():
    logits = torch.empty((0, 151936), device="npu")
    sampled_ids = torch.empty(0, dtype=torch.int64, device="npu")
    result = compute_topk_logprobs(logits, 5, sampled_ids)
    assert result.logprobs.shape == result.logprob_token_ids.shape == (0, 6)
    assert result.selected_token_ranks.numel() == 0

    logits = torch.empty((3, 151936), device="npu")
    token_ids = torch.empty((3, 0), dtype=torch.int64, device="npu")
    assert compute_token_logprobs(logits, token_ids).shape == (3, 0)
