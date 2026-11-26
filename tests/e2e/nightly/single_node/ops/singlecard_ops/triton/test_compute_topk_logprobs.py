import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState

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


@pytest.mark.parametrize("logits_mode", [False, True])
def test_compute_topk_scores_modes_and_large_batch(logits_mode):
    """Test compute_topk_scores in both logprobs and raw logits modes with batch > vector cores."""
    init_device_properties_triton()
    torch.manual_seed(7)
    batch_size, vocab_size, num_logprobs = 128, 16384, 5
    logits = torch.randn(batch_size, vocab_size, device="npu", dtype=torch.bfloat16)
    sampled_token_ids = torch.randint(0, vocab_size, (batch_size,), device="npu", dtype=torch.int64)
    cu_tensor = torch.arange(batch_size + 1, device="npu", dtype=torch.int32)

    out = compute_topk_scores(
        logits,
        num_logprobs,
        sampled_token_ids,
        cu_num_logits=cu_tensor,
        logits_mode=logits_mode,
    )
    torch.npu.synchronize()

    topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
    ref_ids = torch.cat((sampled_token_ids.unsqueeze(-1), topk_indices), dim=1)
    if logits_mode:
        ref_scores = logits.gather(-1, ref_ids).to(torch.float32)
    else:
        ref_scores = torch.gather(torch.log_softmax(logits.float(), dim=-1), dim=1, index=ref_ids)
    sampled_logits = torch.gather(logits, 1, sampled_token_ids.unsqueeze(-1))
    ref_ranks = (logits >= sampled_logits).sum(dim=1).to(torch.int64)

    assert torch.equal(out.logprob_token_ids, ref_ids)
    assert torch.equal(out.selected_token_ranks, ref_ranks)
    assert out.cu_num_generated_tokens is None
    assert out.cu_num_generated_tokens_tensor is cu_tensor
    torch.testing.assert_close(out.logprobs, ref_scores, rtol=1e-3, atol=1e-3)


def test_compute_topk_scores_custom_logprob_token_ids():
    """Verify per-request custom logprob_token_ids override via _fill_logprob_token_ids_kernel."""
    init_device_properties_triton()
    torch.manual_seed(11)
    device = torch.device("npu")
    batch_size, vocab_size = 2, 1024
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    sampled = torch.tensor([3, 7], device=device, dtype=torch.int64)

    state = LogprobTokenIdsState(max_num_reqs=4, device=device)
    state.add_request(0, SamplingParams(logprobs=3, logprob_token_ids=[10, 20, 30]))
    state.add_request(1, SamplingParams(logprobs=2))
    state.apply_staged_writes()

    expanded_idx = torch.tensor([0, 1], device=device, dtype=torch.int32)
    out = compute_topk_scores(
        logits,
        num_logprobs=2,
        sampled_token_ids=sampled,
        logprob_token_ids_state=state,
        expanded_idx_mapping=expanded_idx,
        max_per_req_token_ids=3,
        logits_mode=False,
    )
    torch.npu.synchronize()

    assert out.logprob_token_ids[0].cpu().tolist() == [3, 10, 20, 30]
    ref_lp = torch.log_softmax(logits, dim=-1)
    torch.testing.assert_close(out.logprobs[0], ref_lp[0, [3, 10, 20, 30]], atol=1e-4, rtol=1e-4)
    # Row 1 only requested top-2 logprobs, so column 3 is masked to -inf.
    assert torch.isneginf(out.logprobs[1, 3])


def test_compute_token_logprobs_chunked_large_topk():
    """Verify chunked topk loop handles num_logprobs > _MAX_TOPK_BLOCK (1024) without UB overflow."""
    torch.manual_seed(19)
    logits = torch.randn(4, 4096, device="npu", dtype=torch.float32)
    token_ids = torch.randint(0, 4096, (4, 2048), device="npu", dtype=torch.int64)
    actual = compute_token_logprobs(logits, token_ids)
    expected = torch.gather(torch.log_softmax(logits, dim=-1), dim=1, index=token_ids)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
