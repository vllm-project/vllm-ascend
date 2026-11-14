# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.logprob import compute_topk_scores

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
def test_compute_topk_scores(batch_size, vocab_size, num_logprobs):
    """Test compute_topk_scores for correctness of IDs, logprobs, and ranks.
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
    triton_output = compute_topk_scores(logits, num_logprobs, sampled_token_ids)
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

    output = compute_topk_scores(logits, 0, sampled_token_ids)
    torch.npu.synchronize()

    ranks = output.selected_token_ranks.cpu().tolist()
    assert ranks[0] == 1, f"the argmax must rank 1, got {ranks[0]}"
    assert ranks[1] == 2, f"the runner-up must rank 2, got {ranks[1]}"
    assert ranks[2] == 2, f"each of a two-way top tie must report 2, got {ranks[2]}"
    assert all(rank >= 1 for rank in ranks), f"ranks must be 1-based, got {ranks}"


@pytest.mark.parametrize("entry", ["direct", "sampler", "prompt_logprob", "rejection_sampler"])
@pytest.mark.parametrize("num_logprobs", [0, 2])
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("tensor_metadata", [False, True])
def test_custom_token_scores(num_logprobs, logits_mode, tensor_metadata, entry):
    """Mixed requests preserve requested order, expanded mapping and padding."""
    init_device_properties_triton()
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [1.0, 3.0, 5.0, 0.0, 2.0, 4.0],
            [4.0, 2.0, 0.0, 5.0, 3.0, 1.0],
        ],
        device="npu",
    )
    sampled = torch.tensor([3, 1, 2, 0], device="npu")
    mapping = torch.tensor([2, 0, 1, 2], dtype=torch.int32, device="npu")
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(gpu=torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")),
        token_ids=SimpleNamespace(gpu=torch.tensor([[5, 0], [0, 0], [1, 0]], dtype=torch.int32, device="npu")),
    )
    metadata = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu") if tensor_metadata else [0, 2, 4]
    score_fn = compute_topk_scores
    if entry != "direct":
        importlib.import_module("vllm_ascend.patch.worker.patch_triton")
        path = (
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
            if entry == "rejection_sampler"
            else f"vllm.v1.worker.gpu.sample.{entry}"
        )
        score_fn = importlib.import_module(path).compute_topk_scores
        assert score_fn is compute_topk_scores
    result = score_fn(logits, num_logprobs, sampled, metadata, state, mapping, 2, logits_mode)
    expected_ids = torch.zeros((4, 3), dtype=sampled.dtype, device="npu")
    valid = torch.zeros((4, 3), dtype=torch.bool, device="npu")
    expected_ids[:, 0] = sampled
    valid[:, 0] = True
    # Rows 0 and 3 map to request 2, row 1 to request 0, row 2 uses top-k.
    expected_ids[0, 1] = expected_ids[3, 1] = 1
    valid[0, 1] = valid[3, 1] = True
    expected_ids[1, 1:] = torch.tensor([5, 0], device="npu")
    valid[1, 1:] = True
    if num_logprobs:
        expected_ids[2, 1:] = torch.tensor([2, 5], device="npu")
        valid[2, 1:] = True
    scores = logits if logits_mode else logits.log_softmax(-1)
    expected = scores.gather(-1, expected_ids).masked_fill(~valid, -float("inf"))
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(result.logprobs, expected, rtol=1e-4, atol=1e-4)
    if tensor_metadata:
        assert result.cu_num_generated_tokens is None
        assert result.cu_num_generated_tokens_tensor is metadata
    else:
        assert result.cu_num_generated_tokens is metadata
        assert result.cu_num_generated_tokens_tensor is None


def test_custom_duplicate_candidates():
    init_device_properties_triton()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device="npu")
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(gpu=torch.tensor([3], dtype=torch.int32, device="npu")),
        token_ids=SimpleNamespace(gpu=torch.tensor([[2, 2, 0]], dtype=torch.int32, device="npu")),
    )
    result = compute_topk_scores(
        logits,
        0,
        torch.tensor([3], device="npu"),
        logprob_token_ids_state=state,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device="npu"),
        max_per_req_token_ids=3,
    )
    expected_ids = torch.tensor([[3, 2, 2, 0]], device="npu")
    torch.testing.assert_close(result.logprob_token_ids, expected_ids)
    torch.testing.assert_close(result.logprobs, logits.log_softmax(-1).gather(1, expected_ids))


@pytest.mark.parametrize("num_logprobs", [8192, 32768, 152064])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("logits_mode", [False, True])
def test_large_token_scores(num_logprobs, custom, logits_mode):
    """Large candidates stream across blocks through the actual sampler binding."""
    init_device_properties_triton()
    importlib.import_module("vllm_ascend.patch.worker.patch_triton")
    score_fn = importlib.import_module("vllm.v1.worker.gpu.sample.sampler").compute_topk_scores
    assert score_fn is compute_topk_scores
    torch.manual_seed(42)
    logits = torch.randn(2, 152064, device="npu", dtype=torch.float32)
    sampled = torch.tensor([11, 12], device="npu", dtype=torch.int64)
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(gpu=torch.tensor([0, 3], device="npu", dtype=torch.int32)),
        token_ids=SimpleNamespace(gpu=torch.tensor([[0, 0, 0], [5, 6, 7]], device="npu", dtype=torch.int32)),
    )
    mapping = torch.tensor([1, 0], device="npu", dtype=torch.int32)
    result = score_fn(
        logits,
        num_logprobs,
        sampled,
        logprob_token_ids_state=state if custom else None,
        expanded_idx_mapping=mapping if custom else None,
        max_per_req_token_ids=3 if custom else 0,
        logits_mode=logits_mode,
    )
    expected_ids = torch.cat((sampled[:, None], logits.topk(num_logprobs, dim=-1).indices), dim=1)
    valid = torch.ones_like(expected_ids, dtype=torch.bool)
    if custom:
        expected_ids[0, 1:] = 0
        expected_ids[0, 1:4] = state.token_ids.gpu[1].to(torch.int64)
        valid[0, 4:] = False
    scores = logits if logits_mode else logits.log_softmax(-1)
    expected_scores = scores.gather(-1, expected_ids).masked_fill(~valid, -float("inf"))
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=1e-4, atol=1e-4)
    expected_ranks = (logits >= logits.gather(1, sampled[:, None])).sum(-1)
    torch.testing.assert_close(result.selected_token_ranks, expected_ranks, rtol=0, atol=0)
