import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs


@pytest.mark.skipif(
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
    # A rank is the number of entries whose logit is at or above the sampled
    # token's, so the argmax ranks 1. Using `>` here would mirror a `>`
    # implementation and hide the divergence from vLLM; the contract itself is
    # pinned by test_selected_token_rank_is_one_based below.
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
    """The selected token's rank counts entries at or above its logit.

    The tie behaviour is pinned by vLLM's implementation, not by its docstring:
    MRV2's `_ranks_kernel` computes `tl.sum((logits >= x))` and the V1 helper
    `batched_count_greater_than` computes `(x >= values).sum(-1)`. So the rank is
    the number of vocabulary entries whose logit is at or above the sampled
    token's: 1 for a unique argmax, 2 for the runner-up, and 2 for *both*
    members of a two-way tie at the top, because a tied entry counts its other
    member too. The documented "(>=1)" only rules out 0; it does not by itself
    imply 2 for a tie, which is why this test cites the comparison.

    The difference from a strict `>` is exactly the multiplicity of the sampled
    logit, so a tie is off by the tie size rather than uniformly by one.

    Asserted directly rather than against a locally recomputed reference, since
    a `>`-based reference and a `>`-based kernel agree with each other while
    both disagree with vLLM.
    """
    init_device_properties_triton()

    vocab_size = 128
    torch.manual_seed(0)
    logits = torch.randn(3, vocab_size, device="npu", dtype=torch.float32)

    # Row 2: force a two-way tie at the top so the tie semantics are covered.
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


def test_selected_token_rank_excludes_padding_lanes():
    """A rank never exceeds the vocabulary size, even at a -inf selected logit.

    The kernel loads out-of-range lanes of the last tile as `-inf`. With `>=`,
    a selected logit of `-inf` also satisfies `-inf >= -inf`, so those padding
    lanes would be counted and the rank would jump to the padded width instead
    of `vocab_size`. `128256`-style vocabularies make this reachable because they
    are not multiples of the kernel's block size.
    """
    init_device_properties_triton()

    # Deliberately not a multiple of the kernel block size (8192), so the last
    # tile is partially padding.
    vocab_size = 1519
    torch.manual_seed(0)
    logits = torch.randn(1, vocab_size, device="npu", dtype=torch.float32)

    # Select a token whose logit is -inf: every real entry then satisfies
    # `logit >= x`, so the only correct rank is exactly vocab_size.
    sampled_token_id = 7
    logits[0, sampled_token_id] = float("-inf")

    output = compute_topk_logprobs(logits, 0, torch.tensor([sampled_token_id], device="npu", dtype=torch.int64))
    torch.npu.synchronize()

    rank = output.selected_token_ranks.cpu().item()
    assert rank == vocab_size, (
        f"a -inf selected logit must rank exactly vocab_size ({vocab_size}), got {rank}; "
        "a larger value means padding lanes were counted"
    )
