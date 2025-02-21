import pytest
import torch
from vllm_ascend.ops.fused_moe import fused_moe


def test_fused_moe():
    # Since we are using native PyTorch operations in the function, the most reliable ground truth
    # for comparison is the manually computed output. By using hardcoded data, we can ensure
    # that the function produces the expected results and validate its correctness against a known reference.

    # Step 1: Constructing inputs
    hidden_states = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    # w1: [3, 4, 3] (num_experts=3, intermediate_size*2=4, hidden_size=3)
    w1 = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [2.0, 1.0, 0.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0]],
            [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [2.0, -2.0, 2.0], [1.0, 0.0, -1.0]],
            [[-2.0, -1.0, 1.0], [2.0, -1.0, 1.0], [-1.0, 2.0, 1.0], [1.0, 1.0, -1.0]],
        ]
    )

    # w2: [3, 3, 2] (num_experts=3, hidden_size=3, intermediate_size=2)
    w2 = torch.tensor(
        [
            [[1.0, 0.5], [2.0, -1.0], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 1.0], [1.0, -0.0]],
            [[-2.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        ]
    )

    # gating_output: [2, 3] (num_tokens=2, num_experts=3)
    gating_output = torch.tensor([[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]])

    topk = 2

    global_num_experts = 3

    # Only has the first two experts
    expert_map = torch.tensor([0, 1, -1])

    renormalize = False

    use_grouped_topk = False

    # Step 2: Expected output calculation

    # We use topk=2, which means we select the top 2 experts based on gating_output.
    # For sample 1, gating_output = [0.1, 0.7, 0.2], topk_weights = [0.7, 0.2], selected experts = 1, 2
    # For sample 2, gating_output = [0.5, 0.4, 0.1], topk_weights = [0.5, 0.4], selected experts = 0, 1

    # 1. Calculate linear transformation of hidden_states with w1[0] -> F.linear(hidden_states, w1[0])
    # 2. Apply gating function to get gate values -> F.silu(x[:, :intermediate_size])
    # 3. Apply second linear transformation with w2[0] -> F.linear(x, w2[0])
    # 4. Use the topk_weights for each sample and add the weighted outputs of experts 1 and 2

    expected_hidden_states = torch.tensor([[4.6763, -7.3797, 6.0280], [7.1232, 0.6220, 6.1364]])

    # Step 3: Running the fused_moe function
    final_output = fused_moe(
        hidden_states, w1, w2, gating_output, topk, global_num_experts, expert_map, renormalize, use_grouped_topk
    )

    # Step 4: Check the shape and values (this should match the expected result you computed manually)
    assert (
        final_output.shape == hidden_states.shape
    ), f"Expected shape {hidden_states.shape}, but got {final_output.shape}"

    assert torch.allclose(final_output, expected_hidden_states, atol=1e-4), "Output does not match expected result"
