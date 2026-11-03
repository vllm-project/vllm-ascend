# SPDX-License-Identifier: Apache-2.0
"""Previous acceptance selects an initial state independently of query length."""

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import recurrent_kda


@pytest.mark.parametrize("accepted", [1, 3, 5, 6])
@pytest.mark.parametrize("query_len", [1, 2, 6])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_shortened_kda_query_reads_previous_accepted_state(accepted, query_len, state_dtype):
    device = torch.device("npu:0")
    q = torch.zeros((1, query_len, 1, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    gate = torch.zeros_like(q)
    beta = torch.zeros((1, query_len, 1), device=device)
    state = torch.arange(8, device=device, dtype=torch.float32).view(8, 1, 1, 1)
    state = state.expand(8, 1, 128, 128).to(state_dtype).contiguous()
    indices = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32, device=device)

    output, _ = recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        state,
        cu_seqlens=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        ssm_state_indices=indices,
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device=device),
        scale=1.0,
        state_v_first=True,
        inplace_final_state=True,
    )

    # Zero key and gate preserve the selected state at every output slot.
    torch.testing.assert_close(output, torch.full_like(output, accepted), rtol=0, atol=0)
    torch.testing.assert_close(state[1 : query_len + 1], torch.full_like(state[1 : query_len + 1], accepted))


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kda_graph_replay_changes_query_length_and_previous_acceptance(state_dtype):
    device = torch.device("npu:0")
    q = torch.zeros((1, 6, 1, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    k, v, gate = (torch.zeros_like(q) for _ in range(3))
    beta = torch.zeros((1, 6, 1), device=device)
    initial = torch.arange(8, device=device, dtype=torch.float32).view(8, 1, 1, 1)
    initial = initial.expand(8, 1, 128, 128).to(state_dtype).contiguous()
    state = initial.clone()
    starts = torch.tensor([0, 6], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    indices = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32, device=device)

    def run():
        return recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            state,
            cu_seqlens=starts,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            scale=1.0,
            state_v_first=True,
            inplace_final_state=True,
        )[0]

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for query_len, previous_accepted in [(1, 3), (2, 5), (6, 1), (1, 6)]:
        state.copy_(initial)
        starts[1] = query_len
        accepted.fill_(previous_accepted)
        output.fill_(-123)
        graph.replay()
        torch.npu.synchronize()
        active = output[:, :query_len]
        torch.testing.assert_close(active, torch.full_like(active, previous_accepted), rtol=0, atol=0)
        written = state[1 : query_len + 1]
        torch.testing.assert_close(written, torch.full_like(written, previous_accepted), rtol=0, atol=0)
