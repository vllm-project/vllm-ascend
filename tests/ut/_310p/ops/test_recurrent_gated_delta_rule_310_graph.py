import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import enable_custom_op, is_310p

torch_npu.npu.set_compile_mode(jit_compile=False)


pytestmark = [
    pytest.mark.skipif(not hasattr(torch, "npu"), reason="torch.npu is required"),
    pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device is required"),
    pytest.mark.skipif(not is_310p(), reason="310P device is required"),
]


def _run_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=query,
        key=key,
        value=value,
        g=None,
        gk=None,
        beta=beta,
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        scale_value=query.shape[-1] ** -0.5,
    )


def _build_inputs(
    *,
    active_tokens: int = 3,
    padded_tokens: int = 5,
    num_states: int = 12,
    nk: int = 1,
    nv: int = 1,
    dk: int = 128,
    dv: int = 128,
):
    query = torch.randn(padded_tokens, nk, dk, device="npu", dtype=torch.float16).contiguous()
    key = torch.randn_like(query)
    value = torch.randn(padded_tokens, nv, dv, device="npu", dtype=torch.float16).contiguous()
    beta = torch.randn(padded_tokens, nv, device="npu", dtype=torch.float16).contiguous()
    state = torch.randn(num_states, nv, dv, dk, device="npu", dtype=torch.float32).contiguous()
    actual_seq_lengths = torch.ones(active_tokens, device="npu", dtype=torch.int32)
    ssm_state_indices = torch.tensor(
        [2, 5, 7] + [-1] * (padded_tokens - active_tokens),
        device="npu",
        dtype=torch.int32,
    ).contiguous()
    return query, key, value, beta, state, actual_seq_lengths, ssm_state_indices


def test_recurrent_gated_delta_rule_310_aclgraph_preserves_padded_state_rows():
    enable_custom_op()

    (
        query,
        key,
        value,
        beta,
        state,
        actual_seq_lengths,
        ssm_state_indices,
    ) = _build_inputs()
    query_initial = query.clone()
    key_initial = key.clone()
    value_initial = value.clone()
    beta_initial = beta.clone()
    state_initial = state.clone()

    (
        query_alt,
        key_alt,
        value_alt,
        beta_alt,
        state_alt,
        _,
        _,
    ) = _build_inputs()

    eager_state = state.clone()
    eager_out = _run_op(
        query.clone(),
        key.clone(),
        value.clone(),
        beta.clone(),
        eager_state,
        actual_seq_lengths,
        ssm_state_indices,
    )
    eager_state_expected = eager_state.clone()

    eager_state_alt = state_alt.clone()
    eager_out_alt = _run_op(
        query_alt.clone(),
        key_alt.clone(),
        value_alt.clone(),
        beta_alt.clone(),
        eager_state_alt,
        actual_seq_lengths,
        ssm_state_indices,
    )
    eager_state_alt_expected = eager_state_alt.clone()

    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        graph_out = _run_op(
            query,
            key,
            value,
            beta,
            state,
            actual_seq_lengths,
            ssm_state_indices,
        )
    torch.npu.synchronize()

    query.copy_(query_initial)
    key.copy_(key_initial)
    value.copy_(value_initial)
    beta.copy_(beta_initial)
    state.copy_(state_initial)
    graph.replay()
    torch.npu.synchronize()

    active_tokens = actual_seq_lengths.size(0)
    torch.testing.assert_close(
        graph_out[:active_tokens], eager_out[:active_tokens], rtol=3e-3, atol=1e-2
    )
    torch.testing.assert_close(state, eager_state_expected, rtol=3e-3, atol=1e-2)

    query.copy_(query_alt)
    key.copy_(key_alt)
    value.copy_(value_alt)
    beta.copy_(beta_alt)
    state.copy_(state_alt)
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(
        graph_out[:active_tokens], eager_out_alt[:active_tokens], rtol=3e-3, atol=1e-2
    )
    torch.testing.assert_close(state, eager_state_alt_expected, rtol=3e-3, atol=1e-2)
