import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.utils import enable_custom_op, is_310p

torch_npu.npu.set_compile_mode(jit_compile=False)


pytestmark = [
    pytest.mark.skipif(not hasattr(torch, "npu"), reason="torch.npu is required"),
    pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device is required"),
    pytest.mark.skipif(not is_310p(), reason="310P device is required"),
]


def _to_int64_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None:
        return None
    return t.to(torch.int64).contiguous()


def _run_decode_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    initial_state_mode: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_causal_conv1d_310(
        x,
        weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=None,
        cache_indices=_to_int64_tensor(cache_indices),
        initial_state_mode=_to_int64_tensor(initial_state_mode),
        num_accepted_tokens=None,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    )


def _build_inputs(
    *,
    batch_size: int = 4,
    dim: int = 2048,
    width: int = 4,
    seqlen: int = 1,
    total_entries: int = 40,
):
    # The 310P op expects:
    #   x: (batch, seqlen, dim) or (cu_seqlen, dim)
    #   weight: (width, dim)
    #   conv_states: (cache_lines, state_len, dim)
    x = torch.randn(batch_size, seqlen, dim, device="npu", dtype=torch.float16).contiguous()
    weight = torch.randn(width, dim, device="npu", dtype=torch.float16).contiguous()
    bias = torch.randn(dim, device="npu", dtype=torch.float16)
    conv_states = torch.randn(total_entries, width, dim, device="npu", dtype=torch.float16).contiguous()
    cache_indices = torch.randperm(total_entries, device="npu", dtype=torch.int32)[:batch_size]
    initial_state_mode = torch.ones(batch_size, device="npu", dtype=torch.int64)
    return x, weight, bias, conv_states, cache_indices, initial_state_mode


def test_causal_conv1d_310_decode_aclgraph_smoke():
    enable_custom_op()

    x, weight, bias, conv_states, cache_indices, initial_state_mode = _build_inputs()
    x_alt, _, _, conv_states_alt, _, _ = _build_inputs()

    eager_conv_states = conv_states.clone()
    eager_out = _run_decode_op(
        x.clone(),
        weight,
        bias,
        eager_conv_states,
        cache_indices,
        initial_state_mode,
    )
    eager_conv_states_expected = eager_conv_states.clone()

    eager_conv_states_alt = conv_states_alt.clone()
    eager_out_alt = _run_decode_op(
        x_alt.clone(),
        weight,
        bias,
        eager_conv_states_alt,
        cache_indices,
        initial_state_mode,
    )
    eager_conv_states_alt_expected = eager_conv_states_alt.clone()

    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        graph_out = _run_decode_op(
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            initial_state_mode,
        )
    torch.npu.synchronize()

    torch.testing.assert_close(graph_out, eager_out, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(conv_states, eager_conv_states_expected, rtol=3e-3, atol=1e-2)

    x.copy_(x_alt)
    conv_states.copy_(conv_states_alt)
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(graph_out, eager_out_alt, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(conv_states, eager_conv_states_alt_expected, rtol=3e-3, atol=1e-2)
