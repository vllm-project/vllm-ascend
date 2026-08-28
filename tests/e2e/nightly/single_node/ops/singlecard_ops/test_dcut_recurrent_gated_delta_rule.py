import torch
import torch_npu


torch_npu.npu.set_compile_mode(jit_compile=False)


def _dcut_recurrent_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_float = query.float() * scale
    key_float = key.float()
    value_float = value.float()
    beta_float = beta.float()
    gate_float = g.float().exp()
    state_float = state.float().clone()
    output = torch.zeros_like(value_float)

    num_value_heads = value.shape[1]
    num_key_heads = key.shape[1]
    value_heads_per_key_head = num_value_heads // num_key_heads
    for request_index in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request_index])
        end = int(query_start_loc[request_index + 1])
        accepted_index = int(num_accepted_tokens[request_index]) - 1
        initial_state_index = int(ssm_state_indices[request_index, accepted_index])

        for value_head in range(num_value_heads):
            key_head = value_head // value_heads_per_key_head
            recurrent_state = state_float[initial_state_index, value_head].clone()
            for local_index, token_index in enumerate(range(start, end)):
                recurrent_state *= gate_float[token_index, value_head]
                projected_value = torch.mv(recurrent_state, key_float[token_index, key_head])
                delta = (value_float[token_index, value_head] - projected_value) * beta_float[
                    token_index, value_head
                ]
                recurrent_state += torch.outer(delta, key_float[token_index, key_head])
                output[token_index, value_head] = torch.mv(
                    recurrent_state,
                    query_float[token_index, key_head],
                )
                output_state_index = int(ssm_state_indices[request_index, local_index])
                state_float[output_state_index, value_head] = recurrent_state

    return output.to(value.dtype), state_float.to(state.dtype)


def test_dcut_recurrent_uses_fixed_state_rows_and_zero_padding() -> None:
    torch.manual_seed(42)
    dtype = torch.bfloat16
    num_tokens = 4
    num_key_heads = 2
    num_value_heads = 4
    head_dim = 64

    query = torch.nn.functional.normalize(
        torch.rand(num_tokens, num_key_heads, head_dim),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand(num_tokens, num_key_heads, head_dim),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand(num_tokens, num_value_heads, head_dim).to(dtype)
    beta = torch.rand(num_tokens, num_value_heads).to(dtype)
    g = torch.rand(num_tokens, num_value_heads, dtype=torch.float32)

    # Only the first three token rows are active. The fourth row exercises the
    # graph-safe zero-padded output path.
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    ssm_state_indices = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dtype=torch.int32,
    )
    # Select state from the previous verifier step, independently of this
    # step's [2, 1] segment lengths.
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32)
    state = torch.rand(
        8,
        num_value_heads,
        head_dim,
        head_dim,
        dtype=dtype,
    )
    scale = head_dim**-0.5

    expected_output, expected_state = _dcut_recurrent_golden(
        query,
        key,
        value,
        state,
        beta,
        scale,
        query_start_loc,
        ssm_state_indices,
        num_accepted_tokens,
        g,
    )

    state_npu = state.npu()
    output_npu = torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
        query.npu(),
        key.npu(),
        value.npu(),
        state_npu,
        beta=beta.npu(),
        scale=scale,
        query_start_loc=query_start_loc.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
        g=g.npu(),
        zero_padded_output=True,
    )

    torch.testing.assert_close(output_npu.cpu(), expected_output, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(state_npu.cpu(), expected_state, rtol=3e-3, atol=1e-2)
    assert torch.count_nonzero(output_npu[-1]).item() == 0
