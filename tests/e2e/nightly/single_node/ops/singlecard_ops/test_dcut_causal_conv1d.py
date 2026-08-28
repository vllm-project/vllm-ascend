import torch
import torch_npu


torch_npu.npu.set_compile_mode(jit_compile=False)


def test_dcut_causal_conv1d_matches_update_mode() -> None:
    """D-Cut keeps the stock update-mode math and state mutation."""
    torch.manual_seed(42)
    device = "npu"
    dtype = torch.bfloat16

    dim = 64
    width = 4
    state_len = 8
    num_cache_slots = 4
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    # These counts describe the previous verifier step and deliberately exceed
    # the current segment length for the second request.
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32, device=device)

    x = torch.randn(3, dim, dtype=dtype, device=device)
    weight = torch.randn(width, dim, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)
    initial_conv_state = torch.randn(
        num_cache_slots,
        state_len,
        dim,
        dtype=dtype,
        device=device,
    )
    reference_conv_state = initial_conv_state.clone()
    dcut_conv_state = initial_conv_state.clone()
    reference_output = torch.empty_like(x)
    dcut_output = torch.empty_like(x)

    torch.ops._C_ascend.npu_causal_conv1d_custom(
        reference_output,
        x,
        weight,
        conv_state=reference_conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    torch.ops._C_ascend.npu_dcut_causal_conv1d(
        dcut_output,
        x,
        weight,
        conv_state=dcut_conv_state,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=-1,
    )

    torch.testing.assert_close(dcut_output, reference_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dcut_conv_state, reference_conv_state, rtol=1e-2, atol=1e-2)
