# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def cpu_mix_reference(prefix, blocks, projection, gamma, epsilon):
    values = torch.cat((blocks.cpu(), prefix.cpu().unsqueeze(1)), dim=1).float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    logits = (normalized * gamma.cpu().float() * projection.cpu().float()).sum(-1)
    return (logits.softmax(-1).unsqueeze(-1) * values).sum(1).bfloat16().to(prefix.device)


def assert_mix_close(actual, expected, max_abs_error=1):
    error = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert (error <= (1 + expected.float().abs()) / 64).float().mean() >= 0.99
    assert error.max() <= max_abs_error


@pytest.mark.parametrize(
    "num_tokens,num_blocks,hidden_size",
    [
        (1, 1, 128),
        (7, 3, 256),
        (32, 8, 4096),
        (3, 64, 4096),
        (2, 1, 7168),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@pytest.mark.parametrize("layout", ["contiguous", "block_slice"])
@pytest.mark.parametrize("optimize_prefill", [False, True])
def test_attn_res_fused_without_add(num_tokens, num_blocks, hidden_size, epsilon, layout, optimize_prefill):
    generator = torch.Generator().manual_seed(42)
    prefix = torch.randn(num_tokens, hidden_size, generator=generator, dtype=torch.bfloat16)
    blocks = torch.randn(num_tokens, num_blocks, hidden_size, generator=generator, dtype=torch.bfloat16)
    projection = (torch.randn(1, hidden_size, generator=generator) / hidden_size**0.5).bfloat16()
    gamma = torch.randn(hidden_size, generator=generator, dtype=torch.bfloat16)

    inputs = [tensor.npu() for tensor in (prefix, blocks, projection, gamma)]
    if layout == "block_slice":
        storage = inputs[1].new_empty(num_tokens, num_blocks + 2, hidden_size)
        inputs[1] = storage[:, 1 : num_blocks + 1, :]
        inputs[1].copy_(blocks)
    actual, raw, _ = torch.ops._C_ascend.attn_res_fwd.fused(
        inputs[0], None, inputs[1], inputs[2], inputs[3], epsilon, num_blocks, optimize_prefill=optimize_prefill
    )
    expected = cpu_mix_reference(inputs[0], inputs[1], inputs[2], inputs[3], epsilon)

    assert actual.shape == prefix.shape
    assert actual.dtype == prefix.dtype
    torch.testing.assert_close(raw, inputs[0], rtol=0, atol=0)
    assert_mix_close(actual, expected)


@pytest.mark.parametrize("tokens,valid,hidden", [(1, 0, 32), (1, 1, 7168), (4, 7, 7168), (8, 8, 7168)])
@pytest.mark.parametrize("with_add,with_norm", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("bank_slice", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("optimize_prefill", [False, True])
@torch.inference_mode()
def test_attn_res_fused_chain_preserves_prefix_materialized_bank_and_graph(
    tokens, valid, hidden, with_add, with_norm, bank_slice, graph, optimize_prefill
):
    import torch_npu

    torch.manual_seed(38)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix) if with_add else None
    bank_storage = torch.randn(tokens, 10 if bank_slice else 8, hidden, device="npu", dtype=torch.bfloat16)
    bank = bank_storage[:, 1:9] if bank_slice else bank_storage
    bank[:, valid:] = float("nan")
    projection = (torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16()
    gamma = torch.randn(hidden, device="npu", dtype=torch.bfloat16)
    output_gamma = torch.randn_like(gamma) if with_norm else None
    write_idx = valid if valid < 8 else -1

    def fused():
        return torch.ops._C_ascend.attn_res_fwd.fused(
            prefix,
            addend,
            bank,
            projection,
            gamma,
            1e-5,
            valid,
            output_gamma,
            1e-5,
            write_idx,
            True,
            optimize_prefill=optimize_prefill,
        )

    # Eager warmup before capture resolves ACLNN executor resources.
    fused()
    torch.npu.synchronize()
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            results = fused()

    for replay in range(3):
        if replay:
            prefix.copy_(torch.randn_like(prefix))
            if addend is not None:
                addend.copy_(torch.randn_like(addend))
        before = prefix.clone()
        before_addend = addend.clone() if addend is not None else None
        expected_prefix = prefix + addend if with_add else prefix.clone()
        expected_mix = (
            cpu_mix_reference(expected_prefix, bank[:, :valid], projection, gamma, 1e-5) if valid else expected_prefix
        )
        expected_output = torch_npu.npu_rms_norm(expected_mix, output_gamma, 1e-5)[0] if with_norm else expected_mix
        if graph:
            captured.replay()
        else:
            results = fused()
        output, raw_prefix, materialized = results
        # These two BF16 observations must remain bitwise equivalent.
        torch.testing.assert_close(raw_prefix, expected_prefix, rtol=0, atol=0)
        assert_mix_close(materialized, expected_mix)
        # CANN RMSNorm may use a different floating-point reduction tree.
        torch.testing.assert_close(output, expected_output, rtol=1e-2, atol=1e-2)
        if write_idx >= 0:
            torch.testing.assert_close(bank[:, write_idx], expected_prefix, rtol=0, atol=0)
        torch.testing.assert_close(prefix, before, rtol=0, atol=0)
        if with_add:
            torch.testing.assert_close(addend, before_addend, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 4, 8])
@torch.inference_mode()
def test_attn_res_fused_pp_materialization(tokens):
    prefix = torch.randn(tokens, 7168, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix)
    bank = torch.empty(tokens, 8, 7168, device="npu", dtype=torch.bfloat16)
    projection = torch.empty(1, 7168, device="npu", dtype=torch.bfloat16)
    gamma = torch.empty(7168, device="npu", dtype=torch.bfloat16)
    output, raw, _ = torch.ops._C_ascend.attn_res_fwd.fused(prefix, addend, bank, projection, gamma, 1e-5, 0, mix=False)
    torch.testing.assert_close(output, prefix + addend, rtol=0, atol=0)
    torch.testing.assert_close(raw, prefix + addend, rtol=0, atol=0)


@pytest.mark.parametrize("optimize_prefill", [False, True])
@pytest.mark.parametrize(
    "invalid",
    ["valid", "write_slot", "addend", "projection", "output_norm_eps"],
)
def test_attn_res_fused_rejects_invalid_arguments(optimize_prefill, invalid):
    prefix = torch.empty(2, 128, device="npu", dtype=torch.bfloat16)
    addend = torch.empty_like(prefix)
    bank = torch.empty(2, 3, 128, device="npu", dtype=torch.bfloat16)
    projection = torch.empty(1, 128, device="npu", dtype=torch.bfloat16)
    norm = torch.empty(128, device="npu", dtype=torch.bfloat16)
    valid, write_idx, output_eps = 2, -1, 1e-5
    if invalid == "valid":
        valid = 4
    elif invalid == "write_slot":
        write_idx = 1
    elif invalid == "addend":
        addend = addend[:, :64]
    elif invalid == "projection":
        projection = projection[:, :64]
    else:
        output_eps = 0.0

    with pytest.raises(RuntimeError, match="attn_res_fwd.fused: invalid"):
        torch.ops._C_ascend.attn_res_fwd.fused(
            prefix,
            addend,
            bank,
            projection,
            norm,
            1e-5,
            valid,
            norm,
            output_eps,
            write_idx,
            optimize_prefill=optimize_prefill,
        )


@pytest.mark.parametrize("valid", [0, 1, 2, 4, 8])
@torch.inference_mode()
def test_attn_res_prefill_cached_norm_is_reloaded_on_graph_replay(valid):
    """More than 64 tokens exercises per-core reuse; UB must not outlive a call."""
    import torch_npu

    tokens, hidden = 129, 7168
    torch.manual_seed(9219 + valid)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix)
    storage = torch.randn(tokens, 10, hidden, device="npu", dtype=torch.bfloat16)
    bank = storage[:, 1:9]
    bank[:, valid:] = float("nan")
    proj = (torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16()
    gamma = torch.randn(hidden, device="npu", dtype=torch.bfloat16)
    output_gamma = torch.randn_like(gamma)

    def call():
        return torch.ops._C_ascend.attn_res_fwd.fused(
            prefix,
            addend,
            bank,
            proj,
            gamma,
            1e-5,
            valid,
            output_gamma,
            1e-5,
            -1,
            True,
            optimize_prefill=True,
        )

    call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, raw, materialized = call()
    for replay in range(3):
        if replay:
            output_gamma.copy_(torch.randn_like(output_gamma))
            prefix.copy_(torch.randn_like(prefix))
            addend.copy_(torch.randn_like(addend))
        expected_raw = prefix + addend
        expected_mix = cpu_mix_reference(expected_raw, bank[:, :valid], proj, gamma, 1e-5) if valid else expected_raw
        expected = torch_npu.npu_rms_norm(expected_mix, output_gamma, 1e-5)[0]
        graph.replay()
        torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
        assert_mix_close(materialized, expected_mix)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("valid,hidden", [(n, 7168) for n in range(9)] + [(63, 128), (64, 128)])
@torch.inference_mode()
def test_attn_res_fused_register_reductions_changed_inputs(valid, hidden):
    """Check the small-bank path and its register boundary against the plain op."""
    import torch_npu

    torch.manual_seed(921)
    prefix = torch.randn(8, hidden, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix)
    bank = torch.randn(8, valid + 1, hidden, device="npu", dtype=torch.bfloat16)
    # The next bank slot must never contribute to the reduction.
    bank[:, valid:] = float("nan")
    projection = (torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16()
    gamma = torch.randn(hidden, device="npu", dtype=torch.bfloat16)
    output_gamma = torch.randn_like(gamma)

    def call():
        return torch.ops._C_ascend.attn_res_fwd.fused(
            prefix,
            addend,
            bank,
            projection,
            gamma,
            1e-6,
            valid,
            output_gamma,
            1e-6,
            -1,
            True,
            optimize_prefill=True,
        )

    call()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, raw, materialized = call()
    for scale in (1.0, 0.0, 1000.0, 0.001):
        prefix.copy_(torch.randn_like(prefix) * scale)
        addend.copy_(torch.randn_like(addend) * scale)
        rounded = prefix + addend
        expected = cpu_mix_reference(rounded, bank[:, :valid], projection, gamma, 1e-6) if valid else rounded
        expected_norm = torch_npu.npu_rms_norm(expected, output_gamma, 1e-6)[0]
        graph.replay()
        torch.testing.assert_close(raw, rounded, rtol=0, atol=0)
        # BF16 spacing grows with magnitude in the 1000x stress case.
        assert_mix_close(materialized, expected, max_abs_error=2 if scale == 1000.0 else 1)
        torch.testing.assert_close(output, expected_norm, rtol=1e-2, atol=1e-2)
        assert torch.isnan(bank[:, valid:]).all()
