# SPDX-License-Identifier: Apache-2.0
"""A2/A3/A5 whole-row and H-split paths, plus bank mutation coverage."""

import pytest
import torch
import torch_npu

COOPERATIVE_SOC_VERSIONS = frozenset(range(220, 226)) | frozenset(range(250, 257)) | {260}


@pytest.fixture(scope="module", autouse=True)
def require_cooperative_custom_operator():
    if not torch.npu.is_available() or torch_npu.npu.get_soc_version() not in COOPERATIVE_SOC_VERSIONS:
        pytest.skip("Cooperative AttnResFwd requires an A2/A3/A5 build and device")
    if not hasattr(torch.ops._C_ascend, "attn_res_fwd"):
        from vllm_ascend.device.device_config import check_ascend_device_type
        from vllm_ascend.utils import bootstrap_custom_op_env

        check_ascend_device_type()
        bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "attn_res_fwd")


@pytest.mark.parametrize(
    "tokens,valid,hidden,with_add,with_norm,save_materialized,padded_bank,write_bank",
    [
        (1, 8, 7168, True, True, False, False, True),
        (4, 8, 7168, True, True, True, True, True),
        (4, 4, 7168, False, True, False, True, True),
        (1, 1, 7168, True, False, False, False, False),
        (1, 16, 7168, False, True, True, False, False),
        (8, 8, 7168, True, True, True, True, True),
        (4, 8, 1537, True, True, False, True, True),
        (2, 2, 7169, True, False, True, True, True),
        (4, 8, 1025, False, True, True, True, False),
        (4, 8, 7168, True, False, False, False, False),
        (10, 4, 7168, False, True, False, True, True),
        (10, 7, 7168, True, True, True, True, True),
        (4, 1, 16384, True, True, True, True, True),
        (1, 64, 1024, True, True, True, False, True),
        (1, 8, 512, False, False, True, False, False),
        (1, 2, 7169, True, True, True, True, True),
        (1, 3, 8193, False, True, True, True, True),
        (1, 8, 32768, True, True, True, True, True),
        (4, 8, 32768, True, True, True, True, True),
        (4, 8, 32769, False, False, False, True, True),
        (1, 1, 7168, True, True, True, True, False),
        (16, 4, 1537, True, True, True, True, True),
        (16, 64, 1537, False, True, True, True, True),
        (1, 2048, 1, True, False, True, False, True),
        (1, 2048, 32, True, False, True, False, True),
        (1, 8, 1538, True, True, True, True, True),
        (1, 63, 1024, True, True, True, True, True),
        (1, 127, 1024, True, True, True, True, True),
        (1, 2039, 1, True, False, True, False, True),
        (1, 2040, 1, True, False, True, False, True),
        (1, 2041, 32, True, False, True, False, True),
    ],
)
@torch.inference_mode()
def test_attn_res_fwd_cooperative(
    tokens, valid, hidden, with_add, with_norm, save_materialized, padded_bank, write_bank, record_property
):
    # Cover score-only workers, multiple tasks per core, odd H, and full-row
    # output ownership. Every launched core joins the one readiness barrier.
    torch.manual_seed(42 + tokens + valid)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix) if with_add else None
    capacity = max(8, valid + 1)
    storage = torch.randn(tokens, capacity + (2 if padded_bank else 0), hidden, device="npu", dtype=torch.bfloat16)
    bank = storage[:, :capacity]
    bank[:, valid:] = float("nan")
    projection = (torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16()
    gamma = torch.randn(hidden, device="npu", dtype=torch.bfloat16)
    output_gamma = torch.randn_like(gamma) if with_norm else None
    prefix_before = prefix.cpu().clone()
    addend_before = addend.cpu().clone() if with_add else None
    bank_before = storage.cpu().clone()
    norm_eps, output_eps = 1e-5, 1e-4

    # Independent CPU reference, including both BF16 rounding boundaries and
    # distinct epsilons for residual scoring and output normalization.
    raw_ref = (prefix_before + addend_before) if with_add else prefix_before
    values = torch.cat((bank_before[:, :valid], raw_ref[:, None]), dim=1).float()
    keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + norm_eps)
    logits = (keys * gamma.cpu().float() * projection.cpu().float()).sum(-1)
    mixture = (logits.softmax(-1)[..., None] * values).sum(1).bfloat16()
    output_ref = mixture
    if with_norm:
        x = mixture.float()
        output_ref = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + output_eps) * output_gamma.cpu().float()
        ).bfloat16()

    write_idx = valid if write_bank else -1
    output, raw, materialized = torch.ops._C_ascend.attn_res_fwd(
        prefix,
        addend,
        bank,
        projection,
        gamma,
        norm_eps,
        valid,
        output_gamma,
        output_eps,
        write_idx,
        save_materialized,
    )
    torch_npu.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), raw_ref, rtol=0, atol=0)
    comparisons = [("output", output, output_ref)]
    if save_materialized:
        comparisons.append(("materialized", materialized, mixture))
    for name, actual, expected in comparisons:
        error = (actual.float().cpu() - expected.float()).abs()
        matched_ratio = (error <= (1 + expected.float().abs()) / 64).float().mean().item()
        max_abs_error = error.max().item()
        record_property(f"{name}_matched_ratio", matched_ratio)
        record_property(f"{name}_max_abs_error", max_abs_error)
        assert torch.isfinite(actual).all()
        assert matched_ratio >= 0.99
        assert max_abs_error <= 1
    if not save_materialized:
        assert materialized.data_ptr() == output.data_ptr()
    if not with_add:
        assert raw.data_ptr() == prefix.data_ptr()
    if write_bank:
        bank_before[:, write_idx] = raw_ref
    torch.testing.assert_close(storage.cpu(), bank_before, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(prefix.cpu(), prefix_before, rtol=0, atol=0)
    if with_add:
        torch.testing.assert_close(addend.cpu(), addend_before, rtol=0, atol=0)


@pytest.mark.parametrize("aliased_input", ["prefix", "addend", "projection", "norm", "output_norm"])
@torch.inference_mode()
def test_attn_res_fwd_rejects_writable_bank_aliases(aliased_input):
    torch.manual_seed(7)
    hidden, valid = 512, 8
    bank = torch.randn(1, valid + 1, hidden, device="npu", dtype=torch.bfloat16)
    write_row = bank[:, valid]
    inputs = {
        "prefix": torch.randn_like(write_row),
        "addend": torch.randn_like(write_row),
        "projection": torch.randn_like(write_row),
        "norm": torch.randn(hidden, device="npu", dtype=torch.bfloat16),
        "output_norm": torch.randn(hidden, device="npu", dtype=torch.bfloat16),
    }
    inputs[aliased_input] = write_row[0] if aliased_input in ("norm", "output_norm") else write_row
    before = bank.clone()
    with pytest.raises(RuntimeError, match="writable bank must not alias"):
        torch.ops._C_ascend.attn_res_fwd(
            inputs["prefix"],
            inputs["addend"],
            bank,
            inputs["projection"],
            inputs["norm"],
            1e-5,
            valid,
            inputs["output_norm"],
            1e-4,
            valid,
            False,
        )
    torch.testing.assert_close(bank, before, rtol=0, atol=0)
    # Sharing storage is safe when there is no bank write; this must remain
    # accepted rather than rejecting every read-only alias.
    output, _, _ = torch.ops._C_ascend.attn_res_fwd(
        inputs["prefix"],
        inputs["addend"],
        bank,
        inputs["projection"],
        inputs["norm"],
        1e-5,
        valid,
        inputs["output_norm"],
        1e-4,
        -1,
        False,
    )
    torch_npu.npu.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(bank, before, rtol=0, atol=0)


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), -1.0, 0.0, 1e300, 1e-300])
@pytest.mark.parametrize("normalize_output", [False, True])
@torch.inference_mode()
def test_attn_res_fwd_rejects_invalid_norm_epsilon(epsilon, normalize_output):
    prefix = torch.ones(1, 512, device="npu", dtype=torch.bfloat16)
    bank = torch.ones(1, 2, 512, device="npu", dtype=torch.bfloat16)
    gamma = torch.ones(512, device="npu", dtype=torch.bfloat16)
    message = "invalid output norm" if normalize_output else "norm_eps must be finite"
    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.attn_res_fwd(
            prefix,
            None,
            bank,
            prefix,
            gamma,
            1e-5 if normalize_output else epsilon,
            2,
            gamma if normalize_output else None,
            epsilon if normalize_output else 1e-5,
        )


@pytest.mark.parametrize("tokens", [1, 16])
@torch.inference_mode()
def test_attn_res_fwd_softmax_extreme_negative_scores(tokens):
    # R=65 exercises the padded half-interval maximum reduction in both
    # cooperative and token-local paths. Every finite logit is below -1e20.
    hidden, valid = 512, 64
    prefix = torch.ones(tokens, hidden, device="npu", dtype=torch.bfloat16)
    bank = torch.ones(tokens, valid, hidden, device="npu", dtype=torch.bfloat16)
    projection = torch.full((1, hidden), -1e25, device="npu", dtype=torch.bfloat16)
    gamma = torch.ones(hidden, device="npu", dtype=torch.bfloat16)
    output, _, _ = torch.ops._C_ascend.attn_res_fwd(prefix, None, bank, projection, gamma, 1e-5, valid)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(output, prefix, rtol=0, atol=0)


@torch.inference_mode()
def test_attn_res_fwd_rejects_oversized_empty_bank():
    # A zero-token view needs no backing allocation, but capacity * H would
    # overflow int64 before the native validator in the previous adapter.
    hidden = 7168
    capacity = torch.iinfo(torch.int64).max // hidden + 1
    prefix = torch.empty(0, hidden, device="npu", dtype=torch.bfloat16)
    bank = torch.empty_strided((0, capacity, hidden), (0, hidden, 1), device="npu", dtype=torch.bfloat16)
    projection = torch.ones(1, hidden, device="npu", dtype=torch.bfloat16)
    gamma = torch.ones(hidden, device="npu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="invalid bank extent/write slot"):
        torch.ops._C_ascend.attn_res_fwd(prefix, None, bank, projection, gamma, 1e-5, 0)


@pytest.mark.parametrize("with_add", [False, True])
@torch.inference_mode()
def test_attn_res_fwd_accepts_empty_batch(with_add):
    hidden, valid = 512, 2
    prefix = torch.empty(0, hidden, device="npu", dtype=torch.bfloat16)
    bank = torch.empty(0, valid, hidden, device="npu", dtype=torch.bfloat16)
    projection = torch.ones(1, hidden, device="npu", dtype=torch.bfloat16)
    gamma = torch.ones(hidden, device="npu", dtype=torch.bfloat16)
    outputs = torch.ops._C_ascend.attn_res_fwd(
        prefix, torch.empty_like(prefix) if with_add else None, bank, projection, gamma, 1e-5, valid, gamma, 1e-4
    )
    for output in outputs:
        assert output.shape == prefix.shape and output.numel() == 0


@pytest.mark.parametrize("tokens", [0, 1, 4, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
@pytest.mark.parametrize("argument", ["all", "prefix", "bank", "projection", "norm", "addend", "output_norm"])
@torch.inference_mode()
def test_attn_res_fwd_rejects_unsupported_dtype(tokens, dtype, argument):
    hidden, valid = 512, 2
    inputs = {
        "prefix": torch.ones(tokens, hidden, device="npu", dtype=torch.bfloat16),
        "bank": torch.ones(tokens, valid + 1, hidden, device="npu", dtype=torch.bfloat16),
        "projection": torch.ones(1, hidden, device="npu", dtype=torch.bfloat16),
        "norm": torch.ones(hidden, device="npu", dtype=torch.bfloat16),
        "addend": torch.ones(tokens, hidden, device="npu", dtype=torch.bfloat16),
        "output_norm": torch.ones(hidden, device="npu", dtype=torch.bfloat16),
    }
    for name in inputs:
        if argument in ("all", name):
            inputs[name] = inputs[name].to(dtype)
    bank_before = inputs["bank"].clone()
    message = {
        "addend": "invalid addend",
        "output_norm": "invalid output norm",
    }.get(argument, "inputs must be BF16")
    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.attn_res_fwd(
            inputs["prefix"],
            inputs["addend"],
            inputs["bank"],
            inputs["projection"],
            inputs["norm"],
            1e-5,
            valid,
            inputs["output_norm"],
            1e-4,
            valid,
            True,
        )
    torch_npu.npu.synchronize()
    torch.testing.assert_close(inputs["bank"], bank_before, rtol=0, atol=0)
