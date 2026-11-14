# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.fixture(scope="module", autouse=True)
def _enable_nz_format():
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    yield
    torch.npu.config.allow_internal_format = previous == b"enable"


def _full(shape, value, dtype=torch.bfloat16):
    return torch.full(shape, value, dtype=dtype, device="npu")


def _run_mlapo(tokens, q_rank=1536, nope=128, variant="qdown", epsilons=None, through_sfa=False):
    hidden, kv_rank, rope, heads = 7168, 512, 64, 2
    width = q_rank + kv_rank + rope
    dtype = torch.float16 if variant == "fp16" else torch.bfloat16
    unquantized = variant == "no_quant"
    weight_dtype = dtype if unquantized else torch.int8
    pack = 16 if unquantized else 32
    input_value = 0.001 / hidden if unquantized else 1.0
    x = _full((tokens, hidden), input_value, dtype)
    wd = torch_npu.npu_format_cast(_full((1, hidden // pack, width, pack), int(unquantized), weight_dtype), 29)
    wu = torch_npu.npu_format_cast(_full((1, q_rank // pack, heads * (nope + rope), pack), 0, weight_dtype), 29)
    pattern = torch.arange(width, dtype=torch.int32) % 4 + 1
    scale = _full((width,), 0.001, torch.float32)
    gamma_q = torch.linspace(0.75, 1.25, q_rank).to(dtype)
    gamma_kv = torch.linspace(0.5, 1.5, kv_rank).to(dtype)
    cache = _full((1, 128, kv_rank), float("nan"), dtype)
    rope_cache = _full((1, 128, rope), float("nan"), dtype)
    slots = torch.arange(tokens, dtype=torch.int32, device="npu")
    cos, sin = _full((tokens, rope), 1, dtype), _full((tokens, rope), 0, dtype)
    wuk = _full((heads, nope, kv_rank), 0, dtype)
    inner = _full((tokens, q_rank), float("nan"), dtype)
    positional = (
        x,
        wd,
        None if unquantized else scale,
        gamma_q.npu(),
        None if unquantized else _full((q_rank,), 0, dtype),
        wu,
        None if unquantized else _full((heads * (nope + rope),), 1, torch.float32),
        gamma_kv.npu(),
        cos,
        sin,
        wuk,
        cache,
        rope_cache,
        slots,
    )
    kwargs = dict(
        quant_scale0=None if unquantized else _full((1,), 1, dtype),
        quant_offset0=None if unquantized else _full((1,), 0, torch.int8),
        bias0=None if unquantized else pattern.npu(),
        quant_scale1=None if unquantized else _full((1,), 1, dtype),
        quant_offset1=None if unquantized else _full((1,), 0, torch.int8),
        bias1=None if unquantized else _full((heads * (nope + rope),), 0, torch.int32),
        ctkv_scale=_full((1,), 1, dtype),
        q_nope_scale=_full((heads,), 1, dtype),
        cache_mode="krope_ctkv",
        quant_mode="no_quant" if unquantized else "per_tensor_quant_asymm",
        enable_inner_out=variant == "qdown",
        q_out0=_full((tokens, heads, kv_rank), float("nan"), dtype),
        kv_cache_out0=cache,
        q_out1=_full((tokens, heads, rope), float("nan"), dtype),
        kv_cache_out1=rope_cache,
        inner_out=inner,
    )
    if through_sfa:
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.q_lora_rank = q_rank
        impl.q_a_layernorm = SimpleNamespace(variance_epsilon=epsilons[0])
        impl.kv_a_layernorm = SimpleNamespace(variance_epsilon=epsilons[1])
        for name, tensor in zip(
            ("wd_qkv", "deq_scale_qkv", "gamma1", "beta1", "wu_q", "qb_deq_scl", "gamma2"),
            positional[1:8],
        ):
            setattr(impl, name, tensor)
        impl.W_UK_T = wuk
        for name, key in (("quant_bias_qkv", "bias0"), ("qb_qt_bias", "bias1")):
            setattr(impl, name, kwargs[key])
        for name in ("quant_scale0", "quant_offset0", "quant_scale1", "quant_offset1", "ctkv_scale", "q_nope_scale"):
            setattr(impl, name, kwargs[name])
        _, q_out, rope_out, inner = impl._sfa_preprocess_mlapo(
            x, (cache, rope_cache), cos, sin, slots, num_input_tokens=tokens
        )
    else:
        if epsilons is not None:
            kwargs.update(q_epsilon=epsilons[0], kv_epsilon=epsilons[1])
        q_out, _, rope_out, _, inner = torch.ops._C_ascend.mla_preprocess(*positional, **kwargs)
    torch.npu.synchronize()
    assert torch.isnan(cache[0, tokens:]).all(), "Unwritten KV cache rows changed"
    # Zero weights make the quantized GEMM accumulator exactly bias0.
    # The no-quant GEMM uses ones and has a constant, independently known sum.
    if unquantized:
        projected = torch.full((width,), x[0, 0].cpu().float().item() * hidden).to(dtype).float()
    else:
        projected = pattern.float() * scale.cpu()
    return {
        "q": inner.cpu().float() if variant == "qdown" else None,
        "kv": cache[0, :tokens].cpu().float(),
        "q_out": q_out.cpu(),
        "rope_out": rope_out.cpu(),
        "q_input": projected[:q_rank],
        "kv_input": projected[q_rank : q_rank + kv_rank],
        "q_gamma": gamma_q.float(),
        "kv_gamma": gamma_kv.float(),
    }


def _assert_norm(result, name, epsilon):
    values = result[name + "_input"]
    reference = values * torch.rsqrt(values.square().mean() + epsilon) * result[name + "_gamma"]
    actual = result[name]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference.expand_as(actual), atol=0.005, rtol=0.01)


@pytest.mark.parametrize("q_rank,nope", [(1536, 128), (2048, 192)])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("epsilons", [(1e-6, 1e-6), (1e-5, 1e-5), (1e-5, 1e-4)])
@pytest.mark.parametrize("norm", ["q", "kv"])
@torch.inference_mode()
def test_sfa_mlapo_honors_model_epsilons(q_rank, nope, tokens, epsilons, norm):
    result = _run_mlapo(tokens, q_rank, nope, epsilons=epsilons, through_sfa=True)
    _assert_norm(result, norm, epsilons[0 if norm == "q" else 1])


@pytest.mark.parametrize("variant", ["bf16", "fp16", "no_quant"])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("epsilons", [(1e-6, 1e-6), (1e-5, 1e-4)])
@torch.inference_mode()
def test_mlapo_other_variants_honor_kv_epsilon(variant, tokens, epsilons):
    result = _run_mlapo(tokens, variant=variant, epsilons=epsilons)
    _assert_norm(result, "kv", epsilons[1])


@pytest.mark.parametrize("variant", ["qdown", "bf16", "fp16", "no_quant"])
@torch.inference_mode()
def test_mlapo_default_epsilons_preserve_outputs(variant):
    implicit = _run_mlapo(1, variant=variant)
    explicit = _run_mlapo(1, variant=variant, epsilons=(1e-6, 1e-6))
    for name in ("q", "kv", "q_out", "rope_out"):
        if implicit[name] is not None:
            torch.testing.assert_close(implicit[name], explicit[name], atol=0, rtol=0)


@pytest.mark.parametrize("device", ["meta", "npu"])
@pytest.mark.parametrize("parameter", ["q_epsilon", "kv_epsilon"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), 1e-300, 1e300])
def test_mlapo_rejects_invalid_epsilon(device, parameter, value):
    tensor = torch.empty(1, device=device)
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match=parameter + " must be positive and finite"):
        torch.ops._C_ascend.mla_preprocess(
            tensor,
            tensor,
            None,
            tensor,
            None,
            tensor,
            None,
            tensor,
            None,
            None,
            tensor,
            tensor,
            tensor,
            slots,
            quant_scale0=None,
            quant_offset0=None,
            bias0=None,
            quant_scale1=None,
            quant_offset1=None,
            bias1=None,
            ctkv_scale=None,
            q_nope_scale=None,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=False,
            q_out0=tensor,
            kv_cache_out0=tensor,
            q_out1=tensor,
            kv_cache_out1=tensor,
            inner_out=tensor,
            **{parameter: value},
        )
