# SPDX-License-Identifier: Apache-2.0
"""A5 C8 non-absorbed prefill preparation and native attention gates."""

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.ops.flash_attn_c8_quant import fake_quant_flash_attn_c8, quantize_flash_attn_c8


def _inputs(qtokens, ktokens, heads, shared_rope):
    torch.manual_seed(20260921)
    query_width = 224 if shared_rope else 192
    query = (torch.randn(qtokens, heads, query_width) * 0.5).bfloat16().to("npu")[..., :192]
    projected = (torch.randn(ktokens, heads, 256) * 0.5).bfloat16().to("npu")
    # Deliberately different distributions/scales for every head and K vs V.
    projected.mul_(torch.linspace(0.5, 2.0, heads, device="npu")[None, :, None])
    key, value = projected.split(128, dim=-1)
    value.mul_(2)
    if ktokens > 4096:
        # Make the last, partial statistics stripe own the global maxima.
        key[-1].mul_(32)
        value[-1].mul_(64)
    rope = (torch.randn(ktokens, 1 if shared_rope else heads, 64) * 0.5).bfloat16().to("npu")
    return query, key, value, rope


def _assert_fp8_codes(actual, expected, normalized):
    actual = actual.cpu()
    expected = expected.cpu()
    different = actual.float() != expected.float()
    if different.any():
        # A last-bit FP32 divide discrepancy may choose either FP8 neighbor
        # at a rounding midpoint. No general quantization tolerance applies.
        distance = (actual.view(torch.uint8).int() - expected.view(torch.uint8).int()).abs()
        assert torch.all(distance[different] == 1)
        midpoint = (actual.float() + expected.float()) * 0.5
        roundoff = 2 * torch.finfo(torch.float32).eps * normalized.abs()
        assert torch.all((normalized - midpoint).abs()[different] <= roundoff[different])


def _assert_bf16_division_rounding(actual, expected, exact):
    """Allow only adjacent BF16 choices at a two-FP32-division midpoint.

    A CPU/NPU last-bit divide difference can cross a BF16 tie. This is not an
    output tolerance: a differing code must be the immediate BF16 neighbor,
    and the FP64 quotient must lie within the accumulated FP32 rounding bound
    of their exact midpoint. Two divisions each allow one FP32 relative ULP.
    """
    actual, expected = actual.cpu(), expected.cpu()
    different = actual != expected
    if different.any():
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        distance = (actual.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
        assert torch.all(distance[different] == 1)
        midpoint = (actual.double() + expected.double()) * 0.5
        epsilon = torch.finfo(torch.float32).eps
        roundoff = (2 * epsilon / (1 - 2 * epsilon)) * exact.abs()
        assert torch.all((exact - midpoint).abs()[different] <= roundoff[different])


def test_c8_prefill_quant_bf16_midpoint_guard():
    expected = torch.tensor([1.0], dtype=torch.bfloat16)
    neighbor = torch.tensor([1.0078125], dtype=torch.bfloat16)
    midpoint = torch.tensor([1.00390625], dtype=torch.float64)
    _assert_bf16_division_rounding(neighbor, expected, midpoint)
    with pytest.raises(AssertionError):
        _assert_bf16_division_rounding(neighbor, expected, expected.double())
    with pytest.raises(AssertionError):
        _assert_bf16_division_rounding(torch.tensor([1.015625], dtype=torch.bfloat16), expected, midpoint)


@pytest.mark.parametrize(
    "qtokens,ktokens,heads",
    [(1, 1, 1), (17, 129, 12), (257, 257, 12), (4097, 4097, 12), (17, 32769, 12)],
)
@pytest.mark.parametrize("shared_rope", [False, True])
@torch.inference_mode()
def test_c8_prefill_quantize_strided_projection_views(qtokens, ktokens, heads, shared_rope):
    inputs = _inputs(qtokens, ktokens, heads, shared_rope)
    q8, k8, v8, qr, kr, sq, sk, sv = quantize_flash_attn_c8(*inputs)
    query, key, value, key_rope = (tensor.cpu().float() for tensor in inputs)
    expected_sq = query[..., :128].abs().amax(-1) * (1.0 / 448.0)
    expected_sk = key.abs().amax((0, 2)) * (1.0 / 448.0)
    expected_sv = value.abs().amax((0, 2)) * (1.0 / 448.0)
    for actual, expected in ((sq, expected_sq), (sk, expected_sk), (sv, expected_sv)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    for actual, source, scale in (
        (q8, query[..., :128], expected_sq[..., None]),
        (k8, key, expected_sk[None, :, None]),
        (v8, value, expected_sv[None, :, None]),
    ):
        normalized = (source / scale).clamp(-448, 448)
        _assert_fp8_codes(actual, normalized.to(torch.float8_e4m3fn), normalized)
    expected_qr = (query[..., 128:] / expected_sq[..., None] / expected_sk[None, :, None]).bfloat16()
    exact_qr = query[..., 128:].double() / expected_sq[..., None].double() / expected_sk[None, :, None].double()
    _assert_bf16_division_rounding(qr, expected_qr, exact_qr)
    torch.testing.assert_close(kr.cpu().float(), key_rope.expand(ktokens, heads, 64), rtol=0, atol=0)
    assert all(tensor.is_contiguous() for tensor in (q8, k8, v8, qr, kr, sq, sk, sv))

    fake_q, fake_k, fake_v = fake_quant_flash_attn_c8(*inputs)
    expected_q = torch.cat((q8.float() * sq[..., None], qr.float() * sq[..., None] * sk[None, :, None]), -1).bfloat16()
    expected_k = torch.cat((k8.float() * sk[None, :, None], kr.float()), -1).bfloat16()
    expected_v = (v8.float() * sv[None, :, None]).bfloat16()
    for actual, expected in ((fake_q, expected_q), (fake_k, expected_k), (fake_v, expected_v)):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_c8_prefill_quant_graph_replay_zero_heads_preserves_rope():
    inputs = _inputs(129, 129, 12, True)
    quantize_flash_attn_c8(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = quantize_flash_attn_c8(*inputs)
    inputs[0][..., :128].zero_()
    inputs[1].zero_()
    inputs[2].zero_()
    graph.replay()
    torch.npu.synchronize()
    q8, k8, v8, qr, kr, sq, sk, sv = (tensor.cpu() for tensor in result)
    for tensor in (q8, k8, v8):
        assert torch.count_nonzero(tensor.float()) == 0
    for scale in (sq, sk, sv):
        assert torch.all(scale == 1)
    torch.testing.assert_close(qr, inputs[0][..., 128:].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(kr, inputs[3].expand(-1, 12, -1).cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(qr) > 0


@pytest.mark.parametrize(
    "query_lengths,kv_lengths,causal,zero_nope",
    [
        ((1, 33, 127, 129), (1, 33, 127, 129), True, False),
        ((65, 257), (65, 257), True, True),
        ((17, 65), (129, 513), False, False),
        ((17, 65), (129, 513), True, False),
        ((63, 64), (129, 257), False, False),
        ((63, 64), (129, 257), True, False),
        ((385,), (385,), True, False),
        ((319,), (319,), True, False),
        ((257,), (512,), False, False),
    ],
)
@torch.inference_mode()
def test_native_c8_prefill_reference(query_lengths, kv_lengths, causal, zero_nope):
    from flash_attn_c8_prefill_qualification import C8PrefillCase

    case = C8PrefillCase(query_lengths, kv_lengths, causal=causal, zero_nope=zero_nope).to_npu()
    case.check(*case.run())


@torch.inference_mode()
def test_native_c8_prefill_graph_replay_changed_query_and_scales():
    from flash_attn_c8_prefill_qualification import C8PrefillCase

    case = C8PrefillCase((65, 129)).to_npu()
    case.run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, lse = case.run()
    case.q_cpu = (-case.q_cpu.float()).to(torch.float8_e4m3fn)
    case.q.copy_(case.q_cpu)
    case.sq_cpu.mul_(1.25)
    case.sq.copy_(case.sq_cpu)
    case.scale_v.mul_(0.75)
    case.sv.copy_(case.scale_v)
    graph.replay()
    torch.npu.synchronize()
    case.check(output, lse)
