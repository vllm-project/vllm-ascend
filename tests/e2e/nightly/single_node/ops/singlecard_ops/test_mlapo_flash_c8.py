# SPDX-License-Identifier: Apache-2.0
"""A5 K3 C8 prolog: scale semantics, mixed cache strides, and graph replay."""

import pytest
import torch
import torch_npu

from vllm_ascend.worker.flash_kv_cache import split_flash_mla_c8_cache


def make_mxfp8_prolog_inputs(tokens, heads):
    torch.manual_seed(16464)
    device, dtype = "npu", torch.bfloat16

    def weight(input_size, output_size):
        value = torch.randn(output_size, input_size, dtype=dtype, device=device) * 0.02
        quantized, scale = torch_npu.npu_dynamic_mx_quant(value, dst_type=torch.float8_e4m3fn)
        return torch_npu.npu_format_cast(quantized.T.contiguous(), 29), scale.flatten(1).view(torch.float8_e8m0fnu)

    w_dq, s_dq = weight(7168, 1536)
    w_dkv, s_dkv = weight(7168, 576)
    w_uq, s_uq = weight(1536, heads * 192)
    x, scale_x = torch_npu.npu_dynamic_mx_quant(
        torch.randn(tokens, 7168, dtype=dtype, device=device), dst_type=torch.float8_e4m3fn
    )
    # Start at the last token of a page; skip the last row in multi-token cases.
    slots = list(range(127, 127 + tokens))
    if tokens > 1:
        slots[-1] = -1
    kwargs = dict(
        token_x=x,
        weight_dq=w_dq,
        weight_dkv_kr=w_dkv,
        weight_uq_qr=w_uq,
        weight_uk=torch.randn(heads, 128, 512, dtype=dtype, device=device) * 0.02,
        rmsnorm_gamma_cq=torch.ones(1536, dtype=dtype, device=device),
        rmsnorm_gamma_ckv=torch.ones(512, dtype=dtype, device=device),
        rope_sin=torch.empty(0, 64, dtype=dtype, device=device),
        rope_cos=torch.empty(0, 64, dtype=dtype, device=device),
        dequant_scale_x=scale_x.flatten(1).view(torch.float8_e8m0fnu),
        dequant_scale_w_dq=s_dq,
        dequant_scale_w_dkv_kr=s_dkv,
        dequant_scale_w_uq_qr=s_uq,
        cache_index=torch.tensor(slots, dtype=torch.int64, device=device),
        cache_mode="PA_BSND",
        weight_quant_mode=3,
        rmsnorm_epsilon_cq=1e-6,
        rmsnorm_epsilon_ckv=1e-6,
    )
    return kwargs, slots


@pytest.mark.parametrize(
    "tokens,heads,kv_descale",
    [(1, 12, 0.02), (3, 12, 0.1), (32, 12, 0.02), (64, 96, 0.02), (64, 96, 0.1)],
)
@torch.inference_mode()
def test_mxfp8_prolog_per_tensor_flash_cache(tokens, heads, kv_descale):
    if not hasattr(torch.ops._C_ascend, "npu_mla_prolog_v3"):
        pytest.skip("requires the A5 K3 C8 prolog build")
    device, dtype = "npu", torch.bfloat16
    kwargs, slots = make_mxfp8_prolog_inputs(tokens, heads)
    op = torch.ops._C_ascend.npu_mla_prolog_v3
    bf16_kv = torch.zeros(4, 128, 1, 512, dtype=dtype, device=device)
    bf16_kr = torch.zeros(4, 128, 1, 64, dtype=dtype, device=device)
    bf16 = op(kv_cache=bf16_kv, kr_cache=bf16_kr, **kwargs)
    c8_kwargs = dict(
        kwargs,
        kv_cache_quant_mode=1,
        query_quant_mode=1,
        quant_scale_ckv=torch.tensor([1.0 / kv_descale], dtype=torch.float32, device=device),
    )

    # Each kernel page contains dense FP8 CKV and BF16 KR planes. Different
    # layers share the backing, so only the page axis has non-contiguous gaps.
    storage = torch.full((4, 2, 128, 640), 42, dtype=torch.uint8, device=device)
    cache_bytes = storage[:, 0]
    kv_cache, kr_cache = split_flash_mla_c8_cache(cache_bytes.view(torch.float8_e4m3fn))
    kv_ref, kr_ref = kv_cache.contiguous(), kr_cache.contiguous()
    reference = op(kv_cache=kv_ref, kr_cache=kr_ref, **c8_kwargs)

    def run():
        return op(kv_cache=kv_cache, kr_cache=kr_cache, **c8_kwargs)

    actual = run()
    torch.npu.synchronize()
    for index in (0, 1, 2):
        torch.testing.assert_close(actual[index].float().cpu(), reference[index].float().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(kv_cache.float().cpu(), kv_ref.float().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(kr_cache.cpu(), kr_ref.cpu(), rtol=0, atol=0)
    assert torch.all(storage[:, 1].cpu() == 42)

    untouched = torch.ones(4 * 128, dtype=torch.bool)
    untouched[[slot for slot in slots if slot >= 0]] = False
    assert torch.all(kv_cache.cpu().contiguous().view(torch.uint8).reshape(4 * 128, 512)[untouched] == 42)
    assert torch.all(kr_cache.cpu().contiguous().view(torch.uint8).reshape(4 * 128, 128)[untouched] == 42)

    # Compare quantization to the same MXFP8 projection without KV/Q quantization.
    # Q goes through the same BF16 Qn matmul in both modes, so the independent
    # CPU per-token/head quantizer must match the C8 path.
    q_bf16 = bf16[0].float().cpu()
    # CANN multiplies by a rounded FP32 reciprocal. Division differs by one
    # scale ULP and can choose another FP8 code at exact rounding midpoints.
    expected_scale = q_bf16.abs().amax(dim=-1, keepdim=True) * (1.0 / 448.0)
    normalized_q = (q_bf16 / expected_scale).clamp(-448, 448)
    expected_q = normalized_q.to(torch.float8_e4m3fn)
    torch.testing.assert_close(actual[2].cpu(), expected_scale, rtol=1e-5, atol=1e-7)
    actual_q = actual[0].cpu()
    different = actual_q.float() != expected_q.float()
    if different.any():
        # NPU FP32 division can differ from CPU by one ULP. Only adjacent FP8
        # codes at such a midpoint are valid; no general FP8 tolerance applies.
        actual_codes = actual_q.view(torch.uint8).int()
        expected_codes = expected_q.view(torch.uint8).int()
        assert torch.all((actual_codes - expected_codes).abs()[different] == 1)
        midpoint = (actual_q.float() + expected_q.float()) * 0.5
        roundoff = 2 * torch.finfo(torch.float32).eps * normalized_q.abs()
        assert torch.all((normalized_q - midpoint).abs()[different] <= roundoff[different])
    # The standalone CANN quantizer is an independent same-device reference,
    # with identical FP32 division behavior and therefore an exact-code gate.
    quantized_q, descale_q = torch_npu.npu_dynamic_quant(bf16[0].flatten(0, 1), dst_type=torch.float8_e4m3fn)
    torch.testing.assert_close(actual_q.float(), quantized_q.reshape_as(actual_q).float().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual[2].cpu(), descale_q.reshape_as(actual[2]).cpu(), rtol=0, atol=0)
    expected_qr = (bf16[1].float().cpu() / expected_scale / kv_descale).to(dtype)
    torch.testing.assert_close(actual[1].cpu(), expected_qr, rtol=0.008, atol=0.01)

    selected = torch.tensor([slot for slot in slots if slot >= 0], device=device)
    torch.testing.assert_close(
        kr_ref.flatten(0, 1)[selected].cpu(), bf16_kr.flatten(0, 1)[selected].cpu(), rtol=0, atol=0
    )
    # C8 quantizes RMSNorm's FP32 result directly; mode 7 rounds it to BF16.
    # Compare reconstructed latent values, allowing one FP8 quantization step.
    torch.testing.assert_close(
        kv_ref.float().flatten(0, 1)[selected].cpu() * kv_descale,
        bf16_kv.float().flatten(0, 1)[selected].cpu(),
        rtol=0.07,
        atol=kv_descale / 512,
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = run()
    graph.replay()
    torch.npu.synchronize()
    for index in (0, 1, 2):
        torch.testing.assert_close(replayed[index].float().cpu(), reference[index].float().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(kv_cache.float().cpu(), kv_ref.float().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(kr_cache.cpu(), kr_ref.cpu(), rtol=0, atol=0)
    assert torch.all(storage[:, 1].cpu() == 42)


@torch.inference_mode()
def test_mxfp8_prolog_c8_zero_hidden_stays_finite():
    kwargs, _ = make_mxfp8_prolog_inputs(1, 12)
    kwargs["token_x"].zero_()
    storage = torch.zeros(4, 2, 128, 640, dtype=torch.uint8, device="npu")
    kv_cache, kr_cache = split_flash_mla_c8_cache(storage[:, 0].view(torch.float8_e4m3fn))

    def run():
        return torch.ops._C_ascend.npu_mla_prolog_v3(
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            kv_cache_quant_mode=1,
            query_quant_mode=1,
            quant_scale_ckv=torch.tensor([50.0], device="npu"),
            **kwargs,
        )

    result = run()
    torch.testing.assert_close(result[2].cpu(), torch.ones_like(result[2].cpu()), rtol=0, atol=0)
    for tensor in (*result[:2], kv_cache[0, 127], kr_cache[0, 127]):
        result = tensor.float().cpu()
        assert result.isfinite().all()
        assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("zero_heads", [1, 12])
@torch.inference_mode()
def test_mxfp8_prolog_c8_zero_latent_preserves_bf16_component(zero_heads):
    kwargs, _ = make_mxfp8_prolog_inputs(3, 12)
    kwargs["weight_uk"][:zero_heads].zero_()
    bf16_kv = torch.zeros(4, 128, 1, 512, dtype=torch.bfloat16, device="npu")
    bf16_kr = torch.zeros(4, 128, 1, 64, dtype=torch.bfloat16, device="npu")
    op = torch.ops._C_ascend.npu_mla_prolog_v3
    bf16 = op(kv_cache=bf16_kv, kr_cache=bf16_kr, **kwargs)
    storage = torch.zeros(4, 2, 128, 640, dtype=torch.uint8, device="npu")
    kv_cache, kr_cache = split_flash_mla_c8_cache(storage[:, 0].view(torch.float8_e4m3fn))
    quant_scale = torch.tensor([50.0], device="npu")

    def run():
        return op(
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            kv_cache_quant_mode=1,
            query_quant_mode=1,
            quant_scale_ckv=quant_scale,
            **kwargs,
        )

    actual = run()
    q, qr, scale = (tensor.cpu() for tensor in actual[:3])
    assert torch.count_nonzero(q[:, :zero_heads].float()) == 0
    assert torch.all(scale[:, :zero_heads] == 1)
    assert torch.count_nonzero(bf16[1][:, :zero_heads].cpu()) > 0
    expected_qr = (bf16[1].float().cpu() / scale * 50.0).to(torch.bfloat16)
    torch.testing.assert_close(qr, expected_qr, rtol=0.008, atol=0.01)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = run()
    graph.replay()
    torch.npu.synchronize()
    for left, right in zip(replayed[:3], actual[:3]):
        torch.testing.assert_close(left.float().cpu(), right.float().cpu(), rtol=0, atol=0)
