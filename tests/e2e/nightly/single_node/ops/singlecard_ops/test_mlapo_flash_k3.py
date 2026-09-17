# SPDX-License-Identifier: Apache-2.0
"""A5 K3 TP8 prolog writes FlashMLA's combined, strided KV cache in place."""

import pytest
import torch
import torch_npu


@pytest.mark.parametrize("tokens", [4, 8, 16])
@pytest.mark.parametrize("heads", [12, 96])
@torch.inference_mode()
def test_mxfp8_prolog_interleaved_flash_cache(tokens, heads):
    if not hasattr(torch.ops._C_ascend, "npu_mla_prolog_v3"):
        pytest.skip("requires the A5 K3 prolog build")
    device, dtype = "npu", torch.bfloat16
    torch.manual_seed(16464)

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
        cache_index=torch.tensor([127, 128] + list(range(129, 129 + tokens - 3)) + [-1], device=device),
        cache_mode="PA_BSND",
        weight_quant_mode=3,
        rmsnorm_epsilon_cq=1e-6,
        rmsnorm_epsilon_ckv=1e-6,
    )
    storage = torch.full((4, 2, 128, 576), 42, dtype=dtype, device=device)
    cache = storage[:, 0]
    kv_ref = cache[..., :512].unsqueeze(2).contiguous()
    kr_ref = cache[..., 512:].unsqueeze(2).contiguous()
    expected = torch.ops._C_ascend.npu_mla_prolog_v3(kv_cache=kv_ref, kr_cache=kr_ref, **kwargs)

    def run():
        return torch.ops._C_ascend.npu_mla_prolog_v3(
            kv_cache=cache[..., :512].unsqueeze(2), kr_cache=cache[..., 512:].unsqueeze(2), **kwargs
        )

    actual = run()
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(cache[..., :512].unsqueeze(2), kv_ref, rtol=0, atol=0)
    torch.testing.assert_close(cache[..., 512:].unsqueeze(2), kr_ref, rtol=0, atol=0)
    assert torch.all(storage[:, 1] == 42)

    if heads == 96:
        # Replicated Q must match eight independent TP8 head projections.
        uq_nd = torch_npu.npu_format_cast(w_uq, 2)
        expected_parts = []
        for rank in range(8):
            local = dict(kwargs)
            start, end = rank * 12, (rank + 1) * 12
            local["weight_uq_qr"] = torch_npu.npu_format_cast(uq_nd[:, start * 192 : end * 192].contiguous(), 29)
            local["dequant_scale_w_uq_qr"] = s_uq[start * 192 : end * 192].contiguous()
            local["weight_uk"] = kwargs["weight_uk"][start:end].contiguous()
            expected_parts.append(torch.ops._C_ascend.npu_mla_prolog_v3(kv_cache=kv_ref, kr_cache=kr_ref, **local))
        for output_index in (0, 1):
            expected_q = torch.cat([part[output_index].reshape(tokens, 12, -1) for part in expected_parts], dim=1)
            torch.testing.assert_close(actual[output_index].reshape(tokens, 96, -1), expected_q, rtol=0.02, atol=0.02)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = run()
    graph.replay()
    torch.testing.assert_close(replayed[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(cache[..., :512].unsqueeze(2), kv_ref, rtol=0, atol=0)
    assert torch.all(storage[:, 1] == 42)
