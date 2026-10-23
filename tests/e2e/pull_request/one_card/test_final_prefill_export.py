# SPDX-License-Identifier: Apache-2.0
"""NPU numerical coverage of the P C8 -> D TQ4 export boundary.

This is a component test; a full model P/D run is still required for TTFT.
"""

import pytest
import torch
import torch_npu

from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import pack_c8_kv, unpack_c8_kv
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent


def quantize_c8(nope, rope):
    values, scales = torch_npu.npu_dynamic_block_quant(
        nope.reshape(-1, 1, 512), dst_type=torch.int8, row_block_size=1, col_block_size=128,
    )
    return torch.cat((values.reshape(-1, 512), rope.reshape(-1, 64).contiguous().view(torch.int8),
                      scales.float().reshape(-1, 4).contiguous().view(torch.int8)), -1)


@pytest.mark.parametrize("rows", [1, 17, 257, 4096])
@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_c8_export_matches_reference_quantizer(rows, fused):
    torch.manual_seed(211)
    nope = torch.randn(rows, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(rows, 64, dtype=torch.bfloat16, device="npu")
    packed_c8 = quantize_c8(nope, rope)
    before = packed_c8.cpu()
    # Independent CPU unpack: verify signed bytes, FP32 scales and unmodified RoPE.
    scales = before[:, 640:].contiguous().view(torch.float32)
    reference_nope = (before[:, :512].float().reshape(rows, 4, 128)
                      * scales.unsqueeze(-1)).reshape(rows, 512).bfloat16().npu()
    decoded, decoded_rope = unpack_c8_kv(packed_c8)
    torch.testing.assert_close(decoded, reference_nope, rtol=0, atol=0)
    torch.testing.assert_close(decoded_rope, rope, rtol=0, atol=0)
    quantizer = TurboQuantLatent(fused_finalize=fused)
    actual = pack_c8_kv(packed_c8, quantizer)
    online = quantizer.compress(reference_nope).reshape(rows, 258)
    scale = online[:, 256:].contiguous().view(torch.float16)
    online_rope = (rope.float() / (scale.float() + 1e-20)).bfloat16()
    expected = torch.cat((online[:, :256], online_rope.view(torch.uint8), online[:, 256:]), -1)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(packed_c8.cpu(), before, rtol=0, atol=0)
    assert actual.shape == (rows, 386)
    assert torch.isfinite(scale).all()


@torch.inference_mode()
def test_converted_cache_is_consumable_by_tq_attention():
    init_device_properties_triton()
    torch.manual_seed(212)
    rows = 2048
    nope = torch.randn(rows, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(rows, 64, dtype=torch.bfloat16, device="npu")
    store = TurboQuantLatent(fused_finalize=True)
    packed = pack_c8_kv(quantize_c8(nope, rope), store)
    cache = packed.view(rows // 256, 256, 1, 386).view(torch.int8)
    query = torch.randn(1, 64, 576, dtype=torch.bfloat16, device="npu")
    # Query's latent coordinates must match the exported signed-Hadamard K.
    query[:, :, :512].copy_(store.forward(query[:, :, :512]))
    indices = torch.arange(rows, device="npu", dtype=torch.int32).view(1, 1, rows)
    common = dict(
        block_table=torch.arange(rows // 256, dtype=torch.int32, device="npu").view(1, -1),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device="npu"),
        actual_seq_lengths_kv=torch.tensor([rows], dtype=torch.int32, device="npu"),
        scale_value=576**-0.5,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        key_quant_mode=3,
        value_quant_mode=3,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    actual = torch_npu.npu_kv_quant_sparse_flash_attention(query, cache, cache, indices, **common)
    actual = actual[0] if isinstance(actual, tuple) else actual
    assert torch.isfinite(actual).all()
    # Independently decode the byte ABI and compare attention in its rotated
    # coordinates, with a valid, unique sparse index for every key.
    codes = torch.stack((packed[:, :256] & 15, packed[:, :256] >> 4), -1).flatten(1).long()
    scale = packed[:, 384:].contiguous().view(torch.float16).float()
    decoded_nope = (store.centroids.bfloat16().float()[codes] * scale).bfloat16()
    decoded_rope = (packed[:, 256:384].contiguous().view(torch.bfloat16).float() * scale).bfloat16()
    selected = indices[0, 0].long()
    keys = torch.cat((decoded_nope, decoded_rope), -1)[selected].float()
    expected = (query[0].float() @ keys.T * 576**-0.5).softmax(-1) @ decoded_nope[selected].float()
    error = (actual.float().reshape_as(expected) - expected).norm() / expected.norm()
    assert error.item() < 0.03


@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_colocated_c8_update_preserves_source_and_skips_graph_padding(graph):
    from vllm_ascend.quantization.methods.kv_cache.turboquant.final_prefill import (
        C8_SLOT_BYTES, TQ_SLOT_BYTES, ColocatedFinalPrefill, colocated_cache_views,
    )
    raw = torch.zeros(256 * (C8_SLOT_BYTES + TQ_SLOT_BYTES), dtype=torch.int8, device="npu")
    cache = colocated_cache_views(raw, 2, 128)
    source = quantize_c8(torch.randn(256, 512, dtype=torch.bfloat16, device="npu"),
                         torch.randn(256, 64, dtype=torch.bfloat16, device="npu"))
    cache[0].copy_(source.reshape_as(cache[0]))
    state = ColocatedFinalPrefill(TurboQuantLatent())
    state.bind(cache)
    slots = torch.tensor([3, 129, -1, -1], device="npu", dtype=torch.int64)
    state.update_decode_tokens(slots)
    if graph:
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            state.update_decode_tokens(slots)
        cache[1].zero_()
        g.replay()
    torch.npu.synchronize()
    expected = pack_c8_kv(source[[3, 129]], state.quantizer)
    target = cache[1].view(256, TQ_SLOT_BYTES)
    torch.testing.assert_close(target[[3, 129]].view(torch.uint8), expected, rtol=0, atol=0)
    assert not target[0].count_nonzero()
    torch.testing.assert_close(cache[0].flatten(0, 2), source, rtol=0, atol=0)


@torch.inference_mode()
def test_incremental_c8_pipeline_joins_before_cache_reuse():
    from vllm_ascend.quantization.methods.kv_cache.turboquant.final_prefill import (
        C8_SLOT_BYTES, TQ_SLOT_BYTES, ColocatedFinalPrefill,
        PrefillQuantizationPipeline, colocated_cache_views,
    )
    device = torch.device("npu:0")
    pipeline = PrefillQuantizationPipeline(device)
    raw = torch.zeros(512 * (C8_SLOT_BYTES + TQ_SLOT_BYTES), dtype=torch.int8, device=device)
    source = quantize_c8(torch.randn(512, 512, dtype=torch.bfloat16, device=device),
                         torch.randn(512, 64, dtype=torch.bfloat16, device=device))
    state = ColocatedFinalPrefill(TurboQuantLatent())
    state.quantizer._initialize(device)
    state.bind(colocated_cache_views(raw, 4, 128))
    for start, end in [(0, 129), (129, 257), (257, 511)]:
        state.cache[0].view(-1, C8_SLOT_BYTES)[start:end].copy_(source[start:end])
        state.prefill_slots = torch.arange(start, end, device=device)
        pipeline.submit(state)
        pipeline.join()
        assert not pipeline.pending
        expected = pack_c8_kv(source[start:end], state.quantizer)
        torch.testing.assert_close(state.cache[1].view(-1, TQ_SLOT_BYTES)[start:end].view(torch.uint8),
                                   expected, rtol=0, atol=0)
    torch.testing.assert_close(state.cache[0].view(-1, C8_SLOT_BYTES)[:511], source[:511], rtol=0, atol=0)
