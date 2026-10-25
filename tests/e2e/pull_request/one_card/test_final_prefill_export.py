# SPDX-License-Identifier: Apache-2.0
"""NPU numerical coverage of the P C8 -> D TQ4 export boundary.

This is a component test; a full model P/D run is still required for TTFT.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill import (
    C8_TO_TQ4, FinalPrefillExporter, describe_layout,
)
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


@pytest.mark.parametrize("logical_block,chunk_tokens", [(128, 256), (256, 4096)])
@torch.inference_mode()
def test_bulk_export_reused_indices_and_cross_layer_overlap_are_byte_exact(logical_block, chunk_tokens):
    torch.manual_seed(213)
    device = torch.device("npu")
    exporter = FinalPrefillExporter.__new__(FinalPrefillExporter)
    exporter.device = device
    exporter.stream = torch.npu.Stream(device=device)
    exporter.quantizer = TurboQuantLatent()
    exporter.quantizer._initialize(device)
    exporter.incremental = True
    exporter.chunk_tokens = chunk_tokens
    exporter.staging_buffers = [torch.empty(chunk_tokens, 386, dtype=torch.uint8, device=device) for _ in range(2)]
    exporter.caches, exporter.metadata = {}, {}
    layers, expected, targets, originals, regions = {}, {}, {}, {}, list(exporter.staging_buffers)
    count = 2 * (chunk_tokens // logical_block) + 1
    source_ids = list(reversed(range(count)))
    destination_ids = list(range(1, count + 1))

    def layout(tensor, group, producer):
        meta = SimpleNamespace(tensor_group_idx=[group], kv_caches_base_addr=[tensor.data_ptr()],
                               block_len=[tensor[0].numel() * tensor.element_size()],
                               block_size_scale=[logical_block // tensor.shape[1]])
        spec = SimpleNamespace(block_size=logical_block, cache_sparse_sfa_turboquant=not producer,
                               cache_sparse_sfa_c8=producer)
        describe_layout(meta, (tensor,), spec, C8_TO_TQ4, producer)
        return meta

    for layer, scale in [("main.a", 2), ("main.b", 2), ("draft", 1)]:
        physical_tokens = logical_block // scale
        blocks = (count + 2) * scale
        raw = torch.full((blocks, physical_tokens * 656 + 32), 7, dtype=torch.int8, device=device)
        cache = raw[:, :physical_tokens * 656].view(blocks, physical_tokens, 1, 656)
        nope = torch.randn(blocks * physical_tokens, 512, dtype=torch.bfloat16, device=device)
        rope = torch.randn(blocks * physical_tokens, 64, dtype=torch.bfloat16, device=device)
        cache.copy_(quantize_c8(nope, rope).view_as(cache))
        target = torch.full((count + 2, logical_block, 1, 386), 7, dtype=torch.uint8, device=device)
        local = layout(cache, 0 if layer != "draft" else 1, True)
        remote = layout(target, 4 if layer != "draft" else 7, False)
        ids = source_ids if layer != "draft" else source_ids[1:] + source_ids[:1]
        exporter.caches[layer] = (cache,)
        exporter.metadata[layer] = local
        layers[layer] = (remote, ids, destination_ids)
        targets[layer] = target
        originals[layer] = raw, raw.clone()
        regions.extend([raw, target])
        reference = torch.full_like(target, 7)
        pages = chunk_tokens // logical_block
        for start in range(0, count, pages):
            selected = [block * scale + offset for block in ids[start:start + pages] for offset in range(scale)]
            values = cache.index_select(0, torch.tensor(selected, dtype=torch.int64, device=device)).reshape(-1, 656)
            packed = pack_c8_kv(values, exporter.quantizer).view(-1, logical_block, 1, 386)
            for index, dst in enumerate(destination_ids[start:start + pages]):
                reference[dst].copy_(packed[index])
        expected[layer] = reference

    def region_view(address, length):
        for tensor in regions:
            begin = tensor.data_ptr()
            if begin <= address and address + length <= begin + tensor.numel() * tensor.element_size():
                return tensor.view(torch.uint8).reshape(-1)[address - begin:address - begin + length]
        raise AssertionError("Transfer outside live staging/destination storage")

    def write(session, sources, destinations, lengths):
        for src, dst, length in zip(sources, destinations, lengths):
            region_view(dst, length).copy_(region_view(src, length))
        # Emulate the completion/ownership guarantee of a synchronous transport.
        torch.npu.current_stream().synchronize()

    exporter._write = write
    exporter.ready = torch.npu.Event()
    exporter.ready.record()
    for _ in range(2):
        for target in targets.values():
            target.fill_(7)
        torch.npu.synchronize()
        exporter.export_layers(layers, "test-only-copy-transport")
        for layer in layers:
            torch.testing.assert_close(targets[layer].cpu(), expected[layer].cpu(), rtol=0, atol=0)
            raw, original = originals[layer]
            torch.testing.assert_close(raw.cpu(), original.cpu(), rtol=0, atol=0)

    def failed_write(*args):
        raise RuntimeError("test transport failure")

    exporter._write = failed_write
    with pytest.raises(RuntimeError, match="test transport failure"):
        exporter.export_layers(layers, "failure")
    assert exporter.stream.query()  # No next-layer conversion can still read source pages.
