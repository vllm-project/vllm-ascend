# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION, SPARSE_ATTENTION, HYV4FlashConfig, HYV4FlashVLConfig
from vllm.models.hy_v4_flash.reference import (
    decode_e4m3fn,
    hash_ngrams,
    indexpool_fp8,
    mla_attention_probs,
    sparse_candidates,
    window_candidates,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows, pack_c8_components
from vllm_ascend.models.hy_v4_flash.aclgraph import FlashAclGraphManager
from vllm_ascend.models.hy_v4_flash.attention import AscendHYV4FlashAttention, FlashMetadataBuilder
from vllm_ascend.models.hy_v4_flash.dfly_speculator import build_reuse_indices


def test_flat_vl_config_resolves_architecture_and_per_layer_geometry():
    config = HYV4FlashConfig.from_dict(
        {
            "model_type": "hy_v4_flash",
            "architectures": ["HYV4FlashForConditionalGeneration"],
            "hidden_size": 3072,
            "num_hidden_layers": 2,
            "layer_types": ["sparse", "sliding_attention"],
            "num_attention_heads": 32,
            "swa_num_attention_heads": 64,
            "vision_config": {"hidden_size": 1152},
            "rope_parameters": {
                "sparse": {"rope_theta": 10000},
                "sliding_attention": {"rope_theta": 20000},
            },
        },
    )
    assert isinstance(config, HYV4FlashVLConfig)
    assert config.architectures == ["HYV4FlashVLForConditionalGeneration"]
    text = config.get_text_config()
    assert text.layer_types == [SPARSE_ATTENTION, SLIDING_ATTENTION]
    assert text.rope_for_layer(1)["rope_theta"] == 20000
    assert text.rope_for_layer(0)["mrope_section"] == [12, 10, 10]
    assert text.swa_num_attention_heads == 64
    assert config.vision_config.out_hidden_size == 3072


def test_indexpool_fp8_matches_cast_after_hadamard_transform():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(8, 128, generator=generator).bfloat16()
    matrix = torch.ones((1, 1))
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    transformed = x.float() @ matrix * 128**-0.5
    step = torch.exp2(torch.ceil(torch.log2(transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    expected = (transformed / step).to(torch.float8_e4m3fn).float() * step
    torch.testing.assert_close(indexpool_fp8(x), expected, atol=0, rtol=0)


def test_indexpool_scores_relu_before_weighted_reduction_and_masks_future_groups():
    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).expand(2, -1, -1)
    keys = torch.tensor([[2.0, -100.0], [1.0, 0.0], [500.0, 500.0]])
    result = sparse_candidates(queries, torch.ones(2, 2), keys, torch.tensor([0, 9]), 4, 1)
    assert result.tolist() == [[-1, -1, -1, -1, 0, -1, -1, -1], [0, 1, 2, 3, 8, 9, -1, -1]]


def test_indexpool_decode_batch_matches_per_request_and_masks_padding():
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(3, 4, 8, generator=generator)
    weights = torch.randn(3, 4, generator=generator)
    keys = torch.randn(3, 5, 8, generator=generator)
    positions = torch.tensor([19, 6, -1])
    selected = sparse_candidates(query, weights, keys, positions, 4, 2)
    expected = torch.cat(
        [sparse_candidates(query[i : i + 1], weights[i : i + 1], keys[i], positions[i : i + 1], 4, 2) for i in range(3)]
    )
    torch.testing.assert_close(selected, expected)
    assert (selected[-1] < 0).all()


def test_flash_decode_graph_dispatch_pads_uniform_decode_batches_only():
    manager = FlashAclGraphManager.__new__(FlashAclGraphManager)
    manager._graphs_captured = True
    graph = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=3, num_reqs=3, uniform_token_count=1)
    verify = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=6, num_reqs=1, uniform_token_count=6)
    manager._candidates = {}
    manager._decode_descs = {(1, 0, 1): [graph] * 4, (6, 0, 1): [verify] * 2}
    manager.graphs = {graph: object(), verify: object()}

    assert manager.dispatch(3, 3, 1, 0) is graph
    assert manager.dispatch(1, 6, 6, 0) is verify
    assert manager.dispatch(2, 6, 3, 0).cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(2, 2, 1, 0) is graph
    assert manager.dispatch(3, 3, None, 0).cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(1, 3, None, 0).num_reqs == 1
    del manager.graphs[verify]
    assert manager.dispatch(1, 6, 6, 0).cg_mode == CUDAGraphMode.NONE


def test_dfly_reuses_anchor_topk_and_masks_discarded_target_and_future_draft_tokens():
    target_rows = torch.tensor([[1, 4, 6, 7, 8, -1], [0, 1, 2, 3, -1, -1]])
    selected = build_reuse_indices(
        target_rows, torch.tensor([7, 4]), width=3, window=2, active=torch.tensor([True, False])
    )
    assert selected[0].tolist() == [
        [1, 4, -1, -1, -1, -1, 5, 6, 7, -1, -1],
        [1, 4, -1, -1, -1, -1, 5, 6, 7, 8, -1],
        [1, 4, -1, -1, -1, -1, 5, 6, 7, 8, 9],
    ]
    assert selected[1].eq(-1).all()


def test_precise_decode_metadata_excludes_short_prefills():
    builder = FlashMetadataBuilder.__new__(FlashMetadataBuilder)
    builder.precise_decode = True
    common = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 3, 9, 10]),
        seq_lens_cpu=torch.tensor([64, 134, 22]),
        is_prefilling=torch.tensor([True, False, False]),
        block_table_tensor=torch.empty(0),
        slot_mapping=torch.empty(0),
    )
    assert builder.build(0, common).precise_decode_reqs == (False, True, True)
    builder.precise_decode = False
    assert builder.build(0, common).precise_decode_reqs == ()


def test_full_decode_metadata_uses_live_device_lengths():
    builder = FlashMetadataBuilder.__new__(FlashMetadataBuilder)
    builder.precise_decode = False
    seq_lens = torch.tensor([6, 6])
    common = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 6, 12]),
        seq_lens_cpu=seq_lens.clone(),
        seq_lens=seq_lens,
        block_table_tensor=torch.empty(0),
        slot_mapping=torch.empty(0),
    )
    assert builder.build(0, common).decode_seq_lens is None
    captured = builder.build_for_cudagraph_capture(common)
    seq_lens.copy_(torch.tensor([511, 2052]))
    assert captured.decode_seq_lens is seq_lens
    assert captured.decode_seq_lens.tolist() == [511, 2052]


def test_window_boundary_and_zero_value_sink():
    indices = window_candidates(torch.tensor([0, 511, 512]), 512)
    assert (indices[:, 0].tolist(), indices[:, -1].tolist()) == ([-1, 0, 1], [0, 511, 512])
    probs = mla_attention_probs(
        torch.tensor([[[0.0]]]),
        torch.tensor([[[7.0], [100.0]]]),
        torch.tensor([[0, -1]]),
        torch.tensor([0.0]),
        scale=1.0,
    )
    assert probs.tolist() == [[[0.5, 0.0]]]


def test_engram_hash_matches_across_prefill_chunk_boundary():
    tokens = torch.tensor([2, 6, 9, 4, 3])
    tok_map = torch.arange(16, dtype=torch.int32)
    multipliers = torch.tensor([[7, 11, 13], [3, 5, 7]])
    primes = torch.tensor([[[11, 13], [17, 19]], [[5, 7], [11, 13]]])
    offsets = torch.tensor([[0, 11, 24, 41], [0, 5, 12, 23]])
    args = (tok_map, multipliers, primes, offsets, 0, 0, (9,))
    whole = hash_ngrams(tokens, torch.tensor([-1, -1]), *args)
    tail = hash_ngrams(tokens[2:], torch.tensor([6, 2]), *args)
    assert torch.equal(whole[:, 2:], tail)
    assert whole[:, 2].eq(-1).all()
    ngram = 4 * 7 ^ 9 * 11 ^ 6 * 13
    assert whole[0, 3, 2] == ngram % 17 + 24


def test_engram_fp8_row_bytes_match_native_decode():
    bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    native = bits.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(decode_e4m3fn(bits), native, atol=0, rtol=0, equal_nan=True)


def test_fp8_cache_reads_selected_rows_with_per_tile_scales_and_bf16_rope():
    payload = (torch.arange(3 * 512).reshape(3, 512) % 57 - 28).to(torch.float8_e4m3fn)
    rope = (torch.arange(3 * 64).reshape(3, 64) / 32).bfloat16()
    scales = torch.tensor([[0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    packed = pack_c8_components(payload, rope, scales)
    cache = torch.zeros((8, C8_ROW_BYTES + 8), dtype=torch.uint8)
    slots = torch.tensor([5, 1, 4])
    cache[:, :C8_ROW_BYTES].index_copy_(0, slots, packed)
    order = torch.tensor([[1, 2], [0, 1]])
    rows = cache[:, :C8_ROW_BYTES][slots[order]]
    expected = torch.cat((payload.float() * scales.repeat_interleave(128, dim=-1), rope.float()), dim=-1)
    torch.testing.assert_close(dequantize_c8_rows(rows), expected[order], atol=0, rtol=0)


def test_paged_cache_gathers_fields_in_logical_order():
    block_table = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]], dtype=torch.int32)
    positions = torch.tensor([[0, 5, 5], [3, 6, 1]])
    slots = torch.tensor([[4, 7, 7], [7, 4, 3]])
    for dtype, kv_width, sidecar_width in ((torch.bfloat16, 576, 140), (torch.uint8, C8_ROW_BYTES, 280)):
        width = kv_width + sidecar_width
        cache = (torch.arange(8 * width).reshape(8, width) % 127).to(dtype)
        for columns in (slice(kv_width), slice(kv_width, None)):
            expected = cache[slots][..., columns]
            rows = AscendHYV4FlashAttention._gather(cache, block_table, 2, positions, columns)
            torch.testing.assert_close(rows, expected, atol=0, rtol=0)
            assert rows.is_contiguous()
            single = AscendHYV4FlashAttention._gather(cache, block_table[0], 2, positions[0], columns)
            torch.testing.assert_close(single, expected[0], atol=0, rtol=0)


def test_ihc_rms_uses_global_query_boundaries_on_sequence_shards(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import ihc

    metadata = SimpleNamespace(query_start_loc=(0, 7, 11), precise_decode_reqs=(False, True))
    monkeypatch.setattr(ihc, "get_flash_attention_metadata", lambda: metadata)
    rows = torch.arange(16 * 33, dtype=torch.float32).view(16, 33) / 7
    eps = 1e-5
    means = [rows[:7].square().mean(-1, keepdim=True)]
    means.extend(row.square().mean(-1, keepdim=True) for row in rows[7:11].split(1))
    means.append(rows[11:].square().mean(-1, keepdim=True))
    expected = (torch.cat(means) + eps).rsqrt()
    shards = [ihc._inverse_rms(shard, eps, rank * 2) for rank, shard in enumerate(rows.split(2))]
    torch.testing.assert_close(torch.cat(shards), expected, rtol=0, atol=0)
