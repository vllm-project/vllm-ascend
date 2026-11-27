# SPDX-License-Identifier: Apache-2.0
"""Synthetic QSA tests; run only on a reserved Ascend 910B/A3 device."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

enable_custom_op()

from vllm_ascend.models.qwen4_exp.c8_attention import qsa_c8_attention  # noqa: E402
from vllm_ascend.models.qwen4_exp.lightning_indexer import (  # noqa: E402
    qsa_select_paged_tokens_lightning_v2,
    qsa_select_paged_tokens_quantized,
)
from vllm_ascend.models.qwen4_exp.quantization import (  # noqa: E402
    qsa_c8_indexer_cache_views,
    quantize_qsa_rows,
    quantize_qsa_values,
)
from vllm_ascend.ops.triton.qwen4_exp.c8 import (  # noqa: E402
    prepare_qli_metadata_inputs,
    prepare_qli_query,
    quantize_qsa_query,
    store_qsa_c8_rows,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _initialize_triton():
    init_device_properties_triton()


@pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.LIGHTNING_INDEXER_V2_PAGED_KEY_OFFSETS),
    reason="Requires the A2 paged LightningIndexerV2 kernel",
)
@torch.inference_mode()
def test_bf16_lightning_v2_paged_ragged_requests_graph_replay():
    torch.manual_seed(31)
    query_lengths = [4, 1, 6, 3, 0, 2]
    lengths = [4, 5203, 131071, 8202, 0, 2689]
    rows = sum(query_lengths)
    heads, head_dim, page_size, page_count = 4, 128, 128, 256
    query = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="npu")
    # Offset and page stride both differ from a standalone contiguous cache.
    key = torch.randn(page_count * 2, page_size, 1, head_dim, dtype=torch.bfloat16, device="npu")[1::2]
    page_ids = torch.arange(page_count, dtype=torch.int32, device="npu")
    request_ids = torch.arange(len(lengths), dtype=torch.int64).repeat_interleave(torch.tensor(query_lengths)).npu()
    query_start_loc = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32).npu()

    def positions_for(lengths):
        positions = [pos for length, count in zip(lengths, query_lengths) for pos in range(length - count, length)]
        return torch.tensor(positions, device="npu", dtype=torch.int64)

    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        block_table=torch.stack([torch.roll(page_ids, request * 37) for request in range(len(lengths))]),
        token_to_req=request_ids,
        logical_positions=positions_for(lengths),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
        query_start_loc=query_start_loc,
    )

    def run(out=None):
        return qsa_select_paged_tokens_lightning_v2(query, key, metadata, 2048, 4, out)

    def check(output):
        # BSND evaluates each row separately using explicit causal visibility.
        visible = ((metadata.logical_positions + 1) // 4).to(torch.int32)
        weights = torch.full((rows, 1, heads), 1 / math.sqrt(head_dim), dtype=torch.bfloat16, device="npu").float()
        groups, _ = torch.ops.cann_ops_transformer.lightning_indexer(
            query.unsqueeze(1),
            key,
            weights,
            512,
            seqused_k=visible,
            block_table=metadata.block_table[request_ids],
            max_seqlen_q=1,
            layout_q="BSND",
            layout_k="PA_BBND",
        )
        torch.testing.assert_close(output[:, :512], groups[:, 0, 0], rtol=0, atol=0)
        assert (output[:, 515:] == -1).all().item()

    output = torch.empty(rows, 2051, dtype=torch.int32, device="npu")
    check(run(output))
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run(output)

    lengths = [5, 5200, 131072, 8199, 0, 2694]
    query.mul_(0.5)
    metadata.logical_positions.copy_(positions_for(lengths))
    metadata.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
    graph.replay()
    torch.npu.synchronize()
    check(output)
    torch.testing.assert_close(output, run(), rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("indexer", [False, True])
def test_c8_store_rounding_strides_and_invalid_slots(indexer):
    torch.manual_seed(1)
    raw = torch.full((4, 2, 128, 1, 132 if indexer else 128), 19, dtype=torch.int8, device="npu")
    cache = raw[:, 1]
    rows = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, 127, 128, 511, -1, 512], dtype=torch.int64, device="npu")
    expected_codes, expected_scales = quantize_qsa_rows(rows)
    if indexer:
        store_qsa_c8_rows(cache, slots, rows, indexer=True)
        codes, scales = qsa_c8_indexer_cache_views(cache)
    else:
        key_scales = torch.full((4, 1, 128), 0.25, device="npu")
        store_qsa_c8_rows(cache, slots, rows, key_scales=key_scales)
        codes, scales = cache, key_scales.transpose(1, 2)
        assert (scales[2].cpu() == 0.25).all()
    for row, slot in enumerate([0, 127, 128, 511]):
        torch.testing.assert_close(codes[slot // 128, slot % 128].cpu(), expected_codes[row].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(scales[slot // 128, slot % 128].cpu(), expected_scales[row].to(scales.dtype).cpu())
    assert (raw[:, 0].cpu() == 19).all()
    assert (cache[2].cpu() == 19).all()


def _dense_reference(query, key_cache, value_cache, key_scales, table, lengths, v_scale):
    codes_q, scales_q = quantize_qsa_rows(query.cpu())
    q = codes_q.float() * scales_q.unsqueeze(-1)
    kc, vc = key_cache.cpu(), value_cache.cpu()
    ks = key_scales.cpu().transpose(1, 2)
    expected = torch.zeros_like(q)
    repeats = query.shape[1] // kc.shape[2]
    for row, length in enumerate(lengths):
        token = torch.arange(length)
        page = table[row, token // 128].long().cpu()
        keys = (kc[page, token % 128].float() * ks[page, token % 128].unsqueeze(-1)).repeat_interleave(repeats, 1)
        values = (vc[page, token % 128].float() * v_scale.cpu()).repeat_interleave(repeats, 1)
        score = torch.einsum("hd,shd->hs", q[row], keys) / query.shape[-1] ** 0.5
        exp = (score - score.amax(-1, keepdim=True)).exp()
        expected[row] = torch.einsum("hs,shd->hd", (exp * 127).round() / 127, values) / exp.sum(-1, keepdim=True)
    return expected


@torch.inference_mode()
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("row_count", [2, 129])
@pytest.mark.parametrize("kv_heads,query_heads", [(2, 4), (1, 3)])
def test_c8_attention_eager_and_changed_input_graph_replay(dim, row_count, kv_heads, query_heads):
    torch.manual_seed(2)
    query = torch.randn(row_count, query_heads, dim, dtype=torch.bfloat16, device="npu")
    # Preserve the page stride imposed by the shared K/V allocation.
    backing = torch.zeros(4, 2, 128, kv_heads, dim, dtype=torch.int8, device="npu")
    key_cache, value_cache = backing[:, 0], backing[:, 1]
    key_scales = torch.empty(4, kv_heads, 128, dtype=torch.float32, device="npu")
    slots = torch.arange(512, dtype=torch.int64, device="npu")
    v_scale = torch.rand(kv_heads, dim, device="npu") * 0.01 + 0.01
    p_scale = torch.tensor([127.0], device="npu")
    store_qsa_c8_rows(
        key_cache,
        slots,
        torch.randn(512, kv_heads, dim, dtype=torch.bfloat16, device="npu"),
        key_scales=key_scales,
    )
    store_qsa_c8_rows(
        value_cache, slots, torch.randn(512, kv_heads, dim, dtype=torch.bfloat16, device="npu"), v_scale=v_scale
    )
    row_indices = torch.arange(row_count, dtype=torch.int32, device="npu")
    table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device="npu")[row_indices.long() % 2]
    requests = row_indices
    initial_lengths = [131 + row % 7 for row in range(row_count)]
    lengths = torch.tensor(initial_lengths, dtype=torch.int32, device="npu")
    positions = lengths - 1
    compact = torch.full((row_count, 515), -1, dtype=torch.int32, device="npu")
    compact[:, :34] = torch.arange(34, dtype=torch.int32, device="npu")
    output = torch.empty_like(query)

    def run():
        return qsa_c8_attention(
            query,
            key_cache,
            value_cache,
            compact,
            table,
            requests,
            positions,
            lengths,
            output,
            v_scale,
            p_scale,
            key_scales=key_scales,
        )

    expected = _dense_reference(query, key_cache, value_cache, key_scales, table, initial_lengths, v_scale)
    torch.testing.assert_close(run().float().cpu(), expected, rtol=2e-2, atol=2e-2)
    # Warm up compilation and ACLNN tiling before entering graph capture.
    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # Static shapes and addresses, changed query, visibility and physical pages.
    query.mul_(0.75)
    table.copy_(torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device="npu")[row_indices.long() % 2])
    replay_lengths = [129 + row % 5 for row in range(row_count)]
    lengths.copy_(torch.tensor(replay_lengths, dtype=torch.int32, device="npu"))
    positions.copy_(lengths - 1)
    graph.replay()
    torch.npu.synchronize()
    expected = _dense_reference(query, key_cache, value_cache, key_scales, table, replay_lengths, v_scale)
    torch.testing.assert_close(output.float().cpu(), expected, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize("rows,length", [(512, 512), (4096, 8195)])
def test_c8_prefill_grouped_attention_matches_independent_rows(rows, length):
    torch.manual_seed(34)
    pages = (length + 127) // 128
    kv = torch.randint(-127, 128, (pages + 4, 2, 128, 1, 256), dtype=torch.int8, device="npu")
    key, value = kv.unbind(1)
    table = torch.randperm(pages + 4, dtype=torch.int32, device="npu").unsqueeze(0)
    key_scales = torch.rand(pages + 4, 1, 128, device="npu") * 0.01 + 0.001
    v_scale = torch.rand(1, 256, device="npu") * 0.01 + 0.001
    p_scale = torch.tensor([127.0], device="npu")
    query = torch.randn(rows, 3, 256, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(length - rows, length, dtype=torch.int32, device="npu")
    lengths = torch.tensor([length], dtype=torch.int32, device="npu")
    requests = torch.zeros(rows, dtype=torch.int32, device="npu")
    compact = torch.full((rows, 515), -1, dtype=torch.int32, device="npu")
    groups = (positions + 1).floor_divide(4).unsqueeze(1)
    columns = torch.arange(512, dtype=torch.int32, device="npu").unsqueeze(0)
    selected = (columns + positions.unsqueeze(1) * 23) % groups.clamp_min(1)
    compact[:, :512] = torch.where(columns < groups, selected, -1)

    def run(prefill_kv_length):
        return qsa_c8_attention(
            query,
            key,
            value,
            compact,
            table,
            requests,
            positions,
            lengths,
            torch.empty_like(query),
            v_scale,
            p_scale,
            key_scales=key_scales,
            prefill_kv_length=prefill_kv_length,
        )

    torch.testing.assert_close(run(length), run(None), rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("width", [1, 509, 1024])
def test_c8_qli_metadata_preparation_capture_tracks_replay_inputs(width):
    block_table = torch.arange(4 * width * 2, dtype=torch.int32, device="npu").view(4, -1)[:, ::2]
    query_positions = torch.tensor([-1, 1, 3, 15, 131, 65535], device="npu", dtype=torch.int64)
    request_ids = torch.tensor([0, 2, 1, 2, -1, 3], device="npu", dtype=torch.int32)
    lengths = torch.tensor([0, 3, 21, 65533], device="npu", dtype=torch.int32)

    def check(results):
        owners = request_ids.cpu().long()
        expected = (
            torch.arange(7, dtype=torch.int32),
            torch.minimum((query_positions.cpu() + 1).clamp_min(0) // 4, lengths.cpu()[owners] // 4).int(),
            block_table.cpu()[owners].contiguous(),
        )
        for actual, reference in zip(results, expected, strict=True):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def prepare():
        return prepare_qli_metadata_inputs(block_table, request_ids, query_positions, lengths, 4)

    check(prepare())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = prepare()
    request_ids.copy_(torch.tensor([1, 0, 2, 3, 2, -2], dtype=torch.int32, device="npu"))
    lengths.add_(13)
    query_positions.sub_(7)
    block_table.add_(4)
    graph.replay()
    check(captured)


@torch.inference_mode()
def test_c8_indexer_matches_all_visible_group_set():
    # With <512 visible groups, all complete groups must be selected exactly
    # once regardless of floating-point score rounding or tie order.
    query = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(4, 32, 1, 132, dtype=torch.int8, device="npu")
    store_qsa_c8_rows(
        cache,
        torch.arange(128, dtype=torch.int64, device="npu"),
        torch.randn(128, 1, 128, dtype=torch.bfloat16, device="npu"),
        indexer=True,
    )
    selected = qsa_select_paged_tokens_quantized(
        query,
        cache,
        torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device="npu"),
        torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        torch.tensor([130, 136], dtype=torch.int32, device="npu"),
        torch.tensor([131, 137], dtype=torch.int32, device="npu"),
        torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
        2048,
        4,
    ).cpu()
    for row, groups in enumerate((32, 34)):
        torch.testing.assert_close(selected[row, :groups].sort().values, torch.arange(groups, dtype=torch.int32))
        assert (selected[row, groups:512] == -1).all()


@pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.INT8_QLI_V2_REQUEST_BATCHED_DECODE),
    reason="Requires the A3 request-batched INT8 QLI v2 kernel",
)
@torch.inference_mode()
def test_c8_mtp_indexer_request_batch_matches_singletons_after_replay():
    # Each physical key row has a unique, exactly representable positive score.
    # Distinct page orders catch accidental reuse of another request's table.
    width, pages, page_size = 4, 20, 32
    query = torch.ones(16, 4, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(pages, page_size, 1, 132, dtype=torch.int8, device="npu")
    key, scales = qsa_c8_indexer_cache_views(cache)
    key.fill_(1)
    scales.copy_(torch.arange(1, pages * page_size + 1, device="npu").reshape_as(scales))
    page_ids = torch.arange(pages, dtype=torch.int32, device="npu")
    table = torch.stack([page_ids.roll(i * 5) for i in range(4)])
    lengths = torch.tensor([2300, 2301, 2302, 2303], dtype=torch.int32, device="npu")
    boundaries = torch.arange(0, 17, width, dtype=torch.int32, device="npu")
    owners = torch.arange(4, dtype=torch.int32, device="npu").repeat_interleave(width)
    offsets = torch.arange(width, dtype=torch.int32, device="npu").repeat(4) - width
    positions = lengths[owners.long()] + offsets

    def run(decode_width):
        return qsa_select_paged_tokens_quantized(
            query,
            cache,
            table,
            owners,
            positions,
            lengths,
            boundaries,
            2048,
            4,
            decode_width=decode_width,
        )

    def check(actual):
        reference = run(0).cpu()
        actual = actual.cpu()
        torch.testing.assert_close(actual[:, :512].sort().values, reference[:, :512].sort().values)
        torch.testing.assert_close(actual[:, 512:], reference[:, 512:])
        # Check the selected set independently of both QLI schedules.
        for row, pos in enumerate(positions.tolist()):
            if pos < 0:
                assert (actual[row] == -1).all()
                continue
            groups = torch.arange((pos + 1) // 4)
            scores = table[row // width].cpu()[groups // page_size] * page_size + groups % page_size
            expected = groups[scores.topk(512).indices].int().sort().values
            torch.testing.assert_close(actual[row, :512].sort().values, expected)

    check(run(width))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run(width)
    # Keep the graph shape but shorten the live batch and change page ownership.
    boundaries[-1:].copy_(boundaries[-2:-1])
    lengths.copy_(torch.tensor([2423, 2422, 2421, 0], dtype=torch.int32, device="npu"))
    positions.copy_(lengths[owners.long()] + offsets)
    positions[-width:].fill_(-1)
    table.copy_(table.roll(1, 0))
    graph.replay()
    check(captured)


@torch.inference_mode()
def test_c8_static_v_store_channel_scales_and_saturation():
    cache = torch.full((3, 128, 2, 256), 37, dtype=torch.int8, device="npu")
    rows = torch.randn(5, 2, 256, dtype=torch.bfloat16, device="npu") * 100
    scales = torch.rand(2, 256, device="npu") * 0.1 + 0.01
    slots = torch.tensor([0, 127, 128, -1, 384], dtype=torch.int64, device="npu")
    store_qsa_c8_rows(cache, slots, rows, v_scale=scales)
    expected = quantize_qsa_values(rows, scales)
    for row, slot in enumerate((0, 127, 128)):
        torch.testing.assert_close(cache[slot // 128, slot % 128].cpu(), expected[row].cpu(), atol=0, rtol=0)
    assert (cache[2].cpu() == 37).all()


@pytest.mark.parametrize("heads, dim", [(8, 128), (3, 256)])
@pytest.mark.parametrize("tokens", [3, 96, 4096])
@torch.inference_mode()
def test_c8_query_fusion_matches_reference(heads, dim, tokens):
    generator = torch.Generator().manual_seed(84359)
    query = torch.randn(tokens, dim, heads, generator=generator).transpose(1, 2)
    query[0].zero_()
    query[1].mul_(1e-9)
    query[2, :, :6] = torch.tensor([127.0, -127.0, 0.5, 1.5, -0.5, -1.5])
    query = query.to(device="npu", dtype=torch.bfloat16)
    assert not query.is_contiguous()
    torch.testing.assert_close(quantize_qsa_query(query), quantize_qsa_rows(query), rtol=0, atol=0)


@pytest.mark.parametrize("heads, dim", [(8, 128), (3, 256)])
@torch.inference_mode()
def test_c8_query_fusion_graph_replay_matches_reference(heads, dim):
    torch.manual_seed(17)
    query = torch.randn(96, heads, dim, device="npu", dtype=torch.bfloat16)
    quantize_qsa_query(query)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = quantize_qsa_query(query)

    query.mul_(0.25)
    query[0].zero_()
    query[1, :, :6] = torch.tensor([127.0, -127.0, 0.5, 1.5, -0.5, -1.5], device="npu", dtype=torch.bfloat16)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, quantize_qsa_rows(query), rtol=0, atol=0)


@pytest.mark.parametrize("tokens,heads", [(0, 4), (1, 4), (4, 4), (96, 32), (4096, 4), (4, 64)])
@torch.inference_mode()
def test_c8_qli_query_padding_and_graph_replay(tokens, heads):
    query = torch.randn(tokens, 128, heads, dtype=torch.bfloat16, device="npu").transpose(1, 2)

    def check(outputs):
        codes, scales, weights = outputs
        ref_codes, ref_scales = quantize_qsa_rows(query)
        assert codes.shape == (tokens, 64, 128) and codes.dtype == torch.int8
        assert scales.shape == weights.shape == (tokens, 64) and scales.dtype == weights.dtype == torch.float16
        torch.testing.assert_close(codes[:, :heads], ref_codes, rtol=0, atol=0)
        torch.testing.assert_close(scales[:, :heads], ref_scales.half(), rtol=0, atol=0)
        torch.testing.assert_close(weights[:, :heads], torch.full_like(ref_scales.half(), 1 / math.sqrt(128)))
        assert not codes[:, heads:].any().item()
        assert not weights[:, heads:].any().item()
        assert (scales[:, heads:] == 1).all().item()

    check(prepare_qli_query(query))
    if tokens:
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = prepare_qli_query(query)
        query[0].zero_()
        query[-1, :, :6] = torch.tensor([127, -127, 0.5, 1.5, -0.5, -1.5], device="npu", dtype=torch.bfloat16)
        graph.replay()
        torch.npu.synchronize()
        check(captured)
