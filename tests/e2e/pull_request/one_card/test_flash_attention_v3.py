# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FA3 paged-cache and graph replay regression tests on a real Ascend device."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

pytest.importorskip("flash_attn_npu_3")

from vllm_ascend.attention import flash_attention_v3 as fa3
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.updatable_graph import UpdatableGraph

BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_REQS = 4
MAX_BLOCKS = 4
SCALE = 0.071  # Deliberately differs from 1 / sqrt(head_size).


@pytest.fixture(autouse=True)
def forward_context():
    with (
        patch.object(fa3, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch.object(fa3, "flash_attn_varlen_func", wraps=fa3.flash_attn_varlen_func) as varlen,
        patch.object(fa3, "flash_attn_with_kvcache", wraps=fa3.flash_attn_with_kvcache) as paged,
        patch.object(fa3.AscendAttentionBackendImpl, "forward_impl", side_effect=AssertionError("FIA fallback")),
        patch.object(
            fa3.AscendAttentionBackendImpl, "forward_fused_infer_attention", side_effect=AssertionError("FIA")
        ),
        patch.object(fa3.AscendAttentionBackendImpl, "forward_paged_attention", side_effect=AssertionError("PA")),
    ):
        yield SimpleNamespace(varlen=varlen, paged=paged)


def make_attention(dtype, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.device = torch.device("npu")
    builder.model_runner_type = "generate"
    builder.max_num_reqs = MAX_REQS + 1
    builder.graph_buffers = {}
    builder.scheduler_buffers = {}
    builder.scheduler_specs = {(num_heads, num_kv_heads, HEAD_SIZE, dtype, SCALE, 0.0)}
    builder.capture_sizes = {6}
    builder.block_size = BLOCK_SIZE
    impl = object.__new__(fa3.AscendFlashAttentionImpl)
    impl.num_heads = num_heads
    impl.num_kv_heads = num_kv_heads
    impl.head_size = HEAD_SIZE
    impl.scale = SCALE
    impl.logits_soft_cap = 0.0
    impl.key_cache = torch.randn(MAX_REQS * MAX_BLOCKS, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, device="npu", dtype=dtype)
    impl.value_cache = torch.randn_like(impl.key_cache)
    impl.kv_sharing_target_layer_name = None
    impl.is_kv_producer = False
    impl.pcp_enabled = False
    impl.attn_type = "decoder"
    common = SimpleNamespace(
        num_reqs=0,
        num_input_tokens=0,
        num_actual_tokens=0,
        query_start_loc=torch.zeros(MAX_REQS + 2, dtype=torch.int32, device="npu"),
        seq_lens=torch.zeros(MAX_REQS, dtype=torch.int32, device="npu"),
        block_table_tensor=torch.zeros(MAX_REQS, MAX_BLOCKS, dtype=torch.int32, device="npu"),
        slot_mapping=None,
        max_query_len=0,
        attn_state=AscendAttentionState.ChunkedPrefill,
        causal=True,
    )
    return builder, impl, common


def prepare_case(common, query_lens, context_lens, query, key, value):
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    # Non-contiguous physical pages make a wrong block-table row observable.
    pages = torch.randperm(MAX_REQS * MAX_BLOCKS).view(MAX_REQS, MAX_BLOCKS)
    seq_lens = torch.tensor(query_lens) + torch.tensor(context_lens)
    offsets = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    common.num_reqs = num_reqs
    common.num_actual_tokens = num_tokens
    common.num_input_tokens = query.shape[0]
    common.max_query_len = max(query_lens)
    common.query_start_loc.zero_()
    common.query_start_loc[: num_reqs + 1].copy_(offsets)
    common.seq_lens.zero_()
    common.seq_lens[:num_reqs].copy_(seq_lens)
    common.block_table_tensor.copy_(pages)
    slots = []
    for row, (length, context) in enumerate(zip(query_lens, context_lens)):
        for position in range(context, context + length):
            slots.append(pages[row, position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE)
    slots = torch.tensor(slots, dtype=torch.int64, device="npu")
    if common.slot_mapping is None:
        common.slot_mapping = slots
    else:
        common.slot_mapping.fill_(-1)
        common.slot_mapping[:num_tokens].copy_(slots)
    query.normal_()
    key.normal_()
    value.normal_()
    return pages, seq_lens


def reference(impl, query, pages, query_lens, seq_lens):
    expected = []
    offset = 0
    for row, (q_len, kv_len) in enumerate(zip(query_lens, seq_lens.tolist())):
        cache_pages = pages[row, : (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE].to("npu")
        k = impl.key_cache[cache_pages].flatten(0, 1)[:kv_len].float().cpu()
        v = impl.value_cache[cache_pages].flatten(0, 1)[:kv_len].float().cpu()
        q = query[offset : offset + q_len].float().cpu()
        k = k.repeat_interleave(impl.num_heads // impl.num_kv_heads, dim=1)
        v = v.repeat_interleave(impl.num_heads // impl.num_kv_heads, dim=1)
        mask = torch.arange(kv_len)[None, :] <= torch.arange(q_len)[:, None] + kv_len - q_len
        expected.append(
            F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=SCALE
            ).transpose(0, 1)
        )
        offset += q_len
    return torch.cat(expected)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 2), (16, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "query_lens,context_lens,state",
    [
        ([7, 19], [0, 0], AscendAttentionState.PrefillNoCache),
        ([1, 1, 1], [127, 128, 259], AscendAttentionState.DecodeOnly),
        ([3, 1, 7], [128, 257, 0], AscendAttentionState.ChunkedPrefill),
        ([5, 11], [256, 128], AscendAttentionState.PrefillCacheHit),
        ([4, 2, 3], [127, 256, 129], AscendAttentionState.SpecDecoding),
    ],
)
@torch.inference_mode()
def test_fa3_precision(dtype, query_lens, context_lens, state, num_heads, num_kv_heads, forward_context):
    torch.manual_seed(42)
    builder, impl, common = make_attention(dtype, num_heads, num_kv_heads)
    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, HEAD_SIZE, device="npu", dtype=dtype)
    key = torch.empty(num_tokens, num_kv_heads, HEAD_SIZE, device="npu", dtype=dtype)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    pages, seq_lens = prepare_case(common, query_lens, context_lens, query, key, value)
    common.attn_state = state
    if state == AscendAttentionState.PrefillCacheHit:
        pages[1, 0] = pages[0, 0]
        common.block_table_tensor.copy_(pages)
    metadata = builder.build(0, common)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    impl.forward(layer, query, key, value, (impl.key_cache, impl.value_cache), metadata, output)
    torch.npu.synchronize()
    expected = reference(impl, query, pages, query_lens, seq_lens)
    torch.testing.assert_close(output.float().cpu(), expected, atol=0.02, rtol=0.02)
    if state == AscendAttentionState.PrefillNoCache:
        forward_context.varlen.assert_called_once()
        forward_context.paged.assert_not_called()
    else:
        forward_context.paged.assert_called_once()
        forward_context.varlen.assert_not_called()


@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 2), (16, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fa3_graph_changes_requests_lengths_and_pages(dtype, num_heads, num_kv_heads, forward_context):
    torch.manual_seed(43)
    builder, impl, common = make_attention(dtype, num_heads, num_kv_heads)
    query = torch.empty(6, num_heads, HEAD_SIZE, device="npu", dtype=dtype)
    key = torch.empty(6, num_kv_heads, HEAD_SIZE, device="npu", dtype=dtype)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    prepare_case(common, [2, 2, 2], [128, 256, 127], query, key, value)
    metadata = builder.build(0, common)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    caches = (impl.key_cache, impl.value_cache)
    for _ in range(3):
        impl.forward(layer, query, key, value, caches, metadata, output)
    torch.npu.synchronize()
    graph = UpdatableGraph()
    with patch.object(fa3, "_EXTRA_CTX", SimpleNamespace(capturing=True)), torch.npu.graph(graph):
        impl.forward(layer, query, key, value, caches, metadata, output)
    # FA3 replay must not register any host-side per-layer update tasks.
    assert not graph.tasks
    for query_lens, context_lens in [
        ([1, 1, 4], [129, 257, 128]),
        ([3, 3], [254, 126]),
        ([6], [256]),
        ([2], [127]),
    ]:
        pages, seq_lens = prepare_case(common, query_lens, context_lens, query, key, value)
        builder.build(0, common)
        graph.replay()
        torch.npu.synchronize()
        expected = reference(impl, query, pages, query_lens, seq_lens)
        torch.testing.assert_close(output[: sum(query_lens)].float().cpu(), expected, atol=0.02, rtol=0.02)
    # Three warmups and one capture execute the real FA3 Python wrapper;
    # subsequent replays execute the captured NPU kernels directly.
    assert forward_context.paged.call_count == 4
    forward_context.varlen.assert_not_called()
