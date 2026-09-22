# SPDX-License-Identifier: Apache-2.0
"""Prefill chunk planning and mixed decode device metadata regressions."""

import ast
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "vllm_ascend/attention/mla_prefill.py"
spec = importlib.util.spec_from_file_location("flash_mla_prefill_test_module", MODULE_PATH)
prefill_ops = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prefill_ops
spec.loader.exec_module(prefill_ops)


def load_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)


@pytest.mark.parametrize("is_c8", [False, True])
@pytest.mark.parametrize(
    "decodes,prefills,dcp,pcp,has_context,causal,enabled,expect_full",
    [
        (1, 0, False, False, True, True, True, False),
        (64, 0, False, False, True, True, True, False),
        (64, 0, True, False, True, True, True, False),
        (1, 0, True, False, False, True, True, False),
        (0, 1, True, False, True, True, True, False),
        (1, 1, True, False, True, True, True, False),
        (0, 1, False, True, True, True, True, False),
        (0, 1, False, False, False, True, True, False),
        (0, 1, False, False, True, False, True, False),
        (0, 1, False, False, True, True, False, False),
        (0, 1, False, False, True, True, True, True),
        (1, 1, False, False, True, True, True, True),
    ],
)
def test_builder_preserves_decode_and_context_parallel_fallback(
    is_c8, decodes, prefills, dcp, pcp, has_context, causal, enabled, expect_full
):
    path = ROOT / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    builder_class = next(node for node in tree.body if getattr(node, "name", None) == "AscendMLAMetadataBuilder")
    build = next(node for node in builder_class.body if getattr(node, "name", None) == "build")
    native_calls, full_calls = [], []
    original_flash, full_metadata, decode_flash = object(), object(), object()
    decode_tokens = decodes * 4
    requests = decodes + prefills
    tokens = decode_tokens + prefills * 128
    common = SimpleNamespace(
        positions=torch.arange(tokens),
        block_table_tensor=torch.zeros(requests, 8, dtype=torch.int32),
        seq_lens=torch.full((requests,), 131072),
        max_seq_len=131072,
        causal=causal,
        num_computed_prefill_tokens_cpu=torch.zeros(requests) if has_context else None,
        num_actual_tokens=tokens,
        num_input_tokens=tokens,
        slot_mapping=torch.arange(tokens),
        query_start_loc=torch.arange(requests + 1),
        attn_state=object(),
    )

    def native_metadata(builder, received, **kwargs):
        assert received is common and builder.flash_c8_prefill is is_c8
        native_calls.append(kwargs)
        return original_flash

    def full_prefill(received, flash, num_decodes, num_decode_tokens):
        assert received is common and flash is original_flash
        full_calls.append((num_decodes, num_decode_tokens))
        return full_metadata, decode_flash

    def split(received, **kwargs):
        assert received is common
        assert kwargs == {"decode_threshold": 4, "treat_short_extends_as_decodes": True}
        return decodes, prefills, decode_tokens, tokens - decode_tokens

    scope = {
        "envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "split_decodes_and_prefills": split,
        "_build_flash_attention_metadata": native_metadata,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(build), str(path), "exec"), scope)
    builder = SimpleNamespace(
        decode_threshold=4,
        decode_metadata_cls=SimpleNamespace,
        metadata_cls=SimpleNamespace,
        flash_unabsorbed_prefill=enabled,
        flash_c8_prefill=is_c8,
        dcp_enabled=dcp,
        pcp_enabled=pcp,
        _build_full_flash_prefill=full_prefill,
    )
    result = scope["build"](builder, 0, common)
    assert native_calls == [{"is_mla": True, "allow_unabsorbed": not expect_full, "metadata_only": expect_full}]
    assert result.flash is original_flash
    assert full_calls == ([(decodes, decode_tokens)] if expect_full else [])
    assert result.flash_full_prefill is (full_metadata if expect_full else None)
    assert result.flash_decode is (decode_flash if expect_full else None)
    if decodes:
        torch.testing.assert_close(result.decode.seq_lens, common.seq_lens[:decodes])
        torch.testing.assert_close(result.decode.input_positions, common.positions[:decode_tokens])
    else:
        assert result.decode is None


@pytest.mark.parametrize(
    "contexts,budget", [([0, 0, 0], 128), ([0, 127, 129, 515], 512), ([131071, 129761], 16384), ([500] * 7, 256)]
)
def test_prefix_partition_preserves_all_tokens_and_memory_bound(contexts, budget):
    lengths = tuple(5 + i for i in range(len(contexts)))
    offsets = [0, 4, 8]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    plan = prefill_ops.build_prefill_plan(offsets, 2, contexts, budget)
    assert plan.num_decode_tokens == 8
    assert plan.query_lengths == lengths
    covered = [[] for _ in contexts]
    for chunk in plan.chunks:
        assert 0 < chunk.num_tokens <= budget
        assert all(start % 128 == 0 for start in chunk.starts)
        assert all(length > 0 for length in chunk.lengths)
        assert chunk.query_lengths == tuple(lengths[i] for i in chunk.requests)
        for req, start, length in zip(chunk.requests, chunk.starts, chunk.lengths):
            covered[req].extend(range(start, start + length))
    assert covered == [list(range(length)) for length in contexts]
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"))
    assert metadata.current.query_cu[-1].item() == sum(lengths)


def test_lse_layout_preserves_query_and_head_order():
    lse = torch.arange(15).reshape(5, 3, 1).float()
    torch.testing.assert_close(prefill_ops.canonical_lse(lse, 5, 3), lse)
    torch.testing.assert_close(prefill_ops.canonical_lse(lse[..., 0].T, 5, 3), lse)


def test_equal_history_chunks_share_readonly_metadata_within_batch():
    calls = []

    def schedule(call):
        calls.append(call)
        return torch.tensor([len(calls)])

    plan = prefill_ops.build_prefill_plan([0, 5, 14], 0, [131072, 131072], 256)
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"), schedule)
    assert len(metadata.history) == 1024
    assert len(calls) == 2  # One history shape and the causal current chunk.
    first = metadata.history[0]
    assert all(chunk.call is first.call for chunk in metadata.history)
    assert metadata.current is not first.call
    assert metadata.history[-1].starts.tolist() == [130944, 130944]
    assert metadata.history[-1].starts is not first.starts


def test_history_metadata_cache_distinguishes_ragged_lengths_and_batches():
    calls = []

    def schedule(call):
        calls.append(call)
        return torch.tensor([len(calls)])

    plan = prefill_ops.build_prefill_plan([0, 5, 14], 0, [385, 257], 256)
    first = prefill_ops.prepare_metadata(plan, torch.device("cpu"), schedule)
    assert first.history[0].call is first.history[1].call
    assert first.history[2].call.kv_lengths == (128, 1)
    assert first.history[3].call.kv_lengths == (1,)
    assert first.history[2].call is not first.history[0].call
    assert first.history[3].call is not first.history[2].call
    assert len(calls) == 4
    second = prefill_ops.prepare_metadata(plan, torch.device("cpu"), schedule)
    assert len(calls) == 8
    assert second.current is not first.current
    for before, after in zip(first.history, second.history):
        assert before.call is not after.call
        assert before.call.schedule is not after.call.schedule


def test_current_and_history_do_not_share_different_mask_modes():
    plan = prefill_ops.build_prefill_plan([0, 128], 0, [128], 128)
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"))
    history = metadata.history[0].call
    assert history.query_lengths == metadata.current.query_lengths
    assert history.kv_lengths == metadata.current.kv_lengths
    assert history.mask_mode == 0 and metadata.current.mask_mode == 3
    assert history is not metadata.current


def test_reject_decode_lengths_in_prefill_contract():
    with pytest.raises(ValueError, match="metadata mismatch"):
        prefill_ops.build_prefill_plan([0, 4, 8, 13], 2, [9999, 9999, 129], 128)


@pytest.mark.parametrize("is_c8", [False, True])
def test_mixed_decode_uses_live_device_lengths_not_prefill_cpu_counts(is_c8):
    scope = dict(
        torch=torch,
        dataclass=dataclass,
        replace=replace,
        cdiv=lambda a, b: (a + b - 1) // b,
        build_prefill_plan=prefill_ops.build_prefill_plan,
        prepare_metadata=prefill_ops.prepare_metadata,
        native_flash_adapters=lambda *a, **kw: (lambda call: None, None),
        _flash_attention_schedule=lambda builder, flash, **kw: torch.stack(
            (flash.used_q.sum(), flash.cache_lens.sum())
        ),
        DeviceMetadataStage=SimpleNamespace(ATTENTION=1),
        DeviceMetadataTask=lambda stage, fn, group: SimpleNamespace(fn=fn, group=group),
    )
    load_functions(ROOT / "vllm_ascend/attention/flash_metadata.py", {"AscendFlashAttentionMetadata"}, scope)
    load_functions(ROOT / "vllm_ascend/attention/mla_v1.py", {"_build_full_flash_prefill"}, scope)
    builder = SimpleNamespace(
        flash_num_heads=12,
        _flash_attn_mask=None,
        device=torch.device("cpu"),
        decode_threshold=4,
        chunked_prefill_workspace_size=131072,
        _device_metadata_enabled=True,
        _device_metadata_tasks=(),
    )
    common = SimpleNamespace(
        num_reqs=5,
        num_actual_tokens=170,
        query_start_loc_cpu=torch.tensor([0, 4, 8, 41, 170, 170]),
        num_computed_prefill_tokens_cpu=torch.tensor([-999, -999, 129762, 0, -999]),
    )
    flash = scope["AscendFlashAttentionMetadata"](
        query=torch.empty(0, 12, 576),
        schedule=torch.empty(0),
        cu=torch.tensor([0, 4, 8, 41, 170, 170]),
        used_q=torch.tensor([4, 4, 33, 129, 0]),
        cache_lens=torch.tensor([100, 200, 129795, 129, 0]),
        block_table=torch.zeros((5, 1024), dtype=torch.int32),
        slots=torch.empty(176, dtype=torch.int64),
        live_boundaries=torch.empty(177),
        token_live=torch.empty(176, dtype=torch.bool),
        positions=torch.empty(176),
        attn_mask=None,
        max_query_len=129,
        max_seq_len=131072,
        is_prefill=True,
        is_c8=is_c8,
    )
    prefill, decode = scope["_build_full_flash_prefill"](builder, common, flash, 2, 8)
    assert prefill.plan.context_lengths == (129762, 0)
    assert prefill.plan.query_lengths == (33, 129)
    assert prefill.plan.workspace_tokens == 16384
    assert max(chunk.num_tokens for chunk in prefill.plan.chunks) <= 16384
    assert decode.query.shape == (8, 12, 576)
    assert decode.block_table.shape == (2, 1024)
    flash.used_q[:2].copy_(torch.tensor([2, 1]))
    flash.cache_lens[:2].copy_(torch.tensor([98, 197]))
    flash.cu[:3].copy_(torch.tensor([0, 2, 3]))
    flash.token_live[:8].copy_(torch.tensor([True, True, True, False, False, False, False, False]))
    builder._device_metadata_tasks[-1].fn()
    assert decode.schedule.tolist() == [3, 292 if is_c8 else 295]
    if is_c8:
        assert decode.split_kv and not decode.unabsorbed
        assert decode.current_cache.shape == (3, 1, 128, 576)
        assert decode.cache_lens.tolist() == [96, 196]
        assert decode.current_slots.tolist() == [0, 1, 128, -1, -1, -1, -1, -1]
        assert decode.current_block_table[:, 0].tolist() == [0, 1]
    assert prefill.plan.context_lengths == (129762, 0)


@pytest.mark.parametrize("is_c8", [False, True])
def test_full_prefill_avoids_absorbed_query_allocation_and_schedule(is_c8):
    def unexpected_schedule(*args, **kwargs):
        raise AssertionError("full prefill should only build expanded chunk schedules")

    scope = dict(torch=torch, dataclass=dataclass, _flash_attention_schedule=unexpected_schedule)
    load_functions(
        ROOT / "vllm_ascend/attention/flash_metadata.py",
        {"AscendFlashAttentionMetadata", "_build_flash_attention_metadata"},
        scope,
    )
    builder = SimpleNamespace(
        flash_num_heads=12,
        flash_num_kv_heads=1,
        flash_is_c8=is_c8,
        flash_unabsorbed_prefill=True,
        decode_threshold=4,
        device=torch.device("cpu"),
        _flash_attn_mask=None,
        _flash_buffers={},
        kv_cache_spec=SimpleNamespace(dtype=torch.bfloat16, block_size=128),
        kernel_block_size=128,
        _device_metadata_enabled=False,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=15959,
        num_input_tokens=15960,
        block_table_tensor=torch.zeros((1, 128), dtype=torch.int32),
        causal=True,
        max_query_len=15959,
        query_start_loc=torch.tensor([0, 15959]),
        seq_lens=torch.tensor([15959]),
        slot_mapping=torch.arange(15960),
        positions=torch.arange(15960),
    )
    metadata = scope["_build_flash_attention_metadata"](
        builder,
        common,
        is_mla=True,
        allow_unabsorbed=False,
        metadata_only=True,
    )
    assert metadata.query.numel() == metadata.schedule.numel() == 0
    assert metadata.current_cache is None
    assert metadata.current_schedule is None
    assert not metadata.split_kv
    assert metadata.cache_lens.tolist() == [15959, 0]
    assert metadata.slots[-1].item() == -1
    assert metadata.token_live.sum().item() == 15959


@pytest.mark.parametrize("contexts", [(0, 0), (127, 129), (257, 1), (0, 385)])
@pytest.mark.parametrize("scale", [0.03125, 0.25])
@pytest.mark.parametrize("fused", [False, True])
def test_c8_full_prefill_matches_expanded_reference_with_strided_prefix(contexts, scale, fused):
    """Compare ragged C8 history expansion and chunk merging with a dense BF16 oracle."""
    torch.manual_seed(311)
    heads, lengths, budget = 2, (5, 7), 128
    tokens, pages = sum(lengths), 8
    query = (torch.randn(tokens, heads, 192) * 0.15).to(torch.bfloat16)
    current_latent = (torch.randn(tokens, 1, 512) * 0.25).to(torch.bfloat16)
    current_rope = (torch.randn(tokens, 1, 64) * 0.15).to(torch.bfloat16)
    latent_storage = torch.randn(pages, 3, 128, 1, 512).to(torch.float8_e4m3fn)
    rope_storage = (torch.randn(pages, 2, 128, 1, 64) * 0.15).to(torch.bfloat16)
    cache = latent_storage[:, 1], rope_storage[:, 1]
    original_latent, original_rope = latent_storage.float().clone(), rope_storage.clone()
    table = torch.tensor([[6, 2, 7, 3], [5, 1, 4, 0]], dtype=torch.int32)
    weights = torch.randn(512, heads * 256) * 0.015
    plan = prefill_ops.build_prefill_plan([0, lengths[0], tokens], 0, contexts, budget)
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"))
    projected_dtypes, gathered = [], []

    def project(latent):
        projected_dtypes.append(latent.dtype)
        return (latent.float() @ weights).to(torch.bfloat16), None

    def gather(kc, kr, blocks, used, starts, *, key, value):
        assert kc is cache[0] and kr is cache[1]
        assert not kc.is_contiguous() and not kr.is_contiguous()
        assert key.dtype == torch.float8_e4m3fn and value.dtype == torch.bfloat16
        offset = 0
        for request, (length, start) in enumerate(zip(used.tolist(), starts.tolist())):
            locations = torch.arange(start, start + length)
            physical = blocks[request, locations // 128].long()
            key[offset : offset + length].copy_(kc.float()[physical, locations % 128])
            value[offset : offset + length].copy_(kr[physical, locations % 128])
            offset += length
        gathered.append(offset)

    def attention(q, kn, v, kr, call):
        k = torch.cat((kn, kr.expand(-1, heads, -1)), -1).float()
        out = torch.empty(q.shape[0], heads, 128)
        lse = torch.empty(heads, q.shape[0])
        q_start = k_start = 0
        for nq, nk in zip(call.query_lengths, call.kv_lengths):
            logits = torch.einsum("thd,shd->hts", q[q_start : q_start + nq].float(), k[k_start : k_start + nk])
            logits *= 192**-0.5
            if call.mask_mode:
                logits.masked_fill_(torch.ones(nq, nk, dtype=torch.bool).triu(1), -float("inf"))
            out[q_start : q_start + nq] = torch.einsum(
                "hts,shd->thd", logits.softmax(-1), v[k_start : k_start + nk].float()
            )
            lse[:, q_start : q_start + nq] = torch.logsumexp(logits, -1)
            q_start, k_start = q_start + nq, k_start + nk
        return out, lse

    def update(lses, outputs, mode):
        assert mode == 1
        lse = torch.logaddexp(*lses)
        output = sum((part_lse - lse).exp()[:, None] * part for part_lse, part in zip(lses, outputs))
        return output, lse

    def fused_gather(kc, kr, blocks, cumulative, used, starts, descale, *, num_tokens, max_seq_len):
        assert num_tokens == int(cumulative[-1]) and max_seq_len == int(used.max())
        latent = torch.empty(num_tokens, 1, 512, dtype=torch.float8_e4m3fn)
        rope = torch.empty(num_tokens, 1, 64, dtype=torch.bfloat16)
        gather(kc, kr, blocks, used, starts, key=latent, value=rope)
        return (latent.float() * descale).to(torch.bfloat16), rope

    actual, actual_lse = prefill_ops.full_prefill(
        query[..., :128],
        query[..., 128:],
        current_latent,
        current_rope,
        cache,
        table,
        metadata,
        project,
        attention,
        gather,
        update,
        query=query,
        cache_scale=torch.tensor([scale]),
        history_load=fused_gather if fused else None,
    )
    expected, expected_lse = [], []
    start = 0
    for request, (nq, context) in enumerate(zip(lengths, contexts)):
        positions = torch.arange(context)
        physical = table[request, positions // 128].long()
        history = (cache[0].float()[physical, positions % 128] * scale).to(torch.bfloat16)
        latent = torch.cat((history, current_latent[start : start + nq]), 0)
        rope = torch.cat((cache[1][physical, positions % 128], current_rope[start : start + nq]), 0)
        expanded = project(latent.view(-1, 512))[0].view(-1, heads, 256)
        key = torch.cat((expanded[..., :128], rope.expand(-1, heads, -1)), -1).float()
        value = expanded[..., 128:].float()
        for token in range(nq):
            logits = torch.einsum("hd,shd->hs", query[start + token].float(), key[: context + token + 1])
            logits *= 192**-0.5
            expected.append(torch.einsum("hs,shd->hd", logits.softmax(-1), value[: context + token + 1]))
            expected_lse.append(torch.logsumexp(logits, -1))
        start += nq
    torch.testing.assert_close(actual.float(), torch.stack(expected).to(actual.dtype).float(), atol=0.001, rtol=0.01)
    torch.testing.assert_close(actual_lse.squeeze(-1), torch.stack(expected_lse), atol=1e-5, rtol=1e-5)
    assert set(projected_dtypes) == {torch.bfloat16}
    assert all(0 < count <= budget for count in gathered)
    torch.testing.assert_close(latent_storage.float(), original_latent)
    torch.testing.assert_close(rope_storage, original_rope)


@pytest.mark.parametrize("full", [False, True])
def test_c8_dispatch_selects_full_prefill_before_decode_fallback(full):
    scope = {}
    load_functions(ROOT / "vllm_ascend/attention/mla_v1.py", {"_forward_flash"}, scope)
    calls = []
    impl = SimpleNamespace(
        fa_quant_layer=True,
        _forward_flash_full_prefill=lambda *args: calls.append("full"),
        _forward_flash_c8=lambda *args: calls.append("c8"),
    )
    scope["_forward_flash"](
        impl, "layer", None, None, SimpleNamespace(flash_full_prefill=object() if full else None), output=object()
    )
    assert calls == ["full" if full else "c8"]


def test_native_c8_adapter_passes_split_projection_views_without_bf16_staging(monkeypatch):
    heads, tokens = 2, 5
    q = torch.randn(tokens, heads, 192).to(torch.bfloat16)
    kv = torch.randn(tokens, heads, 256).to(torch.bfloat16)
    k, v = kv.split(128, -1)
    kr = torch.randn(tokens, 1, 64).to(torch.bfloat16)
    quantized = tuple(torch.tensor([i]) for i in range(8))
    seen = []
    schedules = []

    def quantize(query, key, value, *, key_rope):
        assert query is q and key is k and value is v and key_rope is kr
        assert not key.is_contiguous() and not value.is_contiguous()
        return quantized

    def native(*args, **kwargs):
        assert all(actual is expected for actual, expected in zip(args[:8], quantized))
        seen.append((args, kwargs))
        return torch.empty(tokens, heads, 128), torch.empty(heads, tokens)

    ops = ModuleType("cann_ops_transformer.ops")
    ops.flash_attn = lambda *args, **kwargs: pytest.fail("C8 must use the quantized native kernel")

    def native_metadata(*args, **kwargs):
        schedules.append(kwargs)
        return torch.tensor([17])

    ops.flash_attn_metadata = native_metadata
    quant_module = ModuleType("vllm_ascend.ops.flash_attn_c8_quant")
    quant_module.quantize_flash_attn_c8 = quantize
    quant_module.fake_quant_flash_attn_c8 = lambda *args: pytest.fail("direct FP8 must not fake-quantize")
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", ModuleType("cann_ops_transformer"))
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, quant_module.__name__, quant_module)
    monkeypatch.setattr(torch.ops._C_ascend, "flash_attn_c8", native, raising=False)
    mask = torch.ones(2048, 2048, dtype=torch.int8)
    schedule, attention = prefill_ops.native_flash_adapters(heads, 0.125, mask, c8=True, bf16_prepare=True)
    metadata = prefill_ops.prepare_metadata(
        prefill_ops.build_prefill_plan([0, tokens], 0, [0], 128), torch.device("cpu"), schedule
    )
    attention(q, k, v, kr, metadata.current)
    args, kwargs = seen[0]
    assert args[8] is metadata.current.query_cu and args[9] is metadata.current.kv_cu
    assert args[10] is metadata.current.schedule
    assert schedules[0]["max_seqlen_q"] == 65
    assert schedules[0]["cu_seqlens_q"] is metadata.current.query_cu
    assert schedules[0]["seqused_q"] is metadata.current.query_used
    assert kwargs["max_seqlen_q"] == tokens
    assert kwargs["softmax_scale"] == 0.125 and kwargs["mask_mode"] == 3
    assert kwargs["attn_mask"] is mask and kwargs["seqused_q"] is metadata.current.query_used


@pytest.mark.parametrize("per_head_rope", [False, True])
def test_native_bf16_adapter_packs_projection_views_without_copying_query(monkeypatch, per_head_rope):
    heads, tokens = 2, 5
    query = torch.randn(tokens, heads, 192).bfloat16()
    kv = torch.randn(tokens, heads, 256).bfloat16()
    key, value = kv.split(128, dim=-1)
    rope = torch.randn(tokens, heads if per_head_rope else 1, 64).bfloat16()
    packed_key = torch.cat((key, rope.expand(-1, heads, -1)), dim=-1)
    packed_value = value.contiguous()
    output = torch.empty(tokens, heads, 128)
    lse = torch.empty(heads, tokens)
    events = []

    def prepare(actual_key, actual_value, actual_rope):
        assert actual_key is key and actual_value is value and actual_rope is rope
        assert not actual_key.is_contiguous() and not actual_value.is_contiguous()
        events.append("prepare")
        return packed_key, packed_value

    def attention(actual_query, actual_key, actual_value, **kwargs):
        assert actual_query is query and actual_key is packed_key and actual_value is packed_value
        assert kwargs["softmax_scale"] == 0.125 and kwargs["mask_mode"] == 3
        events.append("attention")
        return output, lse

    ops = ModuleType("cann_ops_transformer.ops")
    ops.flash_attn = attention
    ops.flash_attn_metadata = lambda *args, **kwargs: torch.tensor([17])
    prepare_module = ModuleType("vllm_ascend.ops.flash_mla_bf16_prepare")
    prepare_module.prepare_flash_mla_bf16 = prepare
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", ModuleType("cann_ops_transformer"))
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, prepare_module.__name__, prepare_module)
    schedule, invoke = prefill_ops.native_flash_adapters(heads, 0.125, None, bf16_prepare=True)
    metadata = prefill_ops.prepare_metadata(
        prefill_ops.build_prefill_plan([0, tokens], 0, [0], 128), torch.device("cpu"), schedule
    )
    actual_output, actual_lse = invoke(query, key, value, rope, metadata.current)
    assert actual_output is output and actual_lse is lse
    assert events == ["prepare", "attention"]


def test_c8_full_prefill_mixed_batch_projects_output_once():
    """Keep mixed-batch cache writes, connector notification and final projection singular."""
    tokens, heads, decode_tokens, prefill_tokens = 9, 2, 2, 6
    x = torch.randn(tokens, 576).to(torch.bfloat16)
    query = torch.randn(tokens, heads, 192).to(torch.bfloat16)
    cache = (torch.empty(2, 128, 1, 512, dtype=torch.float8_e4m3fn), torch.empty(2, 128, 1, 64, dtype=torch.bfloat16))
    scale = torch.tensor([0.25])
    events = []
    flash = SimpleNamespace(
        slots=torch.tensor([*range(tokens - 1), -1]),
        schedule=torch.empty(0),
        attn_mask=None,
        token_live=torch.tensor([True] * (tokens - 1) + [False]),
        block_table=torch.zeros(2, 1, dtype=torch.int32),
    )

    @dataclass
    class Metadata:
        flash: object
        flash_decode: object
        flash_full_prefill: object
        num_decode_tokens: int = decode_tokens
        num_decodes: int = 1

    meta = Metadata(
        flash,
        object(),
        SimpleNamespace(plan=SimpleNamespace(num_tokens=prefill_tokens, query_lengths=(prefill_tokens,))),
    )

    def full(qn, qr, latent, rope, actual_cache, table, metadata, *args, **kwargs):
        assert actual_cache is cache and kwargs["cache_scale"] is scale
        assert qn.shape[0] == latent.shape[0] == prefill_tokens
        events.append("full")
        return torch.ones(prefill_tokens, heads, 128, dtype=x.dtype), None

    def decode(layer, hidden, actual_cache, metadata, output, *, return_projected, projected_inputs):
        assert hidden.shape[0] == decode_tokens and actual_cache is cache
        assert metadata.flash is meta.flash_decode and return_projected
        q_nope, q_rope, latent, rope = projected_inputs
        torch.testing.assert_close(q_nope, query[:decode_tokens, :, :128])
        torch.testing.assert_close(q_rope, query[:decode_tokens, :, 128:])
        torch.testing.assert_close(latent[:, 0], x[:decode_tokens, :512])
        torch.testing.assert_close(rope[:, 0], x[:decode_tokens, 512:])
        events.append("decode")
        return torch.full((decode_tokens, heads * 128), 2, dtype=x.dtype)

    def kv_projection(value):
        events.append("kv_proj")
        return value, None

    def query_projection(*args, **kwargs):
        events.append("q_proj")
        return query

    def latent_normalization(value):
        events.append("kv_norm")
        return value

    def gate_projection(value):
        events.append("gate")
        return torch.zeros(tokens, heads * 128, dtype=x.dtype), None

    def output_projection(projected, **kwargs):
        assert projected.shape[0] == tokens and kwargs["is_prefill"]
        events.append("o_proj")
        return projected, None

    def quantize(latent, rope, kc, kr, slots, inverse):
        assert kc is cache[0] and kr is cache[1] and slots is flash.slots
        events.append("write")

    def adapters(*args, **kwargs):
        assert kwargs == {"c8": False, "fake_quant": False, "bf16_prepare": True}
        return None, None

    scope = dict(
        torch=torch,
        replace=replace,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=1),
        wait_for_device_metadata=lambda *args: None,
        wait_for_kv_layer_from_connector=lambda *args: events.append("wait"),
        quantize_mla_kv=quantize,
        gather_dequant_mla_prefill=object(),
        notify_kv_cache_written=lambda *args: events.append("notify"),
        native_flash_adapters=adapters,
        full_prefill=full,
        DeviceOperator=SimpleNamespace(kv_cache_load=None),
        torch_npu=SimpleNamespace(npu_attention_update=None),
        KimiOProjMMReduceScatterOp=type("KimiOProjMMReduceScatterOp", (), {}),
        flash_attention_output=lambda result, live, output: output.copy_(result),
    )
    load_functions(ROOT / "vllm_ascend/attention/mla_v1.py", {"_forward_flash_full_prefill"}, scope)
    impl = SimpleNamespace(
        fa_quant_layer=True,
        layerwise_kv_cache_hook=SimpleNamespace(wait_for_layer=lambda *args: events.append("hook_wait")),
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=kv_projection,
        _project_query=query_projection,
        kv_a_layernorm=latent_normalization,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        num_heads=heads,
        v_head_dim=128,
        use_mla_rope=False,
        use_output_gate=True,
        g_proj=gate_projection,
        fak_descale_float=scale,
        fak_descale_reciprocal=1 / scale,
        scale=192**-0.5,
        kv_b_proj=None,
        _forward_flash_c8=decode,
        o_proj=output_projection,
    )
    output = torch.empty(tokens, heads * 128, dtype=x.dtype)
    actual = scope["_forward_flash_full_prefill"](impl, "layer", x, cache, meta, output)
    assert actual is output
    assert events == [
        "wait",
        "hook_wait",
        "kv_proj",
        "q_proj",
        "kv_norm",
        "write",
        "notify",
        "full",
        "decode",
        "gate",
        "o_proj",
    ]
    torch.testing.assert_close(actual[:decode_tokens], torch.ones_like(actual[:decode_tokens]))
    torch.testing.assert_close(actual[decode_tokens:-1], torch.full_like(actual[decode_tokens:-1], 0.5))
    assert not actual[-1].count_nonzero()


def test_c8_mixed_decode_reuses_projection_without_cache_write_or_connector_callback(monkeypatch):
    """Verify the decode prefix consumes projections already persisted by its prefill parent."""
    tokens, heads = 3, 2
    events = []

    def forbidden(*args, **kwargs):
        raise AssertionError("mixed decode must reuse the parent's projection and completed cache write")

    query_nope = torch.randn(tokens, heads, 128).bfloat16()
    query_rope = torch.randn(tokens, heads, 64).bfloat16()
    latent = torch.randn(tokens, 1, 512).bfloat16()
    rope = torch.randn(tokens, 1, 64).bfloat16()
    scale = torch.ones(1)
    weights = torch.randn(heads, 128, 512).bfloat16()
    cache = (torch.empty(1, 128, 1, 512, dtype=torch.float8_e4m3fn), torch.empty(1, 128, 1, 64).bfloat16())
    flash = SimpleNamespace(
        query=torch.empty(tokens, heads, 576).bfloat16(),
        schedule=torch.empty(0),
        current_schedule=torch.empty(0),
        unabsorbed=False,
        is_prefill=False,
        causal=True,
        split_kv=True,
        dcp_size=1,
        slots=torch.arange(tokens),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.tensor([128]),
        cu=torch.tensor([0, tokens]),
        used_q=torch.tensor([tokens]),
        attn_mask=None,
        max_query_len=tokens,
        max_seq_len=131,
        current_cache=torch.empty(1, 1, 128, 576).bfloat16(),
        current_block_table=torch.zeros(1, 1, dtype=torch.int32),
        current_slots=torch.arange(tokens),
    )

    def dynamic_quant(value, **kwargs):
        torch.testing.assert_close(value, torch.bmm(query_nope.transpose(0, 1), weights).transpose(0, 1))
        events.append("quant_query")
        return torch.zeros_like(value, dtype=torch.float8_e4m3fn), torch.ones(tokens, heads)

    def scatter(**kwargs):
        assert kwargs["key"].data_ptr() == latent.data_ptr() and kwargs["value"].data_ptr() == rope.data_ptr()
        assert kwargs["slot_mapping"] is flash.current_slots
        events.append("scatter_current")

    def attend(query, actual_cache, **kwargs):
        current = query.dtype == torch.bfloat16
        assert actual_cache is (flash.current_cache if current else cache[0])
        assert kwargs["mask_mode"] == (3 if current else 0)
        events.append("current" if current else "history")
        return torch.ones(tokens, heads, 512).bfloat16(), torch.zeros(heads, tokens)

    def merge(history, lse, *args, **kwargs):
        events.append("merge")
        return history

    def project_value(value):
        events.append("v_up")
        return value[..., :128].reshape(tokens, heads * 128)

    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache", attend, raising=False)
    scope = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_dynamic_quant=dynamic_quant, npu_scatter_pa_kv_cache=scatter),
        DeviceMetadataStage=SimpleNamespace(ATTENTION=1),
        wait_for_device_metadata=lambda *args: events.append("metadata_wait"),
        wait_for_kv_layer_from_connector=forbidden,
        notify_kv_cache_written=forbidden,
        quantize_mla_kv=forbidden,
        scale_mla_query_rope=lambda value, *args: value.contiguous(),
        merge_flash_attention_output=merge,
    )
    load_functions(ROOT / "vllm_ascend/attention/mla_v1.py", {"_forward_flash_c8"}, scope)
    impl = SimpleNamespace(
        q_proj=SimpleNamespace(qrep_active=False),
        W_UK_T=weights,
        _q_proj_and_k_up_proj=forbidden,
        fused_qkv_a_proj=forbidden,
        kv_a_proj_with_mqa=forbidden,
        kv_a_layernorm=forbidden,
        layerwise_kv_cache_hook=SimpleNamespace(wait_for_layer=forbidden),
        fak_descale_float=scale,
        scale=192**-0.5,
        _v_up_proj_batch_major=project_value,
        use_output_gate=True,
        g_proj=forbidden,
        o_proj=forbidden,
    )
    output = scope["_forward_flash_c8"](
        impl,
        "layer",
        torch.zeros(tokens, 576).bfloat16(),
        cache,
        SimpleNamespace(flash=flash),
        torch.empty(tokens, heads * 128).bfloat16(),
        return_projected=True,
        projected_inputs=(query_nope, query_rope, latent, rope),
    )
    assert output.shape == (tokens, heads * 128)
    assert events == ["metadata_wait", "quant_query", "scatter_current", "history", "current", "merge", "v_up"]


@pytest.mark.parametrize("quantized", [False, True])
def test_flash_without_optional_prolog_retains_projection_weights(quantized):
    """The ordinary C8 path needs calibrated scales but no MLAPO weight packing."""

    class LinearMethod:
        pass

    calls = []
    q_weight = torch.randn(3, 4)
    kv_weight = torch.randn(3, 4)
    impl = SimpleNamespace(
        use_flash_mla=True,
        enable_mlapo=False,
        fa_quant_layer=quantized,
        num_heads=2,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        v_head_dim=2,
        kv_b_proj=SimpleNamespace(quant_method=LinearMethod(), weight=torch.randn(10, 4)),
        q_proj=SimpleNamespace(weight=q_weight),
        fused_qkv_a_proj=SimpleNamespace(weight=kv_weight),
        _load_fa_quant_scales=lambda: calls.append("scales"),
        _process_weights_for_fused=lambda _: pytest.fail("unused prolog weight packing"),
    )
    scope = dict(
        torch=torch,
        UnquantizedLinearMethod=LinearMethod,
        ACL_FORMAT_FRACTAL_ND=2,
        torch_npu=SimpleNamespace(npu_format_cast=lambda value, _: value),
        maybe_trans_nz=lambda value: value,
    )
    load_functions(ROOT / "vllm_ascend/attention/mla_v1.py", {"process_weights_after_loading"}, scope)
    scope["process_weights_after_loading"](impl, torch.bfloat16)
    assert calls == (["scales"] if quantized else [])
    assert impl.q_proj.weight is q_weight
    assert impl.fused_qkv_a_proj.weight is kv_weight
    torch.testing.assert_close(impl.W_UK_T, impl.kv_b_proj.weight.T.reshape(4, 2, 5)[..., :3].permute(1, 2, 0))
