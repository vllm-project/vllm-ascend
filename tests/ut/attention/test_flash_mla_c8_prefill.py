# SPDX-License-Identifier: Apache-2.0
"""C8 history/BF16 current orchestration, independent of an NPU runtime."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_prefill(scope):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_flash_c8")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


def _dynamic_quant(query, **kwargs):
    scale = query.float().abs().amax(-1) / 448
    nonzero_scale = torch.where(scale == 0, 1.0, scale)
    return (query.float() / nonzero_scale[..., None]).to(torch.float8_e4m3fn), scale


def _scale_query_rope(rope, query_scale, kv_scale):
    # CPU stand-in for the fused device op. Native coverage validates the
    # actual kernel; here we exercise the production call and data dependency.
    query_scale.masked_fill_(query_scale == 0, 1)
    return (rope.float() / (query_scale[..., None] * kv_scale)).to(torch.bfloat16)


def _quantize_query(query, rope, kv_scale):
    quantized, scale = _dynamic_quant(query)
    scaled_rope = _scale_query_rope(rope, scale, kv_scale)
    return quantized, scaled_rope, scale


@pytest.mark.parametrize("history_lengths", [(0, 0), (127, 128), (128, 129)])
@pytest.mark.parametrize("kv_scale", [0.03125, 0.25])
@pytest.mark.parametrize("dcp_size,q_replicated", [(1, False), (8, False), (8, True)])
@pytest.mark.parametrize("is_prefill", [True, False])
def test_prefill_quantized_history_bf16_current_and_merge(
    monkeypatch,
    history_lengths,
    kv_scale,
    dcp_size,
    q_replicated,
    is_prefill,
    mlapo=False,
    overlap=False,
    owner_rank=3,
):
    torch.manual_seed(314)
    heads, pages = 2, 8
    second_length = 5 if is_prefill else 4
    live_tokens = 4 + second_length
    tokens = live_tokens + 1
    local_start = (3 if dcp_size > 1 else 0) * heads
    local_slice = slice(local_start, local_start + heads)
    # The first row is a four-token verification step in a mixed prefill
    # batch. The last physical token belongs to the padding request.
    cu = torch.tensor([0, 4, live_tokens, tokens], dtype=torch.int32)
    used = torch.tensor([4, second_length, 0], dtype=torch.int32)
    query = (torch.randn(tokens, heads * dcp_size, 576) * 0.2).to(torch.bfloat16)
    # Real DynamicQuant returns scale=0 on all-zero latent rows. Exercise an
    # active fully-zero query and an active rope-only query, not just padding.
    query[0].zero_()
    query[1, :, :512].zero_()
    projection_query = query if q_replicated else query[:, local_slice]
    current = (torch.randn(tokens, 1, 576) * 0.2).to(torch.bfloat16)
    current[-1] = float("nan")
    latent_storage = (torch.randn(pages, 3, 128, 1, 512) * 3).to(torch.float8_e4m3fn)
    rope_storage = (torch.randn(pages, 2, 128, 1, 64) * 0.2).to(torch.bfloat16)
    latent_cache, rope_cache = latent_storage[:, 1], rope_storage[:, 1]
    before_latent, before_rope = latent_storage.clone(), rope_storage.clone()
    table = torch.tensor([[2, 5], [1, 6], [0, 0]], dtype=torch.int32)
    slots = torch.full((tokens,), -1, dtype=torch.int64)
    for request, history in enumerate(history_lengths):
        for i in range(int(used[request])):
            logical = history + i
            slots[request * 4 + i] = table[request, logical // 128] * 128 + logical % 128
    if mlapo:
        slots[torch.arange(tokens) % dcp_size != owner_rank] = -1
    current_cache = torch.full((3, 1, 128, 576), 37, dtype=torch.bfloat16)
    current_slots = torch.tensor([0, 1, 2, 3, *range(128, 128 + second_length), -1])
    b = SimpleNamespace(
        query=torch.full_like(query, float("nan")),
        split_kv=True,
        unabsorbed=False,
        dcp_size=dcp_size,
        is_prefill=is_prefill,
        causal=True,
        schedule=torch.tensor([8]),
        current_schedule=torch.tensor([3]),
        current_cache=current_cache,
        current_block_table=torch.tensor([[0], [1], [0]], dtype=torch.int32),
        current_slots=current_slots,
        slots=slots,
        block_table=table,
        cache_lens=torch.tensor([*history_lengths, 0], dtype=torch.int32),
        used_q=used,
        cu=cu,
        max_query_len=tokens,
        max_seq_len=256,
        token_live=torch.arange(tokens) < live_tokens,
        attn_mask=torch.triu(torch.ones(128, 128, dtype=torch.int8), diagonal=1),
    )
    events, results = [], []

    def write_quantized(c, pe, kc, kr, dest, inverse, *, source_slots=None):
        assert kc.data_ptr() == latent_cache.data_ptr()
        assert kr.data_ptr() == rope_cache.data_ptr()
        assert not kc.is_contiguous() and not kr.is_contiguous()
        events.append("write")
        values = (c.float() * inverse).clamp(-448, 448).to(torch.float8_e4m3fn)
        for row, slot in enumerate(dest.tolist()):
            if slot >= 0:
                if source_slots is None:
                    value, component = values[row], pe[row]
                else:
                    src = int(source_slots[row])
                    assert src >= 0
                    value, component = values[src // 128, src % 128], pe[src // 128, src % 128]
                kc[slot // 128, slot % 128] = value
                kr[slot // 128, slot % 128] = component

    def quantize_query_and_kv(
        q, qr, scale, c, pe, kc, kr, dest, inverse, *, source_slots=None, local_query=None, local_head_start=0
    ):
        # The CPU reference preserves the complete combined helper contract.
        # Its fused target and fallback are qualified separately on NPU.
        write_quantized(c, pe, kc, kr, dest, inverse, source_slots=source_slots)
        if local_query is not None:
            end = local_head_start + local_query.shape[1]
            local_query.copy_(torch.cat((q[:, local_head_start:end], qr[:, local_head_start:end]), -1))
        return _quantize_query(q, qr, scale)

    def preprocess(x, cache, metadata):
        assert metadata.flash.current_cache is current_cache
        assert cache[0] is latent_cache and cache[1] is rope_cache
        events.append("prolog")
        for row, slot in enumerate(current_slots.tolist()):
            if slot >= 0:
                current_cache[slot // 128, 0, slot % 128] = current[row, 0]
        return SimpleNamespace(ql_nope=query[..., :512], q_pe=query[..., 512:]), None

    def scatter(key, value, key_cache, value_cache, slot_mapping, **kwargs):
        assert key.dtype == value.dtype == torch.bfloat16
        for row, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                key_cache[slot // 128, slot % 128] = key[row]
                value_cache[slot // 128, slot % 128] = value[row]

    def flash(q, cache, **kwargs):
        history = q.dtype == torch.float8_e4m3fn
        events.append("history" if history else "current")
        assert kwargs["return_softmax_lse"]
        assert kwargs["mask_mode"] == (0 if history else 3)
        if history:
            # The original non-contiguous caches reach the operator unchanged.
            assert cache.data_ptr() == latent_cache.data_ptr()
            assert kwargs["key_rope"].data_ptr() == rope_cache.data_ptr()
            assert cache.stride(0) == latent_cache.stride(0)
            ql = q.float() * kwargs["dequant_scale_query"][..., None]
            qr = kwargs["query_rope"].float() * kwargs["dequant_scale_query"][..., None] * kv_scale
            assert (kwargs["dequant_scale_query"][:2] == 1).all()
            assert torch.isfinite(qr).all()
            torch.testing.assert_close(qr[:2], query[:2, :, 512:].float(), rtol=0.01, atol=0.002)
        else:
            assert cache is current_cache
            assert q.dtype == torch.bfloat16
            ql, qr = q.float().split([512, 64], -1)
        output = torch.zeros(tokens, q.shape[1], 512, dtype=torch.float32)
        lse = torch.full((q.shape[1], tokens), -float("inf"))
        logits_by_row, values_by_row = [], []
        for request in range(2):
            length = int(kwargs["cache_seqlens"][request])
            locations = kwargs["block_table"][request, torch.arange(length) // 128] * 128 + torch.arange(length) % 128
            if history:
                assert kwargs["layout_kv"] == "PA_BBND"
                keys = cache.float()[locations // 128, locations % 128, 0] * kv_scale
                pe = kwargs["key_rope"][locations // 128, locations % 128, 0].float()
            else:
                keys, pe = cache[locations // 128, 0, locations % 128].float().split([512, 64], -1)
            for local in range(int(used[request])):
                row = request * 4 + local
                limit = length if history else local + 1
                logits = (ql[row] @ keys[:limit].T + qr[row] @ pe[:limit].T) * kwargs["softmax_scale"]
                logits_by_row.append(logits)
                values_by_row.append(keys[:limit])
                if limit:
                    lse[:, row] = torch.logsumexp(logits, -1)
                    output[row] = torch.softmax(logits, -1) @ keys[:limit]
        results.append((logits_by_row, values_by_row))
        return output, lse

    def merge(history, history_lse, group, *, current_output, current_lse, raw_row_words=257):
        assert raw_row_words == 257
        assert (group is None) == (dcp_size == 1)
        # Emulate DCP's head scatter with all other history ranks empty. The
        # local BF16 current contribution must be counted only once.
        history = history[:, local_slice]
        history_lse = history_lse[local_slice]
        merged_lse = torch.logaddexp(history_lse, current_lse)
        hweight = (history_lse - merged_lse).exp().T[..., None]
        cweight = (current_lse - merged_lse).exp().T[..., None]
        merged = hweight * history + cweight * current_output
        # An independent concatenated-logits reference catches missing scale,
        # duplicate current tokens and inconsistent history/current LSE.
        for row in range(live_tokens):
            logits = torch.cat((results[0][0][row][local_slice], results[1][0][row]), dim=-1)
            values = torch.cat((results[0][1][row], results[1][1][row]), dim=0)
            expected = torch.softmax(logits, -1) @ values
            torch.testing.assert_close(merged[row], expected, rtol=2e-5, atol=2e-6)
        return merged

    exchanged, stream_events = [], []

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            event = len(stream_events)
            stream_events.append((self.name, "record", event))
            return event

        def wait_event(self, event):
            stream_events.append((self.name, "wait", event))

    main_stream, side_stream = Stream("main"), Stream("side")
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: main_stream, stream=lambda _: nullcontext()), raising=False
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    def exchange(history, lse, group, *, raw_row_words=257):
        assert raw_row_words == 257
        assert group.world_size == dcp_size
        events.append("pack")
        exchanged.append((history, lse, group))
        return torch.empty(1)

    def combine(recv, head_dim, **kwargs):
        assert head_dim == 512
        return merge(*exchanged[0], **kwargs)

    scope = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_dynamic_quant=_dynamic_quant, npu_scatter_pa_kv_cache=scatter),
        wait_for_device_metadata=lambda *args: events.append("metadata"),
        DeviceMetadataStage=SimpleNamespace(ATTENTION=0),
        MLAPO_MAX_SUPPORTED_TOKENS=128,
        wait_for_kv_layer_from_connector=lambda *args: events.append("wait_kv"),
        quantize_mla_kv=write_quantized,
        scale_mla_query_rope=_scale_query_rope,
        quantize_mla_query=_quantize_query,
        quantize_mla_query_and_kv=quantize_query_and_kv,
        _dcp_mtp_comm_stream=lambda: side_stream,
        exchange_flash_attention_output=exchange,
        combine_flash_attention_output=combine,
        notify_kv_cache_written=lambda *args: events.append("notify"),
        merge_flash_attention_output=merge,
        KimiOProjMMReduceScatterOp=type("ProjectionOp", (), {}),
        flash_attention_output=lambda result, live, out: out.copy_(torch.where(live[:, None], result, 0)),
    )
    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache", flash, raising=False)
    impl = SimpleNamespace(
        enable_mlapo=mlapo,
        mlapo_weight_quant_mode=3,
        mla_preprocess_only_decode=preprocess,
        flash_dcp_overlap=overlap,
        fused_qkv_a_proj=lambda x: (torch.cat((current.new_zeros(tokens, 1), current[:, 0]), -1), None),
        q_lora_rank=1,
        q_a_layernorm=lambda x: x,
        kv_a_layernorm=lambda x: x,
        _q_proj_and_k_up_proj=lambda x: (projection_query[..., :512], projection_query[..., 512:]),
        q_proj=SimpleNamespace(qrep_active=q_replicated, _local_view=lambda q: q[:, local_slice].contiguous()),
        _dcp_all_gather=lambda q, axis: query,
        dcp_group=SimpleNamespace(world_size=dcp_size),
        use_mla_rope=False,
        layerwise_kv_cache_hook=None,
        fak_descale_float=torch.tensor([kv_scale]),
        fak_descale_reciprocal=torch.tensor([1 / kv_scale]),
        scale=576**-0.5,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=False,
        o_proj=lambda x, **kwargs: (x, None),
    )
    output = torch.empty(tokens, heads * 512, dtype=torch.bfloat16)
    _load_prefill(scope)(
        impl,
        "layer",
        torch.ones(tokens, 1, dtype=torch.bfloat16),
        (latent_cache, rope_cache),
        SimpleNamespace(flash=b),
        output,
    )
    assert events == [
        "metadata",
        *(["wait_kv"] if is_prefill else []),
        *(["prolog"] if mlapo else []),
        "write",
        "notify",
        "history",
        *(["pack"] if overlap else []),
        "current",
    ]
    if overlap:
        assert stream_events == [("main", "record", 0), ("side", "wait", 0), ("side", "record", 2), ("main", "wait", 2)]
    if mlapo:
        assert b.query.isnan().all(), "Mode 7 must not materialize a full replicated BF16 query."
        expected = before_latent.clone()
        expected_rope = before_rope.clone()
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                expected[slot // 128, 1, slot % 128] = (
                    (current[row, :, :512].float() / kv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                expected_rope[slot // 128, 1, slot % 128] = current[row, :, 512:]
        torch.testing.assert_close(latent_storage.float(), expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(rope_storage, expected_rope, rtol=0, atol=0)
    assert torch.isfinite(output).all() and (output[-1] == 0).all()
    torch.testing.assert_close(latent_storage[:, 0].float(), before_latent[:, 0].float(), rtol=0, atol=0)
    torch.testing.assert_close(latent_storage[:, 2].float(), before_latent[:, 2].float(), rtol=0, atol=0)
    torch.testing.assert_close(rope_storage[:, 0], before_rope[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(current_cache[0, 0, :4], current[:4, 0], rtol=0, atol=0)
    torch.testing.assert_close(current_cache[1, 0, :second_length], current[4:live_tokens, 0], rtol=0, atol=0)
    assert (current_cache[2] == 37).all()


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("owner_rank", [0, 3, 7])
def test_c8_mode7_current_owned_history_and_multistream(monkeypatch, overlap, owner_rank):
    test_prefill_quantized_history_bf16_current_and_merge(
        monkeypatch,
        (127, 129),
        0.03125,
        8,
        True,
        False,
        mlapo=True,
        overlap=overlap,
        owner_rank=owner_rank,
    )


@pytest.mark.parametrize("tokens", [4, 32])
@pytest.mark.parametrize("causal", [True, False])
def test_c8_decode_preserves_prolog_quantized_query_and_scale(monkeypatch, tokens, causal):
    heads = 12
    source = torch.randn(tokens, heads, 576).to(torch.bfloat16)
    query, query_scale = _dynamic_quant(source[..., :512])
    kv_scale = torch.tensor([0.125])
    query_rope = (source[..., 512:] / query_scale[..., None] / kv_scale).to(torch.bfloat16)
    kv = (
        torch.empty(4, 128, 1, 512, dtype=torch.float8_e4m3fn),
        torch.empty(4, 128, 1, 64, dtype=torch.bfloat16),
    )
    b = SimpleNamespace(
        query=torch.full((tokens, heads, 576), float("nan"), dtype=torch.bfloat16),
        split_kv=False,
        unabsorbed=False,
        dcp_size=1,
        is_prefill=False,
        causal=causal,
        schedule=torch.empty(1, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        cache_lens=torch.tensor([260], dtype=torch.int32),
        used_q=torch.tensor([tokens], dtype=torch.int32),
        cu=torch.tensor([0, tokens], dtype=torch.int32),
        max_query_len=tokens,
        max_seq_len=512,
        token_live=torch.ones(tokens, dtype=torch.bool),
        attn_mask=torch.empty(128, 128, dtype=torch.int8),
    )
    calls = []

    def flash(q, cache, **kwargs):
        calls.append("flash")
        assert q is query
        assert kwargs["query_rope"] is query_rope
        assert kwargs["dequant_scale_query"] is query_scale
        assert kwargs["dequant_scale_key"] is kv_scale
        assert kwargs["mask_mode"] == (3 if causal else 0)
        assert kwargs["layout_out"] == "NTD"
        assert not kwargs["return_softmax_lse"]
        return torch.ones(heads, tokens, 512, dtype=torch.bfloat16), None

    prolog = SimpleNamespace(ql_nope=query, q_pe=query_rope, dequant_scale_q_nope=query_scale)
    scope = dict(
        torch=torch,
        wait_for_device_metadata=lambda *args: None,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=0),
        MLAPO_MAX_SUPPORTED_TOKENS=128,
        notify_kv_cache_written=lambda *args: calls.append("written"),
        KimiOProjMMReduceScatterOp=type("ProjectionOp", (), {}),
        flash_attention_output=lambda result, live, output: output.copy_(result),
    )
    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache", flash, raising=False)
    impl = SimpleNamespace(
        enable_mlapo=True,
        mlapo_weight_quant_mode=3,
        q_proj=SimpleNamespace(qrep_active=False),
        layerwise_kv_cache_hook=None,
        mla_preprocess_only_decode=lambda *args: (prolog, None),
        fak_descale_float=kv_scale,
        scale=576**-0.5,
        _v_up_proj=lambda latent: latent.transpose(0, 1).reshape(tokens, -1),
        use_output_gate=False,
        o_proj=lambda x, **kwargs: (x, None),
    )
    output = torch.empty(tokens, heads * 512, dtype=torch.bfloat16)
    _load_prefill(scope)(
        impl, "layer", torch.empty(tokens, 1, dtype=torch.bfloat16), kv, SimpleNamespace(flash=b), output
    )
    assert calls == ["written", "flash"]
    assert (output == 1).all()
    assert torch.isnan(b.query).all()  # Never cast FP8 Q into this BF16 scratch.


@pytest.mark.parametrize(
    "dcp_size,is_prefill,overlap,output_gate,projection",
    [
        (8, False, True, True, "generic"),
        (8, False, True, True, "reduce_scatter"),
        (8, False, False, True, "generic"),
        (8, False, True, False, "generic"),
        (8, True, True, True, "generic"),
        (1, False, True, True, "direct"),
        (1, False, True, True, "generic"),
    ],
)
def test_c8_gate_projection_overlaps_history_exchange(
    monkeypatch, dcp_size, is_prefill, overlap, output_gate, projection
):
    """Run the production method and assert dependency order and BF16 gating."""
    tokens = 4
    events = []
    hidden = torch.arange(tokens * 3, dtype=torch.bfloat16).view(tokens, 3) / 10
    original = torch.arange(tokens * 4, dtype=torch.bfloat16).view(tokens, 4) / 8
    gate_weight = torch.tensor([[1, -2, 3, -4], [2, 1, -1, 1], [-1, 2, 1, -2]], dtype=torch.bfloat16)
    query = torch.zeros(tokens, dcp_size, 512, dtype=torch.bfloat16)
    rope = torch.zeros(tokens, dcp_size, 64, dtype=torch.bfloat16)
    live = torch.tensor([True, True, True, False])

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            return self.name

        def wait_event(self, event):
            events.append(f"{self.name}_wait")

    main, side = Stream("main"), Stream("side")
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: main, stream=lambda _: nullcontext()), raising=False
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    def flash(q, cache, **kwargs):
        events.append("current" if kwargs["layout_kv"] == "PA_BNBD" else "history")
        return torch.zeros(tokens, dcp_size, 512, dtype=torch.bfloat16), torch.zeros(dcp_size, tokens)

    def exchange(*args, raw_row_words=257):
        assert raw_row_words == 257
        events.append("exchange")
        return torch.empty(1)

    def merge(*args, raw_row_words=257, **kwargs):
        assert raw_row_words == 257
        events.append("merge")
        return torch.empty(tokens, 1, 512, dtype=torch.bfloat16)

    def v_up(*args):
        events.append("v_up")
        return original.clone()

    def gate(x):
        events.append("gate")
        torch.testing.assert_close(x, hidden)
        return x @ gate_weight, None

    def apply_gate(projected, logits, token_live):
        events.append("apply_gate")
        projected.mul_(torch.sigmoid(logits))
        projected.masked_fill_(~token_live[:, None], 0)

    class ProjectionOp:
        tp_size, tp_rank = 2, 0

        def apply_into(self, projected, out):
            events.append("reduce_scatter")
            return out.copy_(projected[: tokens // 2])

    class Unquantized:
        pass

    class Projection:
        input_is_parallel = projection == "direct"
        reduce_results = False
        bias = None
        custom_op = ProjectionOp() if projection == "reduce_scatter" else None
        quant_method = Unquantized()
        weight = torch.eye(4, dtype=torch.bfloat16)

        def __call__(self, value, **kwargs):
            events.append("o_proj")
            return value, None

    b = SimpleNamespace(
        query=torch.zeros(tokens, dcp_size, 576, dtype=torch.bfloat16),
        unabsorbed=False,
        is_prefill=is_prefill,
        split_kv=dcp_size > 1,
        dcp_size=dcp_size,
        causal=True,
        schedule=torch.empty(1),
        current_schedule=torch.empty(1),
        current_cache=torch.empty(1, 1, 128, 576, dtype=torch.bfloat16),
        current_slots=torch.arange(tokens),
        current_block_table=torch.zeros(1, 1, dtype=torch.int32),
        slots=torch.arange(tokens),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.tensor([128]),
        cu=torch.tensor([0, tokens]),
        used_q=torch.tensor([tokens]),
        max_query_len=tokens,
        max_seq_len=128,
        token_live=live,
        attn_mask=torch.empty(128, 128, dtype=torch.int8),
    )
    scope = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_dynamic_quant=_dynamic_quant, npu_scatter_pa_kv_cache=lambda **kwargs: None),
        wait_for_device_metadata=lambda *args: None,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=0),
        MLAPO_MAX_SUPPORTED_TOKENS=128,
        wait_for_kv_layer_from_connector=lambda *args: None,
        quantize_mla_kv=lambda *args: None,
        scale_mla_query_rope=_scale_query_rope,
        notify_kv_cache_written=lambda *args: None,
        _dcp_mtp_comm_stream=lambda: side,
        exchange_flash_attention_output=exchange,
        combine_flash_attention_output=merge,
        merge_flash_attention_output=merge,
        flash_attention_gate=apply_gate,
        flash_attention_output=lambda value, mask, out: out.copy_(torch.where(mask[:, None], value, 0)),
        KimiOProjMMReduceScatterOp=ProjectionOp,
        UnquantizedLinearMethod=Unquantized,
    )
    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache", flash, raising=False)
    impl = SimpleNamespace(
        enable_mlapo=False,
        q_proj=SimpleNamespace(qrep_active=True, _local_view=lambda q: q[:, :1].contiguous()),
        fused_qkv_a_proj=lambda x: (torch.zeros(tokens, 577, dtype=torch.bfloat16), None),
        q_lora_rank=1,
        q_a_layernorm=lambda x: x,
        kv_a_layernorm=lambda x: x,
        _q_proj_and_k_up_proj=lambda x: (query, rope),
        use_mla_rope=False,
        layerwise_kv_cache_hook=None,
        fak_descale_float=torch.tensor([0.03125]),
        fak_descale_reciprocal=torch.tensor([32.0]),
        scale=576**-0.5,
        flash_dcp_overlap=overlap,
        dcp_group=SimpleNamespace(world_size=dcp_size),
        _v_up_proj_batch_major=v_up,
        _v_up_proj=v_up,
        use_output_gate=output_gate,
        g_proj=gate,
        o_proj=Projection(),
    )
    out = torch.empty(tokens // 2 if projection == "reduce_scatter" else tokens, 4, dtype=torch.bfloat16)
    cache = (torch.empty(1, 128, 1, 512), torch.empty(1, 128, 1, 64))
    _load_prefill(scope)(impl, "layer", hidden, cache, SimpleNamespace(flash=b), out)
    expected = original.clone()
    if output_gate:
        expected.mul_(torch.sigmoid(hidden @ gate_weight))
        assert events.count("gate") == 1
        if dcp_size > 1 and not is_prefill and overlap:
            assert events.index("exchange") < events.index("current") < events.index("gate")
            assert events.index("gate") < events.index("main_wait") < events.index("merge") < events.index("v_up")
        else:
            assert events.index("v_up") < events.index("gate")
    else:
        assert "gate" not in events
    expected.masked_fill_(~live[:, None], 0)
    torch.testing.assert_close(out, expected[: out.shape[0]], rtol=0, atol=0)
