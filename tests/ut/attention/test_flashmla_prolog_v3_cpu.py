# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU composition checks; PROLOG and communication are emulated, not NPU evidence."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"


def load(path, name, scope, owner=None):
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "PROLOG_V3_SUPPORTED_INPUT_SIZES",
                    "PROLOG_V3_MAX_BLOCK_SIZE",
                ):
                    scope[target.id] = ast.literal_eval(node.value)
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), path, "exec"), scope)
    return scope[name]


def scatter(*, key, value, key_cache, value_cache, slot_mapping):
    for row, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            key_cache[slot // 128, slot % 128].copy_(key[row])
            value_cache[slot // 128, slot % 128].copy_(value[row])


def fused_cache():
    stride = 128 * 576 + 32
    raw = torch.full((3 * stride + 64,), -17.0)
    return raw.as_strided((3, 128, 1, 576), (stride, 576, 576, 1), 32), raw


@pytest.mark.parametrize(
    "dcp,causal,speculative,heads",
    [
        (False, True, False, 8),
        (False, False, True, 12),
        (True, True, True, 12),
        (True, False, True, 8),
        (True, True, False, 8),
    ],
)
def test_external_forward_reuses_upstream_dcp_prolog(dcp, causal, speculative, heads):
    split = dcp and causal and speculative
    history, raw = fused_cache()
    before = raw.clone()
    slots = torch.tensor([0, 128, -1, -1])
    tokens = slots.numel()
    attention_output = torch.ones(heads, tokens, 512)
    # Injected padding values exercise valid-row isolation, not the NPU output contract.
    attention_output[:, -1, :4] = torch.tensor([7.0, torch.nan, torch.inf, -torch.inf])
    flash = SimpleNamespace(
        slots=slots,
        current=object() if split else None,
        query=torch.empty(tokens, heads, 576),
        adapter=SimpleNamespace(attention=Mock(return_value=(attention_output, None))),
        block_table=object(),
        cache_lens=object(),
        cu=object(),
        used_q=object(),
        schedule=object(),
        attn_mask=None,
        token_live=torch.tensor([True, True, True, False]),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=tokens,
        # The external metadata owns the live/padded writer slots. A stale
        # common mapping must not reach either direct writes or DCP scatter.
        slot_mapping=torch.tensor([1, 129, 255, 256]),
        external_flashmla=flash,
        decode=SimpleNamespace(cos=None, sin=None),
        prefill=None,
    )
    notify = Mock()
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_mla_prolog_v3=Mock()),
        "_EXTRA_CTX": SimpleNamespace(num_tokens=tokens),
        "notify_kv_cache_written": notify,
        "cdiv": lambda x, y: (x + y - 1) // y,
        "DeviceOperator": SimpleNamespace(reshape_and_cache=scatter),
        "record_attention_compute_start": Mock(),
        "maybe_save_kv_layer_to_connector": Mock(),
        "wait_for_kv_layer_from_connector": Mock(),
        "DecodeMLAPreprocessResult": lambda q, pe, k, kr, **kw: SimpleNamespace(ql_nope=q, q_pe=pe, k=k, kr=kr, **kw),
    }
    impl = SimpleNamespace(
        layer_name="mla",
        dtype=torch.bfloat16,
        layerwise_kv_cache_hook=None,
        num_heads=heads,
        mlapo_num_heads=16,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_mla_rope=True,
        enable_kv_nz=False,
        fa_quant_layer=False,
        use_quant_flashmla=False,
        fused_qkv_a_proj=SimpleNamespace(input_size=7168),
        q_proj=SimpleNamespace(_chunk_size=0),
        q_lora_rank=4,
        supports_prolog_v3_quantization=lambda: True,
        supports_prolog_v3_projection=lambda: True,
        use_output_gate=False,
        use_pcp=False,
        is_pcp_decode_sharded=False,
        use_dcp=dcp,
        enable_mlapo=False,
        _dcp_current_kv_buffers=(torch.empty(1, 128, 1, 512), torch.empty(1, 128, 1, 64), torch.arange(128)),
        get_num_actual_tokens=lambda m: m.num_actual_tokens,
        _prolog_v3_weights={"num_heads": 16},
        o_proj=lambda tensor, **kw: (tensor,),
    )
    for name in (
        "supports_prolog_v3_dtype",
        "_supports_prolog_v3_ops",
        "_supports_prolog_v3_input_size",
        "_supports_strided_mla_projection",
        "supports_prolog_v3_writer",
        "_validate_mla_cache_layout",
        "_cache_requires_prolog_v3_writer",
        "_unpack_prolog_decode_q",
        "mla_preprocess_prolog_v3",
        "_get_dcp_current_kv_workspace",
        "forward",
    ):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    cp_path = "vllm_ascend/attention/context_parallel/mla_cp.py"
    owner = "AscendMlaDCPImpl" if dcp else "AscendMLAImpl"
    for name in ("reorg_decode_q", "_decode_requires_current_kv", "_prefill_requires_current_kv"):
        setattr(impl, name, MethodType(load(cp_path if dcp else MLA, name, scope, owner), impl))

    def gather(q, pe, *, dim):
        assert dim == 1 and q.shape[1] == pe.shape[1] == heads
        return q.repeat(1, 8, 1), pe.repeat(1, 8, 1)

    impl._dcp_all_gather_fragments = Mock(side_effect=gather)

    def prolog(hidden, cos, sin, dest, indices, *, quantize_query):
        assert quantize_query
        torch.testing.assert_close(indices, torch.arange(tokens) if split else slots)
        keys = torch.arange(tokens).float()[:, None, None].expand(tokens, 1, 512) + 2
        scatter(key=keys, value=keys[..., :64] + 10, key_cache=dest[0], value_cache=dest[1], slot_mapping=indices)
        q, pe = torch.ones(tokens, 16, 512), torch.full((tokens, 16, 64), 2.0)
        q[:, heads:] = 99
        pe[:, heads:] = 99
        return q, pe, None

    impl._run_prolog_v3 = Mock(side_effect=prolog)
    impl._mla_preprocess = Mock(side_effect=AssertionError("eligible decode must reuse the dependency's PROLOG"))
    if dcp:
        # Numerical attention and the framework exchange/combine are covered
        # separately by test_flashmla_dcp.py. Here check the handoff to them.
        impl._forward_external_flashmla = Mock(return_value=torch.ones(tokens, heads * 128))
    else:
        impl.W_UV = torch.eye(impl.kv_lora_rank, impl.v_head_dim).expand(heads, -1, -1).contiguous()

        def transpose_batchmatmul(latent, weight, *, perm_y):
            assert perm_y == (1, 0, 2)
            return torch.bmm(latent, weight).transpose(0, 1)

        scope["torch_npu"].npu_transpose_batchmatmul = transpose_batchmatmul
        impl._v_up_proj = MethodType(load(MLA, "_v_up_proj", scope, "AscendMLAImpl"), impl)
        external = load(MLA, "_forward_external_flashmla", scope, "AscendMLAImpl")
        impl._forward_external_flashmla = Mock(wraps=MethodType(external, impl))
    impl._forward_decode = Mock(side_effect=AssertionError("external decode must not call FIA"))
    output = torch.empty(tokens, heads * 128)
    assert impl.forward("mla", torch.zeros(tokens, 16), history, metadata, output) is output
    expected = torch.ones_like(output)
    if not dcp:
        query_arg, cache_arg = flash.adapter.attention.call_args.args
        assert query_arg is flash.query and cache_arg is history
        torch.testing.assert_close(flash.query[..., :512], torch.ones_like(flash.query[..., :512]))
        torch.testing.assert_close(flash.query[..., 512:], torch.full_like(flash.query[..., 512:], 2.0))
    # Only valid query results are consumed; padding outputs need not be zero.
    torch.testing.assert_close(output[flash.token_live], expected[flash.token_live])
    consumed, cache_arg, meta_arg = impl._forward_external_flashmla.call_args.args
    assert cache_arg is history and meta_arg is metadata
    impl._mla_preprocess.assert_not_called()
    impl._run_prolog_v3.assert_called_once()
    assert consumed.ql_nope.shape == (tokens, heads * (8 if dcp else 1), 512)
    assert consumed.q_pe.shape == (tokens, heads * (8 if dcp else 1), 64)
    torch.testing.assert_close(consumed.ql_nope, torch.ones_like(consumed.ql_nope))
    torch.testing.assert_close(consumed.q_pe, torch.full_like(consumed.q_pe, 2.0))
    if not dcp:
        assert not consumed.ql_nope.is_contiguous() and not consumed.q_pe.is_contiguous()
    assert consumed.k.untyped_storage().data_ptr() == history.untyped_storage().data_ptr()
    if dcp:
        impl._dcp_all_gather_fragments.assert_called_once()
    if split:
        # Row 2 is valid but nonlocal; row 3 is padding. Both are excluded
        # from local history; the complete current chunk remains available.
        torch.testing.assert_close(consumed.current_k_nope[:, 0, 0], torch.arange(tokens).float() + 2)
        torch.testing.assert_close(consumed.current_k_pe[:, 0, 0], torch.arange(tokens).float() + 12)
    else:
        assert consumed.current_k_nope is None and consumed.current_k_pe is None
    torch.testing.assert_close(history[:2, 0, 0, 0], torch.tensor([2.0, 3.0]))
    changed = torch.zeros_like(raw, dtype=torch.bool)
    for block in (0, 1):
        start = history.storage_offset() + block * history.stride(0)
        changed[start : start + 576] = True
    torch.testing.assert_close(raw[~changed], before[~changed])
    notify.assert_called_once_with("mla")


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("context_write", [False, True])
def test_prolog_writer_stages_interleaved_cache_and_preserves_guards(interleaved, context_write):
    history, raw = fused_cache()
    if not interleaved:
        page_stride = history.stride(0)
        offset = history.storage_offset()
        history = (
            raw.as_strided((3, 128, 1, 512), (page_stride, 512, 512, 1), offset),
            raw.as_strided((3, 128, 1, 64), (page_stride, 64, 64, 1), offset + 128 * 512),
        )
    else:
        history = (history[..., :512], history[..., 512:])
    before = raw.clone()
    slots = torch.tensor([0, -1, 128, -1])
    buffers = (torch.empty(1, 128, 1, 512), torch.empty(1, 128, 1, 64), torch.arange(4))
    outputs = (object(), object(), object())

    def strict_prolog(**kwargs):
        nope, rope = kwargs["kv_cache"], kwargs["kr_cache"]
        # Check the writer layout chosen by this integration; the mock does
        # not establish which strides the deployed PROLOG version supports.
        for cache in (nope, rope):
            assert cache.stride(1) == cache.shape[2] * cache.shape[3]
            assert cache.stride(2) == cache.shape[3]
        keys = torch.arange(4).float()[:, None, None].expand(4, 1, 512) + 2
        scatter(
            key=keys,
            value=keys[..., :64] + 10,
            key_cache=nope,
            value_cache=rope,
            slot_mapping=kwargs["cache_index"],
        )
        return outputs

    write = Mock(side_effect=strict_prolog)
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_mla_prolog_v3=write),
        "cdiv": lambda x, y: (x + y - 1) // y,
        "DeviceOperator": SimpleNamespace(reshape_and_cache=scatter),
    }
    weights = {name: object() for name in ("weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr")}
    norm = SimpleNamespace(weight=SimpleNamespace(data=object()), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        use_flashmla=True,
        use_mla_rope=True,
        fa_quant_layer=False,
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        _flashmla_writer_buffers=buffers,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        _prepare_prolog_v3_token=lambda x: (x, {}, weights),
        _flatten_prolog_rope_table=lambda x, _: x,
    )
    for name in (
        "_prolog_v3_writer_workspace",
        "_run_prolog_v3",
        "_validate_mla_cache_layout",
        "exec_context_kv_prolog_v3",
    ):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    args = (torch.zeros(4, 16), object(), object(), history, slots)
    if context_write:
        impl.exec_context_kv_prolog_v3(*args)
    else:
        assert impl._run_prolog_v3(*args, quantize_query=True) is outputs
    passed_slots = write.call_args.kwargs["cache_index"]
    torch.testing.assert_close(passed_slots, torch.arange(4) if interleaved else slots)
    torch.testing.assert_close(history[0][:2, 0, 0, 0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(history[1][:2, 0, 0, 0], torch.tensor([12.0, 14.0]))
    assert impl._flashmla_writer_buffers is buffers
    if interleaved:
        changed = torch.zeros_like(raw, dtype=torch.bool)
        for block in (0, 1):
            start = history[0].storage_offset() + block * history[0].stride(0)
            changed[start : start + 576] = True
        torch.testing.assert_close(raw[~changed], before[~changed])


@pytest.mark.parametrize("page_size", [128, 768])
@pytest.mark.parametrize("use_flashmla", [False, True])
@pytest.mark.parametrize("quantized_kv", [False, True])
@pytest.mark.parametrize("quantize_query", [False, True])
def test_prolog_tile_size_is_independent_of_writer_pages(page_size, use_flashmla, quantized_kv, quantize_query):
    operator = Mock(return_value=object())
    scope = {"torch": torch, "torch_npu": SimpleNamespace(npu_mla_prolog_v3=operator)}
    run = load(MLA, "_run_prolog_v3", scope, "AscendMLAImpl")
    cache = (torch.empty(1, page_size, 1, 512), torch.empty(1, page_size, 1, 64))
    slots = torch.tensor([0, 1, page_size - 1], dtype=torch.int64)
    weights = dict.fromkeys(("weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr"), object())
    norm = SimpleNamespace(weight=SimpleNamespace(data=object()), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        use_flashmla=use_flashmla,
        fa_quant_layer=quantized_kv,
        support_fp8_attention=False,
        quant_kscale=torch.ones(1, 1),
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        _prepare_prolog_v3_token=lambda x: (x, {}, weights),
        _flatten_prolog_rope_table=lambda x, _: x,
        _prolog_v3_writer_workspace=lambda *_: (cache, slots, False),
    )

    assert run(impl, torch.zeros(3, 16), object(), object(), cache, slots, quantize_query=quantize_query) is (
        operator.return_value
    )

    operator.assert_called_once()
    passed = operator.call_args.kwargs
    assert passed["tile_size"] == 128
    assert passed["kv_cache_quant_mode"] == int(quantized_kv)
    assert passed["query_quant_mode"] == int(quantize_query and quantized_kv)
    assert passed["cache_mode"] == "PA_BSND"
    assert passed["kv_cache"] is cache[0]
    assert passed["kr_cache"] is cache[1]
    assert cache[0].shape == (1, page_size, 1, 512)
    assert cache[1].shape == (1, page_size, 1, 64)
    torch.testing.assert_close(passed["cache_index"], slots)


@pytest.mark.parametrize("page_size", [128, 1536])
def test_flashmla_writer_workspace_preserves_addresses_and_rejects_overflow(page_size):
    scope = {"torch": torch, "cdiv": lambda x, y: (x + y - 1) // y}
    impl = SimpleNamespace(
        use_flashmla=True,
        use_quant_flashmla=False,
        _flashmla_writer_buffers=None,
        W_UV=torch.empty(1),
        dtype=torch.bfloat16,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=160),
            cache_config=SimpleNamespace(block_size=page_size),
        ),
    )
    for name in ("_prepare_flashmla_writer_buffers", "_prolog_v3_writer_workspace"):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    impl._prepare_flashmla_writer_buffers()
    buffers = impl._flashmla_writer_buffers
    assert buffers[0].shape[1] == min(page_size, 1024)
    impl._prepare_flashmla_writer_buffers()
    assert impl._flashmla_writer_buffers is buffers
    cache, _ = fused_cache()
    views = (cache[..., :512], cache[..., 512:])
    writer, slots, staged = impl._prolog_v3_writer_workspace(160, views, torch.arange(160))
    assert staged and slots.numel() == 160
    assert writer[0].data_ptr() == buffers[0].data_ptr()
    with pytest.raises(RuntimeError, match="scheduled token rows"):
        impl._prolog_v3_writer_workspace(161, views, torch.arange(161))


@pytest.mark.parametrize("fallback", [None, "mixed", "over_capacity", "current_kv", "non_bf16", "dcp"])
def test_quant_flashmla_decode_prolog_writes_c8_cache_and_passes_packed_q(fallback):
    tokens, heads, prolog_heads = 3, 2, 4
    bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
    history = torch.full((2, 128, 1, 576), -8.0).to(fp8)
    flash = SimpleNamespace(
        slots=torch.tensor([1, 128, -1]),
        query=torch.zeros(tokens, heads, 576).to(fp8),
        adapter=SimpleNamespace(attention=Mock(return_value=(torch.ones(heads, tokens, 512), None))),
        block_table=object(),
        cache_lens=object(),
        cu=object(),
        used_q=object(),
        schedule=object(),
        attn_mask=None,
    )
    cos = sin = torch.ones(tokens, 64, dtype=bf16)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decodes=2,
        num_prefills=int(fallback == "mixed"),
        num_decode_tokens=tokens,
        slot_mapping=torch.tensor([0, 2, 3]),
        external_flashmla=flash,
        decode=SimpleNamespace(cos=cos, sin=sin),
    )

    def prolog(**kw):
        assert kw["token_x"].shape == (tokens, 8)
        assert kw["weight_quant_mode"] == kw["query_quant_mode"] == kw["kv_cache_quant_mode"] == 0
        assert kw["quant_scale_ckv"] is None
        assert kw["kv_cache"].dtype == kw["kr_cache"].dtype == bf16
        torch.testing.assert_close(kw["cache_index"], torch.arange(tokens))
        kv_values = torch.arange(1, tokens + 1, dtype=bf16).view(tokens, 1, 1)
        kw["kv_cache"].flatten(0, 1)[:tokens].copy_(kv_values)
        kw["kr_cache"].flatten(0, 1)[:tokens].copy_(kv_values + 3)
        q_nope = torch.full((tokens, prolog_heads, 512), 50, dtype=bf16)
        q_rope = torch.full((tokens, prolog_heads, 64), 50, dtype=bf16)
        q_nope[:, :heads] = 1
        q_rope[:, :heads] = 2
        return q_nope, q_rope, None, None, None

    def static_quant(x, scale, offset, dtype, axis, div_mode):
        assert offset is None and axis == -1 and not div_mode and dtype == fp8
        assert x.dtype == torch.float32 and x.is_contiguous()
        return (x.float() * scale).to(fp8)

    def dynamic_quant(x, *, dst_type):
        assert x.dtype == bf16 and x.is_contiguous() and dst_type == fp8
        torch.testing.assert_close(x[..., :512], torch.ones_like(x[..., :512]))
        torch.testing.assert_close(x[..., 512:], torch.full_like(x[..., 512:], 4.0))
        return x.to(fp8), torch.arange(1, 1 + x.size(0) * x.size(1), dtype=torch.float32).view(x.shape[:2])

    npu = SimpleNamespace(
        npu_mla_prolog_v3=Mock(side_effect=prolog),
        npu_quantize=Mock(side_effect=static_quant),
        npu_dynamic_quant=Mock(side_effect=dynamic_quant),
    )
    scope = {
        "torch": torch,
        "torch_npu": npu,
        "_EXTRA_CTX": SimpleNamespace(num_tokens=tokens),
        "notify_kv_cache_written": Mock(),
        "cdiv": lambda x, y: (x + y - 1) // y,
        "DeviceOperator": SimpleNamespace(reshape_and_cache=Mock(side_effect=scatter)),
        "record_attention_compute_start": Mock(),
        "maybe_save_kv_layer_to_connector": Mock(),
        "wait_for_kv_layer_from_connector": Mock(),
        "DecodeMLAPreprocessResult": SimpleNamespace,
    }
    weights = dict.fromkeys(("weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr"), object())
    norm = SimpleNamespace(weight=SimpleNamespace(data=object()), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        layer_name="mla",
        layerwise_kv_cache_hook=None,
        num_heads=heads,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_flashmla=True,
        use_quant_flashmla=True,
        use_pcp=False,
        use_dcp=False,
        is_pcp_decode_sharded=False,
        fa_quant_layer=True,
        enable_mlapo=False,
        use_output_gate=False,
        _quant_flashmla_prolog_workspace=(
            torch.empty(1, 128, 1, 512, dtype=bf16),
            torch.empty(1, 128, 1, 64, dtype=bf16),
            torch.arange(tokens - int(fallback == "over_capacity")),
        ),
        _quant_flashmla_writer_scales=(torch.ones(64), torch.full((512,), 2.0)),
        fak_descale_float=torch.tensor([0.5]),
        _prolog_v3_weights={"num_heads": prolog_heads},
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        _prepare_prolog_v3_token=lambda x: (x, {"weight_quant_mode": 0}, weights),
        _flatten_prolog_rope_table=lambda x, _: x,
        get_num_actual_tokens=lambda m: m.num_actual_tokens,
        _decode_requires_current_kv=lambda m: fallback == "current_kv",
        _prefill_requires_current_kv=lambda m: False,
        _cache_requires_prolog_v3_writer=lambda cache: False,
        _uses_strided_kv_adapter=lambda: False,
        _mla_preprocess=Mock(return_value=(None, None, None)),
        _v_up_proj=lambda latent: torch.ones(tokens, heads * 128),
        o_proj=lambda x, **_: (x,),
    )
    for name in ("_quantize_flashmla_query", "_quantize_flashmla_kv_rows"):
        load(MLA, name, scope)
    for name in (
        "_forward_external_flashmla",
        "_unpack_prolog_decode_q",
        "_preprocess_quant_flashmla_with_prolog",
        "_prolog_v3_writer_workspace",
        "_run_prolog_v3",
        "_validate_mla_cache_layout",
        "forward",
    ):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))

    if fallback == "dcp":
        impl.use_dcp = True
        impl._decode_requires_current_kv = lambda _: True
        impl.reorg_decode_q = Mock(side_effect=lambda *parts: tuple(torch.cat([part] * 8, dim=1) for part in parts))
        impl._forward_external_flashmla = Mock(return_value=torch.zeros(tokens, heads * 128))
    dtype = torch.float32 if fallback == "non_bf16" else bf16
    output = torch.empty(tokens, heads * 128, dtype=dtype)
    assert impl.forward("mla", torch.zeros(tokens, 8, dtype=dtype), history, metadata, output) is output
    if fallback == "dcp":
        impl.reorg_decode_q.assert_called_once()
        result = impl._forward_external_flashmla.call_args.args[0]
        assert result.ql_nope.shape == (tokens, heads * 8, 512)
        assert result.q_pe.shape == (tokens, heads * 8, 64)
        assert result.dequant_scale_q_nope.shape == (tokens, heads * 8)
        # The last slot is non-owned in history but participates in current attention.
        torch.testing.assert_close(result.current_k_nope[-1].float(), torch.full((1, 512), 6.0))
        torch.testing.assert_close(result.current_k_pe[-1].float(), torch.full((1, 64), 6.0))
        torch.testing.assert_close(history[0, 0, 0].float(), torch.full((576,), -8.0))
        return
    if fallback is not None:
        impl._mla_preprocess.assert_called_once()
        npu.npu_mla_prolog_v3.assert_not_called()
        return

    impl._mla_preprocess.assert_not_called()
    npu.npu_mla_prolog_v3.assert_called_once()
    assert npu.npu_quantize.call_count == 2
    scope["notify_kv_cache_written"].assert_called_once_with("mla")
    passed_query, passed_cache = flash.adapter.attention.call_args.args
    assert passed_query.is_contiguous() and passed_cache is history and passed_query is not flash.query
    torch.testing.assert_close(passed_query.float()[..., :512], torch.ones(tokens, heads, 512))
    torch.testing.assert_close(passed_query.float()[..., 512:], torch.full((tokens, heads, 64), 4.0))
    torch.testing.assert_close(
        flash.adapter.attention.call_args.kwargs["q_descale"],
        torch.arange(1, 7, dtype=torch.float32).view(tokens, heads),
    )
    torch.testing.assert_close(history[0, 1, 0].float(), torch.cat((torch.full((512,), 2.0), torch.full((64,), 4.0))))
    torch.testing.assert_close(history[1, 0, 0].float(), torch.cat((torch.full((512,), 4.0), torch.full((64,), 5.0))))
    torch.testing.assert_close(history[0, 0, 0].float(), torch.full((576,), -8.0))
