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
DSPARK = "vllm_ascend/models/kimi_k3_dspark.py"


def load(path, name, scope, owner=None):
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
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


@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_normalizes_token_fused_views_before_selecting_writer(enabled):
    fused, _ = fused_cache()
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_mla_prolog_v3=Mock()),
        "get_cos_and_sin_mla": lambda _: (object(), object()),
        "_log_dspark_mla_cache_diagnostic": lambda *args: None,
    }
    normalize = load(DSPARK, "_normalize_dspark_mla_cache", scope)
    scope["_normalize_dspark_mla_cache"] = normalize
    write = load(DSPARK, "precompute_and_store_context_kv", scope, "AscendK3DSparkModel")
    impl = SimpleNamespace(
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        num_kv_heads=1,
        qk_rope_head_dim=64,
        use_mla_rope=enabled,
        fa_quant_layer=False,
        enable_kv_nz=False,
        supports_prolog_v3_quantization=lambda: True,
        supports_prolog_v3_projection=lambda: True,
        fused_qkv_a_proj=object(),
        q_proj=SimpleNamespace(_chunk_size=0),
        q_lora_rank=4,
        exec_context_kv_prolog_v3=Mock(),
        exec_kv_prefill=Mock(),
    )
    for name in (
        "supports_prolog_v3_dtype",
        "supports_prolog_v3_writer",
        "_validate_prolog_v3_cache",
        "_cache_requires_prolog_v3_writer",
    ):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    choose = Mock(wraps=impl._cache_requires_prolog_v3_writer)
    impl._cache_requires_prolog_v3_writer = choose
    attn = SimpleNamespace(
        impl=impl, kv_cache=fused, q_lora_rank=4, fused_qkv_a_proj=Mock(return_value=(torch.zeros(2, 580),))
    )
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    slots = torch.tensor([1, 129])
    write(model, torch.zeros(2, 16), torch.arange(2), slots)
    views = choose.call_args.args[0]
    assert tuple(t.shape for t in views) == ((3, 128, 1, 512), (3, 128, 1, 64))
    assert all(t.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr() for t in views)
    assert views[1].storage_offset() - views[0].storage_offset() == 512
    if enabled:
        assert impl.exec_context_kv_prolog_v3.call_args.args[3] is views
        impl.exec_kv_prefill.assert_not_called()
        attn.fused_qkv_a_proj.assert_not_called()
    else:
        assert impl.exec_kv_prefill.call_args.args[3] is views
        impl.exec_context_kv_prolog_v3.assert_not_called()


@pytest.mark.parametrize(
    "dcp,causal,speculative",
    [(False, True, False), (False, False, True), (True, True, True), (True, False, True), (True, True, False)],
)
@pytest.mark.parametrize("heads", [8, 12])
def test_external_forward_reuses_upstream_dcp_prolog(dcp, causal, speculative, heads):
    split = dcp and causal and speculative
    history, raw = fused_cache()
    before = raw.clone()
    slots = torch.tensor([0, 128, -1, -1])
    tokens = slots.numel()
    attention_output = torch.ones(heads, tokens, 512)
    # Injected unused-row values check this consumer, not the NPU output contract.
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
        fused_qkv_a_proj=object(),
        q_proj=SimpleNamespace(_chunk_size=0),
        q_lora_rank=4,
        supports_prolog_v3_quantization=lambda: True,
        supports_prolog_v3_projection=lambda: True,
        use_output_gate=False,
        pcp_enabled=False,
        is_pcp_decode_sharded=False,
        dcp_enabled=dcp,
        enable_mlapo=False,
        _dcp_current_kv_buffers=(torch.empty(1, 128, 1, 512), torch.empty(1, 128, 1, 64), torch.arange(128)),
        get_num_actual_tokens=lambda m: m.num_actual_tokens,
        _prolog_v3_weights={"num_heads": 16},
        o_proj=lambda tensor, **kw: (tensor,),
    )
    for name in (
        "supports_prolog_v3_dtype",
        "supports_prolog_v3_writer",
        "_validate_prolog_v3_cache",
        "_cache_requires_prolog_v3_writer",
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

        def project(latent):
            torch.testing.assert_close(latent[:, :-1], torch.ones_like(latent[:, :-1]))
            torch.testing.assert_close(latent[:, -1], torch.zeros_like(latent[:, -1]))
            return latent[..., :128].transpose(0, 1).reshape(tokens, heads * 128)

        impl._v_up_proj = project
        external = load(MLA, "_forward_external_flashmla", scope, "AscendMLAImpl")
        impl._forward_external_flashmla = Mock(wraps=MethodType(external, impl))
    impl._forward_decode = Mock(side_effect=AssertionError("external decode must not call FIA"))
    output = torch.empty(tokens, heads * 128)
    assert impl.forward("mla", torch.zeros(tokens, 16), history, metadata, output) is output
    expected = torch.ones_like(output)
    if not dcp:
        expected[-1].zero_()
        query_arg, cache_arg = flash.adapter.attention.call_args.args
        assert query_arg is flash.query and cache_arg is history
        torch.testing.assert_close(flash.query[..., :512], torch.ones_like(flash.query[..., :512]))
        torch.testing.assert_close(flash.query[..., 512:], torch.full_like(flash.query[..., 512:], 2.0))
    torch.testing.assert_close(output, expected)
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


@pytest.mark.parametrize("use_v2_runner", [False, True])
@pytest.mark.parametrize(
    "connector,module,expected",
    [
        (None, None, True),
        ("MooncakeConnectorV2", None, True),
        ("MooncakePullConnector", None, True),
        ("MooncakeConnectorV1", None, False),
        ("MooncakeConnectorV2", "custom.connector", False),
    ],
)
def test_external_selection_retains_token_fused_pd(connector, module, expected, use_v2_runner):
    scope = {
        "torch": torch,
        "HardwareCapability": SimpleNamespace(MLA_FLASH="mla"),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "MLA_FLASH_SUPPORTED_Q_HEADS": {8, 12, 64, 96},
        "FLASHMLA_V_DIM": 512,
        "FLASHMLA_QK_DIM": 576,
        "enable_sfa": lambda _: False,
        "get_flashmla_ops": Mock(return_value=(Mock(), Mock())),
    }
    for name in ("supports_component_major_mla_pd",):
        load("vllm_ascend/core/kv_cache_interface.py", name, scope)
    select = load(MLA, "_can_use_flashmla", scope, "AscendMLAImpl")
    transfer = None if connector is None else SimpleNamespace(kv_connector=connector, kv_connector_module_path=module)
    impl = SimpleNamespace(
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
        supports_prolog_v3_quantization=lambda: True,
        supports_prolog_v3_projection=lambda: True,
        q_proj=SimpleNamespace(_chunk_size=0),
        vllm_config=SimpleNamespace(
            use_v2_model_runner=use_v2_runner,
            speculative_config=None,
            kv_transfer_config=transfer,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
    )
    expected = expected and use_v2_runner
    assert select(impl) is expected
    if not expected:
        scope["get_flashmla_ops"].assert_not_called()


@pytest.mark.parametrize("mode", ["normal", "dcp", "pcp", "quantization", "chunked_projection", "no_q_lora"])
def test_external_selection_reaches_mrv2_single_raw_allocation(mode):
    class Spec:
        supports_single_raw_backing = True

    class Layer:
        impl: SimpleNamespace

    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=8 if mode == "dcp" else 1),
    )
    impl = SimpleNamespace(
        vllm_config=config,
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=mode == "pcp",
        dcp_enabled=mode == "dcp",
        is_pcp_decode_sharded=False,
        supports_prolog_v3_quantization=lambda: mode != "quantization",
        supports_prolog_v3_projection=lambda: mode != "no_q_lora",
        q_proj=SimpleNamespace(_chunk_size=128 if mode == "chunked_projection" else 0),
    )
    layer = Layer()
    layer.impl = impl
    scope = {
        "torch": torch,
        "AscendMLAAttentionSpec": Spec,
        "MLAAttention": Layer,
        "AttentionLayerBase": Layer,
        "get_layers_from_vllm_config": lambda *args: {"mla": layer},
        "HardwareCapability": SimpleNamespace(MLA_FLASH="mla"),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "MLA_FLASH_SUPPORTED_Q_HEADS": {8, 12, 64, 96},
        "FLASHMLA_V_DIM": 512,
        "FLASHMLA_QK_DIM": 576,
        "enable_sfa": lambda _: False,
        "get_flashmla_ops": lambda: (Mock(), Mock()),
    }
    for name in ("supports_component_major_mla_pd",):
        load("vllm_ascend/core/kv_cache_interface.py", name, scope)
    select = load(MLA, "_can_use_flashmla", scope, "AscendMLAImpl")
    impl.use_flashmla = select(impl)
    allocate = load("vllm_ascend/worker/v2/attn_utils.py", "_uses_single_raw_mla_cache", scope)
    single_raw = allocate(config, "mla", Spec())
    assert select(impl) is (mode != "pcp")
    assert single_raw is (mode != "pcp")


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
        "_validate_context_prolog_v3_cache",
        "_validate_prolog_v3_cache",
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


def test_flashmla_writer_workspace_preserves_addresses_and_rejects_overflow():
    scope = {"torch": torch, "cdiv": lambda x, y: (x + y - 1) // y}
    impl = SimpleNamespace(
        use_flashmla=True,
        _flashmla_writer_buffers=None,
        W_UV=torch.empty(1),
        dtype=torch.bfloat16,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=160),
            cache_config=SimpleNamespace(block_size=128),
        ),
    )
    for name in ("_prepare_flashmla_writer_buffers", "_prolog_v3_writer_workspace"):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    impl._prepare_flashmla_writer_buffers()
    buffers = impl._flashmla_writer_buffers
    impl._prepare_flashmla_writer_buffers()
    assert impl._flashmla_writer_buffers is buffers
    cache, _ = fused_cache()
    views = (cache[..., :512], cache[..., 512:])
    writer, slots, staged = impl._prolog_v3_writer_workspace(160, views, torch.arange(160))
    assert staged and slots.numel() == 160
    assert writer[0].data_ptr() == buffers[0].data_ptr()
    with pytest.raises(RuntimeError, match="scheduled token rows"):
        impl._prolog_v3_writer_workspace(161, views, torch.arange(161))


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("block_size", [128, 768])
def test_prolog_tile_is_not_derived_from_external_current_workspace(external, block_size):
    # Native PROLOG retains main's geometry-derived attribute. External DCP and
    # context writers retain tile128 even when their private dense cache is768.
    write = Mock(return_value=(object(), object(), object()))
    scope = {"torch": torch, "torch_npu": SimpleNamespace(npu_mla_prolog_v3=write)}
    weights = {name: object() for name in ("weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr")}
    norm = SimpleNamespace(weight=SimpleNamespace(data=object()), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        use_flashmla=external,
        fa_quant_layer=False,
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        _prepare_prolog_v3_token=lambda x: (x, {}, weights),
        _flatten_prolog_rope_table=lambda x, _: x,
        _prolog_v3_writer_workspace=lambda tokens, cache, slots: (cache, slots, False),
    )
    run = load(MLA, "_run_prolog_v3", scope, "AscendMLAImpl")
    cache = (torch.empty(1, block_size, 1, 512), torch.empty(1, block_size, 1, 64))
    run(impl, torch.zeros(1, 16), object(), object(), cache, torch.tensor([0]), quantize_query=False)
    assert write.call_args.kwargs["tile_size"] == (128 if external else block_size)
