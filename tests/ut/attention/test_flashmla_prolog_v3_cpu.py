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
    nodes = ast.parse((ROOT / path).read_text()).body
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
        num_kv_heads=1,
        qk_rope_head_dim=64,
        use_mla_rope=enabled,
        enable_kv_nz=False,
        supports_prolog_v3_quantization=lambda: True,
        supports_prolog_v3_projection=lambda: True,
        fused_qkv_a_proj=object(),
        q_proj=SimpleNamespace(_chunk_size=0),
        q_lora_rank=4,
        exec_context_kv_prolog_v3=Mock(),
        exec_kv_prefill=Mock(),
    )
    for name in ("_validate_prolog_v3_cache", "_cache_requires_prolog_v3_writer"):
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


@pytest.mark.parametrize("dcp,causal", [(False, True), (False, False), (True, True), (True, False)])
def test_external_forward_respects_upstream_prolog_exclusions(dcp, causal):
    history, raw = fused_cache()
    before = raw.clone()
    slots = torch.tensor([0, 128, -1])
    flash = SimpleNamespace(
        slots=slots,
        query=torch.empty(3, 8, 576),
        adapter=SimpleNamespace(attention=Mock(return_value=(torch.ones(8, 3, 512), None))),
        block_table=object(),
        cache_lens=object(),
        cu=object(),
        used_q=object(),
        schedule=object(),
        attn_mask=None,
        token_live=torch.tensor([True, True, False]),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=3,
        slot_mapping=slots,
        external_flashmla=flash,
        decode=SimpleNamespace(cos=None, sin=None),
    )
    notify = Mock()
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_mla_prolog_v3=Mock()),
        "_EXTRA_CTX": SimpleNamespace(num_tokens=3),
        "notify_kv_cache_written": notify,
        "record_attention_compute_start": Mock(),
        "maybe_save_kv_layer_to_connector": Mock(),
        "DecodeMLAPreprocessResult": lambda q, pe, k, kr, **kw: SimpleNamespace(ql_nope=q, q_pe=pe, k=k, kr=kr, **kw),
    }
    impl = SimpleNamespace(
        layer_name="mla",
        layerwise_kv_cache_hook=None,
        num_heads=8,
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
        get_num_actual_tokens=lambda m: m.num_actual_tokens,
        _decode_requires_current_kv=lambda _: dcp and causal,
        _prolog_v3_weights={"num_heads": 8},
        o_proj=lambda tensor, **kw: (tensor,),
    )
    for name in (
        "_validate_prolog_v3_cache",
        "_cache_requires_prolog_v3_writer",
        "mla_preprocess_prolog_v3",
        "forward",
    ):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))

    def prolog(hidden, cos, sin, dest, indices, *, quantize_query):
        assert quantize_query
        torch.testing.assert_close(indices, slots)
        keys = torch.arange(3).float()[:, None, None].expand(3, 1, 512) + 2
        scatter(key=keys, value=keys[..., :64] + 10, key_cache=dest[0], value_cache=dest[1], slot_mapping=indices)
        return torch.ones(3, 8 * 512), torch.ones(3, 8 * 64), None

    impl._run_prolog_v3 = Mock(side_effect=prolog)
    native_result = object()
    impl._mla_preprocess = Mock(return_value=(native_result, None))
    impl.mla_preprocess_prolog_v3 = Mock(wraps=impl.mla_preprocess_prolog_v3)
    if dcp:
        # The DCP implementation has separate dense-reference/collective tests.
        impl._forward_external_flashmla = Mock(return_value=torch.ones(3, 8 * 128))
    else:
        impl._v_up_proj = lambda latent: latent[..., :128].transpose(0, 1).reshape(3, 8 * 128)
        external = load(MLA, "_forward_external_flashmla", scope, "AscendMLAImpl")
        impl._forward_external_flashmla = Mock(wraps=MethodType(external, impl))
    impl._forward_decode = Mock(side_effect=AssertionError("external decode must not call FIA"))
    output = torch.empty(3, 8 * 128)
    assert impl.forward("mla", torch.zeros(3, 16), history, metadata, output) is output
    expected = torch.ones_like(output)
    if not dcp:
        expected[-1].zero_()
        query_arg, cache_arg = flash.adapter.attention.call_args.args
        assert query_arg is flash.query and cache_arg is history
        torch.testing.assert_close(flash.query, torch.ones_like(flash.query))
    torch.testing.assert_close(output, expected)
    consumed, cache_arg, meta_arg = impl._forward_external_flashmla.call_args.args
    assert cache_arg is history and meta_arg is metadata
    if dcp:
        impl.mla_preprocess_prolog_v3.assert_not_called()
        impl._mla_preprocess.assert_called_once()
        assert consumed is native_result
        torch.testing.assert_close(raw, before)
    else:
        impl._mla_preprocess.assert_not_called()
        impl._run_prolog_v3.assert_called_once()
        assert consumed.ql_nope.shape == (3, 8, 512)
        assert consumed.q_pe.shape == (3, 8, 64)
        assert consumed.k.untyped_storage().data_ptr() == history.untyped_storage().data_ptr()
        torch.testing.assert_close(history[:2, 0, 0, 0], torch.tensor([2.0, 3.0]))
        # Prefix, physical page padding, and unused rows remain untouched.
        changed = torch.zeros_like(raw, dtype=torch.bool)
        for block in (0, 1):
            start = history.storage_offset() + block * history.stride(0)
            changed[start : start + 576] = True
        torch.testing.assert_close(raw[~changed], before[~changed])
        notify.assert_called_once_with("mla")


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
def test_external_selection_retains_token_fused_pd(connector, module, expected):
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
    for name in ("supports_component_major_mla_pd", "pd_uses_component_major_mla_wire_layout"):
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
            use_v2_model_runner=True,
            speculative_config=None,
            kv_transfer_config=transfer,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
    )
    assert select(impl) is expected
    if not expected:
        scope["get_flashmla_ops"].assert_not_called()


@pytest.mark.parametrize("mode", ["normal", "dcp", "pcp", "quantization", "chunked_projection", "no_q_lora"])
@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_external_selection_reaches_single_raw_allocation(mode, runner):
    class Spec:
        supports_single_raw_backing = True

    class Layer:
        pass

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
    for name in ("supports_component_major_mla_pd", "pd_uses_component_major_mla_wire_layout"):
        load("vllm_ascend/core/kv_cache_interface.py", name, scope)
    select = load(MLA, "_can_use_flashmla", scope, "AscendMLAImpl")
    impl.use_flashmla = select(impl)
    if runner == "v2":
        allocate = load("vllm_ascend/worker/v2/attn_utils.py", "_uses_single_raw_mla_cache", scope)
        single_raw = allocate(config, "mla", Spec())
    else:
        allocate = load("vllm_ascend/worker/model_runner_v1.py", "_uses_single_raw_mla_cache", scope, "NPUModelRunner")
        worker = SimpleNamespace(
            vllm_config=config, use_sparse=False, sparse_kv_offload_enabled=False, use_compress=False
        )
        single_raw = allocate(worker, "mla", Spec(), layer)
    assert select(impl) is (mode != "pcp")
    assert single_raw is (mode != "pcp")
