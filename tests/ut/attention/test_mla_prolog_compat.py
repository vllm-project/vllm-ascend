# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration contracts extracted from the actual SFA/Indexer source.

This compiles selected AST methods, not complete attention modules. The opaque
MLA boundary test uses the real packaged dispatcher with test-only CPU kernels;
vLLM runner integration and NPU execution are not exercised.
Run with --confcutdir=tests/ut/attention to bypass global NPU fixtures.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SFA_PATH = ROOT / "vllm_ascend/attention/sfa_v1.py"
INDEXER_PATH = ROOT / "vllm_ascend/attention/indexer_kpool.py"
DEVICE_PATH = ROOT / "vllm_ascend/device/device_op.py"
MLA_PATH = ROOT / "vllm_ascend/ops/mla.py"
HIDDEN_SIZE = 1024
Q_LORA_RANK = 640
KV_LORA_RANK = 512
NUM_HEADS = 2


class Mxfp8Method:
    group_size = 32
    dynamic_mx_quant_scale_alg = 0


class DynamicMethod:
    pass


class StaticMethod:
    pass


class UnquantizedMethod:
    pass


def _extract_class(path, class_name, names, namespace):
    """Execute unchanged selected methods; fail if production names disappear."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == set(names)
    # Parsing supplies version-specific AST fields (e.g. type_params on 3.12+).
    extracted = ast.parse("class Extracted: pass").body[0]
    assert isinstance(extracted, ast.ClassDef)
    extracted.name = class_name
    extracted.body = [*methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), extracted],
        type_ignores=[],
    )
    namespace = dict(namespace, torch=torch)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


def _quant_globals():
    return dict(
        MX_GROUP_SIZE=32,
        AscendW8A8MXFP8DynamicLinearMethod=Mxfp8Method,
        AscendW8A8DynamicLinearMethod=DynamicMethod,
        AscendW8A8LinearMethod=StaticMethod,
        AscendUnquantizedLinearMethod=UnquantizedMethod,
    )


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("cache_quant", [False, True])
@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("indexer_mode", ["mx", "bf16", "skip_topk", "no_runtime", "zero_local"])
def test_shared_forward_nope_shapes_and_indexer_scales(tokens, cache_quant, with_scale, indexer_mode):
    x_quant = torch.zeros((tokens, HIDDEN_SIZE), dtype=torch.uint8).view(torch.float8_e4m3fn)
    x_scale = torch.zeros((tokens, HIDDEN_SIZE // 32), dtype=torch.uint8)
    q_norm = torch.zeros((tokens, Q_LORA_RANK), dtype=torch.uint8).view(torch.float8_e4m3fn)
    q_scale = torch.zeros((tokens, Q_LORA_RANK // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    if not with_scale:
        q_scale = torch.empty(0, dtype=torch.float8_e8m0fnu)
    adapter = Mock(
        return_value=(
            torch.empty((tokens, NUM_HEADS, KV_LORA_RANK), dtype=torch.bfloat16),
            torch.empty((tokens, NUM_HEADS, 0), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.float32),
            q_norm,
            q_scale,
        )
    )
    npu = SimpleNamespace(npu_dynamic_mx_quant=Mock(return_value=(x_quant, x_scale)))
    cls = _extract_class(
        SFA_PATH,
        "AscendSFAImpl",
        ["_sfa_preprocess_prolog_v3", "_uses_cannbot_prolog"],
        dict(_quant_globals(), mla_prolog=adapter, torch_npu=npu),
    )
    impl = cls()
    impl._quant_type = Mxfp8Method
    impl._mla_prolog_backend = "cannbot"
    impl._mla_prolog_prepared = object()
    impl.enable_sparse_sfa_c8 = cache_quant
    impl.has_indexer = True
    native_query = indexer_mode != "mx"
    impl._uses_native_indexer_query = Mock(return_value=native_query)
    impl.runtime_has_indexer = indexer_mode != "no_runtime"
    impl.skip_topk = indexer_mode == "skip_topk"
    actual_tokens = 0 if indexer_mode == "zero_local" else max(tokens - 1, 1)
    metadata = SimpleNamespace(num_actual_tokens=actual_tokens)
    impl.local_num_heads = NUM_HEADS
    impl.qk_rope_head_dim = 0
    impl.kv_lora_rank = KV_LORA_RANK
    impl.q_lora_rank = Q_LORA_RANK
    impl.q_a_layernorm = Mock()
    impl.q_a_layernorm.weight = torch.ones(Q_LORA_RANK, dtype=torch.bfloat16)
    impl.q_a_layernorm.variance_epsilon = 1e-6
    impl.q_a_layernorm.side_effect = lambda x: torch.nn.functional.rms_norm(
        x, (Q_LORA_RANK,), impl.q_a_layernorm.weight, 1e-6
    )
    source_qkv = (torch.arange(tokens * (Q_LORA_RANK + KV_LORA_RANK)) % 31).to(torch.bfloat16)
    source_qkv = source_qkv.reshape(tokens, Q_LORA_RANK + KV_LORA_RANK)
    impl.fused_qkv_a_proj = Mock(return_value=(source_qkv,))
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(KV_LORA_RANK), variance_epsilon=1e-6)
    impl.sfa_qsfa_tile_size = 128
    impl.sfa_qsfa_k_nope_clip_alpha = None
    impl.sfa_qsfa_kr_cache_dummy = torch.empty(0)
    for name in ("weight_dq", "weight_dkv_kr", "weight_uq_qr", "W_UK_T"):
        setattr(impl, name, object())
    for name in ("weight_dq_scale", "weight_dkv_kr_scale", "weight_uq_qr_scale"):
        setattr(impl, name, torch.zeros(1, dtype=torch.uint8))
    hidden = torch.zeros((tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
    cache = torch.empty((2, 16, 528 if cache_quant else 512), dtype=torch.uint8 if cache_quant else torch.bfloat16)
    slots = torch.arange(tokens, dtype=torch.int64)
    # A NoPE layer has no second (KR) cache; this catches accidental tuple indexing.
    if native_query:
        # Disabled query_norm has no payload, even for a nonempty decode batch.
        adapter.return_value = (*adapter.return_value[:3], torch.empty(0), torch.empty(0))
    hidden_out, query, rope, latent = impl._sfa_preprocess_prolog_v3(
        hidden, (cache,), None, None, slots, attn_metadata=metadata
    )
    assert hidden_out is hidden
    assert query.shape == (tokens, NUM_HEADS, KV_LORA_RANK)
    assert rope.shape == (tokens, NUM_HEADS, 0)
    # (0, Q/32) is a valid T0 MX scale tensor; only (0,) means absent.
    if native_query:
        needs_query = tokens > 0 and indexer_mode == "bf16"
        if needs_query:
            impl.fused_qkv_a_proj.assert_called_once_with(hidden)
            impl.q_a_layernorm.assert_called_once()
            norm_input = impl.q_a_layernorm.call_args.args[0]
            assert norm_input.shape == (tokens, Q_LORA_RANK)  # Keep FULL graph padding.
            assert norm_input.untyped_storage().data_ptr() == source_qkv.untyped_storage().data_ptr()
            expected = torch.nn.functional.rms_norm(
                source_qkv[..., :Q_LORA_RANK], (Q_LORA_RANK,), impl.q_a_layernorm.weight, 1e-6
            )
            torch.testing.assert_close(latent, expected, rtol=0, atol=0)
            assert latent.dtype == torch.bfloat16
        else:
            assert latent is None
            impl.fused_qkv_a_proj.assert_not_called()
            impl.q_a_layernorm.assert_not_called()
    elif with_scale:
        impl.fused_qkv_a_proj.assert_not_called()
        assert isinstance(latent, tuple) and len(latent) == 2
        assert latent[0].shape == (tokens, Q_LORA_RANK)
        assert latent[1].shape == (tokens, Q_LORA_RANK // 32)
        assert latent[1].dtype == torch.float8_e8m0fnu
    else:
        assert isinstance(latent, torch.Tensor)
        assert latent.shape == (tokens, Q_LORA_RANK)
    if tokens:
        npu.npu_dynamic_mx_quant.assert_called_once_with(hidden, dst_type=torch.float8_e4m3fn)
    else:
        npu.npu_dynamic_mx_quant.assert_not_called()
    adapter.assert_called_once()
    call = adapter.call_args.kwargs
    assert call["query_norm_flag"] is (not native_query)
    assert call["backend"] == "cannbot"
    assert call["prepared"] is impl._mla_prolog_prepared
    if tokens:
        assert call["token_x"] is x_quant
    else:
        assert call["token_x"].shape == (0, HIDDEN_SIZE)
        assert call["token_x"].dtype == torch.float8_e4m3fn
    assert call["kv_cache"] is cache
    assert call["rope_sin"] is None and call["rope_cos"] is None
    assert call["kr_cache"] is None or call["kr_cache"].numel() == 0
    assert call["dequant_scale_x"].shape == (tokens, HIDDEN_SIZE // 32)
    assert call["kv_cache_quant_mode"] == (3 if cache_quant else 0)


def _projection(method=None):
    return SimpleNamespace(
        quant_method=SimpleNamespace(quant_method=method or Mxfp8Method()),
        bias=None,
        mxfp8_tp_padding=(0, 0),
        weight_scale=torch.empty(1),
    )


@pytest.fixture
def routing():
    pp = SimpleNamespace(NATIVE=object(), PROLOG_V3=object(), MLAPO=object())
    family = SimpleNamespace(FP8_OPTIMIZED=object())
    profile = SimpleNamespace(device_adaptor_family=family.FP8_OPTIMIZED)
    ascend = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    device_operator = _extract_class(DEVICE_PATH, "A5DeviceAdaptor", ["get_mla_prolog_backend"], {})
    kpool_cls = _extract_class(
        INDEXER_PATH, "Glm5NextKPoolIndexerBackend", ["supports_bf16_query_input"], _quant_globals()
    )
    names = [
        "_select_mla_prolog_backend",
        "_uses_cannbot_prolog",
        "_cannbot_config_unsupported_reasons",
        "_get_layer_quant_method",
        "_resolve_preprocess_type",
        "_fused_preprocess_type",
        "_get_fused_type_unsupported_reasons",
        "_uses_native_indexer_query",
        "_try_enable_type",
    ]
    cls = _extract_class(
        SFA_PATH,
        "AscendSFAImpl",
        names,
        dict(
            _quant_globals(),
            PreprocessType=pp,
            Glm5NextKPoolIndexerBackend=kpool_cls,
            DeviceAdaptorFamily=family,
            DeviceOperator=device_operator,
            get_current_hardware_profile=lambda: profile,
            get_ascend_config=lambda: ascend,
            logger=Mock(),
        ),
    )
    impl = cls()
    impl._mla_prolog_backend = "cannbot"
    impl.qk_rope_head_dim = 0
    impl.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        kv_transfer_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    impl.rl_weight_update_enabled = False
    impl.enable_sp = False
    impl.enable_sparse_sfa_turboquant = False
    impl.enable_sparse_sfa_c8 = False
    impl.c8_cache_dtype = torch.float8_e4m3fn
    impl.fused_qkv_a_proj = _projection()
    impl.q_proj = _projection()
    impl.indexer = SimpleNamespace(wq_b=_projection(), impl=kpool_cls())
    impl.has_indexer = True
    impl.q_a_layernorm = object()
    impl.kv_a_layernorm = object()
    impl.is_kv_consumer = False
    impl.enable_mlapo = False
    impl._process_weights_for_fused_prolog_v3 = Mock()
    impl._process_weights_for_fused_mlapo = Mock()
    impl._mla_prolog_backend = impl._select_mla_prolog_backend()
    return SimpleNamespace(impl=impl, pp=pp, profile=profile, ascend=ascend, device_operator=device_operator)


@pytest.mark.parametrize("tokens", [1, 128, 8192])
@pytest.mark.parametrize(
    "backend,rope,preprocess,state,prefills,expected",
    [
        ("cannbot", 0, "PROLOG_V3", "DecodeOnly", 0, "PROLOG_V3"),
        ("cannbot", 0, "PROLOG_V3", "DecodeOnly", 1, "NATIVE"),
        ("cannbot", 0, "PROLOG_V3", "PrefillNoCache", 1, "NATIVE"),
        ("cannbot", 0, "PROLOG_V3", "PrefillCacheHit", 1, "NATIVE"),
        ("cannbot", 0, "PROLOG_V3", "ChunkedPrefill", 1, "NATIVE"),
        ("cannbot", 0, "PROLOG_V3", "ChunkedPrefill", 0, "NATIVE"),
        ("cannbot", 0, "PROLOG_V3", "SpecDecoding", 0, "NATIVE"),
        ("cannbot", 0, "NATIVE", "DecodeOnly", 0, "NATIVE"),
        ("torch_npu", 0, "PROLOG_V3", "PrefillNoCache", 1, "PROLOG_V3"),
        ("cannbot", 64, "PROLOG_V3", "PrefillNoCache", 1, "PROLOG_V3"),
    ],
)
def test_cannbot_runtime_decode_only_policy(tokens, backend, rope, preprocess, state, prefills, expected):
    pp = SimpleNamespace(NATIVE=object(), PROLOG_V3=object(), MLAPO=object())
    states = SimpleNamespace(DecodeOnly="DecodeOnly")
    cls = _extract_class(
        SFA_PATH,
        "AscendSFAImpl",
        ["_runtime_preprocess_type", "_uses_cannbot_prolog"],
        dict(PreprocessType=pp, AscendAttentionState=states, MLAPO_MAX_SUPPORTED_TOKENS=1024),
    )
    impl = cls()
    impl.preprocess_type = getattr(pp, preprocess)
    impl._mla_prolog_backend = backend
    impl.qk_rope_head_dim = rope
    prepared = impl._mla_prolog_prepared = object()
    metadata = SimpleNamespace(attn_state=state, num_prefills=prefills)
    assert impl._runtime_preprocess_type(metadata, tokens) is getattr(pp, expected)
    # Routing must not invalidate prepared decode weights or persist a prefill fallback.
    assert impl.preprocess_type is getattr(pp, preprocess)
    assert impl._mla_prolog_prepared is prepared
    if backend == "cannbot" and rope == 0 and preprocess == "PROLOG_V3":
        metadata.attn_state, metadata.num_prefills = "DecodeOnly", 0
        assert impl._runtime_preprocess_type(metadata, 128) is pp.PROLOG_V3


@pytest.mark.parametrize("tokens", [1024, 1025])
def test_decode_only_policy_preserves_legacy_mlapo_limit(tokens):
    pp = SimpleNamespace(NATIVE=object(), PROLOG_V3=object(), MLAPO=object())
    cls = _extract_class(
        SFA_PATH,
        "AscendSFAImpl",
        ["_runtime_preprocess_type"],
        dict(PreprocessType=pp, MLAPO_MAX_SUPPORTED_TOKENS=1024),
    )
    impl = cls()
    impl.preprocess_type = pp.MLAPO
    assert impl._runtime_preprocess_type(None, tokens) is (pp.MLAPO if tokens <= 1024 else pp.NATIVE)


def test_forward_selects_runtime_policy_before_prolog_dispatch():
    tree = ast.parse(SFA_PATH.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_impl")
    choices = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fused_type" for t in node.targets)
    ]
    assert len(choices) == 1
    assert ast.unparse(choices[0].value) == "self._runtime_preprocess_type(attn_metadata, num_input_tokens)"
    prolog_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_sfa_preprocess_prolog_v3"
    ]
    assert prolog_calls and all(choices[0].lineno < call.lineno for call in prolog_calls)


@pytest.mark.parametrize("method", [None, DynamicMethod(), StaticMethod()])
def test_a5_nope_non_mxfp8_remains_native(routing, method):
    impl = routing.impl
    impl.fused_qkv_a_proj.quant_method.quant_method = method
    impl._mla_prolog_backend = impl._select_mla_prolog_backend()
    assert impl._mla_prolog_backend == "torch_npu"
    assert not impl._uses_cannbot_prolog()
    assert impl._resolve_preprocess_type(torch.bfloat16) is routing.pp.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_non_a5_nope_mxfp8_remains_native(routing, monkeypatch):
    base = _extract_class(DEVICE_PATH, "BaseDeviceAdaptor", ["get_mla_prolog_backend"], {})
    monkeypatch.setattr(routing.device_operator, "get_mla_prolog_backend", base.get_mla_prolog_backend)
    impl = routing.impl
    impl._mla_prolog_backend = impl._select_mla_prolog_backend()
    assert impl._mla_prolog_backend == "torch_npu"
    assert impl._resolve_preprocess_type(torch.bfloat16) is routing.pp.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_default_nope_a5_mxfp8_selects_prolog(routing, enforce_eager):
    impl = routing.impl
    impl.vllm_config.model_config.enforce_eager = enforce_eager
    assert impl._cannbot_config_unsupported_reasons() == []
    assert impl._resolve_preprocess_type(torch.bfloat16) is routing.pp.PROLOG_V3
    impl._process_weights_for_fused_prolog_v3.assert_called_once_with()
    impl._process_weights_for_fused_mlapo.assert_not_called()


@pytest.mark.parametrize("method", [None, DynamicMethod(), Mxfp8Method()])
def test_rope_never_selects_cannbot(routing, method):
    impl = routing.impl
    impl.qk_rope_head_dim = 64
    impl.fused_qkv_a_proj.quant_method.quant_method = method
    impl._mla_prolog_backend = impl._select_mla_prolog_backend()
    assert impl._mla_prolog_backend == "torch_npu"
    assert not impl._uses_cannbot_prolog()
    expected = routing.pp.NATIVE if method is None else routing.pp.PROLOG_V3
    assert impl._resolve_preprocess_type(torch.bfloat16) is expected


def test_constructor_uses_device_routing_without_environment_override():
    tree = ast.parse(SFA_PATH.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "_mla_prolog_backend" for target in node.targets)
    ]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == "self._select_mla_prolog_backend()"


def test_existing_mla_boundary_hides_runtime_cache_from_fake_tracing():
    tree = ast.parse(MLA_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("mla_forward", "mla_forward_fake")
    ]
    registrations = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "direct_register_custom_op"
    ]
    assert len(registrations) == 1
    registration = {kw.arg: ast.unparse(kw.value) for kw in registrations[0].keywords}
    assert registration == dict(
        op_name="'mla_forward'",
        op_func="mla_forward",
        mutates_args="['output']",
        fake_impl="mla_forward_fake",
        dispatch_key="'PrivateUse1'",
    )
    cache = torch.zeros(4)
    metadata = object()
    backend = Mock()
    layer = SimpleNamespace(mla_attn=SimpleNamespace(layer_name="attention", kv_cache=cache, impl=backend))
    context = SimpleNamespace(no_compile_layers={"layer": layer}, attn_metadata={"attention": metadata})
    namespace = dict(torch=torch, get_forward_context=lambda: context, eager_break_during_capture=lambda fn: fn)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(MLA_PATH), "exec"), namespace)
    hidden, output = torch.ones(4), torch.empty(4)
    namespace["mla_forward_fake"](hidden, output, "layer")
    backend.forward.assert_not_called()
    namespace["mla_forward"](hidden, output, "layer")
    backend.forward.assert_called_once_with("attention", hidden, cache, metadata, output)


def test_aot_opaque_mla_runtime_preserves_nested_packaged_cache():
    # Real packaged schema/Functionalize guard and extracted MLA runtime, but
    # test-only CPU kernels and registration: not an NPU or runner graph test.
    name = "vllm_ascend.ops.packaged_mla_prolog"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_ascend/ops/packaged_mla_prolog.py")
        assert spec is not None and spec.loader is not None
        packaged_module = importlib.util.module_from_spec(spec)
        sys.modules[name] = packaged_module
        spec.loader.exec_module(packaged_module)
    packaged = sys.modules[name]
    cache = torch.zeros((1, 16, KV_LORA_RANK), dtype=torch.bfloat16)
    pointer = cache.untyped_storage().data_ptr()
    calls = []

    def cpu_prolog(*args, **kwargs):
        names = [arg.name for arg in packaged.quant_mla_prolog._schema.arguments]
        values = dict(zip(names, args), **kwargs)
        target = values["kv_cache"]
        calls.append(target.untyped_storage().data_ptr())
        target.add_(1)
        return packaged._quant_mla_prolog_fake(**values)

    def backend_forward(layer_name, hidden, original_cache, metadata, output):
        unused = torch.empty(0, dtype=torch.uint8)
        packaged.quant_mla_prolog(
            hidden,
            unused,
            unused,
            unused,
            torch.empty((NUM_HEADS, 256, KV_LORA_RANK)),
            unused,
            unused,
            unused,
            unused,
            torch.empty(Q_LORA_RANK),
            torch.empty(KV_LORA_RANK),
            original_cache,
            torch.arange(hidden.shape[0]),
            norm_eps=1e-6,
        )
        output.copy_(hidden + original_cache[0, 0, 0])

    layer = SimpleNamespace(
        mla_attn=SimpleNamespace(layer_name="attention", kv_cache=cache, impl=SimpleNamespace(forward=backend_forward))
    )
    context = SimpleNamespace(no_compile_layers={"layer": layer}, attn_metadata=None)
    tree = ast.parse(MLA_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("mla_forward", "mla_forward_fake")
    ]
    namespace = dict(torch=torch, get_forward_context=lambda: context, eager_break_during_capture=lambda fn: fn)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(MLA_PATH), "exec"), namespace)
    cpu = torch.library.Library("vllm_ascend", "IMPL", "CPU")
    outer = torch.library.Library("glm_mla_boundary_test", "DEF")
    try:
        cpu.impl("quant_mla_prolog", cpu_prolog)
        outer.define("forward(Tensor hidden_states, Tensor(a!) output, str layer_name) -> ()")
        outer.impl("forward", namespace["mla_forward"], "CPU")
        torch.library.register_fake("glm_mla_boundary_test::forward", namespace["mla_forward_fake"], lib=outer)

        def forward(hidden):
            output = torch.empty_like(hidden)
            torch.ops.glm_mla_boundary_test.forward(hidden, output, "layer")
            return output

        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
        hidden = torch.ones((3, HIDDEN_SIZE), dtype=torch.bfloat16)
        torch.testing.assert_close(compiled(hidden), hidden + 1)
        torch.testing.assert_close(compiled(hidden), hidden + 2)
    finally:
        outer._destroy()
        cpu._destroy()
    assert calls == [pointer, pointer]
    assert torch.all(cache == 2)


def test_model_wrapper_and_piecewise_split_keep_mla_opaque():
    tree = ast.parse(MLA_PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMultiHeadLatentAttention"
    )
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    calls = [ast.unparse(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call)]
    assert "torch.ops.vllm.mla_forward" in calls
    assert not any("impl" in call for call in calls)
    platform = ast.parse((ROOT / "vllm_ascend/platform.py").read_text(encoding="utf-8"))
    splits = [
        node
        for node in ast.walk(platform)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "compilation_config.splitting_ops.extend"
    ]
    assert any("vllm::mla_forward" in ast.literal_eval(node.args[0]) for node in splits)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("hardware", "A5"),
        ("rl", "RL"),
        ("pd", "PD"),
        ("speculative", "speculative"),
        ("pcp", "parallel"),
        ("dcp", "parallel"),
        ("sp", "parallel"),
        ("offload", "offload"),
        ("turboquant", "TurboQuant"),
    ],
)
def test_unsupported_configuration_fails_without_prepare(routing, case, reason):
    impl = routing.impl
    config = impl.vllm_config
    if case == "hardware":
        routing.profile.device_adaptor_family = object()
    elif case == "rl":
        impl.rl_weight_update_enabled = True
    elif case == "pd":
        config.kv_transfer_config = object()
    elif case == "speculative":
        config.speculative_config = object()
    elif case == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif case == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif case == "sp":
        impl.enable_sp = True
    elif case == "offload":
        routing.ascend.sparse_kv_offload_config.enabled = True
    else:
        impl.enable_sparse_sfa_turboquant = True
    with pytest.raises(ValueError, match=reason):
        impl._resolve_preprocess_type(torch.bfloat16)
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


@pytest.mark.parametrize("projection", ["fused_qkv_a_proj", "q_proj", "indexer"])
@pytest.mark.parametrize("failure", ["quant", "bias", "padding"])
def test_incompatible_projection_is_rejected(routing, projection, failure):
    layer = routing.impl.indexer.wq_b if projection == "indexer" else getattr(routing.impl, projection)
    if failure == "quant":
        layer.quant_method.quant_method = DynamicMethod()
    elif failure == "bias":
        layer.bias = torch.zeros(1)
    else:
        layer.mxfp8_tp_padding = (0, 16)
    with pytest.raises(ValueError, match="MXFP8|bias|padding"):
        routing.impl._resolve_preprocess_type(torch.bfloat16)
    routing.impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_cannbot_rejects_fp16_model(routing):
    with pytest.raises(ValueError, match="BF16"):
        routing.impl._resolve_preprocess_type(torch.float16)
    routing.impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_cannbot_rejects_int8_cache(routing):
    routing.impl.enable_sparse_sfa_c8 = True
    routing.impl.c8_cache_dtype = torch.int8
    with pytest.raises(ValueError, match="FP8|INT8"):
        routing.impl._resolve_preprocess_type(torch.bfloat16)


@pytest.fixture
def indexer():
    cls = _extract_class(
        INDEXER_PATH,
        "Glm5NextKPoolIndexerBackend",
        ["_get_query_input", "supports_bf16_query_input"],
        _quant_globals(),
    )
    impl = cls()
    impl.wq_b = _projection()
    impl.q_lora_rank = Q_LORA_RANK
    return impl


def _query_tuple(tokens=5):
    values = torch.arange(tokens * Q_LORA_RANK).to(torch.uint8).reshape(tokens, Q_LORA_RANK)
    scales = torch.arange(tokens * (Q_LORA_RANK // 32)).to(torch.uint8).reshape(tokens, Q_LORA_RANK // 32)
    return values.view(torch.float8_e4m3fn), scales.view(torch.float8_e8m0fnu)


@pytest.mark.parametrize("tokens", [0, 3, 5])
def test_indexer_slices_both_quantized_values_and_scales_without_copy(indexer, tokens):
    values, scales = _query_tuple()
    result = indexer._get_query_input((values, scales), tokens)
    assert isinstance(result, tuple) and len(result) == 2
    for actual, source in zip(result, (values, scales)):
        assert actual.shape == (tokens, source.shape[1])
        assert actual.dtype == source.dtype
        assert actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert torch.equal(actual.view(torch.uint8), source[:tokens].view(torch.uint8))


def test_indexer_accepts_empty_mx_tuple(indexer):
    values, scales = _query_tuple(tokens=0)
    result = indexer._get_query_input((values, scales), 0)
    assert isinstance(result, tuple)
    assert result[0].shape == (0, Q_LORA_RANK)
    assert result[1].shape == (0, Q_LORA_RANK // 32)
    assert result[1].dtype == torch.float8_e8m0fnu


def test_indexer_preserves_unquantized_input_path(indexer):
    values = torch.arange(5 * Q_LORA_RANK, dtype=torch.float32).reshape(5, Q_LORA_RANK)
    result = indexer._get_query_input(values, 3)
    torch.testing.assert_close(result, values[:3])
    assert result.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


@pytest.mark.parametrize("method", [DynamicMethod(), StaticMethod(), None])
def test_indexer_rejects_tuple_for_other_quantizers(indexer, method):
    indexer.wq_b.quant_method.quant_method = method
    with pytest.raises(TypeError, match="MXFP8"):
        indexer._get_query_input(_query_tuple(), 3)


@pytest.mark.parametrize(
    "case",
    [
        "padding",
        "length",
        "values_dtype",
        "scale_dtype",
        "scale_shape",
        "tokens",
        "width",
        "values_not_tensor",
        "scales_not_tensor",
        "group",
        "values_stride",
        "scales_stride",
        "device",
    ],
)
def test_indexer_rejects_malformed_tuple(indexer, case):
    values, scales = _query_tuple()
    query: tuple[object, ...] = (values, scales)
    tokens = 3
    if case == "padding":
        indexer.wq_b.mxfp8_tp_padding = (16, 0)
    elif case == "length":
        query = (values,)
    elif case == "values_dtype":
        query = (values.view(torch.uint8), scales)
    elif case == "scale_dtype":
        query = (values, scales.view(torch.uint8))
    elif case == "scale_shape":
        query = (values, scales[:, :-1])
    elif case == "width":
        # Internally matching per-32 scales are insufficient if Q is wrong.
        query = (values[:, :512].contiguous(), scales[:, :16].contiguous())
    elif case == "values_not_tensor":
        query = (object(), scales)
    elif case == "scales_not_tensor":
        query = (values, object())
    elif case == "group":
        indexer.wq_b.quant_method.quant_method.group_size = 64
    elif case == "values_stride":
        strided = torch.zeros((5, Q_LORA_RANK * 2), dtype=torch.uint8)[:, ::2].view(torch.float8_e4m3fn)
        assert not strided.is_contiguous()
        query = (strided, scales)
    elif case == "scales_stride":
        strided = torch.zeros((5, Q_LORA_RANK // 16), dtype=torch.uint8)[:, ::2].view(torch.float8_e8m0fnu)
        assert not strided.is_contiguous()
        query = (values, strided)
    elif case == "device":
        # Metadata-only mismatch needs no accelerator or real NPU import.
        query = (values, torch.empty(scales.shape, dtype=scales.dtype, device="meta"))
    else:
        tokens = 6
    with pytest.raises((TypeError, ValueError)):
        indexer._get_query_input(query, tokens)


def _bf16_projection(width=Q_LORA_RANK):
    return SimpleNamespace(
        quant_method=UnquantizedMethod(),
        weight=torch.ones((4, width), dtype=torch.bfloat16),
        bias=None,
        mxfp8_tp_padding=(0, 0),
    )


@pytest.mark.parametrize("skip_topk", [False, True])
@pytest.mark.parametrize("cache_quant", [False, True])
def test_cannbot_admits_bf16_kpool_without_requantizing_weights(routing, skip_topk, cache_quant):
    impl = routing.impl
    impl.indexer.wq_b = _bf16_projection()
    impl._skip_topk = skip_topk
    impl.enable_sparse_sfa_c8 = cache_quant
    weight = impl.indexer.wq_b.weight
    assert impl._resolve_preprocess_type(torch.bfloat16) is routing.pp.PROLOG_V3
    impl._process_weights_for_fused_prolog_v3.assert_called_once()
    assert impl.indexer.wq_b.weight is weight
    assert not hasattr(impl.indexer.wq_b, "weight_scale")


@pytest.mark.parametrize("case", ["backend", "unknown", "missing", "fp16", "fp32", "bias", "padding"])
def test_cannbot_bf16_indexer_exception_is_narrow(routing, case):
    impl = routing.impl
    layer = impl.indexer.wq_b = _bf16_projection()
    if case == "backend":
        impl.indexer.impl = object()
    elif case == "unknown":
        layer.quant_method = SimpleNamespace(quant_method=None)
    elif case == "missing":
        del layer.weight
    elif case in ("fp16", "fp32"):
        layer.weight = layer.weight.to(torch.float16 if case == "fp16" else torch.float32)
    elif case == "bias":
        layer.bias = torch.ones(4, dtype=torch.bfloat16)
    else:
        layer.mxfp8_tp_padding = (16, 0)
    with pytest.raises(ValueError, match="MXFP8|bias|padding"):
        impl._resolve_preprocess_type(torch.bfloat16)
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_non_cannbot_prolog_still_rejects_bf16_indexer(routing):
    impl = routing.impl
    impl._mla_prolog_backend = "torch_npu"
    impl.qk_rope_head_dim = 64
    impl.indexer.wq_b = _bf16_projection()
    reasons = impl._get_fused_type_unsupported_reasons(routing.pp.PROLOG_V3)
    assert "PROLOG_V3 requires a quantized indexer query projection." in reasons


@pytest.mark.parametrize("tokens", [0, 1, 3, 5])
def test_bf16_indexer_slices_native_query_without_conversion(indexer, tokens):
    indexer.wq_b = _bf16_projection()
    query = torch.randn((5, Q_LORA_RANK), dtype=torch.bfloat16)
    result = indexer._get_query_input(query, tokens)
    torch.testing.assert_close(result, query[:tokens], rtol=0, atol=0)
    assert result.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()


@pytest.mark.parametrize("query_type", ["tuple", "fp8", "fp16", "fp32"])
def test_bf16_indexer_rejects_non_native_query(indexer, query_type):
    indexer.wq_b = _bf16_projection()
    values, scales = _query_tuple()
    query = (values, scales) if query_type == "tuple" else values
    if query_type in ("fp16", "fp32"):
        query = torch.zeros(values.shape, dtype=torch.float16 if query_type == "fp16" else torch.float32)
    with pytest.raises(TypeError):
        indexer._get_query_input(query, 3)


@pytest.mark.parametrize("field,value", [("group_size", 64), ("dynamic_mx_quant_scale_alg", 1)])
def test_cannbot_rejects_incompatible_mx_scale_parameters(routing, field, value):
    setattr(routing.impl.q_proj.quant_method.quant_method, field, value)
    with pytest.raises(ValueError, match="per-32|scale_alg"):
        routing.impl._resolve_preprocess_type(torch.bfloat16)
    routing.impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_prepare_hook_splits_loaded_weights_without_mutating_source():
    prepare = Mock(return_value=object())
    npu = SimpleNamespace(npu_format_cast=Mock(side_effect=lambda tensor, fmt: tensor.clone()))
    cls = _extract_class(
        SFA_PATH,
        "AscendSFAImpl",
        ["_uses_cannbot_prolog", "_process_weights_for_fused_prolog_v3", "_prepare_cannbot_prolog_weights"],
        dict(_quant_globals(), torch_npu=npu, prepare_mla_prolog=prepare, ACL_FORMAT_FRACTAL_ND=2, logger=Mock()),
    )
    impl = cls()
    impl._mla_prolog_backend = "cannbot"
    impl.qk_rope_head_dim = 0
    impl.q_lora_rank = Q_LORA_RANK
    width = Q_LORA_RANK + KV_LORA_RANK
    fused = torch.arange(HIDDEN_SIZE * width).to(torch.uint8).reshape(HIDDEN_SIZE, width)
    scales = torch.arange((HIDDEN_SIZE // 64) * width * 2).to(torch.uint8).reshape(HIDDEN_SIZE // 64, width, 2)
    q_weight = torch.zeros((Q_LORA_RANK, NUM_HEADS * 256), dtype=torch.uint8)
    q_scales = torch.zeros((Q_LORA_RANK // 64, NUM_HEADS * 256, 2), dtype=torch.uint8)
    impl.fused_qkv_a_proj = SimpleNamespace(weight=fused.view(torch.float8_e4m3fn), weight_scale=scales)
    impl.q_proj = SimpleNamespace(weight=q_weight.view(torch.float8_e4m3fn), weight_scale=q_scales)
    impl.W_UK_T = torch.zeros((NUM_HEADS, 256, KV_LORA_RANK), dtype=torch.bfloat16)
    impl.q_a_layernorm = SimpleNamespace(weight=torch.ones(Q_LORA_RANK), variance_epsilon=1e-6)
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(KV_LORA_RANK), variance_epsilon=1e-6)
    originals = [impl.fused_qkv_a_proj.weight, scales, impl.q_proj.weight, q_scales, impl.W_UK_T]
    snapshots = [(tensor.clone(), tensor.untyped_storage().data_ptr()) for tensor in originals]
    impl._process_weights_for_fused_prolog_v3()
    prepare.assert_called_once()
    assert impl._mla_prolog_prepared is prepare.return_value
    call = prepare.call_args.kwargs
    assert torch.equal(call["weight_dq"].view(torch.uint8), fused[:, :Q_LORA_RANK])
    assert torch.equal(call["weight_dkv_kr"].view(torch.uint8), fused[:, Q_LORA_RANK:])
    assert torch.equal(call["dequant_scale_w_dq"], scales[:, :Q_LORA_RANK, :])
    assert torch.equal(call["dequant_scale_w_dkv_kr"], scales[:, Q_LORA_RANK:, :])
    assert call["weight_uk"] is impl.W_UK_T
    assert all(call.args[1] == 2 for call in npu.npu_format_cast.call_args_list)
    for original, (snapshot, pointer) in zip(originals, snapshots):
        assert original.untyped_storage().data_ptr() == pointer
        assert torch.equal(original.view(torch.uint8), snapshot.view(torch.uint8))


@pytest.mark.parametrize("compute_topk", [False, True])
@pytest.mark.parametrize("bf16_indexer", [False, True])
@pytest.mark.parametrize("row_shard", [None, (1, 3)])
@pytest.mark.parametrize("fp8_cache", [False, True])
@pytest.mark.parametrize("graph_mode", ["eager", "full"])
def test_indexer_forward_preserves_query_shards_and_full_cache_inputs(
    compute_topk, bf16_indexer, row_shard, fp8_cache, graph_mode
):
    metadata_type = type("Metadata", (SimpleNamespace,), {})
    tail_type = type("TailMetadata", (SimpleNamespace,), {})
    shard = None if row_shard is None else SimpleNamespace(start=row_shard[0], stop=row_shard[1])
    metadata = metadata_type(indexer_row_shard=shard, num_actual_tokens=3, pcp_hidden_restore_indices=None)
    tail = tail_type()
    context = SimpleNamespace(attn_metadata={"tail": tail}, cudagraph_runtime_mode=graph_mode)
    cls = _extract_class(
        INDEXER_PATH,
        "Glm5NextKPoolIndexerBackend",
        ["_get_query_input", "supports_bf16_query_input", "forward"],
        dict(
            _quant_globals(),
            AscendIndexerKPoolMetadata=metadata_type,
            AscendIndexerKPoolTailMetadata=tail_type,
            get_forward_context=lambda: context,
            F=torch.nn.functional,
        ),
    )
    impl = cls()
    impl.tail_cache = SimpleNamespace(prefix="tail")
    impl.k_cache = object()
    impl._key_weight = torch.ones(1)
    impl.n_head = NUM_HEADS
    impl.head_dim = 4
    impl.q_lora_rank = Q_LORA_RANK
    impl.softmax_scale = 1.0
    impl._head_weight_f32 = torch.ones((NUM_HEADS, HIDDEN_SIZE))
    impl.wq_b = Mock(side_effect=lambda x: (torch.ones((x[0].shape[0], NUM_HEADS * impl.head_dim)),))
    impl.wq_b.quant_method = SimpleNamespace(quant_method=Mxfp8Method())
    impl.wq_b.mxfp8_tp_padding = (0, 0)
    if bf16_indexer:
        impl.wq_b.quant_method = UnquantizedMethod()
        impl.wq_b.weight = torch.ones((NUM_HEADS * impl.head_dim, Q_LORA_RANK), dtype=torch.bfloat16)
        impl.wq_b.side_effect = lambda x: (torch.nn.functional.linear(x, impl.wq_b.weight),)
    key_cache, scale_cache, tail_cache = object(), object(), object()
    caches = (key_cache, scale_cache) if fp8_cache else (key_cache,)
    impl._bound_caches = Mock(side_effect=lambda layer: caches if layer is impl.k_cache else (tail_cache,))
    impl._forward_cann = Mock(return_value=object())
    # Upstream SFA owns trimming: FULL graphs retain padded rows; eager callers trim.
    tokens = 5 if graph_mode == "full" else 3
    hidden = (torch.arange(tokens).unsqueeze(1) + 1).expand(tokens, HIDDEN_SIZE).to(torch.bfloat16)
    values = torch.arange(5).unsqueeze(1).expand(5, Q_LORA_RANK).to(torch.uint8).view(torch.float8_e4m3fn)
    scales = (torch.arange(5).unsqueeze(1) + 127).expand(5, Q_LORA_RANK // 32).to(torch.uint8)
    scales = scales.view(torch.float8_e8m0fnu)
    native_query = (torch.arange(5).unsqueeze(1) + 1).expand(5, Q_LORA_RANK).to(torch.bfloat16)
    query = native_query if bf16_indexer else (values, scales)
    if not compute_topk:
        query = None  # Cache-only execution must not inspect or project Query.
    result = impl.forward(hidden, query, hidden, metadata, compute_topk=compute_topk)
    assert result is impl._forward_cann.return_value
    call = impl._forward_cann.call_args
    torch.testing.assert_close(call.args[0], hidden, rtol=0, atol=0)
    assert call.args[0].shape[0] == tokens  # Never shard or re-trim key/cache writes.
    assert call.args[3] is key_cache and call.args[4] is tail_cache
    assert call.kwargs["indexer_scale_cache"] is (scale_cache if fp8_cache else None)
    assert call.kwargs["compute_topk"] is compute_topk
    if compute_topk:
        start, stop = row_shard if row_shard is not None else (0, tokens)
        impl.wq_b.assert_called_once()
        passed_query = impl.wq_b.call_args.args[0]
        if bf16_indexer:
            torch.testing.assert_close(passed_query, native_query[start:stop], rtol=0, atol=0)
            assert passed_query.untyped_storage().data_ptr() == native_query.untyped_storage().data_ptr()
        else:
            for part, original in zip(passed_query, (values, scales)):
                assert torch.equal(part.view(torch.uint8), original[start:stop].view(torch.uint8))
                assert part.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        assert call.args[1].shape == (stop - start, NUM_HEADS, impl.head_dim)
        expected_weights = torch.nn.functional.linear(hidden[start:stop].float(), impl._head_weight_f32)
        expected_weights = expected_weights.to(call.args[1].dtype) * (impl.softmax_scale * impl.n_head**-0.5)
        torch.testing.assert_close(call.args[2], expected_weights, rtol=0, atol=0)
    else:
        impl.wq_b.assert_not_called()
        assert call.args[1] is None and call.args[2] is None


@pytest.mark.parametrize("method_type", [None, DynamicMethod, StaticMethod, Mxfp8Method])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("enable_mlapo", [False, True])
@pytest.mark.parametrize("cache_quant", [False, True])
@pytest.mark.parametrize("kv_consumer", [False, True])
def test_a2_a3_standard_provider_keeps_upstream_routing(
    routing, monkeypatch, method_type, rope_dim, enable_mlapo, cache_quant, kv_consumer
):
    # A2/A3 use BaseDeviceAdaptor, not the A5-only packaged backend override.
    base = _extract_class(DEVICE_PATH, "BaseDeviceAdaptor", ["get_mla_prolog_backend"], {})
    monkeypatch.setattr(routing.device_operator, "get_mla_prolog_backend", base.get_mla_prolog_backend)
    impl = routing.impl
    routing.profile.device_adaptor_family = object()
    method = UnquantizedMethod() if method_type is None else method_type()
    impl.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_consumer=True) if kv_consumer else None
    impl.is_kv_consumer = kv_consumer
    impl.qk_rope_head_dim = rope_dim
    impl.fused_qkv_a_proj = _projection(method)
    impl.q_proj = _projection(method)
    if method_type is None:
        # Unquantized linear methods are the outer method, not a quantizer wrapper.
        impl.fused_qkv_a_proj.quant_method = UnquantizedMethod()
        impl.q_proj.quant_method = UnquantizedMethod()
        assert impl._get_layer_quant_method(impl.fused_qkv_a_proj) is None
    impl._quant_type = method_type
    impl.enable_mlapo = enable_mlapo
    impl.enable_sparse_sfa_c8 = cache_quant
    impl._mla_prolog_backend = impl._select_mla_prolog_backend()
    assert impl._mla_prolog_backend == "torch_npu"
    assert not impl._uses_cannbot_prolog()
    assert not impl._uses_native_indexer_query()
    expected = routing.pp.NATIVE
    if rope_dim:
        if method_type in (DynamicMethod, Mxfp8Method) or (method_type is None and kv_consumer and not cache_quant):
            expected = routing.pp.PROLOG_V3
        elif method_type is StaticMethod and enable_mlapo and not cache_quant:
            expected = routing.pp.MLAPO
    assert impl._resolve_preprocess_type(torch.bfloat16) is expected
    if expected is routing.pp.PROLOG_V3:
        impl._process_weights_for_fused_prolog_v3.assert_called_once()
        impl._process_weights_for_fused_mlapo.assert_not_called()
    elif expected is routing.pp.MLAPO:
        impl._process_weights_for_fused_mlapo.assert_called_once_with(torch.bfloat16)
        impl._process_weights_for_fused_prolog_v3.assert_not_called()
    else:
        impl._process_weights_for_fused_prolog_v3.assert_not_called()
        impl._process_weights_for_fused_mlapo.assert_not_called()


@pytest.mark.parametrize("tokens", [1, 1024, 1025, 8192])
@pytest.mark.parametrize("state", ["DecodeOnly", "PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill", "SpecDecoding"])
@pytest.mark.parametrize("prefills", [0, 2])
def test_torch_npu_prolog_has_no_cannbot_decode_only_restriction(tokens, state, prefills):
    pp = SimpleNamespace(NATIVE=object(), PROLOG_V3=object(), MLAPO=object())
    states = SimpleNamespace(DecodeOnly="DecodeOnly")
    cls = _extract_class(
        SFA_PATH,
        "AscendSFAImpl",
        ["_uses_cannbot_prolog", "_runtime_preprocess_type"],
        dict(PreprocessType=pp, AscendAttentionState=states, MLAPO_MAX_SUPPORTED_TOKENS=1024),
    )
    impl = cls()
    impl.preprocess_type = pp.PROLOG_V3
    impl._mla_prolog_backend = "torch_npu"
    impl.qk_rope_head_dim = 64
    assert (
        impl._runtime_preprocess_type(
            SimpleNamespace(attn_state=getattr(states, state, state), num_prefills=prefills), tokens
        )
        is pp.PROLOG_V3
    )
