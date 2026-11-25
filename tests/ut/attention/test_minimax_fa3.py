# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the optional A2/A3 operator; no NPU accuracy claim."""

import __future__

import ast
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap,sliding_window", [(0.0, None), (30.0, 2048)])
def test_fa3_scheduler_registration_completes_before_runtime(fa3, monkeypatch, dtype, softcap, sliding_window):
    device = NS(type="npu", index=0)
    specs = {(32, 4, 128, dtype, 0.1, softcap, sliding_window)}
    events = []

    def scheduler_metadata(**kwargs):
        events.append("scheduler")
        return torch.zeros(8, dtype=torch.uint8)

    scheduler = Mock(side_effect=scheduler_metadata)
    sync = Mock(side_effect=lambda *_: events.append("synchronize"))
    fake_torch = NS(
        int32=torch.int32,
        arange=Mock(return_value=torch.tensor([0, 1], dtype=torch.int32)),
        ones=Mock(return_value=torch.ones(1, dtype=torch.int32)),
        npu=NS(synchronize=sync),
    )
    namespace = fa3._initialize_fa3_scheduler.__globals__
    monkeypatch.setitem(namespace, "torch", fake_torch)
    monkeypatch.setitem(namespace, "get_scheduler_metadata", scheduler)
    monkeypatch.setitem(namespace, "logger", NS(info=Mock()))
    fa3._initialize_fa3_scheduler(device, specs, 128)
    assert events == ["synchronize", "scheduler", "synchronize"]
    args = scheduler.call_args.kwargs
    assert args["batch_size"] == 1 and args["page_size"] == 128
    assert args["num_heads_q"] == 32 and args["num_heads_kv"] == 4
    assert args["qkv_dtype"] == dtype
    assert args["has_softcap"] is (softcap != 0.0)
    assert args["window_size"] == ((-1, -1) if sliding_window is None else (2047, 2047))
    torch.testing.assert_close(args["cu_seqlens_q"], torch.tensor([0, 1], dtype=torch.int32))
    # Layers sharing an operator contract initialize once per device.
    fa3._initialize_fa3_scheduler(device, specs, 128)
    assert scheduler.call_count == 1 and sync.call_count == 2
    # A different device must complete its own registration.
    fa3._initialize_fa3_scheduler(NS(type="npu", index=1), specs, 128)
    assert scheduler.call_count == 2 and sync.call_count == 4


def test_fa3_initialization_failure_is_not_cached(fa3, monkeypatch):
    namespace = fa3._initialize_fa3_scheduler.__globals__
    sync = Mock(side_effect=[None, RuntimeError("registration failed"), None, None])
    scheduler = Mock(return_value=torch.zeros(8, dtype=torch.uint8))
    monkeypatch.setitem(
        namespace,
        "torch",
        NS(
            int32=torch.int32,
            arange=Mock(return_value=torch.tensor([0, 1], dtype=torch.int32)),
            ones=Mock(return_value=torch.ones(1, dtype=torch.int32)),
            npu=NS(synchronize=sync),
        ),
    )
    monkeypatch.setitem(namespace, "get_scheduler_metadata", scheduler)
    monkeypatch.setitem(namespace, "logger", NS(info=Mock()))
    spec = {(16, 1, 128, torch.bfloat16, 0.1, 0.0, None)}
    device = NS(type="npu", index=0)
    with pytest.raises(RuntimeError, match="registration failed"):
        fa3._initialize_fa3_scheduler(device, spec, 128)
    assert not namespace["_FA3_WARMUP_KEYS"]
    fa3._initialize_fa3_scheduler(device, spec, 128)
    assert scheduler.call_count == 2 and sync.call_count == 4


def execute_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.fixture
def profiles():
    # Execute the real immutable capability matrix without initializing a device.
    namespace: dict[str, Any] = {"__name__": __name__}
    exec((ROOT / "vllm_ascend/device/hardware.py").read_text(), namespace)
    source = (ROOT / "vllm_ascend/device/hardware_profile.py").read_text()
    source = source.replace("from vllm_ascend.device.device_config import get_device_config", "")
    source = source.replace("from vllm_ascend.device.hardware import AscendDeviceType", "")
    namespace["get_device_config"] = lambda: NS(_device_type=namespace["AscendDeviceType"].A2)
    exec(source, namespace)
    return NS(**namespace)


@pytest.fixture
def selection(monkeypatch, profiles):
    runtime = NS(device=profiles.AscendDeviceType.A2, installed=True)
    target = NS(hf_config=NS(architectures=["MiniMaxM3SparseForConditionalGeneration"]))
    runtime.config = NS(
        model_config=target,
        speculative_config=NS(
            method="dspark",
            target_model_config=target,
            draft_model_config=NS(architectures=["DSparkMiniMaxDraftModel"]),
            enable_adaptive_verification=False,
        ),
        parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=NS(cache_dtype="auto"),
        quant_config=None,
    )
    runtime.selector = NS(
        use_mla=False,
        use_sparse=False,
        use_pcp=False,
        attn_type="decoder",
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
    )
    monkeypatch.setitem(sys.modules, "vllm.config", NS(get_current_vllm_config_or_none=lambda: runtime.config))
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", NS(AttentionType=NS(DECODER="decoder")))
    monkeypatch.setitem(sys.modules, "vllm.compilation", NS(backends=NS(model_tag="backbone")))
    namespace = {
        "torch": torch,
        "HardwareCapability": profiles.HardwareCapability,
        "AttentionBackendFamily": profiles.AttentionBackendFamily,
        "FlashAttentionNpu3Model": profiles.FlashAttentionNpu3Model,
        "get_current_hardware_profile": lambda: profiles.get_hardware_profile(runtime.device),
        "util": NS(find_spec=lambda _: object() if runtime.installed else None),
        "_MINIMAX_M3_ARCHITECTURES": frozenset(
            {"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}
        ),
        "_validate_fa3_backend": lambda *args: False,
        "_validate_flash_attn_backend": lambda *args: False,
    }
    path = ROOT / "vllm_ascend/platform.py"
    execute_nodes(
        path,
        {
            "_validate_flash_attn_npu_3",
            "_validate_fa3_common_constraints",
            "_validate_fa3_global_constraints",
            "_uses_minimax_fa3",
            "_uses_step5_fa3",
            "_is_supported_flash_attn_draft",
        },
        namespace,
    )
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_select_attn_backend_cls")
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    runtime.select = lambda: namespace["_select_attn_backend_cls"](None, runtime.selector)
    runtime.profiles = profiles
    return runtime


@pytest.mark.parametrize("device", ["A2", "A3", "A5", "_310P"])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "arch",
    [
        "MiniMaxM3SparseForCausalLM",
        "DSparkMiniMaxDraftModel",
        "Qwen3DSparkModel",
        "MiniMaxM31FlashA3DSparkModel",
        "Other",
    ],
)
def test_hardware_and_model_routing(selection, device, installed, arch):
    r = selection
    r.device = getattr(r.profiles.AscendDeviceType, device)
    r.installed = installed
    r.config.model_config = NS(hf_config=NS(architectures=[arch]))
    actual = r.select()
    if device in ("A2", "A3") and installed and arch not in ("Other", "MiniMaxM3SparseForCausalLM"):
        assert actual == "vllm_ascend.attention.flash_attention_v3.AscendFlashAttentionBackend"
    else:
        assert "flash_attention_v3" not in actual


@pytest.mark.parametrize(
    "fault",
    [
        "no_config",
        "no_model",
        "other_target",
        "other_method",
        "dtype",
        "kv_cache_dtype",
        "cache_dtype",
        "c8",
        "attn_type",
        "use_sparse",
        "use_mla",
        "use_compress",
        "has_sink",
        "use_pcp",
        "use_dcp",
        "use_mm_prefix",
        "use_per_head_quant_scales",
        "use_batch_invariant",
        "use_adaptive_verification",
        "use_rswa",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_unsupported_configurations_fall_back(selection, fault):
    r = selection
    r.config.model_config = NS(hf_config=NS(architectures=["DSparkMiniMaxDraftModel"]))
    if fault == "no_config":
        r.config = None
    elif fault == "no_model":
        r.config.model_config = None
    elif fault == "other_target":
        r.config.speculative_config.target_model_config = NS(hf_config=NS(architectures=["Other"]))
    elif fault == "other_method":
        r.config.speculative_config.method = "mtp"
    elif fault.endswith("parallel_size"):
        setattr(r.config.parallel_config, fault, 2)
    elif fault == "cache_dtype":
        r.config.cache_config.cache_dtype = "fp8"
    elif fault == "c8":
        r.config.quant_config = NS(enable_c8_quant=True)
    else:
        value = {"dtype": torch.float32, "kv_cache_dtype": "fp8", "attn_type": "encoder"}.get(fault, True)
        setattr(r.selector, fault, value)
    # Sparse/compressed keys may be unsupported by the existing default table.
    try:
        result = r.select()
    except KeyError:
        assert fault in ("use_sparse", "use_compress")
    else:
        assert "flash_attention_v3" not in result


@pytest.fixture
def fa3(monkeypatch):
    # Contract tests must stay on CPU even when torch_npu is installed.
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "is_available", lambda: False)

    class Base:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def __class_getitem__(cls, _):
            return cls

    namespace = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "Enum": Enum,
        "AttentionBackend": Base,
        "AttentionMetadata": Base,
        "AttentionMetadataBuilder": Base,
        "AttentionImpl": Base,
        "AttentionType": NS(DECODER="decoder"),
        "CUDAGraphMode": NS(FULL_DECODE_ONLY="full_decode_only"),
        "logger": NS(info_once=lambda *args, **kwargs: None),
        "get_scheduler_metadata": Mock(),
        "get_current_vllm_config_or_none": lambda: None,
        "_FA3_WARMUP_KEYS": set(),
        "flash_attn_with_kvcache": Mock(),
        "envs_vllm": NS(VLLM_KV_CACHE_LAYOUT="NHD"),
        "DeviceOperator": NS(reshape_and_cache=Mock()),
        "notify_kv_cache_written": Mock(),
    }

    @contextmanager
    def transfer():
        yield

    namespace["attention_transfer_window"] = transfer
    names = {
        "_get_window_size",
        "_uses_hnd_kv_cache",
        "_initialize_fa3_scheduler",
        "AscendFlashAttentionBackend",
        "AscendFlashAttentionMetadata",
        "AscendFlashAttentionMetadataBuilder",
        "AscendFlashAttentionImpl",
    }
    execute_nodes(ROOT / "vllm_ascend/attention/flash_attention_v3.py", names, namespace)

    # A module proxy keeps patched operator callables in the production globals.
    class Proxy:
        def __getattr__(self, name):
            try:
                return namespace[name]
            except KeyError:
                raise AttributeError(name) from None

        def __setattr__(self, name, value):
            namespace[name] = value

        def __delattr__(self, name):
            del namespace[name]

    return Proxy()


def test_cache_write_precedes_attention_and_transfer_completion(fa3):
    events = []
    fa3.DeviceOperator.reshape_and_cache.side_effect = lambda **kw: events.append("cache")
    fa3.notify_kv_cache_written.side_effect = lambda *args: events.append("notify")

    @contextmanager
    def transfer():
        events.append("begin")
        yield
        events.append("end")

    fa3.attention_transfer_window = transfer
    query = torch.ones(2, 4, 128, dtype=torch.bfloat16)

    def record_attention(*args, **kwargs):
        events.append("attention")
        return query * 2

    fa3.flash_attn_with_kvcache.side_effect = record_attention
    impl = fa3.AscendFlashAttentionImpl(4, 128, 0.1, num_kv_heads=1, sliding_window=8)
    metadata = NS(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        seq_lens=torch.tensor([3]),
        block_tables=torch.tensor([[0]]),
        query_start_loc=torch.tensor([0, 2]),
        max_query_len=2,
        causal=False,
        scheduler_metadata={(4, 1, 128, torch.bfloat16, 0.1, 0.0, 8): torch.tensor([1])},
    )
    layer = NS(layer_name="draft", _k_scale_float=1.0, _v_scale_float=1.0)
    cache = torch.zeros(2, 1, 128, 1, 128, dtype=torch.bfloat16)
    out = torch.empty_like(query)
    assert impl.forward(layer, query, query[:, :1], query[:, :1], cache, metadata, out) is out
    assert events == ["cache", "notify", "begin", "attention", "end"]
    assert torch.equal(out, query * 2)
    assert fa3.flash_attn_with_kvcache.call_args.kwargs["window_size"] == (7, 7)
    events.clear()
    impl.forward(layer, query, None, None, cache, None, out)
    assert not events and not out.any()


@pytest.mark.parametrize(
    ("device", "eager", "fault"),
    [(device, eager, None) for device in ("A2", "A3") for eager in (False, True)]
    + [
        ("A3", True, fault)
        for fault in [
            "target_tag",
            "other_target",
            "missing_text_config",
            "other_draft",
            "use_pcp",
            "use_dcp",
            "quantized_draft_kv",
            "missing_operator",
            "target_attention",
            "no_model",
        ]
    ],
)
@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm5_next_text"])
def test_glm_dflash_local_draft_routing(selection, device, eager, fault, model_type):
    r = selection
    r.device = getattr(r.profiles.AscendDeviceType, device)
    target_arch = "GlmMoeDsaForCausalLM" if model_type == "glm_moe_dsa" else "Glm5NextForCausalLM"
    target = NS(hf_text_config=NS(model_type=model_type), hf_config=NS(architectures=[target_arch]))
    r.config.model_config = target
    r.config.speculative_config = NS(
        method="dflash",
        enforce_eager=eager,
        target_model_config=target,
        draft_model_config=NS(architectures=["DFlash2DraftModel"]),
        enable_adaptive_verification=False,
    )
    # The target is distributed and C8-quantized; the selected draft layer is
    # local with unquantized KV. Global topology must not reject that layer.
    r.config.parallel_config = NS(
        pipeline_parallel_size=2, prefill_context_parallel_size=8, decode_context_parallel_size=1
    )
    r.config.quant_config = NS(enable_c8_quant=True)
    backends = sys.modules["vllm.compilation"].backends
    backends.model_tag = "dflash_head"
    if fault == "target_tag":
        backends.model_tag = "backbone"
    elif fault == "other_target":
        target.hf_text_config.model_type = "other"
    elif fault == "missing_text_config":
        target.hf_text_config = None
    elif fault == "other_draft":
        r.config.speculative_config.draft_model_config.architectures = ["Other"]
    elif fault in ("use_pcp", "use_dcp"):
        setattr(r.selector, fault, True)
    elif fault == "quantized_draft_kv":
        r.selector.kv_cache_dtype = "int8"
    elif fault == "missing_operator":
        r.installed = False
    elif fault == "target_attention":
        r.selector.use_mla = True
        r.selector.use_sparse = True
    elif fault == "no_model":
        r.config.model_config = None
    actual = r.select()
    assert ("flash_attention_v3" in actual) == (fault is None)


@pytest.mark.parametrize("layout", ["NHD", "HND", "LBHNC", "BLHNC"])
@pytest.mark.parametrize("heads", [1, 8])
def test_physical_cache_layout_write_and_fa3_view(fa3, monkeypatch, layout, heads):
    _check_cache_layout(fa3, monkeypatch, layout, heads, "env", True, None)


@pytest.mark.parametrize("layout_source", ["env", "connector"])
def test_cache_layout_alias_resolution(fa3, monkeypatch, layout_source):
    # Canonical HND is covered by physical writes and the connector regression.
    layout = "LBHNC"
    from vllm.config import CacheConfig
    from vllm.v1.attention.backends import utils

    config = NS(cache_config=CacheConfig(), kv_transfer_config=NS(kv_role="kv_both"))
    monkeypatch.setattr(utils.envs, "VLLM_KV_CACHE_LAYOUT", layout if layout_source == "env" else None)
    monkeypatch.setattr(utils, "get_kv_connector_cache_layout", lambda _: layout)
    resolved = utils.resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    assert resolved.name == "LBHNC"
    assert fa3._uses_hnd_kv_cache(config)


def test_copied_draft_config_uses_worker_layout_outside_context(fa3, monkeypatch):
    _check_cache_layout(fa3, monkeypatch, "HND", 8, "connector", False, "auto")


def _check_cache_layout(fa3, monkeypatch, layout, heads, layout_source, runtime_config_visible, draft_kv_dtype):
    fa3.envs_vllm.VLLM_KV_CACHE_LAYOUT = layout if layout_source == "env" else None
    # Model construction precedes the layout RPC/cache initialization.
    from vllm.config import CacheConfig, replace

    config = NS(cache_config=CacheConfig(), speculative_config=None)
    # Match load_dflash_model: even "auto" creates an independent cache config.
    draft_cache_config = (
        replace(config.cache_config, cache_dtype=draft_kv_dtype) if draft_kv_dtype is not None else config.cache_config
    )
    draft_config = NS(cache_config=draft_cache_config, speculative_config=None)
    assert (draft_config.cache_config is config.cache_config) is (draft_kv_dtype is None)
    fa3.get_current_vllm_config_or_none = lambda: draft_config
    impl = fa3.AscendFlashAttentionImpl(heads, 16, 0.1, num_kv_heads=heads)
    fa3.get_current_vllm_config_or_none = lambda: config
    if layout_source == "connector":
        from vllm.v1.attention.backends import utils

        # Exercise upstream negotiation without an environment override, as in
        # PD. The worker receives the resolved cache config after construction.
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        monkeypatch.setattr(utils.envs, "VLLM_KV_CACHE_LAYOUT", None)
        connector_layout = Mock(return_value=layout)
        monkeypatch.setattr(utils, "get_kv_connector_cache_layout", connector_layout)
        config.kv_transfer_config = NS(kv_role="kv_both")
        resolved = utils.resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
        assert resolved.name == ("LBHNC" if layout in ("HND", "LBHNC") else "LBNHC")
        connector_layout.assert_called_once_with(config)
    if draft_kv_dtype is not None:
        assert draft_config.cache_config.kv_cache_layout is None
    config.scheduler_config = NS(max_num_seqs=2)
    config.compilation_config = NS(
        cudagraph_capture_sizes=[], cudagraph_mode=None, static_forward_context={"draft": NS(impl=impl)}
    )
    fa3.AscendFlashAttentionMetadataBuilder(
        NS(block_size=128, dtype=torch.bfloat16), ["draft"], config, torch.device("cpu")
    )
    hnd = layout in ("HND", "LBHNC", "BLHNC")
    assert impl.use_hnd_kv_cache is hnd
    shape = fa3.AscendFlashAttentionBackend.get_kv_cache_shape(2, 128, heads, 16)
    assert shape == ((2, 2, heads, 128, 16) if hnd else (2, 2, 128, heads, 16))
    cache = torch.zeros(shape, dtype=torch.bfloat16)
    if not runtime_config_visible:
        # Runtime DFlash context-KV updates can run after leaving the config
        # context used for connector negotiation and cache allocation.
        fa3.get_current_vllm_config_or_none = lambda: None
    assert impl.use_hnd_kv_cache is hnd
    query = torch.ones(2, heads, 16, dtype=torch.bfloat16)
    key = torch.arange(2 * heads * 16, dtype=torch.float32).reshape(2, heads, 16).to(torch.bfloat16)
    value = key + 2
    slots = torch.tensor([3, 129])

    def write_cache(**kw):
        assert kw.get("use_bnsd", False) is hnd
        for i, slot in enumerate(kw["slot_mapping"].tolist()):
            block, offset = divmod(slot, 128)
            for name in ("key", "value"):
                dst = kw[name + "_cache"]
                dst = dst[block, :, offset] if hnd else dst[block, offset]
                dst.copy_(kw[name][i])

    def attention(q, k, v, **kwargs):
        assert k.shape == v.shape == (2, 128, heads, 16)
        assert k.data_ptr() == cache[0].data_ptr()
        assert v.data_ptr() == cache[1].data_ptr()
        assert k.stride() == ((heads * 128 * 16, 16, 128 * 16, 1) if hnd else (128 * heads * 16, heads * 16, 16, 1))
        for i, slot in enumerate(slots.tolist()):
            block, offset = divmod(slot, 128)
            assert torch.equal(k[block, offset], key[i])
            assert torch.equal(v[block, offset], value[i])
        return q

    fa3.DeviceOperator.reshape_and_cache.side_effect = write_cache
    fa3.flash_attn_with_kvcache.side_effect = attention
    metadata = NS(
        num_actual_tokens=2,
        slot_mapping=slots,
        seq_lens=torch.tensor([4, 2]),
        block_tables=torch.tensor([[0], [1]]),
        query_start_loc=torch.tensor([0, 1, 2]),
        max_query_len=1,
        causal=False,
        scheduler_metadata={(heads, heads, 16, torch.bfloat16, 0.1, 0.0, None): torch.tensor([1])},
    )
    layer = NS(layer_name="draft", _k_scale_float=1.0, _v_scale_float=1.0)
    output = torch.empty_like(query)
    forward_key, forward_value = key, value
    if not runtime_config_visible:
        # Match propose -> _precompute_context_kv -> do_kv_cache_update, then
        # read those cached values in a forward without new K/V.
        impl.do_kv_cache_update(layer, key, value, cache, slots)
        forward_key = forward_value = None
    assert impl.forward(layer, query, forward_key, forward_value, cache, metadata, output) is output
    fa3.DeviceOperator.reshape_and_cache.assert_called_once()
    fa3.flash_attn_with_kvcache.assert_called_once()
    assert torch.equal(output, query)


@pytest.mark.parametrize("method", [None, "eagle3", "dspark"])
@pytest.mark.parametrize("enabled", [False, True])
def test_graph_fusion_pass_registration(monkeypatch, method, enabled):
    path = ROOT / "vllm_ascend/compilation/graph_fusion_pass_manager.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "GraphFusionPassManager"
    )
    configure = next(n for n in cls.body if getattr(n, "name", None) == "configure")
    flags = NS(fuse_norm_quant=False, fuse_qknorm_rope=enabled, fuse_muls_add=False)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ascend_config", NS(get_ascend_config=lambda: NS(ascend_compilation_config=flags))
    )
    fusion = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.compilation.passes.qknorm_rope_fusion_pass", NS(QKNormRopeFusionPass=fusion)
    )
    namespace: dict[str, Any] = {
        "__package__": "vllm_ascend.compilation",
        "enable_sp": lambda _: False,
        "get_current_hardware_profile": lambda: NS(supports=lambda _: False),
    }
    exec(
        compile(
            ast.Module(body=[configure], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    config = NS(speculative_config=NS(use_dspark=lambda: method == "dspark") if method else None)
    manager = NS(passes=[])
    namespace["configure"](manager, config)
    assert fusion.call_count == len(manager.passes) == int(enabled)


def test_prefill_bounds_causal_storage_and_oversized_tiling(fa3):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.capture_sizes, builder.capture_query_len = {16}, 8
    builder.max_num_reqs, builder.block_size = 2, 128
    spec = (4, 1, 128, torch.bfloat16, 0.1, 0.0, None)
    builder.scheduler_specs, builder.scheduler_buffers = {spec}, {}
    common = NS(
        num_reqs=2,
        num_actual_tokens=16,
        num_input_tokens=16,
        max_query_len=16,
        query_start_loc=torch.tensor([0, 16, 16], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 16, 16], dtype=torch.int32),
        seq_lens=torch.tensor([16, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0], [1]]),
        slot_mapping=torch.arange(16),
        causal=True,
    )
    fa3.get_scheduler_metadata.return_value = torch.zeros(8, dtype=torch.uint8)
    metadata = builder.build(0, common)
    assert metadata.max_query_len == 16  # Same token bucket as decode, but a longer prefill.
    assert not builder.scheduler_buffers
    assert fa3.get_scheduler_metadata.call_args.kwargs["max_seqlen_q"] == 16
    common.query_start_loc.copy_(torch.tensor([0, 8, 16], dtype=torch.int32))
    common.query_start_loc_cpu.copy_(common.query_start_loc)
    common.max_query_len = 8
    causal = builder.build(0, common)
    common.causal = False
    noncausal = builder.build(0, common)
    assert causal.scheduler_metadata[spec].data_ptr() != noncausal.scheduler_metadata[spec].data_ptr()
    address = noncausal.scheduler_metadata[spec].data_ptr()
    fa3.get_scheduler_metadata.return_value = torch.zeros(1024, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="graph-stable buffer"):
        builder.build(0, common)
    assert noncausal.scheduler_metadata[spec].data_ptr() == address


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_minimax_fa3_scheduler_api(fa3, causal, padded, softcap):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.capture_sizes = set()
    builder.capture_query_len = None
    builder.block_size = 128
    builder.max_num_reqs = 3
    scale = 0.123
    spec = (16, 1, 128, torch.bfloat16, scale, softcap, 4096)
    builder.scheduler_specs = {spec}
    builder.scheduler_buffers = {}
    offsets = torch.tensor([0, 8, 8] if padded else [0, 8, 11], dtype=torch.int32)
    common = NS(
        num_reqs=2,
        num_actual_tokens=8 if padded else 11,
        num_input_tokens=11,
        max_query_len=8,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([137, 73], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.arange(11),
        causal=causal,
    )

    # Match the keyword-only subset of the public scheduler API.
    def scheduler(
        *,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype,
        cu_seqlens_q,
        page_size,
        causal,
        window_size,
        has_softcap,
        num_splits,
    ):
        assert has_softcap is (softcap != 0.0)
        assert window_size == (4095, 0 if causal else 4095)
        assert cache_seqlens.numel() == batch_size
        assert cu_seqlens_q.numel() == batch_size + 1
        return torch.tensor([1], dtype=torch.uint8)

    with patch.object(fa3, "get_scheduler_metadata", side_effect=scheduler):
        metadata = builder.build(0, common)
    assert metadata.scheduler_metadata[spec].tolist() == [1]
    assert metadata.query_start_loc.numel() == (2 if padded else 3)
    assert metadata.block_tables.shape[0] == (1 if padded else 2)


def test_minimax_fa3_graph_metadata_keeps_storage_across_batch_sizes(fa3):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.capture_sizes = {16, 24}
    builder.capture_query_len = 8
    builder.max_num_reqs = 3
    builder.block_size = 128
    spec = (16, 1, 128, torch.bfloat16, 0.123, 0.0, 4096)
    builder.scheduler_specs = {spec}
    builder.scheduler_buffers = {}
    offsets = torch.tensor([0, 8, 16, 16], dtype=torch.int32)
    common = NS(
        num_reqs=3,
        num_actual_tokens=16,
        num_input_tokens=24,
        max_query_len=8,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([137, 73, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
        slot_mapping=torch.arange(24),
        causal=True,
    )

    def scheduler(**kwargs):
        # The 950 API appends an int32 cumulative-KV entry per request.
        return torch.full((3584 + 4 * (kwargs["batch_size"] + 1),), kwargs["batch_size"], dtype=torch.uint8)

    addresses = []
    with patch.object(fa3, "get_scheduler_metadata", side_effect=scheduler):
        for batch_size in (2, 1, 3, 2):
            offsets.copy_(torch.tensor([8 * min(i, batch_size) for i in range(4)], dtype=torch.int32))
            common.num_actual_tokens = batch_size * 8
            metadata = builder.build(0, common)
            tiling = metadata.scheduler_metadata[spec]
            addresses.append(tiling.data_ptr())
            assert tiling.numel() == 3584 + 4 * (batch_size + 1)
            assert torch.all(tiling == batch_size)
        common.num_input_tokens = 16
        other_bucket = builder.build(0, common)
    assert len(set(addresses)) == 1
    assert len(builder.scheduler_buffers) == 2
    assert other_bucket.scheduler_metadata[spec].data_ptr() != addresses[0]


@pytest.mark.parametrize(
    ("method", "has_config", "npu_available", "eager", "block_size"),
    [
        ("dflash", True, True, True, None),
        ("dflash", True, True, False, 256),
        ("dspark", True, True, False, 256),
        ("mtp", True, True, True, 256),
        (None, True, True, False, 256),
        ("dflash", False, True, True, 256),
        ("dflash", True, None, True, 256),
        ("dflash", True, False, True, 256),
    ],
)
def test_scheduler_warmup_precedes_metadata_builder(
    fa3, monkeypatch, method, has_config, npu_available, eager, block_size
):
    namespace = fa3.AscendFlashAttentionImpl.__init__.__globals__
    device = NS(type="npu", index=3)
    current_device = Mock(return_value=3)
    fake_torch = NS(device=Mock(return_value=device))
    if npu_available is not None:
        fake_torch.npu = NS(is_available=lambda: npu_available, current_device=current_device)
    monkeypatch.setitem(namespace, "torch", fake_torch)
    warmup = Mock()
    monkeypatch.setitem(namespace, "_initialize_fa3_scheduler", warmup)
    config = NS(
        speculative_config=NS(method=method, enforce_eager=eager) if method else None,
        model_config=NS(dtype=torch.float16),
        cache_config=NS(block_size=block_size),
        scheduler_config=NS(max_num_seqs=4),
        compilation_config=NS(
            cudagraph_capture_sizes=[],
            cudagraph_mode="none",
            static_forward_context={},
        ),
    )
    monkeypatch.setitem(namespace, "get_current_vllm_config_or_none", lambda: config if has_config else None)
    impl = fa3.AscendFlashAttentionImpl(4, 128, 0.1, num_kv_heads=1)
    expected_specs = {(4, 1, 128, torch.float16, 0.1, 0.0, None)}
    if has_config and npu_available:
        warmup.assert_called_once_with(device, expected_specs, block_size or 128)
        current_device.assert_called_once()
    else:
        warmup.assert_not_called()
        current_device.assert_not_called()
    if method == "dflash" and has_config and npu_available and eager:
        config.compilation_config.static_forward_context = {"draft.0": NS(impl=impl), "draft.1": NS(impl=impl)}
        warmup.reset_mock()
        builder = fa3.AscendFlashAttentionMetadataBuilder(
            NS(block_size=256, dtype=torch.float16), ["draft.0", "draft.1"], config, device
        )
        # Builder construction is after profiling and must not retry native init.
        warmup.assert_not_called()
        assert builder.scheduler_specs == expected_specs
        assert not builder.scheduler_buffers


def test_glm_dflash_missing_loader_tag_falls_back(selection):
    from vllm.compilation import backends

    r = selection
    r.device = r.profiles.AscendDeviceType.A3
    r.config.speculative_config.method = "dflash"
    r.config.speculative_config.enforce_eager = True
    r.config.speculative_config.target_model_config = NS(hf_config=NS(architectures=["GlmMoeDsaForCausalLM"]))
    r.config.speculative_config.draft_model_config = NS(hf_config=NS(architectures=["DFlash2DraftModel"]))
    del backends.model_tag
    assert "flash_attention_v3" not in r.select()
