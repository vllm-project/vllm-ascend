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
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def execute_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.fixture
def profiles():
    # Execute the real immutable capability matrix without initializing a device.
    namespace = {"__name__": __name__}
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
        speculative_config=NS(method="dspark", target_model_config=target, enable_adaptive_verification=False),
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
    namespace = {
        "torch": torch,
        "HardwareCapability": profiles.HardwareCapability,
        "AttentionBackendFamily": profiles.AttentionBackendFamily,
        "get_current_hardware_profile": lambda: profiles.get_hardware_profile(runtime.device),
        "util": NS(find_spec=lambda _: object() if runtime.installed else None),
        "_MINIMAX_M3_ARCHITECTURES": frozenset(
            {"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}
        ),
        "_validate_fa3_backend": lambda *args: False,
        "_validate_flash_attn_backend": lambda *args: False,
    }
    path = ROOT / "vllm_ascend/platform.py"
    execute_nodes(path, {"_validate_minimax_ecosystem_fa3", "_uses_minimax_ecosystem_fa3"}, namespace)
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    method = next(n for n in cls.body if getattr(n, "name", None) == "_select_attn_backend_cls")
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    runtime.select = lambda: namespace["_select_attn_backend_cls"](None, runtime.selector)
    runtime.profiles = profiles
    return runtime


@pytest.mark.parametrize("device", ["A2", "A3", "A5", "_310P"])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("arch", ["MiniMaxM3SparseForCausalLM", "DSparkMiniMaxDraftModel", "Qwen3DSparkModel", "Other"])
def test_hardware_and_model_routing(selection, device, installed, arch):
    r = selection
    r.device = getattr(r.profiles.AscendDeviceType, device)
    r.installed = installed
    r.config.model_config = NS(hf_config=NS(architectures=[arch]))
    actual = r.select()
    if device in ("A2", "A3") and installed and arch != "Other":
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
def fa3():
    class Base:
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
        "logger": NS(info_once=lambda *args, **kwargs: None),
        "get_scheduler_metadata": Mock(),
        "flash_attn_with_kvcache": Mock(),
        "DeviceOperator": NS(reshape_and_cache=Mock()),
        "notify_kv_cache_written": Mock(),
    }

    @contextmanager
    def transfer():
        yield

    namespace["attention_transfer_window"] = transfer
    names = {
        "_get_window_size",
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
    fa3.flash_attn_with_kvcache.side_effect = lambda *a, **kw: events.append("attention") or query * 2
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


@pytest.mark.parametrize("method,expected", [(None, True), ("eagle3", True), ("dspark", False)])
def test_sparse_prepare_fusion_is_disabled_for_dspark(method, expected):
    path = ROOT / "vllm_ascend/models/minimax_m3/minimax_m3.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3SparseAttention"
    )
    guard = next(n for n in cls.body if getattr(n, "name", None) == "_can_fuse_sparse_prepare")
    namespace = {"torch": torch, "HAS_TRITON": True, "_MAX_FUSED_QKNORM_ROPE_CACHE_INSERT_TOKENS": 262144}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec"), namespace)
    norm = NS(variance_epsilon=1e-6)
    layer = NS(
        use_dspark=method == "dspark",
        head_dim=128,
        idx_head_dim=128,
        rotary_emb=NS(is_neox_style=True, rotary_dim=64),
        q_norm=norm,
        k_norm=norm,
        index_q_norm=norm,
        index_k_norm=norm,
    )
    hidden = NS(shape=(8, 128), device=NS(type="npu"), dtype=torch.bfloat16)
    assert namespace["_can_fuse_sparse_prepare"](layer, torch.arange(8), hidden) == expected


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
    namespace = {
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
    assert fusion.call_count == len(manager.passes) == int(enabled and method != "dspark")


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
def test_minimax_fa3_scheduler_api(fa3, causal, padded):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.capture_sizes = set()
    builder.capture_query_len = None
    builder.block_size = 128
    builder.max_num_reqs = 3
    scale = 0.123
    spec = (16, 1, 128, torch.bfloat16, scale, 0.0, 4096)
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

    # Match the keyword-only subset of the v0.4.2.post2 public scheduler API.
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
        softcap,
        softmax_scale,
        num_splits,
    ):
        assert softcap == 0.0
        assert softmax_scale == scale
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
