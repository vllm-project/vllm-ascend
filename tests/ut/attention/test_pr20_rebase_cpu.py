# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise PR20/mainline integration without importing the NPU plugin.

Production methods run with dependency boundaries stubbed; these checks do
not validate NPU kernels, graph capture, or inference performance.
"""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"
MODEL = "vllm_ascend/models/kimi_k3.py"


def load(path, owner, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), path, "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pcp", [1, 2])
@pytest.mark.parametrize(
    "mode,runner,draft_prefix",
    [
        ("C16", "generate", False),
        ("C16", "draft", False),
        ("C8", "generate", False),
        ("C8", "generate", True),
        ("C8", "draft", True),
    ],
)
def test_pa_nz_flag_preserves_c8_target_draft_and_pcp_routes(enabled, pcp, mode, runner, draft_prefix):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        model_config=SimpleNamespace(runner_type=runner, dtype=torch.bfloat16),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=128),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=128,
        ),
        _kimi_c8_draft_layer_prefixes=("draft",) if draft_prefix else (),
    )
    scope = {
        "torch": torch,
        "get_current_vllm_config": lambda: config,
        "get_ascend_config": lambda: SimpleNamespace(mla_cache_mode=mode, enable_kv_nz=False),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "HardwareCapability": SimpleNamespace(FP8_ATTENTION="fp8", MLA_FLASH="mla"),
        "is_pcp_decode_sharding_enabled": lambda _: False,
        "enabling_mlapo": lambda _: True,
        "enable_fa_quant": lambda *_: False,
        "kv_cache_dtype_str_to_dtype": lambda *_: torch.bfloat16,
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_ENABLE_PA_NZ=enabled),
        "FLASHMLA_SUPPORTED_BLOCK_SIZES": tuple(range(128, 1025, 128)),
        "MLA_KERNEL_BLOCK_SIZE": 128,
        "FLASHMLA_QK_DIM": 576,
        "FLASHMLA_V_DIM": 512,
        "MLA_FLASH_SUPPORTED_Q_HEADS": frozenset((8, 12, 64, 96)),
        "supports_component_major_mla_pd": lambda _: True,
        "enable_sfa": lambda _: False,
        "get_flashmla_ops": lambda: (object(), object()),
    }
    validate = Mock()
    impl = SimpleNamespace(_validate_kimi_c8_layer=Mock(), _validate_flash_mla_pa_nz_layer=validate)
    impl._can_use_flashmla = MethodType(load(MLA, "AscendMLAImpl", "_can_use_flashmla", scope), impl)
    initialize = load(MLA, "AscendMLAImpl", "__init__", scope)
    initialize(
        impl,
        12,
        576,
        0.125,
        1,
        None,
        None,
        "auto",
        None,
        "decoder",
        None,
        q_lora_rank=4,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        rotary_emb=None,
        fused_qkv_a_proj=object(),
        q_b_proj=object(),
        kv_b_proj=object(),
        o_proj=object(),
        use_mla_rope=False,
        layer_name="draft.layer" if draft_prefix else "target.layer",
    )
    expected_pa_nz = enabled and mode == "C16" and runner != "draft" and pcp == 1
    assert impl.use_flashmla_pa_nz is expected_pa_nz
    assert impl.flash_mla_layout == ("pa_nz" if expected_pa_nz else "pa_bbnd")
    assert impl.kimi_c8_mode is (mode == "C8" and not draft_prefix)
    assert impl.is_draft_cache is (mode == "C8" and draft_prefix)
    assert impl.use_flashmla is (pcp == 1 and not impl.kimi_c8_mode)
    if expected_pa_nz:
        validate.assert_called_once_with()
    else:
        validate.assert_not_called()


def test_layout_metadata_does_not_cross_target_and_draft_readers():
    get_flash = load(MLA, "AscendMLAMetadata", "get_external_flashmla", {})
    native, nz = object(), object()
    metadata = SimpleNamespace(
        external_flashmla=native,
        external_flashmla_by_layout={"pa_bbnd": native, "pa_nz": nz},
    )
    assert get_flash(metadata, "pa_nz") is nz
    assert get_flash(metadata, "pa_bbnd") is native
    assert get_flash(metadata, "missing") is None
    metadata.external_flashmla_by_layout = None
    assert get_flash(metadata, "pa_bbnd") is native


@pytest.mark.parametrize("connector_supported", [False, True])
@pytest.mark.parametrize("mode", ["c8", "native", "pa_nz"])
def test_raw_cache_connector_gate_remains_specific_to_pa_nz(mode, connector_supported):
    class CacheSpec:
        pass

    class Attention:
        pass

    spec = CacheSpec()
    spec.supports_single_raw_backing = True
    spec.static_fp8_pa_nz = mode == "c8"
    spec.block_size = 1664 if spec.static_fp8_pa_nz else 128
    spec.page_size_bytes = spec.block_size * 576
    attention = Attention()
    attention.impl = SimpleNamespace(
        pcp_enabled=False,
        is_pcp_decode_sharded=False,
        use_flashmla=mode != "c8",
        flash_mla_layout="pa_nz" if mode == "pa_nz" else "pa_bbnd",
    )
    validate = Mock()
    scope = {
        "AscendMLAAttentionSpec": CacheSpec,
        "AttentionLayerBase": object,
        "MLAAttention": Attention,
        "enable_sfa": lambda _: False,
        "supports_component_major_mla_pd": lambda _: connector_supported,
        "get_layers_from_vllm_config": lambda *_: {"mla": attention},
        "validate_pa_nz_page_stride": validate,
        "supports_strided_mla_cache": Mock(side_effect=AssertionError("No strided fallback is expected")),
    }
    select = load("vllm_ascend/worker/v2/attn_utils.py", None, "_uses_single_raw_mla_cache", scope)
    assert select(SimpleNamespace(), "mla", spec) is (connector_supported or mode == "pa_nz")
    if mode == "c8" and connector_supported:
        validate.assert_called_once_with(spec.block_size, spec.page_size_bytes)
    else:
        validate.assert_not_called()


@pytest.mark.parametrize("layout,draft", [("pa_bbnd", False), ("pa_bbnd", True), ("pa_nz", False)])
def test_native_pages_are_advertised_only_for_native_target_reader(layout, draft):
    legacy, native = object(), object()

    class LegacyLayer:
        def get_attn_backend(self):
            return legacy

    tree = ast.parse((ROOT / MLA).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAAttention")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"MLAAttention": LegacyLayer, "AscendFlashMLABackend": native}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), MLA, "exec"), scope)
    layer = scope["AscendMLAAttention"]()
    layer.impl = SimpleNamespace(
        use_flashmla=True,
        is_draft_cache=draft,
        flash_mla_layout=layout,
        use_flashmla_pa_nz=layout == "pa_nz",
    )
    assert layer.get_attn_backend() is (native if layout == "pa_bbnd" and not draft else legacy)


@pytest.mark.parametrize("context_available", [False, True])
def test_bf16_metadata_prelaunch_preserves_c8_kda_hook(context_available):
    class KdaAttention:
        pass

    c8_start = Mock()
    kda = KdaAttention()
    c8 = SimpleNamespace(impl=SimpleNamespace(kimi_c8_mode=True, _prelaunch_kimi_c8_metadata=c8_start))
    shared, separate = (SimpleNamespace(metadata_async=True) for _ in range(2))
    starts, metadatas, layers = [], {}, [SimpleNamespace(self_attn=kda), SimpleNamespace(self_attn=c8)]
    for name, flash in (("nz0", shared), ("nz1", shared), ("nz2", separate)):
        start = Mock()
        starts.append(start)
        impl = SimpleNamespace(layer_name=name, flash_mla_layout="pa_nz", start_flash_mla_metadata=start)
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(impl=impl)))
        metadatas[name] = SimpleNamespace(num_decode_tokens=3, get_external_flashmla=lambda _, flash=flash: flash)
    scope = {
        "AscendKimiK3DeltaAttention": KdaAttention,
        "is_forward_context_available": lambda: context_available,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata=metadatas),
    }
    model = SimpleNamespace(layers=layers)
    load(MODEL, "AscendKimiLinearModel", "_bind_c8_metadata_prelaunch", scope)(model)
    kda._kimi_c8_metadata_prelaunch(metadatas)
    c8_start.assert_called_once_with(metadatas)
    load(MODEL, "AscendKimiLinearModel", "_start_flash_mla_metadata", scope)(model)
    if context_available:
        starts[0].assert_called_once_with(metadatas["nz0"])
        starts[1].assert_not_called()
        starts[2].assert_called_once_with(metadatas["nz2"])
    else:
        assert all(not start.called for start in starts)


@pytest.mark.parametrize("heads", [6, 12, 96])
def test_bf16_prolog_transforms_only_unpadded_tp_local_heads(heads):
    q_rank, q_dim, nope_dim, kv_rank = 5, 6, 4, 8
    q_b = torch.arange(heads * q_dim * q_rank, dtype=torch.float32).reshape(heads * q_dim, q_rank)
    w_uk = torch.arange(heads * nope_dim * kv_rank, dtype=torch.float32).reshape(heads, nope_dim, kv_rank)
    replace_parameter = Mock()
    format_cast = Mock(side_effect=lambda value, _: value)
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_format_cast=format_cast),
        "replace_parameter": replace_parameter,
        "ACL_FORMAT_FRACTAL_NZ": 29,
    }
    impl = SimpleNamespace(
        num_heads=heads,
        q_lora_rank=q_rank,
        qk_head_dim=q_dim,
        qk_nope_head_dim=nope_dim,
        kv_lora_rank=kv_rank,
        q_proj=SimpleNamespace(weight=q_b),
        W_UK_T=w_uk,
    )
    prepare = load(MLA, "AscendMLAImpl", "_prepare_flash_mla_bf16_weights", scope)
    prepare(impl)
    assert replace_parameter.call_count == 2
    for call, name, expected in zip(
        replace_parameter.call_args_list, ("flash_weight_uq_qr", "flash_weight_uk"), (q_b.T, w_uk)
    ):
        assert call.args[:2] == (impl, name)
        assert call.kwargs == {"prefer_copy": True}
        assert call.args[2].is_contiguous()
        torch.testing.assert_close(call.args[2], expected)
    format_cast.assert_called_once()
    assert format_cast.call_args.args[0].shape == (q_rank, heads * q_dim)
