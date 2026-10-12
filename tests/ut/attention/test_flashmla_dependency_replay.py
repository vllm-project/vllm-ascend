# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the dependency replay; kernels and vLLM objects are stand-ins.

Run with --confcutdir=tests/ut/attention. These execute production Python
methods, but do not establish NPU, HCCL or ACL graph compatibility.
"""

import __future__

import ast
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_function(relative, name, scope, class_name=None):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), relative, "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope[name]


def test_shared_probe_skips_absent_package_and_propagates_internal_failure():
    scope = {
        "lru_cache": lru_cache,
        "Callable": Callable,
        "find_spec": Mock(return_value=None),
        "import_module": Mock(),
    }
    load_function("vllm_ascend/attention/utils.py", "_has_flashmla_module", scope)
    probe = load_function("vllm_ascend/attention/utils.py", "get_flashmla_ops", scope)
    assert probe() is None
    scope["import_module"].assert_not_called()
    scope["find_spec"].return_value = object()
    scope["_has_flashmla_module"].cache_clear()
    scope["import_module"].return_value = SimpleNamespace(flash_mla_with_kvcache=lambda: None)
    assert probe() is None
    attention, metadata = Mock(), Mock()
    scope["import_module"].return_value = SimpleNamespace(
        flash_mla_with_kvcache=attention, flash_mla_with_kvcache_metadata=metadata
    )
    assert probe() == (attention, metadata)
    scope["import_module"].side_effect = ModuleNotFoundError("broken_dependency", name="broken_dependency")
    with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
        probe()


def make_dependency_cache(use_flashmla, spec_heads, available, hardware, dtype, page_padding_bytes=256):
    class OtherSpec:
        pass

    spec = SimpleNamespace(
        block_size=128,
        num_heads=spec_heads,
        num_kv_heads=1,
        dtype=dtype,
        page_size_bytes=128 * 576 * torch.empty((), dtype=dtype).element_size() + page_padding_bytes,
    )
    layer_name = "mla"
    layer = SimpleNamespace(impl=SimpleNamespace(use_flashmla=use_flashmla))
    scope = {
        "torch": torch,
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        "get_current_vllm_config": lambda: SimpleNamespace(),
        "_is_dsv4_model": lambda config: False,
        "_get_layer_kv_cache_specs": lambda config: {layer_name: spec},
        "is_deepseek_v41_cache": lambda specs: False,
        "get_storage_block_size": lambda spec: spec.block_size,
        "get_layers_from_vllm_config": lambda *args: {layer_name: layer},
        "AttentionLayerBase": object,
        "AscendSFAIndexerCacheSpec": OtherSpec,
        "AscendIndexerKPoolTailSpec": OtherSpec,
        "MambaSpec": OtherSpec,
        "is_hidden_state_cache_spec": lambda spec: False,
        "_uses_single_raw_mla_cache": lambda *args: True,
        "_get_attention_kv_cache_dims": lambda *args: (512, 64),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda capability: hardware),
        "HardwareCapability": SimpleNamespace(MLA_FLASH=object()),
        "MLA_FLASH_SUPPORTED_Q_HEADS": frozenset((8, 12, 64, 96)),
        "get_flashmla_ops": lambda: (Mock(), Mock()) if available else None,
    }
    for name in ("row_major_strides", "get_single_raw_mla_backing", "make_page_strided_cache_view"):
        load_function("vllm_ascend/worker/utils.py", name, scope)
    reshape = load_function("vllm_ascend/worker/v2/attn_utils.py", "_reshape_kv_cache_v2", scope)
    # Nonzero storage offset and a padded block stride must survive either view.
    backing = torch.full((2 * spec.page_size_bytes + 64,), 42, dtype=torch.uint8)
    raw = backing[32:-32]
    snapshot = raw.clone()
    group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=[layer_name])
    result = reshape([group], {layer_name: (raw,)}, "auto", [128], {}, SimpleNamespace(num_blocks=2))[layer_name]
    return SimpleNamespace(spec=spec, backing=backing, raw=raw, snapshot=snapshot, view=result)


@pytest.mark.parametrize(
    "use_flashmla,spec_heads,available,hardware,fused",
    [
        (True, 1, True, True, True),
        (False, 1, True, True, False),
        (False, 8, True, True, True),
        (False, 8, False, True, False),
        (False, 8, True, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_selected_flashmla_reuses_dependency_view_and_fia_layout(
    use_flashmla, spec_heads, available, hardware, fused, dtype
):
    cache = make_dependency_cache(use_flashmla, spec_heads, available, hardware, dtype)
    spec, raw, snapshot, result = cache.spec, cache.raw, cache.snapshot, cache.view
    assert torch.equal(raw, snapshot)
    views = (result,) if fused else result
    assert isinstance(result, torch.Tensor) is fused
    assert isinstance(views, tuple)
    for view in views:
        assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert view.stride(0) * view.element_size() == spec.page_size_bytes
    if fused:
        assert result.shape == (2, 128, 1, 576)
        assert result.storage_offset() * result.element_size() == 32
    else:
        assert [view.shape for view in views] == [(2, 128, 1, 512), (2, 128, 1, 64)]
        assert views[1].storage_offset() * views[1].element_size() == 32 + 128 * 512 * 2
    assert spec.num_heads == spec_heads


@pytest.mark.parametrize("fused", [True, False], ids=["flashmla-fused", "fia-component-major"])
@pytest.mark.parametrize("page_padding_bytes", [0, 256], ids=["full-page", "padded-page"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dependency_cache_reaches_dspark_prolog_writer(fused, page_padding_bytes, dtype):
    """Exercise view creation through the real writer call; the NPU kernel is a stand-in."""
    cache = make_dependency_cache(fused, 1, True, True, dtype, page_padding_bytes)
    expected_views = (cache.view[..., :512], cache.view[..., 512:]) if fused else cache.view
    slots = torch.tensor([0, 127, 128, 255, -1], dtype=torch.int32)
    states = torch.zeros((len(slots), 8), dtype=dtype)
    cos, sin = torch.ones((len(slots), 64), dtype=dtype), torch.zeros((len(slots), 64), dtype=dtype)

    def fake_prolog(**kwargs):
        assert kwargs["cache_mode"] == "PA_BSND"
        assert kwargs["cache_index"].dtype == torch.int64
        torch.testing.assert_close(kwargs["cache_index"], slots.to(torch.int64))
        assert kwargs["token_x"] is states
        for actual, expected in zip((kwargs["kv_cache"], kwargs["kr_cache"]), expected_views):
            assert actual.untyped_storage().data_ptr() == cache.raw.untyped_storage().data_ptr()
            assert actual.shape == expected.shape
            assert actual.stride() == expected.stride()
            assert actual.storage_offset() == expected.storage_offset()
        # Marker writes check aliasing only, not real PROLOG RMSNorm/RoPE numerics.
        for index, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            page, token = divmod(slot, 128)
            kwargs["kv_cache"][page, token, 0].fill_(index + 1)
            kwargs["kr_cache"][page, token, 0].fill_(index + 11)

    prolog = Mock(side_effect=fake_prolog)
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_mla_prolog_v3=prolog),
        "AscendW8A8DynamicLinearMethod": object(),
        "AscendW8A8MXFP8DynamicLinearMethod": object(),
        "get_cos_and_sin_mla": lambda positions: (cos, sin),
        "_log_dspark_mla_cache_diagnostic": lambda *args: None,
    }
    model_path = "vllm_ascend/models/kimi_k3_dspark.py"
    for name in ("_normalize_dspark_mla_cache", "_uses_prolog_v3_context_writer"):
        load_function(model_path, name, scope)
    impl = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_kv_heads=1,
        use_mla_rope=True,
        fa_quant_layer=False,
        mlapo_W_UK_T=object(),
        q_a_layernorm=SimpleNamespace(weight=torch.ones(8, dtype=dtype), variance_epsilon=1e-6),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512, dtype=dtype), variance_epsilon=1e-6),
        _prepare_context_prolog_v3_weights=lambda: {
            "quant_type": None,
            "weight_dq": object(),
            "weight_uq_qr": object(),
            "weight_dkv_kr": object(),
        },
        exec_kv_prefill=Mock(side_effect=AssertionError("Strided DSpark views must reach PROLOG_V3")),
    )
    for name in ("_validate_context_prolog_v3_cache", "exec_context_kv_prolog_v3"):
        method = load_function("vllm_ascend/attention/mla_v1.py", name, scope, "AscendMLAImpl")
        setattr(impl, name, MethodType(method, impl))
    projection = Mock(side_effect=AssertionError("PROLOG_V3 owns the context projection"))
    attn = SimpleNamespace(kv_cache=cache.view, impl=impl, fused_qkv_a_proj=projection, q_lora_rank=8)
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    precompute = load_function(model_path, "precompute_and_store_context_kv", scope, "AscendK3DSparkModel")
    precompute(model, states, torch.arange(len(slots)), slots)
    prolog.assert_called_once()
    projection.assert_not_called()
    impl.exec_kv_prefill.assert_not_called()

    # Independent raw-buffer addresses verify the page boundary, offset and
    # untouched padding; a copied view would leave this original storage stale.
    expected_raw = cache.snapshot.clone()
    expected_elements = expected_raw.view(dtype)
    page_elements = cache.spec.page_size_bytes // expected_elements.element_size()
    for index, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, token = divmod(slot, 128)
        nope_offset = page * page_elements + token * (576 if fused else 512)
        rope_offset = nope_offset + 512 if fused else page * page_elements + 128 * 512 + token * 64
        expected_elements[nope_offset : nope_offset + 512].fill_(index + 1)
        expected_elements[rope_offset : rope_offset + 64].fill_(index + 11)
    assert torch.equal(cache.raw, expected_raw)
    assert torch.all(cache.backing[:32] == 42)
    assert torch.all(cache.backing[-32:] == 42)


def test_flash_scope_preserves_other_provider_and_releases_once_on_failure():
    scope = {"contextmanager": contextmanager}
    bind = load_function("vllm_ascend/worker/v2/attn_utils.py", "flashmla_metadata_scope", scope)
    state = SimpleNamespace(executor=None, defer=False)
    flash_builder = SimpleNamespace(flashmla_state=state)
    target_builder = SimpleNamespace(take_device_metadata_tasks=Mock(), defer_device_metadata=Mock())
    groups = [[SimpleNamespace(get_metadata_builder=lambda index, b=b: b) for b in (flash_builder, target_builder)]]
    executor = SimpleNamespace(submission_in_flight=True, release=Mock())
    with pytest.raises(RuntimeError, match="consumer failed"), bind(groups, executor):
        assert state.executor is executor and state.defer
        with bind(groups, executor):
            pass
        executor.release.assert_not_called()
        raise RuntimeError("consumer failed")
    executor.release.assert_called_once()
    assert state.executor is None and not state.defer
    target_builder.take_device_metadata_tasks.assert_not_called()
    target_builder.defer_device_metadata.assert_not_called()


def test_no_flash_executor_leaves_builders_untouched():
    scope = {"contextmanager": contextmanager}
    bind = load_function("vllm_ascend/worker/v2/attn_utils.py", "flashmla_metadata_scope", scope)
    builder = Mock(side_effect=AssertionError("FIA must not bind FlashMLA state"))
    with bind([[SimpleNamespace(get_metadata_builder=builder)]], None):
        pass
    builder.assert_not_called()
