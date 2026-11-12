# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import vllm

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.ops.fused_moe.routed_experts import EplbExpertTensorList
from vllm_ascend.utils import vllm_version_is


def _native_tree(path):
    return ast.parse((Path(vllm.__file__).parent / path).read_text(encoding="utf-8"))


def _execute(nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<actual installed native contract>", "exec"), scope)


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("engine_fails", [False, True])
def test_weight_transfer_invokes_native_replacement_callback(with_draft, engine_fails):
    events = []
    release = vllm_version_is("0.33.0")
    native = next(
        node
        for node in _native_tree("v1/worker/gpu/model_runner.py").body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    native.body = [
        node
        for node in native.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("reset_lora_state", "on_target_weights_replaced", "get_draft_model")
    ]
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == ("reset_lora_state" if release else "on_target_weights_replaced")
        for node in native.body
    )
    base_methods = {"reset_lora_state": lambda self: events.append("reset release LoRA")} if release else {}
    native.bases = [ast.Name(id="Base", ctx=ast.Load())]
    native.decorator_list = []

    class DraftModelSpeculator:
        model = object()

    scope: dict[str, Any] = {
        "Base": type("Base", (), base_methods),
        "DraftModelSpeculator": DraftModelSpeculator,
        "set_current_vllm_config": lambda config: nullcontext(),
        "refresh_quantized_lm_heads": lambda model: events.append("refresh draft lm heads"),
        "vllm_version_is": vllm_version_is,
    }
    _execute([native], scope)
    runner = scope["GPUModelRunner"]()
    runner.speculator = DraftModelSpeculator() if with_draft else None
    runner.vllm_config = object()
    worker_path = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/worker.py"
    worker_cls = next(
        node
        for node in ast.parse(worker_path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUWorker"
    )
    finish = next(
        node for node in worker_cls.body if isinstance(node, ast.FunctionDef) and node.name == "finish_weight_update"
    )
    _execute([finish], scope)

    def finish_base():
        events.append("finish base weights")
        if engine_fails:
            raise RuntimeError("weight engine failed")

    worker = SimpleNamespace(
        _weight_update_active=True,
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=SimpleNamespace(finish_weight_update=finish_base),
        model_runner=runner,
    )
    if engine_fails:
        with pytest.raises(RuntimeError, match="weight engine failed"):
            scope["finish_weight_update"](worker)
        assert events == ["finish base weights"]
        assert worker._weight_update_active is True
    else:
        scope["finish_weight_update"](worker)
        expected = ["finish base weights"]
        if release:
            expected.append("reset release LoRA")
        if with_draft:
            expected.append("refresh draft lm heads")
        assert events == expected
        assert worker._weight_update_active is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_eplb_copy_back_preserves_independent_experts(dtype):
    dst = [EplbExpertTensorList(torch.zeros(2, 4, dtype=dtype) for _ in range(3)) for _ in range(2)]
    src = [
        EplbExpertTensorList(torch.full((2, 4), float(i + offset + 1), dtype=dtype) for i in range(3))
        for offset in (0, 5)
    ]
    pointers = [[tensor.data_ptr() for tensor in weights] for weights in dst]
    source_snapshot = [[tensor.clone() for tensor in weights] for weights in src]
    assert dst[0].is_cuda is False
    tree = _native_tree("distributed/eplb/rebalance_execute.py")
    scope: dict[str, Any] = {}
    if vllm_version_is("0.33.0"):
        loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and ast.unparse(node.target) == "dst"
            and "w[dst].copy_(b[dst]" in ast.unparse(node)
        )
        scope.update(dest_indices=[2, 0], expert_weights=dst, expert_weights_buffers=src)
        _execute([loop], scope)
        pairs = [(2, 2), (0, 0)]
    else:
        copy = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "copy_rows")
        _execute([copy], scope)
        scope["copy_rows"](dst, [2, 0], src, [0, 2])
        scope["copy_rows"](dst, [0], dst, [0])  # Native add_model warmup is a no-op copy.
        pairs = [(2, 0), (0, 2)]
    for weights, source, snapshot, original_ptrs in zip(dst, src, source_snapshot, pointers):
        assert [tensor.data_ptr() for tensor in weights] == original_ptrs
        for dst_row, src_row in pairs:
            torch.testing.assert_close(weights[dst_row], snapshot[src_row], rtol=0, atol=0)
        assert torch.count_nonzero(weights[1]).item() == 0
        for tensor, original in zip(source, snapshot):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_sfa_satisfies_native_sparse_guard_without_dense_prefill():
    release = vllm_version_is("0.33.0")
    assert AscendSFAImpl.is_sparse is (not release)
    assert AscendSFAImpl.supports_dense_mha_prefill is release
    native = next(
        node
        for node in _native_tree("model_executor/layers/attention/mla_attention.py").body
        if isinstance(node, ast.ClassDef) and node.name == "MLAAttention"
    )
    init = next(node for node in native.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    guard = next(
        (
            node
            for node in init.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "use_sparse and (not self.impl.is_sparse)"
        ),
        None,
    )
    assert (guard is None) == release
    if release:
        return
    state = SimpleNamespace(
        impl=SimpleNamespace(
            is_sparse=AscendSFAImpl.is_sparse, supports_dense_mha_prefill=AscendSFAImpl.supports_dense_mha_prefill
        ),
        attn_backend=SimpleNamespace(get_name=lambda: "ASCEND_SFA"),
        supports_dense_mha_prefill=True,
    )
    scope = dict(self=state, use_sparse=True, prefix="model.layers.0.self_attn.attn", logger=Mock())
    _execute([guard], scope)
    prefill = next(
        node
        for node in init.body
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("self.impl.is_sparse and")
    )
    _execute([prefill], scope)
    assert state.prefill_backend is None
    state.impl.is_sparse = False
    with pytest.raises(RuntimeError, match="requires a sparse attention backend"):
        _execute([guard], scope)


@pytest.mark.parametrize("custom_metadata", [False, True])
def test_sfa_metadata_constructor_keeps_sparse_backend_and_release_clone(custom_metadata):
    release = vllm_version_is("0.33.0")
    backend = Mock() if release else None
    cloned_backend = object()
    if backend is not None:
        backend.clone.return_value = cloned_backend
    layer = SimpleNamespace(
        prefill_backend=backend,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    config: Any = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32, hf_text_config=SimpleNamespace(mla_use_nope=False)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=None,
    )
    spec: Any = SimpleNamespace(block_size=128)
    metadata_cls = type("CustomSFAMetadata", (AscendSFAMetadata,), {}) if custom_metadata else None
    native_module = "vllm.model_executor.layers.attention.mla_attention"
    with (
        patch(f"{native_module}.get_num_attention_heads_from_layers", return_value=1),
        patch(f"{native_module}.get_dcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(AscendSFAMetadataBuilder, "determine_prefill_query_data_type", return_value=torch.float32),
        patch.object(AscendSFAMetadataBuilder, "determine_chunked_prefill_workspace_size", return_value=4),
        patch("vllm_ascend.attention.sfa_v1.is_pcp_decode_sharding_enabled", return_value=False),
    ):
        builder = AscendSFAMetadataBuilder(spec, ["layer"], config, torch.device("cpu"), metadata_cls)
    assert layer.prefill_backend is backend
    assert builder.metadata_cls is (metadata_cls if custom_metadata else AscendSFAMetadata)
    assert builder.model_config is config.model_config
    assert builder.kv_cache_spec is spec
    assert builder.vllm_config is config
    assert builder.device == torch.device("cpu")
    assert builder.decode_threshold == builder.reorder_batch_threshold == 1
    if release:
        assert backend is not None
        backend.clone.assert_called_once_with()
        assert builder._prefill_backend is cloned_backend
    else:
        assert not hasattr(builder, "_prefill_backend")
        assert not hasattr(builder, "chunked_prefill_workspace")
