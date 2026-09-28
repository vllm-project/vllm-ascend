# SPDX-License-Identifier: Apache-2.0
"""CPU tests for worker lifecycle and the actual Kimi prefill dispatch.

The startup compiler and chunk math are mocked only where necessary; Torch is
real and the production module/methods are imported normally from the checkout.
"""

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops import kda_state_copy as production
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention


@pytest.mark.parametrize("backend", ["auto", "torch", "triton"])
def test_backend_config_is_typed_and_defaults_to_auto(backend):
    """The central enum admits only explicit supported backend names."""
    config_args = {"sparse_kv_offload_config": SimpleNamespace(enabled=False)}
    assert AscendConfig(**config_args).kda_state_copy_backend == "auto"
    assert AscendConfig(**config_args, kda_state_copy_backend=backend).kda_state_copy_backend == backend
    with pytest.raises(ValueError):
        AscendConfig(**config_args, kda_state_copy_backend="automatic-fallback")


def test_startup_deduplicates_and_publishes_after_all_seals(monkeypatch):
    """No layer can observe a partly prepared or unsealed startup plan."""
    events = []
    shared = torch.zeros((8, 1, 2, 3))
    compiled = MappingProxyType({"kernel": object()})
    layers = [
        SimpleNamespace(_kda_state_copy_backend="triton", kv_cache=(None, shared.clone()), _ascend_kda_state_copy=None)
        for _ in range(3)
    ]

    def prepare(state, maximum):
        assert maximum == 128
        events.append("prepare")
        return SimpleNamespace(seal=seal, _compiled=compiled)

    def seal():
        assert all(layer._ascend_kda_state_copy is None for layer in layers)
        events.append("seal")

    monkeypatch.setattr(production.KDAStateCopyPlan, "prepare", prepare)
    context = {f"layer_{i}": layer for i, layer in enumerate(layers)}
    production.initialize_kda_state_copy(context, 128)
    assert events == ["prepare", "seal"]
    bound = [layer._ascend_kda_state_copy for layer in layers]
    assert len({id(plan) for plan in bound}) == len(layers)
    assert all(plan._compiled is compiled for plan in bound)
    assert [plan._layer_name for plan in bound] == list(context)


@pytest.mark.parametrize("failure", ["prepare", "seal", "unbound"])
def test_startup_failure_clears_old_plans_and_cannot_fallback(monkeypatch, failure):
    """A startup error leaves every opted-in layer unready, including reinit."""
    layers = [
        SimpleNamespace(
            _kda_state_copy_backend="triton",
            kv_cache=(None, torch.zeros((8, 1, 2, 3))),
            _ascend_kda_state_copy=object(),
        )
        for _ in range(2)
    ]
    if failure == "unbound":
        layers[-1].kv_cache = ()

    def fail(*args):
        raise RuntimeError("deliberate startup failure")

    monkeypatch.setattr(
        production.KDAStateCopyPlan,
        "prepare",
        fail if failure == "prepare" else lambda *args: SimpleNamespace(seal=fail),
    )
    with pytest.raises(RuntimeError):
        production.initialize_kda_state_copy(dict(enumerate(layers)), 128)
    assert all(layer._ascend_kda_state_copy is None for layer in layers)


def test_non_kda_and_non_opted_in_layers_do_not_compile(monkeypatch):
    """Default models and empty PP stages never allocate or compile a plan."""
    prepare = Mock(side_effect=AssertionError("unexpected preparation"))
    monkeypatch.setattr(production.KDAStateCopyPlan, "prepare", prepare)
    production.initialize_kda_state_copy(
        {"other": SimpleNamespace(), "default": SimpleNamespace(_kda_state_copy_backend="torch")}, 128
    )
    prepare.assert_not_called()


@pytest.mark.parametrize("keep", [None, torch.tensor([1], dtype=torch.int64)])
def test_actual_prefill_calls_plan_and_preserves_keep_metadata(monkeypatch, keep):
    """The real method routes gather/scatter, including filtered request IDs."""
    import vllm_ascend.ops.kimi_kda as kimi

    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(attention)
    attention._kda_state_copy_backend = "triton"
    attention.gate_lower_bound = None
    attention.A_log = torch.zeros(1)
    attention.dt_bias = torch.zeros(2)
    selected = 2 if keep is None else 1
    gathered = torch.zeros((selected, 1, 2, 2))
    final_state = gathered + 2
    plan = SimpleNamespace(gather=Mock(return_value=gathered), scatter=Mock())
    attention._ascend_kda_state_copy = plan
    chunk = Mock(return_value=("output", final_state))
    monkeypatch.setattr(kimi, "run_chunk_kda", chunk)
    monkeypatch.setattr(kimi, "clear_ssm_states", Mock(side_effect=AssertionError("Torch fallback")))
    state = torch.zeros((4, 1, 2, 2))
    indices, flags = torch.tensor([1, 3], dtype=torch.int32), torch.tensor([True, False])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 2), cu_seqlens_kern=None, keep_meta=keep, chunk_indices_chunk64_host=(0, 0)
    )
    output = attention._run_prefill(None, None, None, None, None, state, indices, flags, metadata)
    assert output == "output"
    assert chunk.call_args.args[5] is gathered
    torch.testing.assert_close(plan.gather.call_args.args[1], indices if keep is None else indices[keep])
    torch.testing.assert_close(plan.gather.call_args.args[2], flags if keep is None else flags[keep])
    assert plan.scatter.call_args.args[0] is state
    assert plan.scatter.call_args.args[1] is final_state
    torch.testing.assert_close(plan.scatter.call_args.args[2], plan.gather.call_args.args[1])


def test_opted_in_prefill_without_startup_fails_closed():
    """An omitted worker hook cannot silently run the default implementation."""
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(attention)
    attention._kda_state_copy_backend = "triton"
    attention._ascend_kda_state_copy = None
    metadata = SimpleNamespace(cu_seqlens_host=(0, 1), cu_seqlens_kern=None, keep_meta=None)
    with pytest.raises(RuntimeError, match="worker cache initialization"):
        attention._run_prefill(
            None, None, None, None, None, torch.zeros((1, 1, 2, 2)), torch.tensor([0]), torch.tensor([True]), metadata
        )


@pytest.mark.parametrize("relative", ["worker/model_runner_v1.py", "worker/v2/model_runner.py"])
def test_worker_hook_is_inside_cache_initialization(relative):
    """Static wiring regression: both runner paths invoke the startup helper.

    This supplements behavioral helper/prefill tests; it is not a full worker
    boot test and does not claim to instantiate a serving engine.
    """
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "initialize_kv_cache"
    )
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    startup = next(
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "initialize_kda_state_copy"
    )
    bind = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == ("initialize_kv_cache_tensors" if "v1" in relative else "initialize_kv_cache")
    )
    assert startup.lineno > bind.lineno
    guard = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and any(child is startup for child in ast.walk(node))
        and "kda_state_copy_backend" in ast.unparse(node.test)
    )
    assert '"auto"' in ast.unparse(guard.test) or "'auto'" in ast.unparse(guard.test)
    assert '"triton"' in ast.unparse(guard.test) or "'triton'" in ast.unparse(guard.test)


def test_auto_unsupported_contiguous_cache_needs_no_plan(monkeypatch):
    """Auto leaves unsupported contiguous layouts on the ordinary Torch route."""
    state = torch.zeros((4, 1, 2, 2))
    layer = SimpleNamespace(_kda_state_copy_backend="auto", kv_cache=(None, state))
    monkeypatch.setattr(production.KDAStateCopyPlan, "prepare", Mock(side_effect=AssertionError("unexpected compile")))
    production.initialize_kda_state_copy({"layer": layer}, 8)
    assert layer._kda_state_copy_ready
    assert layer._ascend_kda_state_copy is None


def test_auto_eligible_cache_selects_prepared_triton(monkeypatch):
    """Automatic capability routing publishes a sealed plan without opt-in."""
    state = torch.zeros((4, 1, 2, 2))
    layer = SimpleNamespace(_kda_state_copy_backend="auto", kv_cache=(None, state))
    plan = SimpleNamespace(seal=Mock())
    monkeypatch.setattr(production, "supports_kda_state_copy", lambda state: True)
    monkeypatch.setattr(production.KDAStateCopyPlan, "prepare", lambda *args: plan)
    production.initialize_kda_state_copy({"layer": layer}, 8)
    plan.seal.assert_called_once()
    assert layer._ascend_kda_state_copy is not plan
    assert layer._ascend_kda_state_copy._layer_name == "layer"
    assert not hasattr(plan, "_layer_name")
    assert layer._kda_state_copy_ready


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_strided_fallback_rejects_invalid_scheduler_limit(maximum):
    """Both plan types reject invalid limits before touching device state."""
    with pytest.raises(ValueError, match="positive scheduler limit"):
        production.StridedKDAFallbackPlan.prepare(torch.empty((2, 1, 2, 2)), maximum)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_registered_ops_have_fake_outputs_and_mutation_schema(dtype, monkeypatch):
    """Fake tracing allocates metadata only and never resolves a worker plan."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(production, "_context_plan", Mock(side_effect=AssertionError("fake accessed worker")))
    with FakeTensorMode():
        state = torch.empty_strided((8, 2, 3, 4), (64, 12, 4, 1), dtype=dtype)
        indices = torch.empty(3, dtype=torch.int32)
        flags = torch.empty((1, 3), dtype=torch.int32)
        packed = torch.ops.vllm.kda_state_gather(state, indices, flags, "fake.layer")
        assert packed.shape == (3, 2, 3, 4)
        assert packed.dtype == dtype and packed.is_contiguous()
        assert packed.device == state.device
        assert torch.ops.vllm.kda_state_scatter(state, packed, indices, "fake.layer") is None
        with pytest.raises(RuntimeError):
            torch.ops.vllm.kda_state_gather(state, indices, flags[:, :2], "fake.layer")
        with pytest.raises(RuntimeError):
            torch.ops.vllm.kda_state_scatter(state, packed[:2], indices, "fake.layer")
    schema = torch.ops.vllm.kda_state_scatter.default._schema
    assert schema.arguments[0].alias_info.is_write
    assert schema.arguments[1].alias_info is None


def test_compiled_dispatch_requires_current_ready_layer(monkeypatch):
    """Missing lifecycle publication cannot be hidden behind the custom op."""
    layer = SimpleNamespace(_ascend_kda_state_copy=object(), _kda_state_copy_ready=False)
    monkeypatch.setattr(production, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": layer}))
    with pytest.raises(RuntimeError, match="prepared worker layer"):
        production._context_plan("layer")
    layer._kda_state_copy_ready = True
    assert production._context_plan("layer") is layer._ascend_kda_state_copy


@pytest.mark.parametrize("plan_type", [production.KDAStateCopyPlan, production.StridedKDAFallbackPlan])
def test_same_layout_layers_trace_their_own_binding(monkeypatch, plan_type):
    """Shared kernels must not alias layer names in either compiled copy path."""
    state = torch.empty_strided((8, 1, 2, 3), (12, 6, 3, 1))
    layers = {
        name: SimpleNamespace(_kda_state_copy_backend="auto", kv_cache=(None, state))
        for name in ("layer_0", "layer_31")
    }
    template = plan_type()
    template._compiled = MappingProxyType({"kernel": object()})
    template.seal = Mock()
    prepare = Mock(return_value=template)
    monkeypatch.setattr(plan_type, "prepare", prepare)
    monkeypatch.setattr(production, "supports_kda_state_copy", lambda state: plan_type is production.KDAStateCopyPlan)
    production.initialize_kda_state_copy(layers, 8)
    prepare.assert_called_once()
    template.seal.assert_called_once()
    assert not hasattr(template, "_layer_name")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    gather = Mock(return_value=torch.empty((1, 1, 2, 3)))
    scatter = Mock()
    monkeypatch.setattr(torch.ops.vllm, "kda_state_gather", gather)
    monkeypatch.setattr(torch.ops.vllm, "kda_state_scatter", scatter)
    indices, flags = torch.tensor([0]), torch.tensor([True])
    for name, layer in layers.items():
        plan = layer._ascend_kda_state_copy
        assert plan._compiled is template._compiled
        # Simulate a context exposing only this layer, not its layout siblings.
        monkeypatch.setattr(
            production,
            "get_forward_context",
            lambda name=name, layer=layer: SimpleNamespace(no_compile_layers={name: layer}),
        )
        assert production._context_plan(plan._layer_name) is plan
        packed = plan.gather(state, indices, flags)
        plan.scatter(state, packed, indices)
        assert gather.call_args.args[-1] == name
        assert scatter.call_args.args[-1] == name


@pytest.mark.parametrize("cache", [None, (), 3, (None, None), (None, object())])
def test_startup_rejects_unbound_or_non_tensor_cache(monkeypatch, cache):
    """Invalid cache binding fails before compilation and leaves layers unready."""
    layer = SimpleNamespace(_kda_state_copy_backend="auto", kv_cache=cache)
    prepare = Mock(side_effect=AssertionError("invalid cache reached preparation"))
    monkeypatch.setattr(production.KDAStateCopyPlan, "prepare", prepare)
    with pytest.raises(RuntimeError, match="cache.*(bound|torch.Tensor)"):
        production.initialize_kda_state_copy({"layer": layer}, 8)
    assert not layer._kda_state_copy_ready
    assert layer._ascend_kda_state_copy is None
    prepare.assert_not_called()


def test_startup_rejects_missing_cache_attribute():
    """Absent bindings use the same explicit lifecycle error as empty caches."""
    layer = SimpleNamespace(_kda_state_copy_backend="auto")
    with pytest.raises(RuntimeError, match="cache must be bound"):
        production.initialize_kda_state_copy({"layer": layer}, 8)
    assert not layer._kda_state_copy_ready


@pytest.mark.parametrize("backend", ["auto", "triton", "torch"])
@pytest.mark.parametrize("keep", [None, torch.tensor([0])])
def test_prefill_rejects_missing_flags_before_cache_access(backend, keep):
    """Missing prefill metadata must not preserve stale states on any backend."""
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(attention)
    attention._kda_state_copy_backend = backend
    metadata = SimpleNamespace(cu_seqlens_host=(0, 1), cu_seqlens_kern=None, keep_meta=keep)
    state = torch.full((2, 1, 2, 2), 7.0)
    with pytest.raises(ValueError, match="requires has_initial_state"):
        attention._run_prefill(None, None, None, None, None, state, torch.tensor([0]), None, metadata)
    torch.testing.assert_close(state, torch.full_like(state, 7), rtol=0, atol=0)
