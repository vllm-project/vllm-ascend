# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPUModelRunner._check_repair_kv_layout_capability gates the repair policy.

Under kv_load_failure_policy='repair' the gate runs three fail-closed
checks at engine init: the explicit head-major (LBHNC/HND) layout
refusal, the binding-contract verification (this runner's own
_repair_view_binding validated against the fence layer's actual bound
views, through the SAME executor API the repair-time fence uses), and
the legacy fallback that keeps the check active on pre-binding vLLM
bases. A repair stack exposing NEITHER API is a LOUD warning, never a
silent pass -- a missing helper must never turn into successful
admission.

The gate locates the layer through the repair executor's
find_repair_fence_layer (imported from vllm.v1.worker.gpu.repair_executor
behind the same ImportError guard), which accepts the model zoo's
decoder-stack names (model.layers, transformer.encoder.layers,
transformer.h, transformer.blocks, and the language_model multimodal
wrappers) instead of assuming model.model.layers, and refuses clearly when
no decoder layer with the fence's self_attn.attn seam exists.

Stock vLLM CI images carry no repair executor, so the executor module is
injected as a fake exposing the binding-contract API (and, for the
legacy-fallback tests, only the old shape-based derivation).
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_BLOCK_SIZE = 128
_REPAIR_EXECUTOR = "vllm.v1.worker.gpu.repair_executor"
# The container walk mirrors vllm.v1.worker.gpu.repair_executor
# (find_repair_fence_layer); see _mirror_find_repair_fence_layer below.
_CONTAINERS = (
    ("model", "layers"),
    ("transformer", "encoder", "layers"),
    ("transformer", "h"),
    ("transformer", "blocks"),
    ("language_model", "model", "layers"),
    ("language_model", "transformer", "encoder", "layers"),
    ("layers",),
)


class _FakeAttn:
    """The fence seam plus a bound per-K/V view pair (flat form)."""

    def __init__(self):
        self.kv_cache = [
            torch.zeros(4, _BLOCK_SIZE),
            torch.zeros(4, _BLOCK_SIZE),
        ]


class _FakeDecoderLayer:
    """Carries the repair fence's cache seam: self_attn.attn."""

    def __init__(self):
        self.self_attn = SimpleNamespace(attn=_FakeAttn())


def _stack(layers):
    return SimpleNamespace(layers=layers)


def _make_runner(model=None, policy="repair", block_size=_BLOCK_SIZE, kv_cache_layout="LBHNC") -> NPUModelRunner:
    runner = object.__new__(NPUModelRunner)
    kv = None if policy is None else SimpleNamespace(kv_load_failure_policy=policy)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=kv,
        cache_config=SimpleNamespace(block_size=block_size, kv_cache_layout=kv_cache_layout),
    )
    # _repair_view_binding reads the runner's resolved kv cache config
    # (the gate runs at the end of initialize_kv_cache).
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(block_size=block_size)])
    if model is None:
        model = SimpleNamespace(model=_stack([_FakeDecoderLayer()]))
    runner.model = model
    return runner


def _mirror_find_repair_fence_layer(model):
    """Mirrors Core's find_repair_fence_layer for stock vLLM CI images,
    which carry no repair executor to import: same containers in the same
    order, same self_attn.attn seam check, same return shapes."""
    for path in _CONTAINERS:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if not node:
            continue
        for layer in node:
            if getattr(getattr(layer, "self_attn", None), "attn", None) is not None:
                return layer, ".".join(path)
    return None, _CONTAINERS


def _core_fence_resolver():
    """The resolver the gate imports in production: the real Core function
    when the repair stack is installed (the hm vLLM builds the gate
    targets), and the mirror above on stock vLLM images."""
    try:
        from vllm.v1.worker.gpu.repair_executor import (  # type: ignore[import-not-found]
            find_repair_fence_layer,
        )
    except ImportError:
        return _mirror_find_repair_fence_layer
    return find_repair_fence_layer


def _binding(block_size=_BLOCK_SIZE, token_axis=2, layout_name="LBHNC"):
    return SimpleNamespace(
        kind="per_kv",
        block_size=block_size,
        token_axis=token_axis,
        blocks_axis=0,
        layout_name=layout_name,
    )


def _inject_repair_executor(monkeypatch, derived=None, geometry_raises=None):
    """The binding-contract executor module (the API the gate prefers).

    With `derived`/`geometry_raises` the module ALSO exposes the legacy
    shape-based derivation (delete the binding-contract attrs to force the
    reverse pair: a vLLM base without #68377).
    """
    module = ModuleType("repair_executor")
    module.find_repair_fence_layer = _core_fence_resolver()  # type: ignore[attr-defined]
    module.per_kv_repair_binding = (  # type: ignore[attr-defined]
        lambda block_size, token_axis, layout_name=None: _binding(block_size, token_axis, layout_name)
    )

    def _views(layer):
        cache = layer.self_attn.attn.kv_cache
        return list(cache) if isinstance(cache, (list, tuple)) else [cache]

    module._repair_cache_views = _views  # type: ignore[attr-defined]

    def _geometry(binding, views):
        # Mirror the real contract's headline refusal: the binding's
        # block size must be available, and the views must carry it.
        if binding.block_size is None or binding.block_size <= 0:
            raise ValueError("binding's block size is not available")
        if geometry_raises is not None:
            raise geometry_raises
        return SimpleNamespace(
            block_size=binding.block_size,
            token_axis=binding.token_axis,
            is_core_shared=False,
        )

    module._repair_geometry_from_binding = _geometry  # type: ignore[attr-defined]
    module._verify_repair_layer_geometry = lambda layer, geo, binding: None  # type: ignore[attr-defined]
    if geometry_raises is not None:

        def _legacy_block_size(layer):
            raise geometry_raises

        module._repair_block_size = _legacy_block_size  # type: ignore[attr-defined]
    elif derived is not None:
        module._repair_block_size = lambda layer: derived  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, module)
    return module


def test_gate_skipped_without_kv_transfer_config(monkeypatch):
    """A missing kv_transfer_config cannot select the repair policy."""
    _inject_repair_executor(monkeypatch)
    _make_runner(policy=None)._check_repair_kv_layout_capability()


def test_gate_skipped_for_fail_and_recompute(monkeypatch):
    """'fail' and 'recompute' keep booting without consulting the fence."""
    _inject_repair_executor(monkeypatch, geometry_raises=AssertionError("must not be consulted"))
    for policy in ("fail", "recompute"):
        _make_runner(policy=policy)._check_repair_kv_layout_capability()


def test_gate_refuses_lbnhc_layout_explicitly(monkeypatch):
    """The policy-level refusal: repair is validated for head-major
    layouts only, regardless of the geometry the views would admit."""
    _inject_repair_executor(monkeypatch)
    with pytest.raises(RuntimeError, match="head-major KV cache layouts") as excinfo:
        _make_runner(kv_cache_layout="LBNHC")._check_repair_kv_layout_capability()
    assert "HND or" in str(excinfo.value)


def test_gate_refuses_unresolved_layout_with_lbnhc_env(monkeypatch):
    """Pre-resolution the gate falls back to the env and applies the same
    policy-level refusal."""
    _inject_repair_executor(monkeypatch)
    with pytest.raises(RuntimeError, match="head-major KV cache layouts"), pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_KV_CACHE_LAYOUT", "LBNHC")
        _make_runner(kv_cache_layout=None)._check_repair_kv_layout_capability()


def test_gate_noop_when_repair_executor_absent(monkeypatch):
    """Stock vLLM builds carry no repair executor; the gate warns and
    stays dormant (the explicit layout check still applied above)."""
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, None)
    _make_runner()._check_repair_kv_layout_capability()


def test_gate_passes_on_hnd_binding_contract(monkeypatch):
    """The per-K/V binding contract validates against the fence layer's
    views: the gate passes without consulting any shape-derived guess."""
    _inject_repair_executor(monkeypatch)
    _make_runner()._check_repair_kv_layout_capability()


def test_gate_refuses_binding_view_contradiction(monkeypatch):
    """A fence layer whose bound views contradict the binding (the
    binding-contract verification refuses) fails the init gate with the
    same guidance as the repair-time fence."""
    _inject_repair_executor(
        monkeypatch,
        geometry_raises=ValueError("the views contradict the per-K/V binding contract"),
    )
    with pytest.raises(RuntimeError, match="VLLM_KV_CACHE_LAYOUT=HND") as excinfo:
        _make_runner()._check_repair_kv_layout_capability()
    assert "views contradict the per-K/V binding contract" in str(excinfo.value)


def test_gate_legacy_fallback_refuses_mismatched_block_size(monkeypatch):
    """The reverse pair -- a vLLM base without the binding contract: the
    legacy shape-based derivation keeps the check active and refuses the
    default BNSD geometry (it derives num_kv_heads)."""
    module = _inject_repair_executor(monkeypatch, derived=8)
    # Simulate the pre-#68377 module: no binding-contract API.
    for name in (
        "_repair_cache_views",
        "_repair_geometry_from_binding",
        "_verify_repair_layer_geometry",
        "per_kv_repair_binding",
    ):
        delattr(module, name)
    with pytest.raises(RuntimeError, match="VLLM_KV_CACHE_LAYOUT=HND"):
        _make_runner()._check_repair_kv_layout_capability()


def test_gate_legacy_fallback_passes_on_hnd_geometry(monkeypatch):
    """The legacy derivation that reproduces the manager block size
    admits the repair policy on the old API."""
    module = _inject_repair_executor(monkeypatch, derived=_BLOCK_SIZE)
    for name in (
        "_repair_cache_views",
        "_repair_geometry_from_binding",
        "_verify_repair_layer_geometry",
        "per_kv_repair_binding",
    ):
        delattr(module, name)
    _make_runner()._check_repair_kv_layout_capability()


def test_gate_loud_when_neither_api_exists(monkeypatch):
    """A repair executor with neither the binding contract nor the
    legacy derivation must warn, never silently pass the check."""
    module = ModuleType("repair_executor")
    module.find_repair_fence_layer = _core_fence_resolver()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, module)
    _make_runner()._check_repair_kv_layout_capability()


@pytest.mark.parametrize(
    "model,expected_path",
    [
        (SimpleNamespace(model=_stack([_FakeDecoderLayer()])), "model.layers"),
        (
            SimpleNamespace(transformer=SimpleNamespace(encoder=_stack([_FakeDecoderLayer()]))),
            "transformer.encoder.layers",
        ),
        (
            SimpleNamespace(transformer=SimpleNamespace(h=[_FakeDecoderLayer()])),
            "transformer.h",
        ),
        (
            SimpleNamespace(transformer=SimpleNamespace(blocks=[_FakeDecoderLayer()])),
            "transformer.blocks",
        ),
        (
            SimpleNamespace(language_model=SimpleNamespace(model=_stack([_FakeDecoderLayer()]))),
            "language_model.model.layers",
        ),
    ],
)
def test_discovery_accepts_the_model_zoo_container_names(monkeypatch, model, expected_path):
    """The fence layer is found across decoder-stack naming conventions."""
    _inject_repair_executor(monkeypatch, geometry_raises=ValueError("boom"))
    with pytest.raises(RuntimeError) as excinfo:
        _make_runner(model=model)._check_repair_kv_layout_capability()
    assert expected_path in str(excinfo.value)


def test_discovery_skips_stacks_without_the_fence_seam(monkeypatch):
    """Entries without the self_attn.attn seam are never chosen, in any stack."""
    _inject_repair_executor(monkeypatch)
    model = SimpleNamespace(
        transformer=SimpleNamespace(h=[SimpleNamespace(self_attn=SimpleNamespace(attn=None))]),
        model=_stack([_FakeDecoderLayer()]),
    )
    _make_runner(model=model)._check_repair_kv_layout_capability()


def test_discovery_refuses_when_no_decoder_layer_exists(monkeypatch):
    """No seam anywhere: a clear, actionable refusal - never a wrong block size."""
    _inject_repair_executor(monkeypatch)
    with pytest.raises(RuntimeError, match="self_attn") as excinfo:
        _make_runner(
            model=SimpleNamespace(transformer=SimpleNamespace(encoder=_stack([])))
        )._check_repair_kv_layout_capability()
    assert "unsupported" in str(excinfo.value)
    assert "'fail' or 'recompute'" in str(excinfo.value)
