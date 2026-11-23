# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPUModelRunner._check_repair_kv_layout_capability gates the repair policy.

Under kv_load_failure_policy='repair' the worker-side repair executor derives
the manager block size of the first decoder layer from the live KV cache
geometry and refuses every repair unless it reproduces the scheduler's block
size. On the default Ascend BNSD layout ([num_blocks, block_size, heads,
dim] per K/V view) the derivation reads the head count, so every repair
would fail the fence at repair time. The gate moves that refusal to engine
init and names the [B, H, N, C] layout switch (VLLM_KV_CACHE_LAYOUT=HND /
LBHNC).

The gate locates the layer through the repair executor's
find_repair_fence_layer (imported from vllm.v1.worker.gpu.repair_executor
behind the same ImportError guard), which accepts the model zoo's
decoder-stack names (model.layers, transformer.encoder.layers,
transformer.h, transformer.blocks, and the language_model multimodal
wrappers) instead of assuming model.model.layers, and refuses clearly when
no decoder layer with the fence's self_attn.attn seam exists.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

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
    pass


class _FakeDecoderLayer:
    """Carries the repair fence's cache seam: self_attn.attn."""

    def __init__(self):
        self.self_attn = SimpleNamespace(attn=_FakeAttn())


def _stack(layers):
    return SimpleNamespace(layers=layers)


def _make_runner(model=None, policy="repair", block_size=_BLOCK_SIZE) -> NPUModelRunner:
    runner = object.__new__(NPUModelRunner)
    kv = None if policy is None else SimpleNamespace(kv_load_failure_policy=policy)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=kv,
        cache_config=SimpleNamespace(block_size=block_size),
    )
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


def _inject_repair_executor(monkeypatch, derived=None, raises=None):
    module = ModuleType("repair_executor")
    if raises is not None:

        def _raise(layer):
            raise raises

        module._repair_block_size = _raise  # type: ignore[attr-defined]
    else:
        module._repair_block_size = lambda layer: derived  # type: ignore[attr-defined]
    module.find_repair_fence_layer = _core_fence_resolver()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, module)


def test_gate_skipped_without_kv_transfer_config(monkeypatch):
    """A missing kv_transfer_config cannot select the repair policy."""
    _inject_repair_executor(monkeypatch)
    _make_runner(policy=None)._check_repair_kv_layout_capability()


def test_gate_skipped_for_fail_and_recompute(monkeypatch):
    """'fail' and 'recompute' keep booting without consulting the fence."""
    _inject_repair_executor(monkeypatch, raises=AssertionError("must not be consulted"))
    for policy in ("fail", "recompute"):
        _make_runner(policy=policy)._check_repair_kv_layout_capability()


def test_gate_noop_when_repair_executor_absent(monkeypatch):
    """Stock vLLM builds carry no repair executor; the gate is a no-op."""
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, None)
    _make_runner()._check_repair_kv_layout_capability()


def test_gate_passes_on_hnd_geometry(monkeypatch):
    """[B, H, N, C] per-K/V views derive the manager block size exactly."""
    _inject_repair_executor(monkeypatch, derived=_BLOCK_SIZE)
    _make_runner()._check_repair_kv_layout_capability()


def test_gate_refuses_default_bnsd_geometry(monkeypatch):
    """The default layout derives num_kv_heads; every repair would fail.
    The refusal names the layout switch and the discovered layer path."""
    _inject_repair_executor(monkeypatch, derived=8)
    with pytest.raises(RuntimeError, match="VLLM_KV_CACHE_LAYOUT=HND") as excinfo:
        _make_runner()._check_repair_kv_layout_capability()
    assert "block size of 8" in str(excinfo.value)
    assert "model.layers" in str(excinfo.value)


def test_gate_refuses_unrecognised_geometry(monkeypatch):
    """A geometry the fence cannot parse is refused with the same guidance."""
    _inject_repair_executor(monkeypatch, raises=ValueError("unrecognized repair KV cache geometry"))
    with pytest.raises(RuntimeError, match="VLLM_KV_CACHE_LAYOUT=HND") as excinfo:
        _make_runner()._check_repair_kv_layout_capability()
    assert "unrecognized repair KV cache geometry" in str(excinfo.value)


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
    _inject_repair_executor(monkeypatch, derived=8)
    with pytest.raises(RuntimeError) as excinfo:
        _make_runner(model=model)._check_repair_kv_layout_capability()
    assert expected_path in str(excinfo.value)


def test_discovery_skips_stacks_without_the_fence_seam(monkeypatch):
    """Entries without the self_attn.attn seam are never chosen, in any stack."""
    _inject_repair_executor(monkeypatch, derived=_BLOCK_SIZE)
    model = SimpleNamespace(
        transformer=SimpleNamespace(h=[SimpleNamespace(self_attn=SimpleNamespace(attn=None))]),
        model=_stack([_FakeDecoderLayer()]),
    )
    _make_runner(model=model)._check_repair_kv_layout_capability()


def test_discovery_refuses_when_no_decoder_layer_exists(monkeypatch):
    """No seam anywhere: a clear, actionable refusal - never a wrong block size."""
    _inject_repair_executor(monkeypatch, derived=_BLOCK_SIZE)
    with pytest.raises(RuntimeError, match="self_attn") as excinfo:
        _make_runner(
            model=SimpleNamespace(transformer=SimpleNamespace(encoder=_stack([])))
        )._check_repair_kv_layout_capability()
    assert "unsupported" in str(excinfo.value)
    assert "'fail' or 'recompute'" in str(excinfo.value)
