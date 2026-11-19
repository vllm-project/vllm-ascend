# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPUModelRunner._check_repair_kv_layout_capability gates the repair policy.

Under kv_load_failure_policy='repair' the worker-side repair executor derives
the manager block size of layer 0 from the live KV cache geometry and refuses
every repair unless it reproduces the scheduler's block size. On the default
Ascend BNSD layout ([num_blocks, block_size, heads, dim] per K/V view) the
derivation reads the head count, so every repair would fail the fence at
repair time. The gate moves that refusal to engine init and names the
[B, H, N, C] layout switch (VLLM_KV_CACHE_LAYOUT=HND / LBHNC).
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_BLOCK_SIZE = 128
_REPAIR_EXECUTOR = "vllm.v1.worker.gpu.repair_executor"


def _make_runner(policy: str = "repair", block_size: int = _BLOCK_SIZE) -> NPUModelRunner:
    runner = object.__new__(NPUModelRunner)
    kv_transfer_config = None if policy is None else SimpleNamespace(kv_load_failure_policy=policy)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=SimpleNamespace(block_size=block_size),
    )
    runner.model = SimpleNamespace(model=SimpleNamespace(layers=[object()]))
    return runner


def _inject_repair_executor(monkeypatch, derived=None, raises=None):
    module = ModuleType("repair_executor")
    if raises is not None:

        def _raise(layer):
            raise raises

        module._repair_block_size = _raise
    else:
        module._repair_block_size = lambda layer: derived
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, module)


def test_gate_skipped_without_kv_transfer_config(monkeypatch):
    """A missing kv_transfer_config cannot select the repair policy."""
    _inject_repair_executor(monkeypatch)
    _make_runner(None)._check_repair_kv_layout_capability()


def test_gate_skipped_for_fail_and_recompute(monkeypatch):
    """'fail' and 'recompute' keep booting without consulting the fence."""
    _inject_repair_executor(monkeypatch, raises=AssertionError("the repair executor must not be consulted"))
    for policy in ("fail", "recompute"):
        _make_runner(policy)._check_repair_kv_layout_capability()


def test_gate_noop_when_repair_executor_absent(monkeypatch):
    """Stock vLLM builds carry no repair executor; the gate is a no-op."""
    monkeypatch.setitem(sys.modules, _REPAIR_EXECUTOR, None)
    _make_runner("repair")._check_repair_kv_layout_capability()


def test_gate_passes_on_hnd_geometry(monkeypatch):
    """[B, H, N, C] per-K/V views derive the manager block size exactly."""
    _inject_repair_executor(monkeypatch, derived=_BLOCK_SIZE)
    _make_runner("repair")._check_repair_kv_layout_capability()


def test_gate_refuses_default_bnsd_geometry(monkeypatch):
    """The default layout derives num_kv_heads; every repair would fail.
    The refusal must name the layout switch."""
    _inject_repair_executor(monkeypatch, derived=8)
    with pytest.raises(RuntimeError, match="VLLM_KV_CACHE_LAYOUT=HND") as excinfo:
        _make_runner("repair")._check_repair_kv_layout_capability()
    assert "block size of 8" in str(excinfo.value)
    assert f"blocks of {_BLOCK_SIZE}" in str(excinfo.value)


def test_gate_refuses_unrecognised_geometry(monkeypatch):
    """A geometry the fence cannot parse is refused with the same guidance."""
    _inject_repair_executor(monkeypatch, raises=ValueError("unrecognized repair KV cache geometry"))
    with pytest.raises(RuntimeError, match="VLLM_KV_CACHE_LAYOUT=HND") as excinfo:
        _make_runner("repair")._check_repair_kv_layout_capability()
    assert "unrecognized repair KV cache geometry" in str(excinfo.value)
