#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the multimodal pinned staging pool (issue #21094).

Runs on CPU: the pinned backing buffer and the accelerator event are
substituted with plain-CPU / fake equivalents, so the pool logic (slice
bookkeeping, multi-slot contiguous runs, event-gated recycling, fallback
semantics of the patched multimodal functions) is verified without an NPU.
"""

import importlib.machinery
import sys
import types

import pytest
import torch

from vllm_ascend import mm_staging_pool
from vllm_ascend.ascend_config import MMStagingConfig
from vllm_ascend.mm_staging_pool import MMPinnedStagingPool, alloc_view, get_pool

MB = 1 << 20


class FakeEvent:
    """Event double with a programmable completion state."""

    def __init__(self):
        self.done = False
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.done


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Isolate the lazy singletons and the patch marker per test."""
    monkeypatch.setattr(mm_staging_pool, "_POOL", None)
    monkeypatch.setattr(mm_staging_pool, "_POOL_FAILED", False)
    monkeypatch.setattr(mm_staging_pool, "_WARNED_NO_EVENT", False)
    from vllm_ascend.patch.platform import patch_mm_staging_pool as patch_mod

    monkeypatch.setattr(patch_mod, "_PATCH_DONE", False)


@pytest.fixture()
def cpu_backing(monkeypatch):
    """Replace the pinned backing allocation with a plain CPU buffer."""
    monkeypatch.setattr(
        MMPinnedStagingPool,
        "_allocate_backing",
        lambda self, nbytes: torch.zeros(nbytes, dtype=torch.uint8),
    )


@pytest.fixture()
def fake_events(monkeypatch):
    """Replace the accelerator event factory with FakeEvent instances."""
    events = []

    def _new_event():
        ev = FakeEvent()
        events.append(ev)
        return ev

    monkeypatch.setattr(mm_staging_pool, "_new_pinned_event", _new_event)
    return events


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_mm_staging_config_defaults_and_disable():
    cfg = MMStagingConfig()
    assert cfg.pinned_pool_mb == 2048
    assert cfg.slot_mb == 512
    disabled = MMStagingConfig(pinned_pool_mb=0, slot_mb=0)
    assert disabled.pinned_pool_mb == 0


def test_mm_staging_config_rejects_invalid():
    with pytest.raises(ValueError):
        MMStagingConfig(pinned_pool_mb=-1)
    with pytest.raises(ValueError):
        MMStagingConfig(pinned_pool_mb=1024, slot_mb=0)
    with pytest.raises(ValueError):
        MMStagingConfig(pinned_pool_mb=256, slot_mb=512)


# ---------------------------------------------------------------------------
# Pool slice bookkeeping
# ---------------------------------------------------------------------------


def test_pool_alloc_layout_and_no_overlap(cpu_backing):
    pool = MMPinnedStagingPool(4 * MB, MB)
    base = pool._buf.data_ptr()
    a = pool.alloc(MB, torch.float32)
    assert a is not None
    flat_a, ptr_a = a
    assert flat_a.shape == (MB // 4,)  # MB bytes viewed as fp32
    assert flat_a.dtype == torch.float32
    assert ptr_a == base
    b = pool.alloc(MB, torch.uint8)
    assert b is not None
    assert b[1] == base + MB  # next slot, no overlap


def test_pool_rejects_out_of_capacity_and_misaligned(cpu_backing):
    pool = MMPinnedStagingPool(4 * MB, MB)
    assert pool.alloc(5 * MB, torch.uint8) is None
    assert pool.alloc(0, torch.uint8) is None
    assert pool.alloc(3, torch.float32) is None  # not element-aligned


def test_pool_multi_slot_contiguous_run(cpu_backing, fake_events):
    pool = MMPinnedStagingPool(4 * MB, MB)
    base = pool._buf.data_ptr()
    a = pool.alloc(MB, torch.uint8)  # slot 0
    assert a is not None
    # Free slot 1 only (event completes immediately on reclaim).
    b = pool.alloc(MB, torch.uint8)  # slot 1
    assert b is not None
    pool.flush_if_active(b[1])
    fake_events[-1].done = True
    # A 2-slot request must take the contiguous run [1, 2] (slot 0 still active).
    c = pool.alloc(2 * MB, torch.uint8)
    assert c is not None
    assert c[1] == base + MB
    # 3-slot run impossible while slot 0 is still active.
    assert pool.alloc(3 * MB, torch.uint8) is None


def test_pool_event_gated_recycle(cpu_backing, fake_events):
    pool = MMPinnedStagingPool(4 * MB, MB)
    a = pool.alloc(4 * MB, torch.uint8)
    assert a is not None
    pool.flush_if_active(a[1])
    assert len(fake_events) == 1 and fake_events[0].recorded
    # Event not complete: the whole pool stays held.
    assert pool.alloc(MB, torch.uint8) is None
    # Event completes: slots come back on the next alloc attempt.
    fake_events[0].done = True
    assert pool.alloc(4 * MB, torch.uint8) is not None


def test_pool_flush_unknown_ptr_is_noop(cpu_backing, fake_events):
    pool = MMPinnedStagingPool(MB, MB)
    foreign = torch.zeros(4)
    pool.flush_if_active(foreign.data_ptr())
    assert fake_events == []
    assert pool._pending == []


def test_pool_slice_is_fully_writable(cpu_backing):
    pool = MMPinnedStagingPool(2 * MB, MB)
    a = pool.alloc(MB, torch.float32)
    assert a is not None
    flat, _ = a
    flat.fill_(1.0)
    assert torch.all(flat == 1.0)


# ---------------------------------------------------------------------------
# Lazy singleton + config resolution
# ---------------------------------------------------------------------------


def test_get_pool_disabled_by_config(cpu_backing, monkeypatch):
    monkeypatch.setattr(mm_staging_pool, "_resolve_pool_mb", lambda: (0, 512))
    assert get_pool() is None
    assert get_pool() is None  # cached disabled state, no repeated init


def test_get_pool_init_failure_falls_back(monkeypatch):
    def boom(self, nbytes):
        raise RuntimeError("pin_memory unsupported")

    monkeypatch.setattr(MMPinnedStagingPool, "_allocate_backing", boom)
    monkeypatch.setattr(mm_staging_pool, "_resolve_pool_mb", lambda: (2, 1))
    assert get_pool() is None
    assert mm_staging_pool._POOL_FAILED


def test_get_pool_creates_from_config(cpu_backing, monkeypatch):
    monkeypatch.setattr(mm_staging_pool, "_resolve_pool_mb", lambda: (2, 1))
    pool = get_pool()
    assert pool is not None
    assert pool.num_slots == 2
    assert get_pool() is pool  # singleton


def test_resolve_pool_mb_defaults_when_config_unavailable(monkeypatch):
    import vllm_ascend.ascend_config as ascend_config

    def unavailable():
        raise RuntimeError("Ascend config is not initialized")

    monkeypatch.setattr(ascend_config, "get_ascend_config", unavailable)
    assert mm_staging_pool._resolve_pool_mb() == (
        mm_staging_pool.DEFAULT_POOL_MB,
        mm_staging_pool.DEFAULT_SLOT_MB,
    )


def test_alloc_view_shape_and_release(cpu_backing, fake_events):
    monkey_pool = MMPinnedStagingPool(2 * MB, MB)
    mm_staging_pool._POOL = monkey_pool
    view = alloc_view((3, 5), torch.float32)
    assert view is not None
    assert view.shape == (3, 5)
    assert view.dtype == torch.float32
    assert monkey_pool._buf.data_ptr() <= view.data_ptr() < (monkey_pool._buf.data_ptr() + 2 * MB)
    mm_staging_pool.release_after_enqueue(view)
    assert len(fake_events) == 1
    # Non-pool tensors are released as a no-op.
    mm_staging_pool.release_after_enqueue(torch.zeros(4))
    assert len(fake_events) == 1


def test_alloc_view_rejects_empty_and_oversize(cpu_backing):
    mm_staging_pool._POOL = MMPinnedStagingPool(MB, MB)
    assert alloc_view((0,), torch.float32) is None
    assert alloc_view((2 * MB,), torch.float32) is None


# ---------------------------------------------------------------------------
# Patched multimodal functions (fail-open semantics)
# ---------------------------------------------------------------------------


def _fake_mm_inputs_module():
    """Minimal stand-in for vllm.multimodal.inputs with the required surface."""
    mod = types.ModuleType("vllm.multimodal.inputs.ut_stub")

    def json_map_leaves(fn, tree):
        if isinstance(tree, dict):
            return {k: json_map_leaves(fn, v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [json_map_leaves(fn, v) for v in tree]
        return fn(tree)

    mod.json_map_leaves = json_map_leaves
    mod._is_dense = lambda t: True
    mod.is_list_of = lambda batch, tp, check=None: isinstance(batch, list) and all(isinstance(x, tp) for x in batch)
    mod.MultiModalFlatField = type("MultiModalFlatField", (), {})
    mod.MultiModalBatchedField = type("MultiModalBatchedField", (), {})
    mod._nested_tensors_h2d = lambda *args, **kwargs: None
    return mod


@pytest.fixture()
def patched_module(cpu_backing, monkeypatch):
    from vllm_ascend.patch.platform import patch_mm_staging_pool as patch_mod

    # CPU UT: real pin_memory is unavailable, so the direct-allocation
    # fallback (new_empty/new_zeros with pin_memory=True) is relaxed to a
    # plain CPU tensor for the purpose of these tests.
    def _strip_pin(fn):
        def wrapper(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.pop("pin_memory", None)
            return fn(self, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(torch.Tensor, "new_empty", _strip_pin(torch.Tensor.new_empty))
    monkeypatch.setattr(torch.Tensor, "new_zeros", _strip_pin(torch.Tensor.new_zeros))

    mod = _fake_mm_inputs_module()
    patch_mod._patch_inputs_module(mod)
    assert getattr(mod, "_mm_staging_pool_patched", False)
    return mod, patch_mod


def test_patch_is_idempotent(patched_module):
    mod, patch_mod = patched_module
    first = mod._nested_tensors_h2d
    patch_mod._patch_inputs_module(mod)
    assert mod._nested_tensors_h2d is first


def test_patch_fails_open_on_missing_attrs(cpu_backing, caplog):
    from vllm_ascend.patch.platform import patch_mm_staging_pool as patch_mod

    mod = _fake_mm_inputs_module()
    del mod.MultiModalFlatField  # simulate vLLM version drift
    patch_mod._patch_inputs_module(mod)
    assert not getattr(mod, "_mm_staging_pool_patched", False)


def test_patched_h2d_stages_via_pool(patched_module, monkeypatch):
    mod, patch_mod = patched_module
    staged = []
    released = []

    def fake_alloc_view(shape, dtype):
        t = torch.zeros(tuple(shape), dtype=dtype)
        staged.append(t)
        return t

    monkeypatch.setattr(patch_mod, "alloc_view", fake_alloc_view)
    monkeypatch.setattr(patch_mod, "release_after_enqueue", lambda t: released.append(t.data_ptr()))

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert mod._nested_tensors_h2d(x, None, pin_memory=True) is x  # device=None passthrough
    out = mod._nested_tensors_h2d(x, "cpu", pin_memory=True)
    assert len(staged) == 1
    assert torch.equal(out, x)
    assert staged[0].data_ptr() in released  # recycled after enqueue
    assert released[-1] == staged[0].data_ptr()


def test_patched_h2d_falls_back_when_pool_says_no(patched_module, monkeypatch):
    mod, patch_mod = patched_module
    monkeypatch.setattr(patch_mod, "alloc_view", lambda shape, dtype: None)
    released = []
    monkeypatch.setattr(patch_mod, "release_after_enqueue", lambda t: released.append(t.data_ptr()))

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    original_ptr = x.data_ptr()
    out = mod._nested_tensors_h2d(x, "cpu", pin_memory=True)
    assert torch.equal(out, x)
    assert len(released) == 1
    assert released[0] != original_ptr  # the fallback tensor is a fresh allocation


def test_patched_flat_reduce_concat_uses_pool(patched_module, monkeypatch):
    mod, patch_mod = patched_module
    seen_shapes = []

    def fake_alloc_view(shape, dtype):
        seen_shapes.append(tuple(shape))
        return torch.full(tuple(shape), -1, dtype=dtype)

    monkeypatch.setattr(patch_mod, "alloc_view", fake_alloc_view)
    batch = [torch.ones(2, 3), torch.ones(2, 3) * 2]
    field = mod.MultiModalFlatField()
    field.dim = 0
    out = field._reduce_data(batch, pin_memory=True)
    assert seen_shapes == [(4, 3)]  # (*before, concat, *after) requested
    assert torch.equal(out, torch.cat(batch, dim=0))
    assert torch.all(out != -1)  # pool memory fully overwritten

    # pin_memory=False never touches the pool.
    seen_shapes.clear()
    field._reduce_data(batch, pin_memory=False)
    assert seen_shapes == []


def test_patched_flat_reduce_falls_back(patched_module, monkeypatch):
    mod, patch_mod = patched_module
    monkeypatch.setattr(patch_mod, "alloc_view", lambda shape, dtype: None)
    batch = [torch.ones(2, 3), torch.ones(2, 3) * 2]
    field = mod.MultiModalFlatField()
    field.dim = 0
    out = field._reduce_data(batch, pin_memory=True)
    assert torch.equal(out, torch.cat(batch, dim=0))


def test_patched_flat_reduce_variable_len_zeroes_pool_slice(patched_module, monkeypatch):
    mod, patch_mod = patched_module

    def fake_alloc_view(shape, dtype):
        return torch.full(tuple(shape), -1, dtype=dtype)  # stale bytes

    monkeypatch.setattr(patch_mod, "alloc_view", fake_alloc_view)
    batch = [torch.ones(1, 2, 3), torch.ones(1, 2, 4) * 2]
    field = mod.MultiModalFlatField()
    field.dim = 1
    out = field._reduce_data(batch, pin_memory=True)
    expected = torch.zeros(1, 4, 4)
    expected[:, :2, :3] = batch[0]
    expected[:, 2:4, :] = batch[1]
    assert torch.equal(out, expected)  # zero_() preserved torch.zeros semantics


def test_patched_batched_reduce_stack_uses_pool(patched_module, monkeypatch):
    mod, patch_mod = patched_module
    seen_shapes = []

    def fake_alloc_view(shape, dtype):
        seen_shapes.append(tuple(shape))
        return torch.full(tuple(shape), -1, dtype=dtype)

    monkeypatch.setattr(patch_mod, "alloc_view", fake_alloc_view)
    batch = [torch.ones(2, 3), torch.ones(2, 3) * 2]
    field = mod.MultiModalBatchedField()
    out = field._reduce_data(batch, pin_memory=True)
    assert seen_shapes == [(2, 2, 3)]
    assert torch.equal(out, torch.stack(batch))

    monkeypatch.setattr(patch_mod, "alloc_view", lambda shape, dtype: None)
    out = field._reduce_data(batch, pin_memory=True)
    assert torch.equal(out, torch.stack(batch))


# ---------------------------------------------------------------------------
# Import hook (meta_path finder / loader wrapper)
# ---------------------------------------------------------------------------


def test_import_hook_delegates_and_wraps_loader(monkeypatch):
    from vllm_ascend.patch.platform.patch_mm_staging_pool import (
        _MMInputsPatchFinder,
        _MMInputsPatchLoader,
    )

    class StubLoader:
        def create_module(self, spec):
            return _fake_mm_inputs_module()

        def exec_module(self, module):
            pass  # attributes already present on the stub module

    class StubFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.multimodal.inputs":
                return None
            return importlib.machinery.ModuleSpec(fullname, StubLoader())

    monkeypatch.delitem(sys.modules, "vllm.multimodal.inputs", raising=False)
    finder = _MMInputsPatchFinder()
    assert finder.find_spec("vllm.multimodal.cache") is None

    monkeypatch.setattr(sys, "meta_path", [finder, StubFinder()])
    spec = finder.find_spec("vllm.multimodal.inputs")
    assert spec is not None
    assert isinstance(spec.loader, _MMInputsPatchLoader)

    module = spec.loader.create_module(spec)
    spec.loader.exec_module(module)
    assert getattr(module, "_mm_staging_pool_patched", False)
    # The patched h2d comes from the patch module, not the stub original.
    assert module._nested_tensors_h2d.__qualname__ != "StubLoader"


def test_import_hook_skips_when_already_imported(monkeypatch):
    from vllm_ascend.patch.platform.patch_mm_staging_pool import _MMInputsPatchFinder

    monkeypatch.setitem(sys.modules, "vllm.multimodal.inputs", object())
    finder = _MMInputsPatchFinder()
    assert finder.find_spec("vllm.multimodal.inputs") is None
