# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for moe_offload unit tests.

All tests in this suite MUST pass on x86 CPU runners without torch_npu/acl:
the offload manager imports lazily (see manager.py NOTE), and the fixtures
below never touch the NPU paths.
"""

import sys

import pytest


class _BlockNpuImports:
    """Simulate a CPU-only CI runner: any acl/torch_npu import raises."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("acl", "torch_npu"):
            raise ImportError(f"blocked in CPU UT: {name}")


@pytest.fixture(scope="session", autouse=True)
def block_npu_backends():
    sys.meta_path.insert(0, _BlockNpuImports())
    yield
    sys.meta_path.remove(sys.meta_path[0])


def _assemble(m, offset, count, slots, registered=True):
    """Set up manager state without running _initialize() (which builds a
    torch.npu.Stream — unavailable on CPU runners). Mirrors the fields
    _initialize assigns, so slot/topology logic is tested verbatim."""
    from vllm_ascend.moe_offload.manager import compute_folded_topology

    m.moe_layer_offset = offset
    m.moe_layer_count = count
    m.moe_slot_count = slots
    folded = compute_folded_topology(offset, count, slots)
    m._folded_topology = folded
    m.rotating_moe_layer_count = folded.rotating_moe_layer_count
    m.resident_tail_layer = folded.resident_tail_layer
    m.layer_prefixes = [f"model.layers.{i}.mlp" for i in range(offset, offset + count)] if registered else []
    return m


@pytest.fixture
def mgr_glm():
    """GLM-5.2-like: MoE layers 1..77 (78-layer model, 1 dense head)."""
    from vllm_ascend.moe_offload.manager import MoEOffloadManager

    MoEOffloadManager._instance = None
    m = MoEOffloadManager.__new__(MoEOffloadManager)
    _assemble(m, offset=1, count=77, slots=2)
    yield m
    MoEOffloadManager._instance = None


@pytest.fixture
def mgr_ds():
    """DeepSeek-V4-Flash-like: MoE layers 0..42 (43 layers, all MoE)."""
    from vllm_ascend.moe_offload.manager import MoEOffloadManager

    MoEOffloadManager._instance = None
    m = MoEOffloadManager.__new__(MoEOffloadManager)
    _assemble(m, offset=0, count=43, slots=2)
    yield m
    MoEOffloadManager._instance = None


@pytest.fixture
def mgr_slot3():
    from vllm_ascend.moe_offload.manager import MoEOffloadManager

    MoEOffloadManager._instance = None
    m = MoEOffloadManager.__new__(MoEOffloadManager)
    _assemble(m, offset=1, count=77, slots=3)
    yield m
    MoEOffloadManager._instance = None
