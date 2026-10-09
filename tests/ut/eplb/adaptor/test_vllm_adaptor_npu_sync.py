# SPDX-License-Identifier: Apache-2.0
"""NPU sync-behavior regression tests for the EPLB D2D expert update path.

The update path runs between forward passes while the compute stream still
holds the whole enqueued forward. Any host-synchronizing op there (a copy_
with the default non_blocking=False from a pageable CPU tensor, a .cpu(),
a device-tensor .item()) waits for the stream to drain and stalls the
pipeline once per layer update.

Each test enqueues a calibrated busy chain on the stream, then times the
call under test on the host: a synchronous call waits ~BUSY_MS, an
asynchronous one must return well below it. Data correctness is asserted
after a final synchronize so the asynchronous copies are required to land.
"""

import time

import numpy
import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU backend before torch.npu use)

import vllm_ascend.eplb.core.eplb_device_transfer_loader as loader_mod
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor

BUSY_MS = 150.0
ASYNC_BUDGET_MS = 40.0


def _npu_available() -> bool:
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


requires_npu = pytest.mark.skipif(not _npu_available(), reason="requires a real Ascend NPU")


def _enqueue_busy_stream(min_busy_ms: float) -> float:
    """Enqueue a matmul chain worth >= min_busy_ms of device time and return
    the estimated device-busy duration in ms. The caller must keep the NPU
    busy until after it measured the probe (sync at the end of the test)."""
    a = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    a @ b
    torch.npu.synchronize()
    one_ms = max((time.perf_counter() - t0) * 1000.0, 0.05)
    reps = max(1, int(min_busy_ms / one_ms))
    out = a
    for _ in range(reps):
        out = out @ b
    return reps * one_ms


def _make_adaptor(num_layers=2, num_logical=8192, num_local_experts=4):
    """Minimal VllmEplbAdaptor carrying only the state the D2D update path
    touches, with the same tensor placement/dtypes as production: log2phy
    maps live on device as int32, expert maps on CPU, expert weights and
    transfer buffers are device tensors of matching shape. The map length is
    padded well beyond a real layer's expert count so torch_npu's pageable
    H2D staging cost is large enough to observe a synchronous copy."""
    adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
    adaptor.num_moe_layers = num_layers
    adaptor.log2phy_map_per_layer = {i: torch.zeros(num_logical, dtype=torch.int32).npu() for i in range(num_layers)}
    adaptor.expert_map_per_layer_cpu = {i: torch.zeros(num_logical, dtype=torch.int32) for i in range(num_layers)}
    adaptor.expert_weight_key_per_layer = {i: "w13_w2" for i in range(num_layers)}
    adaptor.expert_param_per_layer = {}
    adaptor.buffer_tensor_list = {"w13_w2": []}
    for i in range(num_layers):
        adaptor.expert_param_per_layer[i] = []
        for _ in range(num_local_experts):
            # per-expert weight tensor list, mixed dtypes like the quantized weights
            adaptor.expert_param_per_layer[i].append(
                [torch.full((32, 16), 3.0, dtype=torch.float32).npu(), torch.full((16,), 2.0).npu()]
            )
    for buffer_id in range(num_local_experts):
        adaptor.buffer_tensor_list["w13_w2"].append([torch.full((32, 16), 7.0).npu(), torch.full((16,), 5.0).npu()])
    # warm up the host pinned allocator so the first probe is not charged
    # its one-time initialization
    torch.empty(1, pin_memory=True)
    return adaptor


def _make_loader(adaptor, layer_id=0, recv_experts=(0, 1)):
    from unittest.mock import patch

    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        ldr = loader_mod.D2DExpertWeightLoader()
    ldr.state = loader_mod.ExpertWeightUpdateState.TRANSFERRING
    ldr.comm_op_list = ["enqueued"]
    ldr.layer_id = layer_id
    ldr.recv_expert_list = [(e, e) for e in recv_experts]
    ldr.updated_expert_map = torch.arange(num_logical_of(adaptor), dtype=torch.int64)
    ldr.updated_log2phy_map = torch.arange(num_logical_of(adaptor), dtype=torch.int64)
    ldr.eplb_adaptor = adaptor
    return ldr


def num_logical_of(adaptor):
    return adaptor.log2phy_map_per_layer[0].shape[0]


@requires_npu
class TestEplbUpdatePathAsync:
    def test_do_update_log2phy_map_returns_without_draining_stream(self):
        adaptor = _make_adaptor()
        # the updator hands over a pageable CPU int64 tensor (torch.from_numpy)
        updated = torch.from_numpy(numpy.arange(num_logical_of(adaptor), dtype=numpy.int64))
        busy_ms = _enqueue_busy_stream(BUSY_MS)

        t0 = time.perf_counter()
        adaptor.do_update_log2phy_map(0, updated)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        torch.npu.synchronize()
        assert elapsed_ms < min(ASYNC_BUDGET_MS, 0.4 * busy_ms), (
            f"do_update_log2phy_map blocked the host for {elapsed_ms:.1f}ms while the "
            f"stream held ~{busy_ms:.1f}ms of work: the H2D map commit is synchronous"
        )
        assert torch.equal(adaptor.log2phy_map_per_layer[0].cpu(), updated.to(torch.int32).cpu())

    def test_do_update_expert_weight_returns_without_draining_stream(self):
        adaptor = _make_adaptor()
        busy_ms = _enqueue_busy_stream(BUSY_MS)

        t0 = time.perf_counter()
        adaptor.do_update_expert_weight(0, 0, 0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        torch.npu.synchronize()
        assert elapsed_ms < min(ASYNC_BUDGET_MS, 0.4 * busy_ms), (
            f"do_update_expert_weight blocked the host for {elapsed_ms:.1f}ms"
        )
        for weight, buffer in zip(adaptor.expert_param_per_layer[0][0], adaptor.buffer_tensor_list["w13_w2"][0]):
            assert torch.equal(weight.cpu(), buffer.cpu())

    def test_update_expert_map_and_weight_end_to_end_async(self):
        """Audit the whole per-layer update: nothing inside may wait on the
        stream, otherwise the update drains the enqueued forward per layer."""
        adaptor = _make_adaptor()
        ldr = _make_loader(adaptor)
        updated_expert_map = ldr.updated_expert_map
        updated_log2phy_map = ldr.updated_log2phy_map
        busy_ms = _enqueue_busy_stream(BUSY_MS)

        t0 = time.perf_counter()
        ldr.update_expert_map_and_weight()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        torch.npu.synchronize()
        assert elapsed_ms < min(ASYNC_BUDGET_MS, 0.4 * busy_ms), (
            f"update_expert_map_and_weight blocked the host for {elapsed_ms:.1f}ms while "
            f"the stream held ~{busy_ms:.1f}ms of work"
        )
        assert ldr.state == loader_mod.ExpertWeightUpdateState.WAITING
        assert torch.equal(adaptor.log2phy_map_per_layer[0].cpu(), updated_log2phy_map.to(torch.int32).cpu())
        assert torch.equal(adaptor.expert_map_per_layer_cpu[0], updated_expert_map.to(torch.int32))
