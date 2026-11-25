# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM-Ascend project
"""Real-NPU validation for the GroupCoordinatorPatch.destroy() HCCL path.

This is the NPU counterpart of the CPU unit test in
``tests/ut/patch/worker/patch_common/test_patch_distributed.py``.

It validates, on real Ascend hardware, that the ``destroy()`` path of
``GroupCoordinatorPatch`` behaves identically with and without the
``getattr(self, "backend", None)`` fallback introduced for issue #21966:

* On Ascend the backend is always ``hccl``, so the device process group must
  be released through the HCCL registry (``_release_hccl_resources()``), and
  the ``backend != "hccl"`` branch must not destroy it a second time.
* The gloo CPU group is destroyed directly.
* After ``destroy()`` the registry holds no dangling entry, and a fresh HCCL
  group can be created and used on the same devices.
* A coordinator built without ``__init__`` (the upstream-test shape, so no
  ``backend`` attribute) is still destroyable, and destroys its device group
  directly.

The two ranks run in spawned processes, so the test needs two NPU devices::

    pytest -sv tests/e2e/pull_request/two_card/test_group_coordinator_destroy_npu.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401  (registers the npu backend)
from vllm.utils.network_utils import get_open_port

from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch

_WORLD_SIZE = 2


def _check(condition: bool, message: str) -> None:
    rank = os.environ.get("RANK", "?")
    if condition:
        print(f"[rank{rank}] PASS: {message}", flush=True)
    else:
        print(f"[rank{rank}] FAIL: {message}", flush=True)
        raise AssertionError(message)


def _validate_real_hccl_coordinator(world_size: int, local_rank: int) -> None:
    """The normal Ascend path: backend is hccl, the registry owns the group."""
    coordinator = GroupCoordinatorPatch(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="hccl",
        use_device_communicator=False,
        group_name="npu_destroy_validation",
    )

    _check(coordinator.backend == "hccl", "coordinator backend is 'hccl' on Ascend")
    _check(coordinator.device_group is not None, "real HCCL device group was created")

    # Capture the handles before destroy() deletes the attributes.
    device_group = coordinator.device_group
    cpu_group = coordinator.cpu_group

    # The collective proves the group is genuinely functional, not just created.
    tensor = torch.ones(4, device=f"npu:{local_rank}")
    dist.all_reduce(tensor, group=device_group)
    torch.npu.synchronize()
    _check(
        bool((tensor == world_size).all().item()),
        f"all_reduce over the real HCCL group works (got {tensor[0].item()})",
    )

    _check(
        len(getattr(coordinator, "_acquired_hccl_keys", [])) > 0,
        "coordinator acquired an HCCL registry key",
    )

    # Count the destroy_process_group calls. The HCCL registry's release()
    # also calls torch.distributed.destroy_process_group, so the call count
    # alone does not separate the two paths. What does: on Ascend the device
    # group must be destroyed exactly once (by the registry), and the cpu
    # group exactly once (directly). A second direct destroy of the device
    # group would show up as a count of 2.
    destroy_calls: list[object] = []
    real_destroy = dist.destroy_process_group

    def spy_destroy(group=None, *args, **kwargs):
        destroy_calls.append(group)
        return real_destroy(group, *args, **kwargs)

    dist.destroy_process_group = spy_destroy
    try:
        coordinator.destroy()
    finally:
        dist.destroy_process_group = real_destroy

    device_group_destroy_count = sum(1 for group in destroy_calls if group is device_group)
    _check(
        device_group_destroy_count == 1,
        "hccl device group was destroyed exactly once (by the registry)",
    )
    _check(
        len(destroy_calls) == 2,
        f"destroy() destroyed exactly the device group and the cpu group (calls={len(destroy_calls)})",
    )
    _check(cpu_group in destroy_calls, "gloo cpu group was destroyed directly")

    _check(not hasattr(coordinator, "device_group"), "device_group attribute was deleted")
    _check(not hasattr(coordinator, "cpu_group"), "cpu_group attribute was deleted")
    _check(
        len(getattr(coordinator, "_acquired_hccl_keys", [])) == 0,
        "coordinator holds no HCCL registry key after destroy()",
    )

    # A fresh group must still work, i.e. destroy() left the HCCL domain
    # in a usable state.
    second = GroupCoordinatorPatch(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="hccl",
        use_device_communicator=False,
        group_name="npu_destroy_validation_second",
    )
    _check(second.device_group is not None, "a fresh HCCL group can be created after destroy()")

    tensor2 = torch.ones(4, device=f"npu:{local_rank}")
    dist.all_reduce(tensor2, group=second.device_group)
    torch.npu.synchronize()
    _check(
        bool((tensor2 == world_size).all().item()),
        "all_reduce works on the fresh group after the first destroy()",
    )
    second.destroy()


def _validate_coordinator_without_backend(world_size: int) -> None:
    """The upstream shape: no ``backend`` attribute, so destroy() must not raise."""
    orphan = GroupCoordinatorPatch.__new__(GroupCoordinatorPatch)
    _check(
        not hasattr(orphan, "backend"),
        "orphan coordinator has no 'backend' attribute (upstream shape)",
    )

    orphan_device_group = dist.new_group(list(range(world_size)), backend="hccl")
    orphan_cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    orphan.device_group = orphan_device_group
    orphan.cpu_group = orphan_cpu_group
    orphan.mq_broadcaster = None

    destroy_calls: list[object] = []
    real_destroy = dist.destroy_process_group

    def spy_destroy(group=None, *args, **kwargs):
        destroy_calls.append(group)
        return real_destroy(group, *args, **kwargs)

    dist.destroy_process_group = spy_destroy
    try:
        orphan.destroy()
    finally:
        dist.destroy_process_group = real_destroy

    _check(not hasattr(orphan, "device_group"), "orphan released its device_group attribute")
    _check(not hasattr(orphan, "cpu_group"), "orphan released its cpu_group attribute")
    _check(
        orphan_device_group in destroy_calls,
        "orphan coordinator destroyed its device group directly",
    )
    _check(
        orphan_cpu_group in destroy_calls,
        "orphan coordinator destroyed its cpu group directly",
    )


def _worker(rank: int, port: int) -> None:
    """One of the two ranks: validate the destroy path on a real HCCL group."""
    # The same environment a torchrun launch provides, so the HCCL backend
    # uses the usual rendezvous.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)

    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=_WORLD_SIZE)

    # The HCCL pg options helper reads the current vLLM config.
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        _validate_real_hccl_coordinator(_WORLD_SIZE, rank)
        _validate_coordinator_without_backend(_WORLD_SIZE)

    dist.barrier()
    dist.destroy_process_group()


def test_group_coordinator_destroy_releases_the_hccl_group() -> None:
    """The destroy path releases the HCCL group exactly once, and still works
    for a coordinator that was built without ``__init__``."""
    if torch.npu.device_count() < _WORLD_SIZE:
        pytest.skip(f"requires {_WORLD_SIZE} NPU devices")
    mp.spawn(_worker, args=(get_open_port(),), nprocs=_WORLD_SIZE, join=True)
