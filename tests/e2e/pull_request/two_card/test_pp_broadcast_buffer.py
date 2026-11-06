# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual HCCL PP broadcast group without loading a model."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401


def _broadcast_worker(rank: int, init_method: str) -> None:
    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch

    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    coordinator = None
    try:
        coordinator = GroupCoordinatorPatch(
            group_ranks=[[0, 1]],
            local_rank=rank,
            torch_distributed_backend="hccl",
            use_device_communicator=False,
            group_name="pp",
        )
        tokens = torch.full((8,), rank, dtype=torch.int64, device=f"npu:{rank}")
        torch.npu.synchronize()
        free_before = torch.npu.mem_get_info()[0]

        sibling = coordinator.make_sibling_device_group(group_desc="pp_broadcast")
        assert sibling is not coordinator.device_group
        dist.broadcast(tokens, src=1, group=sibling)
        torch.npu.synchronize()
        free_after = torch.npu.mem_get_info()[0]

        assert torch.equal(tokens.cpu(), torch.ones(8, dtype=torch.int64))
        # Allow runtime overhead above the normal 400 MiB CCL allocation, but
        # reject inheriting the global 4096 MiB setting (about 8 GiB per rank).
        max_group_memory_bytes = 2 * (1 << 30)
        assert 0 < free_before - free_after < max_group_memory_bytes
    finally:
        if coordinator is not None:
            coordinator.destroy()
        dist.destroy_process_group()


def test_pp_broadcast_does_not_inherit_large_global_buffer(monkeypatch, tmp_path):
    monkeypatch.setenv("HCCL_BUFFSIZE", "4096")
    init_method = (tmp_path / "pp-broadcast-store").as_uri()
    mp.spawn(_broadcast_worker, args=(init_method,), nprocs=2, join=True)
