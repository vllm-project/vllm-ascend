# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-owned Smoke cases used by Lite Actions."""

from tests.integration_tests import OverrideDefinitions

SMOKE_DEFAULTS = {
    "worker": "blue",
    "target_pool": "a3_shared_blue",
    "docker": "cann_master_torchtitan_master_xyj",
}

SMOKE_PROFILES = {
    "dispatch_smoke": [
        "worker_os_blue",
        # "worker_os_yellow",  # Enable when the Yellow worker is ready.
        "dispatch_smoke_8p",
        "dispatch_smoke_16p",
        "dispatch_smoke_8p",
    ],
}


def build_smoke_list() -> list[OverrideDefinitions]:
    def worker_os(worker: str) -> OverrideDefinitions:
        return OverrideDefinitions(
            test_name=f"worker_os_{worker}",
            test_descr=f"OS probe on {worker}",
            ci_kind="worker_probe",
            ci_overrides={"worker": worker},
            nnodes=0,
            ngpu=0,
            timeout=90,
            check_loss=False,
        )

    def npu_smoke(nodes: int) -> OverrideDefinitions:
        return OverrideDefinitions(
            test_name=f"dispatch_smoke_{nodes * 8}p",
            test_descr=f"Lite Actions dispatch smoke: {nodes} node(s), {nodes * 8}P HCCL/DP handshake",
            ci_kind="npu_smoke",
            nnodes=nodes,
            ngpu=8,
            timeout=75,
            env_vars={"ASCEND_SET_ENV_PATH": "/usr/local/Ascend/ascend-toolkit/set_env.sh"},
            check_loss=False,
        )

    return [
        worker_os("blue"),
        worker_os("yellow"),
        npu_smoke(1),
        npu_smoke(2),
    ]


def resolve_smoke_profile(profile: str = "dispatch_smoke") -> list[OverrideDefinitions]:
    if profile not in SMOKE_PROFILES:
        raise ValueError(f"Unknown Smoke profile: {profile}")
    definitions = build_smoke_list()
    cases = {test.test_name: test for test in definitions}
    if len(cases) != len(definitions):
        raise ValueError("duplicate Smoke test")
    return [cases[name] for name in SMOKE_PROFILES[profile]]


def main() -> None:
    import os
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401 - registers the HCCL backend

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://", timeout=timedelta(seconds=45))
    try:
        value = torch.tensor([rank], dtype=torch.int32, device=f"npu:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size - 1) // 2
        if int(value.cpu().item()) != expected_sum:
            raise RuntimeError(f"DP all_reduce mismatch: expected {expected_sum}, got {int(value.cpu().item())}")
        dist.barrier()
        if rank == 0:
            print(f"DISPATCH_SMOKE_PASS world_size={world_size} all_reduce_sum={expected_sum}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
