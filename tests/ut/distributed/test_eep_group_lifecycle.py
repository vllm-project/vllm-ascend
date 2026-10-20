# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.
"""N1 qualification: EEP standby-group lifecycle on Ascend NPU (2 -> 4).

Drives vLLM 0.26's REAL elastic-EP group machinery end to end on one node,
mirroring the production scale-up flow (ElasticEPScalingExecutor /
standby_state):

  old workers (ranks 0,1)                  new workers (ranks 2,3)
  ---------------------------------------  --------------------------------
  boot on coord store A (dp=2):            boot on coord store B (dp=4):
    init_distributed_environment             init_distributed_environment
    initialize_model_parallel                initialize_model_parallel
    (local torch world ws=1 + hccl)          (blocks on "dp_0"/"ep_0" fetch
                                              until old rank 0 publishes)
  sanity collective on initial dp group
  create_standby_groups(dp=4, store B)  <== rendezvous: the standby groups
                                            meet the new workers' boot groups
                                            on the SAME store-B keys, so both
                                            sides land in one HCCL comm
  batch_transfer_weights (send r -> r+2)    batch_transfer_weights (recv)
  pop_standby_groups() +
  _replace_active_groups(...)
  final 4-rank collective on the new active dp group (all ranks)

Pass criteria: every stage PASSes on all 4 ranks, received weights match
the sender's values, and the post-switch 4-rank HCCL collectives (all_reduce
over device_group + NPUCommunicator.broadcast) return correct results.

Run (clean cwd is mandatory, see tests/ut/distributed/test_stateless_hccl_pg.py):
  cd /root && pytest /path/to/test_eep_group_lifecycle.py -x -s
"""

import json
import multiprocessing as mp
import os
import socket
import sys
import time
from types import SimpleNamespace

OLD_DP_SIZE = 2
NEW_DP_SIZE = 4
WORLD_TIMEOUT_S = 1500

# Stage names recorded per rank (first failure stops that rank).
STAGES = [
    "boot",
    "initial_collective",
    "standby_created",
    "weights_transferred",
    "switched",
    "final_collective",
    "cleanup",
]


def _find_free_ports(num: int, base: int = 30200) -> list[int]:
    ports: list[int] = []
    port = base
    while len(ports) < num:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                ports.append(port)
            except OSError:
                pass
        port += 1
    return ports


def _make_stub_config(rank: int, dp_size: int, coord_store_port: int):
    """Duck-typed stand-in for VllmConfig covering the EEP code paths.

    Only the fields actually read by init_distributed_environment /
    initialize_model_parallel / _init_elastic_ep_world are provided.
    """
    parallel_config = SimpleNamespace(
        enable_elastic_ep=True,
        data_parallel_rank=rank,
        data_parallel_size=dp_size,
        world_size_across_dp=dp_size,
        world_size=1,
        nnodes=1,
        nnodes_within_dp=1,
        distributed_executor_backend="ray",
        data_parallel_master_ip="127.0.0.1",
        _coord_store_port=coord_store_port,
        enable_eplb=False,
        enable_expert_parallel=True,
        use_sequence_parallel_moe=False,
        prefill_context_parallel_size=1,
        all2all_backend=None,
        distributed_timeout_seconds=None,
        cpu_distributed_timeout_seconds=None,
    )
    model_config = SimpleNamespace(is_moe=True)
    return SimpleNamespace(parallel_config=parallel_config, model_config=model_config)


class _DummyModel:
    """Minimal nn.Module stand-in for batch_transfer_weights.

    Uses a real nn.Module's state_dict() ordering; three float32 params of
    distinct shapes, all on the worker's NPU device.
    """

    NAMES = ("w1", "w2", "w3")

    def __init__(self, device):
        import torch
        import torch.nn as nn

        self._module = nn.Module()
        self._module.w1 = nn.Parameter(torch.zeros(64, device=device))
        self._module.w2 = nn.Parameter(torch.zeros(8, 16, device=device))
        self._module.w3 = nn.Parameter(torch.zeros(4, 4, 4, device=device))

    def state_dict(self):
        return self._module.state_dict()

    def fill_sender(self, sender_rank: int) -> None:
        for i, name in enumerate(self.NAMES):
            self.state_dict()[name].data.fill_(float(i + 1) * 100 + sender_rank)

    def verify_received(self, sender_rank: int) -> None:
        import torch

        for i, name in enumerate(self.NAMES):
            expected = float(i + 1) * 100 + sender_rank
            got = self.state_dict()[name].data
            if not torch.allclose(got, torch.full_like(got, expected)):
                raise RuntimeError(
                    f"param {name}: got {got.flatten()[0].item()}, "
                    f"expected {expected}"
                )


def _worker(rank: int, ports: dict, result_path: str) -> None:
    """One NPU worker process. rank<2: old (scales 2->4); rank>=2: new."""
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    import vllm_ascend  # noqa: F401  (activates the platform hooks)

    # Production applies the global patch chain (incl. the
    # StatelessGroupCoordinator CudaCommunicator->NPUCommunicator swap in
    # patch_stateless_coordinator.py) from NPUPlatform.pre_register_and_update
    # during engine startup; a bare worker process must do it explicitly.
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)

    is_old = rank < OLD_DP_SIZE
    dp_size = OLD_DP_SIZE if is_old else NEW_DP_SIZE
    coord_store_port = ports["store_a"] if is_old else ports["store_b"]
    # NOTE: HCCL port hygiene (per-group HCCL_IF_BASE_PORT windows and the
    # per-device HCCL_NPU_SOCKET_PORT_RANGE) is provided by the platform
    # hook (NPUPlatform.stateless_init_device_torch_dist_pg); the test
    # intentionally does NOT set them so the production path is exercised.

    stages: dict[str, str] = {}
    stage = "boot"

    def finish() -> None:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(stages, f, indent=2)

    try:
        torch.npu.set_device(rank)
        device = torch.device(f"npu:{rank}")

        import vllm.config.vllm as vllm_config_vllm_mod

        stub = _make_stub_config(rank, dp_size, coord_store_port)
        # set_current_vllm_config is a context manager; set the module
        # global directly (it lives in vllm.config.vllm, NOT on the
        # vllm.config package) so it stays active for the worker lifetime.
        vllm_config_vllm_mod._current_vllm_config = stub

        from vllm.distributed.parallel_state import (
            _replace_active_groups,
            get_dp_group,
            get_world_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.distributed.elastic_ep.elastic_execute import (
            batch_transfer_weights,
        )
        from vllm.distributed.elastic_ep.standby_state import (
            create_standby_groups,
            pop_standby_groups,
        )

        # ---- stage: boot ------------------------------------------------
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{ports[f'local{rank}']}",
            local_rank=rank,
            backend="hccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            prefill_context_model_parallel_size=1,
            decode_context_model_parallel_size=1,
            backend="hccl",
        )
        assert get_world_group().rank == rank, (
            f"world rank {get_world_group().rank} != {rank}"
        )
        assert get_dp_group().world_size == dp_size
        assert get_dp_group().device_communicator is not None
        assert get_dp_group().device_communicator.device == device, (
            f"device communicator device "
            f"{get_dp_group().device_communicator.device} != {device}"
        )
        stages["boot"] = "PASS"
        finish()

        # ---- stage: initial collective (old ranks only) -----------------
        stage = "initial_collective"
        if is_old:
            t = torch.full((8,), float(rank + 1), dtype=torch.float32, device=device)
            dist.all_reduce(t, group=get_dp_group().device_group)
            if t[0].item() != 3.0:  # 1 + 2
                raise RuntimeError(f"initial all_reduce got {t[0].item()}")
            b = torch.zeros(4, dtype=torch.float32, device=device)
            if rank == 0:
                b.fill_(42.0)
            get_dp_group().device_communicator.broadcast(b, src=0)
            if b[0].item() != 42.0:
                raise RuntimeError(f"initial broadcast got {b[0].item()}")
        stages["initial_collective"] = "PASS"
        finish()

        # ---- stage: standby groups --------------------------------------
        stage = "standby_created"
        # Old ranks create standby groups on the scale-round store B; new
        # ranks already built their boot groups on the same store-B keys
        # during the boot stage above -- both sides rendezvous there.
        if is_old:
            create_standby_groups(
                new_dp_size=NEW_DP_SIZE,
                new_world_size_across_dp=NEW_DP_SIZE,
                master_ip="127.0.0.1",
                coord_store_port=ports["store_b"],
                enable_eplb=False,
            )
            from vllm.distributed.elastic_ep.standby_state import (
                get_standby_dp_group,
            )

            standby_dp = get_standby_dp_group()
            assert standby_dp is not None and standby_dp.world_size == NEW_DP_SIZE
        stages["standby_created"] = "PASS"
        finish()

        # ---- stage: weight transfer (production pairing: old r -> r+2) --
        stage = "weights_transferred"
        model = _DummyModel(device)
        if is_old:
            model.fill_sender(rank)
            batch_transfer_weights(
                model._module,
                is_sender=True,
                peer_rank=rank + OLD_DP_SIZE,
                dp_group=get_standby_dp_group(),
                expert_weights=[],
            )
        else:
            sender_rank = rank - OLD_DP_SIZE
            batch_transfer_weights(
                model._module,
                is_sender=False,
                peer_rank=sender_rank,
                dp_group=get_dp_group(),
                expert_weights=[],
            )
            model.verify_received(sender_rank)
        stages["weights_transferred"] = "PASS"
        finish()

        # ---- stage: switch active groups (old ranks only) ---------------
        stage = "switched"
        if is_old:
            popped = pop_standby_groups()
            _replace_active_groups(
                world=popped["world"],
                dp=popped["dp"],
                ep=popped["ep"],
                eplb=popped["eplb"],
                node_count=popped["node_count"],
            )
            assert get_dp_group().world_size == NEW_DP_SIZE
        stages["switched"] = "PASS"
        finish()

        # ---- stage: final collective over the new 4-rank dp group -------
        stage = "final_collective"
        t = torch.full((8,), float(rank + 1), dtype=torch.float32, device=device)
        dist.all_reduce(t, group=get_dp_group().device_group)
        expected = 10.0  # 1 + 2 + 3 + 4
        if t[0].item() != expected:
            raise RuntimeError(
                f"final all_reduce got {t[0].item()}, expected {expected}"
            )
        b = torch.zeros(4, dtype=torch.float32, device=device)
        if rank == 0:
            b.fill_(7.0)
        get_dp_group().device_communicator.broadcast(b, src=0)
        if b[0].item() != 7.0:
            raise RuntimeError(f"final broadcast got {b[0].item()}")
        stages["final_collective"] = "PASS"
        finish()

    except Exception as exc:  # noqa: BLE001
        import traceback

        stages[stage] = f"FAIL: {exc!r}\n{traceback.format_exc()}"
        finish()
        return

    # ---- cleanup (best effort, never fails the run) ----------------------
    try:
        import gc

        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        gc.collect()
        torch.npu.empty_cache()
        stages["cleanup"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        stages["cleanup"] = f"WARN: {exc!r}"
    finish()


def test_eep_group_lifecycle_2_to_4() -> None:
    from torch.distributed import TCPStore

    ports = dict(
        zip(
            ["store_a", "store_b", "local0", "local1", "local2", "local3"],
            _find_free_ports(6),
        )
    )

    # Parent hosts both coordination stores: store A for the initial 2-rank
    # world, store B for the scale round (standby groups <-> new boot).
    stores = [
        TCPStore("127.0.0.1", ports["store_a"], is_master=True, wait_for_workers=False),
        TCPStore("127.0.0.1", ports["store_b"], is_master=True, wait_for_workers=False),
    ]

    ctx = mp.get_context("spawn")
    paths = [f"/tmp/n1_rank{r}.json" for r in range(NEW_DP_SIZE)]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

    procs = [
        ctx.Process(target=_worker, args=(r, ports, paths[r]))
        for r in range(NEW_DP_SIZE)
    ]
    for proc in procs:
        proc.start()

    # Poll instead of plain join: a rank that fails early leaves its peers
    # blocked on store fetches (rendezvous never completes), so once any
    # rank has recorded a failure, give the others a short grace period
    # and then stop waiting.
    deadline = time.time() + WORLD_TIMEOUT_S
    fail_seen_at: float | None = None
    while time.time() < deadline:
        if all(not proc.is_alive() for proc in procs):
            break
        any_fail = False
        all_done = True
        for path in paths:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if any(v.startswith("FAIL") for v in data.values()):
                    any_fail = True
                if "cleanup" not in data and not any(
                    v.startswith("FAIL") for v in data.values()
                ):
                    all_done = False  # still mid-run (or waiting on peers)
            except (OSError, json.JSONDecodeError):
                all_done = False
        if any_fail:
            if fail_seen_at is None:
                fail_seen_at = time.time()
            elif time.time() - fail_seen_at > 180:
                break
        elif all_done:
            break
        time.sleep(10)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join()

    del stores  # close the coord stores after the workers are done

    results: list[dict] = []
    for r, path in enumerate(paths):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            results.append({})

    print("\n===== N1 EEP GROUP LIFECYCLE (2 -> 4) =====", flush=True)
    failed = False
    for stage in STAGES:
        marks = []
        for r in range(NEW_DP_SIZE):
            mark = results[r].get(stage, "MISSING")
            if not (mark == "PASS" or mark.startswith("WARN")):
                failed = True
            marks.append(mark if mark.startswith("PASS") or mark.startswith("WARN") else mark)
        line = f"{stage:22s} " + " | ".join(
            m if len(m) <= 40 else m[:40] + "..." for m in marks
        )
        print(line, flush=True)
        for r, m in enumerate(marks):
            if not (m == "PASS" or m.startswith("WARN")):
                print(f"    rank{r} detail: {results[r].get(stage)}", flush=True)

    with open("/tmp/n1_summary.json", "w", encoding="utf-8") as f:
        json.dump({"failed": failed, "results": results}, f, indent=2)

    assert not failed, "N1 EEP group lifecycle failed; see per-stage output above"


if __name__ == "__main__":
    test_eep_group_lifecycle_2_to_4()
    print("N1 PASSED", flush=True)
    sys.exit(0)
