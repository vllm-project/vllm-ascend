# SPDX-License-Identifier: Apache-2.0
"""N0 qualification test for stateless HCCL process groups on Ascend NPU.

This is the gating test for elastic EP (EEP) support on Ascend: vLLM's
elastic EP creates *standby* groups via
``vllm.distributed.utils.stateless_init_torch_distributed_process_group``,
which delegates device PG construction to
``NPUPlatform.stateless_init_device_torch_dist_pg``.

Before building any elastic AFD machinery on top of it, we must prove on
real hardware that a stateless HCCL process group can:

1. be created outside torch.distributed's global world state,
2. run collectives (all_reduce / all_gather),
3. coexist with other stateless groups (subgroups, like standby DP/EP),
4. be destroyed and re-created repeatedly (scale up/down cycles),
5. survive the destruction of a peer group (active -> standby switch).

The test spawns one process per NPU rank with ``multiprocessing`` and
drives all scenarios above. Each worker writes a JSON result file that
the pytest process asserts on.

Usage (requires NPU hardware):

    pytest -sv tests/ut/distributed/test_stateless_hccl_pg.py
    N0_WORLD_SIZE=4 pytest -sv tests/ut/distributed/test_stateless_hccl_pg.py
"""

import json
import multiprocessing as mp
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field

import pytest

try:
    import torch
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

# Number of sequential create -> collective -> destroy cycles per worker.
NUM_CREATE_DESTROY_CYCLES = 3
# Upper bound (seconds) for a worker process to finish all scenarios.
WORKER_TIMEOUT_S = 600
# Default number of NPU ranks used by the test (>= 2, even for subgroups).
DEFAULT_WORLD_SIZE = 4
# Tensor size used for collectives.
COLLECTIVE_TENSOR_NUMEL = 1024


def _scenario_enabled(name: str) -> bool:
    """N0_ONLY="scenario1,scenario2" runs a subset of scenarios.

    Debugging affordance: run one scenario in a fresh process set to
    isolate it from state accumulated by earlier scenarios (HCCL
    daemons/listeners live per process and survive group destruction).
    """
    only = os.getenv("N0_ONLY", "").strip()
    return not only or name in only.split(",")


def _find_free_ports(num_ports: int, base_port: int = 29600) -> list[int]:
    """Reserve *num_ports* free TCP ports near *base_port*.

    Binds sockets to port 0 when the preferred port is taken, then closes
    them. Small TOCTOU window is acceptable for a dedicated test machine.
    """
    ports: list[int] = []
    candidates = []
    port = base_port
    while len(candidates) < num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                candidates.append(port)
            except OSError:
                pass
        port += 1
    # Close-check done; re-bind with port 0 fallback would complicate the
    # rendezvous protocol, so just hand back the checked candidates.
    ports = candidates
    return ports


@dataclass
class ScenarioPorts:
    """TCP ports reserved for each scenario, shared by all workers."""

    # One port per create/destroy cycle.
    cycle_ports: list[int] = field(default_factory=list)
    # Two ports for the two concurrent subgroups (world // 2 ranks each).
    subgroup_ports: list[int] = field(default_factory=list)
    # Two ports for the active/standby switch scenario.
    active_port: int = 0
    standby_port: int = 0
    # One port for the bidirectional p2p (isend/irecv) scenario.
    p2p_port: int = 0


def _n0_worker(
    rank: int,
    world_size: int,
    ports: ScenarioPorts,
    result_path: str,
) -> None:
    """Worker body: run all N0 scenarios on NPU *rank* and dump results."""
    results: dict[str, str] = {}

    def record(name: str, ok: bool, detail: str = "") -> None:
        results[name] = "PASS" if ok else f"FAIL: {detail}"

    try:
        import torch
        import torch.distributed as dist

        import vllm_ascend  # noqa: F401  (register NPU platform plugin)
        from vllm.distributed.utils import (
            stateless_destroy_torch_distributed_process_group,
            stateless_init_torch_distributed_process_group,
        )

        torch.npu.set_device(rank)

        def make_group(port: int, ws: int, r: int, name: str):
            return stateless_init_torch_distributed_process_group(
                host="127.0.0.1",
                port=port,
                rank=r,
                world_size=ws,
                backend="hccl",
                group_name=name,
            )

        def check_all_reduce(pg, ws: int, group_rank: int) -> None:
            tensor = torch.full(
                (COLLECTIVE_TENSOR_NUMEL,),
                float(group_rank + 1),
                dtype=torch.float32,
                device="npu",
            )
            dist.all_reduce(tensor, group=pg)
            expected = float(sum(range(1, ws + 1)))
            if not torch.allclose(tensor, torch.full_like(tensor, expected)):
                raise RuntimeError(
                    f"all_reduce mismatch: got {tensor[0].item()}, "
                    f"expected {expected}"
                )

        def check_all_gather(pg, ws: int, group_rank: int) -> None:
            tensor = torch.full(
                (COLLECTIVE_TENSOR_NUMEL,),
                float(group_rank + 1),
                dtype=torch.float32,
                device="npu",
            )
            gathered = [
                torch.empty_like(tensor) for _ in range(ws)
            ]
            dist.all_gather(gathered, tensor, group=pg)
            for idx in range(ws):
                expected = float(idx + 1)
                if not torch.allclose(
                    gathered[idx], torch.full_like(tensor, expected)
                ):
                    raise RuntimeError(
                        f"all_gather mismatch at index {idx}"
                    )

        # ---- Scenario 1: repeated create -> collective -> destroy ----
        if _scenario_enabled("create_destroy_cycles"):
            try:
                for cycle in range(NUM_CREATE_DESTROY_CYCLES):
                    pg = make_group(
                        ports.cycle_ports[cycle],
                        world_size,
                        rank,
                        f"n0_cycle_{cycle}",
                    )
                    check_all_reduce(pg, world_size, rank)
                    check_all_gather(pg, world_size, rank)
                    stateless_destroy_torch_distributed_process_group(pg)
                record("create_destroy_cycles", True)
            except Exception as exc:  # noqa: BLE001
                record("create_destroy_cycles", False, repr(exc))

        # ---- Scenario 2: two concurrent subgroups ----
        # Mimics standby DP/EP subgroup creation: the world is split into
        # two independent stateless groups with their own rendezvous ports.
        if _scenario_enabled("concurrent_subgroups"):
            try:
                subgroup_size = world_size // 2
                subgroup_idx = rank // subgroup_size
                sub_rank = rank % subgroup_size
                pg = make_group(
                    ports.subgroup_ports[subgroup_idx],
                    subgroup_size,
                    sub_rank,
                    f"n0_subgroup_{subgroup_idx}",
                )
                check_all_reduce(pg, subgroup_size, sub_rank)
                stateless_destroy_torch_distributed_process_group(pg)
                record("concurrent_subgroups", True)
            except Exception as exc:  # noqa: BLE001
                record("concurrent_subgroups", False, repr(exc))

        # ---- Scenario 3: active/standby switch ----
        # The core EEP semantic: a standby group is created while the
        # active group is still alive; the active group is then destroyed
        # and the standby group must remain fully functional afterwards.
        if _scenario_enabled("active_standby_switch"):
            try:
                active_pg = make_group(
                    ports.active_port, world_size, rank, "n0_active"
                )
                check_all_reduce(active_pg, world_size, rank)

                standby_pg = make_group(
                    ports.standby_port, world_size, rank, "n0_standby"
                )
                # Both groups usable concurrently.
                check_all_reduce(active_pg, world_size, rank)
                check_all_reduce(standby_pg, world_size, rank)

                # Retire the active group (as _replace_active_groups would).
                stateless_destroy_torch_distributed_process_group(active_pg)

                # Standby group must survive the peer group's destruction.
                check_all_reduce(standby_pg, world_size, rank)
                check_all_gather(standby_pg, world_size, rank)
                stateless_destroy_torch_distributed_process_group(standby_pg)
                record("active_standby_switch", True)
            except Exception as exc:  # noqa: BLE001
                record("active_standby_switch", False, repr(exc))

        # ---- Scenario 4: bidirectional isend/irecv ----
        # Mimics the elastic EP expert weight transfer
        # (elastic_execute.batch_transfer_weights): every rank concurrently
        # isends a tensor to its neighbor and irecvs from its other
        # neighbor, through the production path
        # NPUCommunicator.batch_isend_irecv (which posts on the HCCL
        # backend directly -- stateless groups are not in torch's _world
        # registry, so dist.isend/irecv would raise). Posting all async ops
        # before waiting is the NPUCommunicator.batch_isend_irecv strategy;
        # this scenario proves it does not deadlock on HCCL.
        if _scenario_enabled("bidirectional_p2p"):
            try:
                from torch.distributed import P2POp

                from vllm_ascend.distributed.device_communicators.npu_communicator import (  # noqa: E501
                    NPUCommunicator,
                )

                pg = make_group(ports.p2p_port, world_size, rank, "n0_p2p")
                comm = NPUCommunicator(
                    cpu_group=pg,
                    device=torch.device(f"npu:{rank}"),
                    device_group=pg,
                    global_ranks=list(range(world_size)),
                    global_world_size=world_size,
                )
                send_dst = (rank + 1) % world_size
                recv_src = (rank - 1) % world_size
                send_tensor = torch.full(
                    (COLLECTIVE_TENSOR_NUMEL,),
                    float(rank + 1),
                    dtype=torch.float32,
                    device="npu",
                )
                recv_tensor = torch.zeros(
                    COLLECTIVE_TENSOR_NUMEL, dtype=torch.float32, device="npu"
                )
                p2p_ops = []
                for op_fn, tensor, peer in (
                    (torch.distributed.isend, send_tensor, send_dst),
                    (torch.distributed.irecv, recv_tensor, recv_src),
                ):
                    op = object.__new__(P2POp)
                    op.op = op_fn
                    op.tensor = tensor
                    op.group_peer = peer
                    p2p_ops.append(op)
                comm.batch_isend_irecv(p2p_ops)
                expected = float(((rank - 1) % world_size) + 1)
                if not torch.allclose(
                    recv_tensor, torch.full_like(recv_tensor, expected)
                ):
                    raise RuntimeError(
                        f"irecv mismatch: got {recv_tensor[0].item()}, "
                        f"expected {expected}"
                    )
                stateless_destroy_torch_distributed_process_group(pg)
                record("bidirectional_p2p", True)
            except Exception as exc:  # noqa: BLE001
                record("bidirectional_p2p", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        results["worker_setup"] = f"FAIL: {exc!r}"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


@pytest.mark.skipif(
    not NPU_AVAILABLE, reason="NPU hardware is required for the N0 test"
)
def test_stateless_hccl_pg_qualification():
    """N0 qualification: stateless HCCL group lifecycle on real NPU."""
    world_size = int(os.getenv("N0_WORLD_SIZE", DEFAULT_WORLD_SIZE))
    if world_size < 2 or world_size % 2 != 0:
        pytest.fail(
            f"N0_WORLD_SIZE must be an even number >= 2, got {world_size}"
        )
    if torch.npu.device_count() < world_size:
        pytest.skip(
            f"need {world_size} NPUs, found {torch.npu.device_count()}"
        )

    total_ports = (
        NUM_CREATE_DESTROY_CYCLES + 2 + 2 + 1  # cycles + subgroups + switch + p2p
    )
    free_ports = _find_free_ports(total_ports)
    ports = ScenarioPorts(
        cycle_ports=free_ports[:NUM_CREATE_DESTROY_CYCLES],
        subgroup_ports=free_ports[
            NUM_CREATE_DESTROY_CYCLES : NUM_CREATE_DESTROY_CYCLES + 2
        ],
        active_port=free_ports[-3],
        standby_port=free_ports[-2],
        p2p_port=free_ports[-1],
    )

    with tempfile.TemporaryDirectory(prefix="n0_stateless_hccl_") as tmpdir:
        result_paths = [
            os.path.join(tmpdir, f"rank{rank}.json")
            for rank in range(world_size)
        ]
        ctx = mp.get_context("spawn")
        workers = [
            ctx.Process(
                target=_n0_worker,
                args=(rank, world_size, ports, result_paths[rank]),
            )
            for rank in range(world_size)
        ]
        for worker in workers:
            worker.start()
        try:
            deadline = time.time() + WORKER_TIMEOUT_S
            for worker in workers:
                remaining = max(deadline - time.time(), 1.0)
                worker.join(timeout=remaining)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join()

        failures = []
        for rank, path in enumerate(result_paths):
            if not os.path.exists(path):
                failures.append(
                    f"rank {rank}: no result file (worker crashed, "
                    f"exitcode={workers[rank].exitcode})"
                )
                continue
            with open(path, encoding="utf-8") as f:
                results = json.load(f)
            for scenario, status in results.items():
                if status != "PASS":
                    failures.append(f"rank {rank} / {scenario}: {status}")

    assert not failures, "N0 qualification failures:\n" + "\n".join(failures)
