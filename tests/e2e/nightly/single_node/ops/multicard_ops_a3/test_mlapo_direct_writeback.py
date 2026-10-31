# SPDX-License-Identifier: Apache-2.0
"""Multi-TP Direct writeback and FSA consumer consistency on real GLM weights.

Uses the production manager's producer-completion and reuse-layer publication
methods, real HCCL ranks and memfabric SHARED pools. The FSA internal planner is
used to force CPU misses: this is not the full owner-layer CPU LRU/service test.
No host synchronization is inserted between Direct and its FSA consumer.

Run on an otherwise idle A3 host from the repository root::

    python -m tests.e2e.nightly.single_node.ops.multicard_ops_a3.test_mlapo_direct_writeback \
        --model-path /data/weights/GLM-5.2-w8a8 --output /actual/tp2.json --tp-size 2
"""

import argparse
import gc
import json
import socket
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch_npu
from memfabric_hybrid import offload
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_fused_sparse_attention_overlap import (
    BLOCK_SIZE,
    _assert_selection_state_matches_full_cache,
    _call_fused,
    _make_case,
)
from tests.e2e.nightly.single_node.ops.singlecard_ops.test_mlapo_direct_equivalence import (
    OUTPUTS,
    direct,
    exact,
    loaded_libraries,
    new_outputs,
    nightly_result_dir,
    original,
    resolve_nightly_weights,
    runtime_provenance,
    sha256,
    weights_for_heads,
)
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
)
from vllm_ascend.utils import enable_custom_op

TOKENS = 6
TOPK = 32
LAYER = "model.layers.0.self_attn.attn"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _manager(rank, group, slots):
    # Only initialize state required by the real methods under test. No methods
    # or device operations are mocked; the expensive LRU owner is out of scope.
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank, manager.tp_size, manager.tp_group = rank, group.world_size, group
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {LAYER: 0}
    manager.offload_layer_names = [LAYER]
    manager.current_kv_by_layer = {}
    manager.producer_writeback_layers = set()
    manager.current_kv_save_stream = torch_npu.npu.Stream()
    manager.fused_plan_stream = torch_npu.npu.Stream()
    manager.fused_plan_status_npu = torch.zeros(1, dtype=torch.int32, device="npu")
    # A different owner exercises the per-layer completion broadcast in the
    # production reuse path, rather than substituting a test-only barrier.
    manager.fused_overlap_plan_owner_layer_id = -1
    manager.fused_plan_current_linear_slots_npu = slots
    return manager


def _case(rank, group, model_path):
    device = f"npu:{rank}"
    d, w, provenance = weights_for_heads(model_path, 0, 64 // group.world_size, device, tp_rank=rank)
    old, old_guards = new_outputs(TOKENS, d, device, paged=True)
    out, guards = new_outputs(TOKENS, d, device)
    x = torch.empty(TOKENS, d["hidden"], dtype=torch.bfloat16, device=device)
    cos = torch.empty(TOKENS, 1, 1, d["rope"], dtype=x.dtype, device=device)
    sin = torch.empty_like(cos)
    compact = torch.arange(TOKENS, dtype=torch.int32, device=device)
    slots = torch.empty(TOKENS, dtype=torch.int32, device=device)

    reference = _make_case(torch.bfloat16, TOPK)
    baseline = (reference.full_kv_cache.cpu(), reference.full_k_rope.cpu())
    shapes = [list(value.shape) for value in baseline]
    pools = tuple(offload.empty(shape, dtype=torch.bfloat16, pin_memory=True) for shape in shapes) if rank == 0 else ()
    addresses = torch.tensor([pool.data_ptr() for pool in pools] if rank == 0 else [0, 0], device=device)
    group.broadcast(addresses, src=0)
    torch.npu.synchronize()
    if rank != 0:
        pools = tuple(
            torch.ops._C_ascend.sparse_kv_restore_bfloat16_tensor(int(ptr), shape)
            for ptr, shape in zip(addresses.cpu().tolist(), shapes)
        )
    assert all(pool.data_ptr() == ptr for pool, ptr in zip(pools, addresses.cpu().tolist()))
    flat_pools = tuple(pool.view(-1, pool.shape[-1]) for pool in pools)
    reference.topk_indices.copy_(torch.arange(TOPK, dtype=torch.int32, device=device).view(1, 1, TOPK))
    reference.query = torch.empty(1, d["heads"], d["kv"] + d["rope"], dtype=x.dtype, device=device)
    actual = replace(
        reference,
        query=torch.empty_like(reference.query),
        full_kv_cache=pools[0],
        full_k_rope=pools[1],
        selection_kv_cache=reference.selection_kv_cache.clone(),
        selection_k_rope=reference.selection_k_rope.clone(),
        selection_block_status=reference.selection_block_status.clone(),
        selection_membership_map=reference.selection_membership_map.clone(),
    )
    selection_slots = torch.empty_like(slots)
    manager = _manager(rank, group, selection_slots)
    shared_block = int(reference.full_block_table.cpu()[0, 0])
    selection_block = int(reference.selection_block_table.cpu()[0, 0])
    slot_ids = []
    expected_pools = []

    def update(iteration):
        # Preparation precedes Direct. Every replay changes the written values
        # and slots, and forces FSA misses so old resident KV cannot mask a race.
        group.barrier()
        generator = torch.Generator(device="cpu").manual_seed(20261028 + iteration)
        x.copy_(torch.randn(x.shape, dtype=x.dtype, generator=generator) * (1.0 + iteration / 4))
        angle = (torch.arange(TOKENS).float()[:, None] + iteration + 1) / 100
        angle = angle * torch.arange(1, 33).float()[None, :]
        cos.copy_(torch.cat((angle.cos(), angle.cos()), dim=-1).to(x.dtype).view_as(cos))
        sin.copy_(torch.cat((angle.sin(), angle.sin()), dim=-1).to(x.dtype).view_as(sin))
        logical = [3 + iteration % 2 + row * 2 for row in range(TOKENS)]
        slot_ids[:] = [shared_block * BLOCK_SIZE + value for value in logical]
        slots.copy_(torch.tensor(slot_ids, dtype=torch.int32, device=device))
        selection_slots.copy_(torch.tensor([selection_block * BLOCK_SIZE + value for value in logical], device=device))
        for case in (actual, reference):
            case.selection_kv_cache.fill_(float("nan"))
            case.selection_k_rope.fill_(float("nan"))
            case.selection_block_status.fill_(-1)
            case.selection_membership_map.fill_(-1)
        original(x, cos, sin, compact, old, w)
        torch.npu.synchronize()
        expected_pools[:] = [value.clone() for value in baseline]
        for expected, name in zip(expected_pools, ("current_k", "current_rope")):
            expected.view(-1, expected.shape[-1])[slot_ids] = old[name].cpu()
        reference.full_kv_cache.copy_(expected_pools[0])
        reference.full_k_rope.copy_(expected_pools[1])
        reference.query.copy_(torch.cat((old["q_nope"][-1:], old["q_rope"][-1:]), dim=-1))
        if rank == 0:
            for pool, initial in zip(pools, baseline):
                pool.copy_(initial)
        torch.npu.synchronize()
        group.barrier()

    def launch(capturing):
        if capturing:
            # This isolated reuse layer has fixed plan inputs. Fork its idle
            # plan stream before Direct so the real join is capture-compatible
            # without adding a dependency on completion of the CPU stores.
            manager.fused_plan_stream.wait_stream(torch.npu.current_stream())
        direct(x, cos, sin, out, w, flat_pools if rank == 0 else None, slots if rank == 0 else None)
        manager.offload_new_kv(
            LAYER,
            slots,
            pools[0],
            pools[1],
            None,
            None,
            out["current_k"],
            out["current_rope"],
            capturing=capturing,
            writeback_in_producer=True,
        )
        actual.query.copy_(torch.cat((out["q_nope"][-1:], out["q_rope"][-1:]), dim=-1))
        manager.inject_current_kv_into_selection(
            LAYER,
            TOKENS,
            actual.selection_kv_cache,
            actual.selection_k_rope,
            capturing=capturing,
        )
        result = _call_fused(actual)
        manager.wait_for_current_kv_writeback(capturing)
        return result

    def check(label, result):
        # Deliberately after the entire Direct -> notification -> FSA chain.
        torch.npu.synchronize()
        values = {name: exact(out[name], old[name], (label, rank, name)) for name in OUTPUTS}
        for pool, expected, name in zip(pools, expected_pools, ("current_k", "current_rope")):
            values[name + "_cpu_pool"] = exact(pool, expected, (label, rank, "shared CPU pool", name))
        _assert_selection_state_matches_full_cache(actual)
        expected = _call_fused(reference)
        torch.npu.synchronize()
        values["fsa"] = exact(result, expected, (label, rank, "FSA CPU pool vs original-MLAPO KV"))
        for name, backing in guards.items():
            exact(backing[[0, -1]], torch.full_like(backing[[0, -1]], -17.5), (label, name, "guard"))
        for name in ("q_nope", "q_rope", "q_c"):
            backing = old_guards[name]
            exact(backing[[0, -1]], torch.full_like(backing[[0, -1]], -17.5), (label, name, "original guard"))
        group.barrier()
        return {"label": label, "rank": rank, "slots": list(slot_ids), "outputs": values}

    rows = []
    update(0)
    rows.append(check("eager", launch(False)))
    update(1)
    graph = UpdatableGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        captured_output = launch(True)
    rows.append(check("graph_capture", captured_output))
    for iteration in (2, 3):
        update(iteration)
        graph.replay()
        rows.append(check(f"graph_replay_{iteration}", captured_output))
    assert len({row["outputs"]["current_k"]["sha256"] for row in rows}) > 1, "KV did not change between replays"
    # Pool owners and peer views remain live until every rank has completed all
    # reads. The caller keeps memfabric initialized until this function returns.
    torch.npu.synchronize()
    group.barrier()
    return {"rank": rank, "weights": provenance, "cases": rows}


def _worker(rank, world_size, group_port, store_port, model_path, result_dir):
    group = None
    initialized = False
    result = {"status": "failed", "rank": rank, "tp_size": world_size}
    try:
        torch_npu.npu.set_device(rank)
        torch.npu.config.allow_internal_format = True
        assert enable_custom_op()
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            backend="hccl",
            distributed_init_method=f"tcp://127.0.0.1:{group_port}",
            timeout=timedelta(minutes=5),
        )
        group = init_model_parallel_group(
            [list(range(world_size))],
            local_rank=rank,
            backend="hccl",
            group_name="mlapo_writeback_test",
            use_device_communicator=False,
        )
        cfg = offload.OffloadConfig()
        cfg.store_url = f"tcp://127.0.0.1:{store_port}"
        cfg.device_id, cfg.world_size, cfg.rank_id = rank, world_size, rank
        cfg.reserve_size, cfg.alloc_size = 1 << 30, (1 << 30) if rank == 0 else 0
        cfg.scene = offload.Scene.SHARED
        assert offload.initialize(cfg) == 0
        initialized = True
        group.barrier()
        with torch.inference_mode():
            result.update(_case(rank, group, model_path))
        result["libraries"] = loaded_libraries()
        result["runtime"] = runtime_provenance()
        result["status"] = "passed"
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if initialized:
                torch.npu.synchronize()
                gc.collect()
                offload.uninitialize()
            if group is not None:
                group.destroy()
            destroy_distributed_environment()
        except BaseException:
            result["status"] = "failed"
            result["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            (result_dir / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(2, 4), default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(args.model_path / "config.json")
    if torch.npu.device_count() < args.tp_size:
        raise RuntimeError(f"requires {args.tp_size} free visible A3 devices")
    result_dir = args.output.parent / (args.output.stem + "-ranks")
    result_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "test_sha256": sha256(__file__), "tp_size": args.tp_size}
    try:
        group_port, store_port = _free_port(), _free_port()
        while store_port == group_port:
            store_port = _free_port()
        mp.spawn(
            _worker,
            args=(args.tp_size, group_port, store_port, args.model_path.resolve(), result_dir),
            nprocs=args.tp_size,
            join=True,
        )
        result["ranks"] = [json.loads((result_dir / f"rank-{rank}.json").read_text()) for rank in range(args.tp_size)]
        assert all(row["status"] == "passed" and len(row["cases"]) == 4 for row in result["ranks"])
        result["status"] = "passed"
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["scope"] = (
            "Real multi-TP Direct/manager reuse-layer completion/FSA consumer; not owner LRU or full service"
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("tp_size", [2, 4])
def test_mlapo_direct_shared_pool_writeback(tp_size):
    main(
        [
            "--model-path",
            str(resolve_nightly_weights()),
            "--output",
            str(nightly_result_dir(f"tp{tp_size}") / "result.json"),
            "--tp-size",
            str(tp_size),
        ]
    )


if __name__ == "__main__":
    main()
