# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Gloo DP/TP collectives around production Engram orchestration.

Only table lookup and the Triton row-copy kernel use CPU references. No
vLLM/NPU initialization or physical host shared-memory allocation is tested.
The CLI supports full 16-rank checks in an isolated CPU Linux container:
  python tests/ut/models/test_engram_dp_pcp.py --tp 4 --dp 4 --pcp 1 --shared
  python tests/ut/models/test_engram_dp_pcp.py --tp 4 --dp 2 --pcp 2
"""

import argparse
import ast
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HASH_HEADS = 24


def _functions(path, names, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]


def _load_parallel(group, context=None, reduce=None):
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm_ascend/models/deepseek_v41/engram/parallel.py"
    names = {"negotiate_engram_token_slot", "gather_engram_hashes"}
    module = ModuleType("engram_dp_collective_methods")
    module.__dict__.update(
        torch=torch,
        dist=SimpleNamespace(all_reduce=reduce or dist.all_reduce, ReduceOp=dist.ReduceOp),
        DEAD_ID=-1,
        get_dp_group=lambda: group,
        get_engram_dp_group=lambda: group if group.world_size > 1 else None,
        get_engram_dp_size=lambda: group.world_size,
        get_forward_context=context or Mock(side_effect=AssertionError("MRV2 read ForwardContext")),
        _engram_select_rows=_cpu_select_rows,
    )
    # Reuse the actual upstream definitions now imported by Ascend, without
    # importing CUDA kernels or initializing hardware in the CPU harness.
    vllm = pytest.importorskip("vllm")
    upstream_path = Path(vllm.__file__).parent / "models/deepseek_v41/nvidia/engram.py"
    upstream_names = {"engram_gathered_num_tokens", "gather_engram_hashes", "_gather_engram_rows"}
    upstream_tree = ast.Module(body=_functions(upstream_path, upstream_names), type_ignores=[])
    exec(compile(upstream_tree, str(upstream_path), "exec"), module.__dict__)
    module.upstream_gather_engram_hashes = module.gather_engram_hashes
    exec(compile(ast.Module(body=_functions(path, names), type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _cpu_select_rows(gathered, output, source_tokens, token_start, local_width):
    """Reference for the rank-major Triton row copy, not for communication."""
    ranks = gathered.shape[0] // source_tokens
    local_heads, dim = gathered.shape[1:]
    assert local_heads * dim == local_width
    view = gathered.reshape(ranks, source_tokens, local_heads, dim)
    view = view[:, token_start : token_start + output.shape[0]].permute(1, 0, 2, 3)
    output.copy_(view.reshape_as(output))


def _embedding_method(parallel, tp_gather):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/embedding.py"
    namespace = {
        "torch": torch,
        "_gather_engram_rows": parallel._gather_engram_rows,
        "tensor_model_parallel_all_gather": tp_gather,
    }
    body = _functions(path, {"embed_gathered"}, "AscendParallelEngramEmbedding")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["embed_gathered"]


class _GlooGroup:
    def __init__(self, group, size, rank):
        self.cpu_group = group
        self.world_size = size
        self.rank_in_group = rank
        self.gathers = 0

    def all_gather(self, tensor, dim=0):
        self.gathers += 1
        values = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(values, tensor.contiguous(), group=self.cpu_group)
        return torch.cat(values, dim=dim)


def _worker(rank, tp, dp, pcp, shared, rendezvous, output_dir):
    torch.set_num_threads(1)
    world = tp * dp * pcp
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=120))
    dp_rank, remainder = divmod(rank, pcp * tp)
    pcp_rank, tp_rank = divmod(remainder, tp)
    local_dp_group = local_tp_group = None
    # All processes create every group in the same deterministic order.
    for cp in range(pcp):
        for tensor_rank in range(tp):
            ranks = [data_rank * pcp * tp + cp * tp + tensor_rank for data_rank in range(dp)]
            group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=120))
            if rank in ranks:
                local_dp_group = _GlooGroup(group, dp, dp_rank)
    for data_rank in range(dp):
        for cp in range(pcp):
            ranks = list(range(data_rank * pcp * tp + cp * tp, data_rank * pcp * tp + (cp + 1) * tp))
            group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=120))
            if rank in ranks:
                local_tp_group = _GlooGroup(group, tp, tp_rank)
    reductions = []

    def reduce_count(tensor, **kwargs):
        reductions.append(int(tensor[0]))
        dist.all_reduce(tensor, **kwargs)

    parallel = _load_parallel(local_dp_group, reduce=reduce_count)
    embed_gathered = _embedding_method(parallel, local_tp_group.all_gather)
    shard_count = tp if shared else tp * dp
    assert HASH_HEADS % shard_count == 0
    local_heads = HASH_HEADS // shard_count
    head_start = (tp_rank if shared else tp_rank * dp + dp_rank) * local_heads
    table = SimpleNamespace(
        part_n_hash_cols=local_heads, n_hash_cols=HASH_HEADS, dim=1, dp_size=1 if shared else dp, tp_size=tp
    )

    def lookup(indices, out):
        values = indices[:, head_start : head_start + local_heads]
        out.copy_(values.clamp_min(0).to(torch.bfloat16).unsqueeze(-1))

    table.lookup = lookup
    rounds = []
    for step in range(4):
        if step == 0:
            count = 1 + (dp_rank + pcp_rank) % 3
        elif step == 1:
            count = 0 if dp_rank == 0 else dp_rank + pcp_rank + 1
        elif step == 2:
            count = 0
        else:
            count = 1 + (dp_rank + pcp_rank) % 2
        # Distinct PCP markers expose accidental cross-PCP exchanges.
        hashes = torch.arange(count * 2 * HASH_HEADS, dtype=torch.int32).reshape(count, 2, HASH_HEADS)
        hashes += 1 + step * 1000 + pcp_rank * 100 + dp_rank * 10
        before_dp, before_tp, before_reduce = local_dp_group.gathers, local_tp_group.gathers, len(reductions)
        gathered = parallel.gather_engram_hashes(hashes, dp_shared_memory=shared, pre_forward=True)
        for layer in range(2):
            actual = embed_gathered(table, gathered[:, layer], count)
            torch.testing.assert_close(actual, hashes[:, layer].to(torch.bfloat16).unsqueeze(-1), rtol=0, atol=0)
            # Match prepare_engram_inputs' buffer copy, including zero rows:
            # skipping the TP gather must still return the full head width.
            buffer = actual.new_empty((count + 1, HASH_HEADS))
            buffer[:count].copy_(actual.flatten(1))
        actual_counts = (
            local_dp_group.gathers - before_dp,
            local_tp_group.gathers - before_tp,
            len(reductions) - before_reduce,
        )
        expected_counts = (
            3 if dp > 1 and not shared else 0,
            2 if tp > 1 and count > 0 else 0,
            1 if dp > 1 and not shared else 0,
        )
        assert actual_counts == expected_counts, (rank, step, actual_counts, expected_counts)
        rounds.append(
            {
                "step": step,
                "local_rows": count,
                "dp_gathers": actual_counts[0],
                "tp_gathers": actual_counts[1],
                "slot_negotiations": actual_counts[2],
            }
        )
    Path(output_dir, f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "dp": dp_rank, "pcp": pcp_rank, "tp": tp_rank, "rounds": rounds}), encoding="utf-8"
    )
    dist.barrier()
    dist.destroy_process_group()


def run_collectives(tp, dp, pcp, shared):
    with tempfile.TemporaryDirectory(prefix="engram-gloo-") as temporary:
        path = Path(temporary)
        rendezvous = (path / "rendezvous").as_uri()
        mp.spawn(_worker, args=(tp, dp, pcp, shared, rendezvous, temporary), nprocs=tp * dp * pcp, join=True)
        results = [json.loads((path / f"rank-{rank}.json").read_text()) for rank in range(tp * dp * pcp)]
    return {
        "passed": True,
        "backend": "gloo",
        "tp": tp,
        "dp": dp,
        "pcp": pcp,
        "shared": shared,
        "ranks": results,
        "scope": "Real CPU DP/TP collectives; table values and Triton row copy use CPU references",
    }


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("tp,dp,pcp,shared", [(2, 2, 1, False), (2, 2, 1, True), (2, 1, 2, False), (1, 2, 2, False)])
def test_real_gloo_collectives(tp, dp, pcp, shared):
    assert run_collectives(tp, dp, pcp, shared)["passed"]


def test_mrv1_retains_forward_context_slot_and_no_handshake():
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock(side_effect=lambda value, dim: value))
    context = Mock(
        return_value=SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([3, 4])))
    )
    reduce = Mock(side_effect=AssertionError("MRV1 negotiated a new slot"))
    parallel = _load_parallel(group, context=context, reduce=reduce)
    actual = parallel.gather_engram_hashes(torch.tensor([[[7]]], dtype=torch.int32))
    assert actual.flatten().tolist() == [7, -1, -1, -1]
    context.assert_called_once()
    reduce.assert_not_called()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.tp, args.dp, args.pcp) < 1 or args.tp * args.dp * args.pcp > 16:
        parser.error("Positive TP/DP/PCP and at most 16 processes are required")
    if HASH_HEADS % (args.tp if args.shared else args.tp * args.dp):
        parser.error("24 hash heads must divide over TP (shared) or TP*DP (non-shared)")
    report = run_collectives(args.tp, args.dp, args.pcp, args.shared)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
