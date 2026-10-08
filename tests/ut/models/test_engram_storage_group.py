# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram storage topology and real CPU collectives without NPU imports."""

import ast
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_storage_group(world, distributed, same_node, dp_group=None):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/parallel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"EngramStorageGroup", "create_engram_storage_group", "resolve_dp_shared_memory"}
    definitions = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    module = ModuleType("engram_storage_group_methods")
    module.__dict__.update(
        torch=torch,
        dist=distributed,
        get_world_group=lambda: world,
        get_engram_dp_group=lambda: dp_group,
        get_engram_dp_size=lambda: dp_group.world_size if dp_group is not None else 1,
        in_the_same_node_as=same_node,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _config(tp=1, dp=2, pcp=8, **overrides):
    values = dict(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        prefill_context_parallel_size=pcp,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        nnodes=1,
        enable_elastic_ep=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _mock_module(world_size, rank, same_node=None, dp_group=None):
    distributed = SimpleNamespace(
        new_group=Mock(side_effect=lambda ranks, **kwargs: tuple(ranks)),
        destroy_process_group=Mock(),
        all_reduce=Mock(),
        ReduceOp=dist.ReduceOp,
    )
    world = SimpleNamespace(world_size=world_size, rank=rank, cpu_group="world-cpu")
    same_node = same_node or Mock(side_effect=lambda ranks: [True] * len(ranks))
    module = _load_storage_group(world, distributed, same_node, dp_group)
    return module, distributed, same_node


@pytest.mark.parametrize("rank", range(16))
def test_tp1_dp2_pcp8_share_one_group(rank):
    module, distributed, same_node = _mock_module(16, rank)
    group = module.create_engram_storage_group(_config())
    assert group.cpu_group == tuple(range(16))
    assert group.world_size == 16 and group.rank_in_group == rank
    distributed.new_group.assert_called_once_with(list(range(16)), backend="gloo")
    same_node.assert_called_once_with(tuple(range(16)))
    assert distributed.all_reduce.call_args.kwargs == dict(op=dist.ReduceOp.MIN, group="world-cpu")
    group.close()
    group.close()
    distributed.destroy_process_group.assert_called_once_with(tuple(range(16)))


@pytest.mark.parametrize("rank", range(16))
def test_storage_keeps_tp_and_external_dp_coordinates(rank):
    # ExternalDP2 x DP2 x PCP2 x TP2: four independent table shard groups.
    module, distributed, _ = _mock_module(16, rank)
    group = module.create_engram_storage_group(_config(tp=2, dp=2, pcp=2))
    expected = [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]
    assert [call.args[0] for call in distributed.new_group.call_args_list] == expected
    members = next(ranks for ranks in expected if rank in ranks)
    assert group.cpu_group == tuple(members)
    assert group.world_size == 4 and group.rank_in_group == members.index(rank)


def test_pcp1_borrows_dp_group_and_preserves_query_group():
    dp_group = SimpleNamespace(cpu_group="existing-dp", world_size=2, rank_in_group=1)
    module, distributed, same_node = _mock_module(16, 9, dp_group=dp_group)
    group = module.create_engram_storage_group(_config(tp=8, dp=2, pcp=1))
    assert group.cpu_group == "existing-dp" and group.world_size == 2 and group.rank_in_group == 1
    assert module.get_engram_dp_group() is dp_group
    group.close()
    distributed.new_group.assert_not_called()
    distributed.destroy_process_group.assert_not_called()
    same_node.assert_not_called()


def test_owned_group_close_can_retry_after_destroy_failure():
    module, distributed, _ = _mock_module(16, 0)
    group = module.create_engram_storage_group(_config())
    distributed.destroy_process_group.side_effect = [RuntimeError("destroy failed"), None]
    with pytest.raises(RuntimeError, match="destroy failed"):
        group.close()
    assert not group._closed
    group.close()
    group.close()
    assert group._closed
    assert distributed.destroy_process_group.call_count == 2


def test_unshared_singleton_returns_none():
    dp_group = SimpleNamespace(world_size=1)
    module, distributed, _ = _mock_module(1, 0, dp_group=dp_group)
    assert module.create_engram_storage_group(_config(tp=1, dp=1, pcp=1)) is None
    distributed.new_group.assert_not_called()


@pytest.mark.parametrize("edp,pcp,shared", [(1, 1, False), (2, 1, True), (1, 8, True), (2, 8, True)])
def test_resolve_sharing_includes_pcp_peers(edp, pcp, shared):
    module, _, _ = _mock_module(16, 0, dp_group=SimpleNamespace(world_size=edp))
    assert module.resolve_dp_shared_memory(True, pcp) is shared
    assert not module.resolve_dp_shared_memory(False, pcp)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"pipeline_parallel_size": 2}, "PP=DCP=1"),
        ({"decode_context_parallel_size": 2}, "PP=DCP=1"),
        ({"nnodes": 2}, "single-node"),
        ({"enable_elastic_ep": True}, "elastic EP"),
        ({"data_parallel_size": 0}, "positive"),
    ],
)
def test_invalid_configuration_fails_before_group_creation(overrides, message):
    module, distributed, _ = _mock_module(16, 0)
    with pytest.raises(ValueError, match=message):
        module.create_engram_storage_group(_config(**overrides))
    distributed.new_group.assert_not_called()


def test_invalid_world_layout_fails_before_group_creation():
    module, distributed, _ = _mock_module(15, 0)
    with pytest.raises(ValueError, match="divisible"):
        module.create_engram_storage_group(_config())
    distributed.new_group.assert_not_called()


@pytest.mark.parametrize("same_node", [Mock(return_value=[True, False]), Mock(side_effect=RuntimeError("IPC probe"))])
def test_ipc_failure_cleans_up_owned_group(same_node):
    module, distributed, _ = _mock_module(2, 0, same_node=same_node)
    with pytest.raises(ValueError, match="share node IPC"):
        module.create_engram_storage_group(_config(dp=1, pcp=2))
    distributed.destroy_process_group.assert_called_once_with((0, 1))


def test_other_tp_groups_ipc_failure_is_propagated_before_storage_allocation():
    module, distributed, same_node = _mock_module(4, 0)

    def failed_world_check(value, **kwargs):
        value.zero_()

    distributed.all_reduce.side_effect = failed_world_check
    with pytest.raises(ValueError, match="share node IPC"):
        module.create_engram_storage_group(_config(tp=2, dp=1, pcp=2))
    same_node.assert_called_once_with((0, 2))
    distributed.destroy_process_group.assert_called_once_with((0, 2))


def _storage_worker(rank, tp, dp, pcp, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    group = None
    try:
        world = SimpleNamespace(world_size=world_size, rank=rank, cpu_group=dist.group.WORLD)
        # Topology tests above cover the IPC probe's failure handling. This test
        # exercises actual new_group, MIN fencing and storage-group collectives.
        module = _load_storage_group(world, dist, lambda pg: [True] * dist.get_world_size(pg))
        group = module.create_engram_storage_group(_config(tp=tp, dp=dp, pcp=pcp))
        external_rank, remainder = divmod(rank, tp * dp * pcp)
        tp_rank = remainder % tp
        members = [external_rank * tp * dp * pcp + offset * tp + tp_rank for offset in range(dp * pcp)]
        assert dist.get_process_group_ranks(group.cpu_group) == members
        values = [None] * group.world_size
        dist.all_gather_object(values, rank, group=group.cpu_group)
        assert values == members
        for step in range(2):
            value = torch.tensor([rank + step], dtype=torch.int64)
            dist.all_reduce(value, group=group.cpu_group)
            assert int(value[0]) == sum(members) + step * len(members)
        dist.barrier()
        group.close()
        group.close()
        group = None
    finally:
        if group is not None:
            group.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("tp,dp,pcp", [(1, 2, 2), (2, 1, 2), (1, 1, 2)])
def test_real_gloo_storage_collectives(tmp_path, tp, dp, pcp):
    # Four processes also cover two ExternalDP replicas in the final case.
    rendezvous = (tmp_path / "gloo-init").resolve().as_uri()
    mp.spawn(_storage_worker, args=(tp, dp, pcp, 4, rendezvous), nprocs=4, join=True)
