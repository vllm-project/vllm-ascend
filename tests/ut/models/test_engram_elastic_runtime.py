# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.models.deepseek_v41.engram.runtime import ElasticEngramRuntime


def config():
    return SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=False),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(engram_layer_ids=[1, 14]), max_model_len=64),
        parallel_config=SimpleNamespace(world_size_across_dp=4, local_world_size=4, data_parallel_size_local=1),
        load_config=SimpleNamespace(load_format="safetensors"),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )


def runtime():
    model = SimpleNamespace(
        engram_rotation=torch.eye(4),
        _engram_max_tokens=8,
        _engram_overlap_enabled=True,
        engram_hash=SimpleNamespace(lookback_depth=2, block_size=16),
        config=SimpleNamespace(
            image_token_id=30,
            image_pad_token_id=31,
            engram_layer_ids=[1, 14],
            engram_max_ngram_size=3,
            engram_n_heads=1,
        ),
    )
    return ElasticEngramRuntime(model, config())


@pytest.fixture
def topology(monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram import runtime as runtime_mod

    cpu_group = object()
    initialized = Mock(return_value=True)
    world_size = Mock(return_value=4)
    world_group = Mock(return_value=SimpleNamespace(cpu_group=cpu_group))
    same_node = Mock(return_value=[True] * 4)
    monkeypatch.setattr(torch.distributed, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.distributed, "is_initialized", initialized)
    monkeypatch.setattr(torch.distributed, "get_world_size", world_size)
    monkeypatch.setattr(runtime_mod, "get_world_group", world_group)
    monkeypatch.setattr(runtime_mod, "in_the_same_node_as", same_node)
    return SimpleNamespace(
        cpu_group=cpu_group,
        initialized=initialized,
        world_size=world_size,
        world_group=world_group,
        same_node=same_node,
    )


@pytest.fixture
def initialization(monkeypatch, topology):
    from vllm_ascend import ascend_config
    from vllm_ascend.models.deepseek_v41.engram import runtime as runtime_mod

    rt = runtime()
    rt.model._engram_overlap_enabled = False
    # The model buffer deliberately differs from the ambient CPU device.
    rt.model.topk_indices_buffer = torch.empty(1, device="meta")
    rt.model.config.engram_num_embeddings = [64, 64]
    rt.model.config.engram_head_dim = 64
    rt.model.engram_layout = SimpleNamespace(primes=[[[31, 33]], [[29, 35]]])
    rt.model.engram_weight_root = "/checkpoint"
    rt.model.engram_rotated = False
    rt.model.layers = [SimpleNamespace(engram=SimpleNamespace()) for _ in range(15)]
    monkeypatch.setattr(
        ascend_config, "get_ascend_config", lambda: SimpleNamespace(engram_config=SimpleNamespace(group_size=4))
    )
    groups = Mock(side_effect=[("group-1", True), ("group-14", True)])
    monkeypatch.setattr(runtime_mod, "create_engram_process_group", groups)
    tables = [Mock(width=64), Mock(width=64)]
    factory = Mock(side_effect=tables)
    monkeypatch.setattr(runtime_mod, "ElasticEngramEmbedding", factory)
    ops = ModuleType("cann_ops_transformer.ops")
    ops.ElasticBuffer = object()
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", ops)
    return rt, factory, tables, groups


def test_initialize_binds_every_table_to_model_device_and_is_idempotent(initialization, topology):
    rt, factory, tables, groups = initialization
    rt.initialize()
    rt.initialize()
    topology.same_node.assert_called_once_with(topology.cpu_group)
    assert factory.call_count == groups.call_count == 2
    assert rt.tables == dict(zip([1, 14], tables))
    for call, group_call, layer_id, table in zip(factory.call_args_list, groups.call_args_list, [1, 14], tables):
        assert group_call.args == (4,) and group_call.kwargs["expected_world_size"] == 4
        assert call.kwargs["device"] == rt.model.topk_indices_buffer.device
        assert call.kwargs["layer_id"] == layer_id and call.kwargs["minimum_rows"] == 64
        table.load_checkpoint.assert_called_once_with("/checkpoint", f"layers.{layer_id}.engram.embed.weight")
        table.offload_weights.assert_called_once()
        assert rt.model.layers[layer_id].engram.embed_tokens is table


def test_initialize_uses_resolved_target_domain_when_draft_parallel_world_is_smaller(initialization):
    rt, factory, tables, groups = initialization
    rt.vllm_config.parallel_config.world_size_across_dp = 1
    rt.initialize()
    assert factory.call_count == 2
    assert all(call.args == (4,) and call.kwargs["expected_world_size"] == 4 for call in groups.call_args_list)


@pytest.mark.parametrize("runtime_world,cpu_world", [(1, 4), (4, 1)])
def test_initialize_rejects_incomplete_worlds_before_allocating_resources(
    initialization, topology, runtime_world, cpu_world
):
    rt, factory, tables, groups = initialization
    topology.world_size.side_effect = lambda group=None: cpu_world if group is topology.cpu_group else runtime_world
    with pytest.raises(RuntimeError, match="span all configured ranks"):
        rt.initialize()
    topology.same_node.assert_not_called()
    groups.assert_not_called()
    factory.assert_not_called()
    assert not rt.tables


def test_initialize_requires_distributed_startup_before_locality_probe(initialization, topology):
    rt, factory, tables, groups = initialization
    topology.initialized.return_value = False
    with pytest.raises(RuntimeError, match="initialized distributed process group"):
        rt.initialize()
    topology.world_group.assert_not_called()
    topology.same_node.assert_not_called()
    groups.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("same_node", [[True, True, False, False], [True, False, False, False], [True]])
def test_initialize_rejects_cross_node_or_isolated_ipc_before_allocating_resources(initialization, topology, same_node):
    rt, factory, tables, groups = initialization
    topology.same_node.return_value = same_node
    with pytest.raises(RuntimeError, match="same node|same-node probe"):
        rt.initialize()
    topology.same_node.assert_called_once_with(topology.cpu_group)
    groups.assert_not_called()
    factory.assert_not_called()
    assert not rt.tables


def test_initialize_failure_closes_tables_before_returning(initialization):
    rt, factory, tables, groups = initialization
    tables[0].offload_weights.side_effect = RuntimeError("fetch probe failed")
    with pytest.raises(RuntimeError, match="fetch probe failed"):
        rt.initialize()
    tables[0].destroy.assert_called_once()
    assert not rt.tables
    assert factory.call_count == groups.call_count == 1


def test_metadata_addresses_survive_reorder_padding_and_idle_steps():
    rt = runtime()
    ids, positions = torch.tensor([7, 8, 9, 0]), torch.tensor([3, 4, 8, 0])
    first = rt.stage(
        ids,
        positions,
        4,
        torch.tensor([[6, 5], [4, 3]]),
        torch.tensor([0, 2, 3]),
        torch.tensor([1, 2, 7, -1]),
        torch.tensor([[4, 5], [1, 2]]),
        cg_mode=CUDAGraphMode.FULL,
        current_dead_mask=torch.tensor([False, True, False, False]),
    )["engram_elastic_inputs"]
    pointers = {key: value.data_ptr() for key, value in first.items()}
    assert first["valid"].tolist() == [3]
    assert first["query"].tolist() == [0, 2, 3, 3]
    assert first["lookback"].tolist() == [[6, 5], [4, 3], [-1, -1]]
    assert first["dead"][:4].tolist() == [False, True, False, False]
    rt.stage(ids, positions, 4, torch.tensor([[9, 8]]), torch.tensor([0, 1]), cg_mode=CUDAGraphMode.FULL)
    assert first["query"].tolist() == [0, 1, 1, 1]
    assert first["lookback"].tolist() == [[9, 8], [-1, -1], [-1, -1]]
    assert not first["dead"].any() and (first["slots"] == -1).all()
    rt.stage(ids, positions, 4, force_dummy=True, cg_mode=CUDAGraphMode.FULL)
    assert first["valid"].item() == 0 and not first["query"].any()
    assert (first["lookback"] == -1).all()
    assert pointers == {key: value.data_ptr() for key, value in rt.graph_inputs(8)["engram_elastic_inputs"].items()}


def test_piecewise_and_capacity_rejected_before_submission():
    rt = runtime()
    rt.executor = Mock()
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        rt.stage(torch.ones(2), torch.ones(2), cg_mode=CUDAGraphMode.PIECEWISE)
    with pytest.raises(ValueError, match="capacity"):
        rt.stage(torch.ones(9), torch.ones(9))
    rt.executor.begin.assert_not_called()


def test_capture_prime_cannot_reuse_real_request_slot_history():
    rt = runtime()
    staged = rt.stage(
        torch.tensor([1, 2]),
        torch.tensor([5, 6]),
        query_start_loc=torch.tensor([0, 2]),
        slot_mapping=torch.tensor([8, 9]),
    )["engram_elastic_inputs"]
    assert staged["valid"].item() == 2
    rt.graph_inputs(4, prime=True)
    assert staged["valid"].item() == 0
    assert (staged["slots"] == -1).all() and (staged["lookback"] == -1).all()


def test_multimodal_spans_mask_current_and_lookback_using_device_positions():
    rt = runtime()
    # Image spans may contain ordinary IDs after embedding preprocessing.
    staged = rt.stage(
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([10, 11, 12, 13]),
        query_start_loc=torch.tensor([0, 2, 4]),
        lookback_token_ids=torch.tensor([[3, 2], [1, 0]]),
        cg_mode=CUDAGraphMode.FULL,
        multimodal_spans=[[(8, 3)], [(11, 3)]],
    )["engram_elastic_inputs"]
    assert staged["dead"][:4].tolist() == [True, False, True, True]
    assert staged["lookback_dead"].tolist() == [[True, True], [True, False], [False, False]]
    rt.stage(torch.ones(1), torch.tensor([20]), query_start_loc=torch.tensor([0, 1]))
    assert not staged["dead"].any() and not staged["lookback_dead"].any()


def test_hash_reuses_authoritative_device_history_and_preserves_barriers():
    rt = runtime()
    calls = []

    class Hash:
        token_map = torch.arange(32)

        def ensure_cache(self):
            return True

        def __call__(self, *args):
            calls.append(args)
            return torch.zeros((args[0].numel(), 2, 2), dtype=torch.int32)

    rt.model.engram_hash = Hash()
    ids = torch.tensor([7, 30, 31, -1, 9, 0])
    lookback = torch.tensor([[29, 31], [8, 7], [-1, -1]])
    query, slots, blocks = torch.tensor([0, 2, 5, 5]), torch.arange(8), torch.zeros((3, 4), dtype=torch.int32)
    dead = torch.tensor([False, False, False, False, True, False, False, False])
    _, mask = rt.forward_packed(ids, torch.arange(6), query, lookback, slots, blocks, torch.tensor([5]), dead)
    assert mask.tolist() == [True, False, False, False, False, False]
    args = calls[0]
    assert args[2] is query and torch.equal(args[4], lookback) and args[7] is blocks
    assert args[6].data_ptr() == slots.data_ptr()
    assert args[0].tolist() == [7, 30, 31, 0, 9, 0]
    assert args[5].tolist() == [[False, True], [False, False], [False, False]]


def test_close_attempts_every_table_and_can_retry_failed_destroy():
    rt = runtime()
    first, second = Mock(), Mock()
    first.destroy.side_effect = [RuntimeError("unregister"), None]
    rt.tables = {1: first, 14: second}
    with pytest.raises(RuntimeError, match="close all"):
        rt.close()
    second.destroy.assert_called_once()
    assert rt.tables == {1: first}
    rt.close()
    assert not rt.tables
