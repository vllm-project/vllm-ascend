# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
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
