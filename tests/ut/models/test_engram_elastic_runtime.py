# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.models.deepseek_v41.engram.config import validate_elastic_config
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


def ascend(**kwargs):
    return SimpleNamespace(
        **{"enable_engram": True, "engram_offload_backend": "elastic", "engram_tp_size": None} | kwargs
    )


def test_backend_validation_does_not_restrict_existing_storage():
    validate_elastic_config(ascend(enable_engram=False), SimpleNamespace())
    validate_elastic_config(ascend(engram_offload_backend="uva"), SimpleNamespace())
    backend = ascend()
    validate_elastic_config(backend, config())
    assert backend.engram_tp_size == 4


@pytest.mark.parametrize("size", [0, 1, 2, 3, 8])
def test_elastic_rejects_invalid_or_redundant_local_shards(size):
    with pytest.raises(ValueError, match="engram_tp_size|node-local"):
        validate_elastic_config(ascend(engram_tp_size=size), config())


@pytest.mark.parametrize(
    "field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_elastic_parallel_restrictions_are_scoped_to_backend(field):
    vc = config()
    setattr(vc.parallel_config, field, 2)
    with pytest.raises(ValueError, match="PP=PCP=DCP"):
        validate_elastic_config(ascend(), vc)
    validate_elastic_config(ascend(engram_offload_backend="uva"), vc)


@pytest.mark.parametrize("case", ["offload", "sharing", "multinode", "dummy", "mtp", "other"])
def test_elastic_configuration_errors_are_early(case):
    vc = config()
    if case == "offload":
        vc.engram_config.cpu_offload = False
    elif case == "sharing":
        vc.engram_config.dp_shared_memory = True
    elif case == "multinode":
        vc.parallel_config.local_world_size = 2
    elif case == "dummy":
        vc.load_config.load_format = "dummy"
    else:
        vc.speculative_config = SimpleNamespace(method=case)
    with pytest.raises(ValueError):
        validate_elastic_config(ascend(), vc)


@pytest.mark.parametrize("enabled,backend", [(False, "uva"), (False, "elastic"), (True, "uva")])
@pytest.mark.parametrize("method,tokens", [("mtp", 3), ("dspark", 32), ("eagle3", 3)])
def test_speculative_restrictions_only_apply_to_enabled_elastic(enabled, backend, method, tokens):
    vc = config()
    vc.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=tokens, use_dspark=lambda: True)
    validate_elastic_config(ascend(enable_engram=enabled, engram_offload_backend=backend), vc)


@pytest.mark.parametrize("tokens,valid", [(None, False), (0, False), (1, True), (31, True), (32, False)])
def test_elastic_dspark_retains_ring_capacity_check(tokens, valid):
    vc = config()
    vc.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens)
    if valid:
        validate_elastic_config(ascend(), vc)
    else:
        with pytest.raises(ValueError, match="1..31"):
            validate_elastic_config(ascend(), vc)


@pytest.mark.parametrize("schedule", [[(1, 16, 3), (17, 32, 0)], {1: 3, 17: 0}])
def test_elastic_dspark_dynamic_schedule_supports_zero_and_checks_capacity(schedule):
    vc = config()
    vc.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=3, num_speculative_tokens_per_batch_size=schedule
    )
    validate_elastic_config(ascend(), vc)
    for invalid in (-1, 32):
        vc.speculative_config.num_speculative_tokens_per_batch_size = {1: invalid}
        with pytest.raises(ValueError, match="dynamic lengths"):
            validate_elastic_config(ascend(), vc)


def test_elastic_dspark_rejects_inconsistent_method_report():
    vc = config()
    vc.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=3, use_dspark=lambda: False)
    with pytest.raises(ValueError, match="must report"):
        validate_elastic_config(ascend(), vc)


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_elastic_requires_explicit_dspark_method_without_overriding_proposer(method):
    vc = config()
    vc.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=3, use_dspark=lambda: True)
    with pytest.raises(ValueError, match="only method='dspark'"):
        validate_elastic_config(ascend(), vc)


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
