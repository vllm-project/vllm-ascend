# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.request import LoRARequest

from vllm_ascend.lora.moe_manager import AscendMoELoRAModelManager, AscendMoEWorkerLoRAManager


def make_manager(ep_rank=None):
    module = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(module)
    module.moe_config = SimpleNamespace(
        hidden_dim=4,
        intermediate_size_per_partition=6,
        num_experts=4,
        num_local_experts=4 if ep_rank is None else 2,
        moe_parallel_config=SimpleNamespace(use_ep=ep_rank is not None, ep_rank=ep_rank or 0),
    )
    module.tp_size = 1
    manager = AscendMoELoRAModelManager.__new__(AscendMoELoRAModelManager)
    manager.modules = {"model.layers.0.mlp.experts": module}
    manager.packed_modules = {
        name: [f"{name}.{expert}.{projection}" for expert in range(4) for projection in ("w1", "w2", "w3")]
        for name in manager.modules
    }
    manager._enable_moe_shared_loras = manager._enable_mixed_moe_lora_format = False
    manager.is_pooling_model = manager._is_non_gated_moe = manager.supports_mm = False
    return manager


@pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
@pytest.mark.parametrize("ep_rank", [None, 0, 1])
def test_partial_experts_pack_with_original_scaling_and_zero_missing_targets(projection, ep_rank):
    manager = make_manager(ep_rank)
    name = next(iter(manager.modules))
    expert = 3 if ep_rank == 1 else 0
    inputs, outputs = (6, 4) if projection == "w2" else (4, 6)
    target = f"{name}.{expert}.{projection}"
    adapter = LoRAModel(1, 2, {target: LoRALayerWeights(target, 2, 6, torch.ones(2, inputs), torch.ones(outputs, 2))})
    manager._create_merged_loras_inplace(adapter)
    assert list(adapter.loras) == [name]
    packed = adapter.loras[name]
    count = 4 if ep_rank is None else 2
    local_expert = expert % count
    for index, kind in enumerate(("w1", "w2", "w3")):
        a, b = packed.lora_a[index], packed.lora_b[index]
        assert a.shape[0] == b.shape[0] == count
        delta = b @ a
        expected = torch.zeros_like(delta)
        if kind == projection:
            expected[local_expert].fill_(6)
        assert torch.equal(delta, expected)


def test_half_pair_is_rejected_before_expert_packing():
    manager = make_manager()
    name = next(iter(manager.modules)) + ".0.w1"
    adapter = LoRAModel(1, 2, {name: LoRALayerWeights(name, 2, 2, torch.ones(2, 4), None)})
    with pytest.raises(ValueError, match="Incomplete LoRA A/B pair"):
        manager._create_merged_loras_inplace(adapter)


def test_ep_rank_without_owned_adapter_targets_keeps_empty_adapter():
    manager = make_manager(ep_rank=1)
    adapter = LoRAModel(1, 2, {})
    manager._create_merged_loras_inplace(adapter)
    assert adapter.loras == {}


def dp_worker(monkeypatch, gathered):
    import vllm_ascend.lora.moe_manager as module

    worker = AscendMoEWorkerLoRAManager.__new__(AscendMoEWorkerLoRAManager)
    manager = SimpleNamespace(lora_slots=2, lora_index_to_id=[2, 1], deactivate_adapter=Mock())
    worker._adapter_manager = manager
    worker.add_adapter = Mock()
    monkeypatch.setattr(module, "get_dp_group", lambda: SimpleNamespace(world_size=2, cpu_group=object()))
    monkeypatch.setattr(
        module.dist, "all_gather_object", lambda output, value, group: output.__setitem__(slice(None), gathered)
    )
    return worker


def test_dp_receivers_activate_union_in_canonical_slots(monkeypatch):
    a, b = LoRARequest("a", 1, "/a"), LoRARequest("b", 2, "/b")
    worker = dp_worker(monkeypatch, [[b], [a]])
    worker.synchronize_requests({b})
    assert worker._adapter_manager.deactivate_adapter.call_count == 2
    assert [call.args[0] for call in worker.add_adapter.call_args_list] == [a, b]


def test_dp_inplace_reload_flag_is_kept_from_any_rank(monkeypatch):
    plain = LoRARequest("a", 1, "/a")
    reload = LoRARequest("a", 1, "/a", load_inplace=True)
    worker = dp_worker(monkeypatch, [[reload], [plain]])
    worker.synchronize_requests({plain})
    assert worker.add_adapter.call_args.args[0].load_inplace


@pytest.mark.parametrize(
    "dp,expert_lora,blocked", [(1, False, False), (1, True, False), (2, False, False), (2, True, True)]
)
def test_pin_policy_uses_wrapped_experts_after_manager_creation(monkeypatch, dp, expert_lora, blocked):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            get_vocab_size=lambda: 8,
            hf_config=SimpleNamespace(get_text_config=lambda: SimpleNamespace()),
            is_encoder_decoder=False,
        ),
        lora_config=object(),
    )
    worker = AscendMoEWorkerLoRAManager(config, torch.device("cpu"), {})
    modules = make_manager().modules if expert_lora else {"qkv": torch.nn.Linear(4, 4)}
    manager = SimpleNamespace(modules=modules, pin_adapter=Mock(return_value=True))

    def create(instance, model, vllm_config):
        instance._adapter_manager = manager
        return model

    monkeypatch.setattr("vllm.lora.worker_manager.LRUCacheWorkerLoRAManager.create_lora_manager", create)
    model = torch.nn.Module()
    assert worker.create_lora_manager(model, config) is model
    if blocked:
        with pytest.raises(ValueError, match="pinning across DP"):
            worker.pin_adapter(1)
        manager.pin_adapter.assert_not_called()
    else:
        assert worker.pin_adapter(1)
        manager.pin_adapter.assert_called_once_with(1)


@pytest.mark.parametrize(
    "requests,match",
    [
        ([[LoRARequest("a", 1, "/a")], [LoRARequest("a", 1, "/other")]], "Conflicting"),
        ([[LoRARequest("a", 1, "/a")], [LoRARequest("b", 2, "/b"), LoRARequest("c", 3, "/c")]], "exceeding max_loras"),
    ],
)
def test_dp_invalid_union_does_not_mutate_adapter_slots(monkeypatch, requests, match):
    worker = dp_worker(monkeypatch, requests)
    with pytest.raises(ValueError, match=match):
        worker.synchronize_requests(set())
    worker._adapter_manager.deactivate_adapter.assert_not_called()
    worker.add_adapter.assert_not_called()
