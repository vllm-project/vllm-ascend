from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.23.0"):
    pytest.skip(
        "Refactored MoERunner EPLB tests require vLLM newer than 0.23.0.",
        allow_module_level=True,
    )


def test_refactored_runner_exposes_eplb_layer_protocol():
    runner = AscendMoERunner.__new__(AscendMoERunner)
    object.__setattr__(
        runner,
        "moe_config",
        SimpleNamespace(num_local_experts=2, ep_rank=1),
    )
    expert_map_manager = SimpleNamespace(_expert_map=None)
    routed_experts = SimpleNamespace(
        expert_map_manager=expert_map_manager,
        w13_weight_list=[torch.ones(1), torch.ones(1)],
    )
    object.__setattr__(runner, "routed_experts", routed_experts)
    object.__setattr__(runner, "_expert_map", None)
    object.__setattr__(runner, "log2phy", torch.tensor([1, 0]))
    object.__setattr__(runner, "moe_load", torch.ones(2, dtype=torch.int64))
    object.__setattr__(runner, "multi_stage", True)
    object.__setattr__(runner, "load_counter", torch.tensor(3))

    assert runner.local_num_experts == 2
    assert runner.ep_rank == 1
    assert runner.get_eplb_parameter("w13_weight_list") is routed_experts.w13_weight_list
    torch.testing.assert_close(runner.get_log2phy_map(), torch.tensor([1, 0]))

    new_expert_map = torch.tensor([0, -1])
    runner.update_expert_map(new_expert_map)
    assert runner._expert_map is new_expert_map
    assert expert_map_manager._expert_map is new_expert_map

    runner.clear_moe_load()
    torch.testing.assert_close(runner.moe_load, torch.zeros(2, dtype=torch.int64))
    assert runner.load_counter.item() == 0
