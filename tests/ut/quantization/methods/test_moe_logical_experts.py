from types import SimpleNamespace

from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts


def test_get_moe_num_logical_experts_uses_vllm_config_field():
    layer = SimpleNamespace(moe_config=SimpleNamespace(num_logical_experts=128))

    assert get_moe_num_logical_experts(layer, num_experts=130, global_redundant_expert_num=2) == 128


def test_get_moe_num_logical_experts_falls_back_for_older_configs():
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    assert (
        get_moe_num_logical_experts(
            layer,
            num_experts=133,
            global_redundant_expert_num=2,
            num_shared_experts=3,
        )
        == 128
    )
