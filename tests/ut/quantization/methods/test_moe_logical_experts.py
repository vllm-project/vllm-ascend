from types import SimpleNamespace

from vllm_ascend.quantization.methods.base import AscendLinearScheme, AscendMoEScheme, get_moe_num_logical_experts


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


def test_routed_moe_interface_is_not_exposed_on_linear_schemes():
    assert "get_eplb_weight_views" not in AscendLinearScheme.__dict__
    assert "get_eplb_weight_views" in AscendMoEScheme.__dict__
