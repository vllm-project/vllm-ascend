from types import SimpleNamespace

import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend.utils import get_cann_megamoe_buffer_params


def test_dummy_routes_cover_all_experts_across_ep_ranks():
    routed_experts = []
    for ep_rank_id in range(4):
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        active_mask = torch.tensor([1, 0], dtype=torch.int8)

        hidden_states, topk_ids, topk_weights, active_mask, original_num_tokens = _append_cann_megamoe_dummy_tokens(
            hidden_states,
            topk_ids,
            topk_weights,
            active_mask,
            num_experts=8,
            ep_rank_id=ep_rank_id,
            ep_world_size=4,
        )

        assert original_num_tokens == 2
        assert torch.equal(hidden_states[-1], torch.ones(4, dtype=torch.bfloat16))
        assert torch.equal(topk_weights[-1], torch.full((2,), 0.5))
        assert active_mask.tolist() == [1, 0, 1]
        routed_experts.extend(topk_ids[-1].tolist())

    assert sorted(routed_experts) == list(range(8))


def test_receive_bound_uses_documented_worst_case():
    assert get_cann_megamoe_buffer_params(480, 32, 256, 8) == (512, 8, 32, 131072)


def test_a2_mode_2_selects_megamoe(monkeypatch):
    monkeypatch.setattr(afc, "_MEGA_MOE_SUPPORTED", True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=2,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            hidden_size=4096,
            moe_intermediate_size=1536,
            num_experts_per_tok=8,
            quantize="w8a8_dynamic",
        ),
        get_hidden_size=lambda: 4096,
        get_num_experts=lambda: 256,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        lora_config=None,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=32,
            pipeline_parallel_size=1,
        ),
    )

    assert afc.select_moe_comm_method(512, vllm_config) == MoECommType.FUSED_MC2
