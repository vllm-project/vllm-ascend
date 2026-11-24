# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_forward_context import MOE_PRE_SP_PADDING_MASK, MoECommType
from vllm_ascend.models.deepseek_v41 import model as model_module


@pytest.mark.parametrize(
    "section,option,value",
    [
        (None, None, None),
        ("env", "VLLM_MOE_SKIP_PADDING", False),
        ("runner", "v2", False),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "enable_expert_parallel", False),
        ("parallel", "expert_placement_strategy", "round_robin"),
        ("parallel", "enable_eplb", True),
        ("ascend", "mix_placement", True),
        ("ascend", "enable_force_eplb", True),
        ("communication", "method", MoECommType.MC2),
        ("communication", "method", MoECommType.FUSED_MC2),
        ("communication", "method", MoECommType.ALLTOALL),
    ],
)
@pytest.mark.parametrize(
    "dp_size,pcp_size,tp_size,supported",
    [(1, 1, 1, False), (1, 1, 8, True), (2, 1, 1, True), (2, 1, 4, True), (1, 2, 4, False), (2, 2, 2, False)],
)
def test_skip_moe_padding_requires_supported_layout(
    monkeypatch, section, option, value, dp_size, pcp_size, tp_size, supported
):
    parallel = SimpleNamespace(
        data_parallel_size=dp_size,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        tensor_parallel_size=tp_size,
        expert_placement_strategy="linear",
        enable_eplb=False,
    )
    ascend = SimpleNamespace(mix_placement=False, enable_force_eplb=False)
    if section in ("parallel", "ascend"):
        setattr({"parallel": parallel, "ascend": ascend}[section], option, value)
    config = SimpleNamespace(parallel_config=parallel, speculative_config=None)
    ep_size = dp_size * pcp_size * tp_size
    monkeypatch.setattr(model_module, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", section != "env")
    monkeypatch.setattr(model_module, "use_v2_model_runner", lambda config: section != "runner")
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(
        model_module,
        "select_moe_comm_method",
        lambda num_tokens, config: value if section == "communication" else MoECommType.ALLGATHER,
    )
    enabled = section is None and supported
    assert model_module._can_skip_padding_tokens_in_moe(config) is enabled
    if enabled:
        config.speculative_config = SimpleNamespace(enable_adaptive_verification=True)
        assert model_module._can_skip_padding_tokens_in_moe(config)


@pytest.mark.parametrize("skip", [False, True])
def test_sp_mask_seen_by_first_layer(monkeypatch, skip):
    from vllm_ascend.models.common.ops import sequence_parallel as sp_module

    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_rank", lambda: 7)
    monkeypatch.setattr(torch.ops.vllm, "ascend_sp_padding_mask_impl", sp_module._ascend_sp_padding_mask_impl)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(model_module, "sp_shard", lambda rows: rows.chunk(8)[7])
    monkeypatch.setattr(model_module, "sp_all_gather", lambda rows: torch.cat([rows] * 8))

    class Layer:
        layer_idx = 0
        engram = None

        def __call__(self, positions, hidden, pre_mix, unused, input_ids):
            assert hidden.shape[0] == 2
            torch.testing.assert_close(context.is_padding, padding[14:])
            if skip:
                assert context.additional_kwargs[MOE_PRE_SP_PADDING_MASK] is padding
            else:
                assert MOE_PRE_SP_PADDING_MASK not in context.additional_kwargs
            return hidden, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

    class Backbone(SimpleNamespace):
        def __call__(self, *args, **kwargs):
            if skip:
                assert context.additional_kwargs[MOE_PRE_SP_PADDING_MASK] is context.is_padding
            return model_module.DeepseekV41Model.forward(self, *args, **kwargs)

    model = Backbone(
        use_sequence_parallel=True,
        skip_padding_tokens_in_moe=skip,
        layers=[Layer()],
        start_layer=0,
        end_layer=1,
        decoder_replay_layers=None,
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        aux_hidden_state_layers=(),
        hc_mult=1,
        _mtp_hidden_buffer=None,
        norm=lambda x: x,
    )
    wrapper = SimpleNamespace(model=model)
    rows = torch.arange(16)
    for valid in (15, 6):
        padding = rows >= valid
        context = SimpleNamespace(is_padding=padding, additional_kwargs={})
        model_module.AscendDeepseekV41LLMForCausalLM.forward(
            wrapper, rows, rows, inputs_embeds=rows[:, None].float(), engram_lookups={}, engram_mask=padding
        )
