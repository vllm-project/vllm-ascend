# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.models import qwen3_5


def test_qwen3_5_mtp_uses_local_inputs_on_last_pp_rank():
    predictor = SimpleNamespace()
    predictor.num_mtp_layers = 2
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(2, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x + 1)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x + 2)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4] + x[:, 4:])
    layer0 = MagicMock(return_value=(torch.full((2, 4), 3.0), None))
    layer0.use_attn_reduce_scatter_for_moe = False
    layer1 = MagicMock(return_value=(torch.full((2, 4), 5.0), torch.full((2, 4), 6.0)))
    layer1.use_attn_reduce_scatter_for_moe = False
    predictor.layers = [layer0, layer1]
    predictor.norm = MagicMock(return_value=(torch.full((2, 4), 7.0), None))

    with patch(
        "vllm_ascend.models.qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        output = qwen3_5._forward_local_mtp(
            predictor,
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 4),
            inputs_embeds=None,
            spec_step_idx=3,
        )

    predictor.embed_input_ids.assert_called_once()
    layer1.assert_called_once()
    predictor.norm.assert_called_once()
    assert torch.equal(output, torch.full((2, 4), 7.0))


def test_qwen3_5_mtp_returns_intermediate_tensors_on_non_last_pp_rank():
    predictor = SimpleNamespace()
    predictor.num_mtp_layers = 1
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(1, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])
    layer = MagicMock(return_value=(torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)))
    layer.use_attn_reduce_scatter_for_moe = False
    predictor.layers = [layer]
    predictor.norm = MagicMock()

    with patch(
        "vllm_ascend.models.qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=False),
    ):
        output = qwen3_5._forward_local_mtp(
            predictor,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            hidden_states=torch.zeros(1, 4),
            inputs_embeds=None,
            spec_step_idx=0,
        )

    assert isinstance(output, IntermediateTensors)
    assert torch.equal(output["hidden_states"], torch.full((1, 4), 3.0))
    assert torch.equal(output["residual"], torch.full((1, 4), 4.0))
    predictor.norm.assert_not_called()


def test_qwen3_5_architectures_use_ascend_model_registry():
    from vllm_ascend import models

    with patch.object(models.ModelRegistry, "register_model") as register_model:
        models.register_model()

    registrations = {
        call.args[0]: call.args[1] for call in register_model.call_args_list if call.args[0].startswith("Qwen3_5")
    }
    assert registrations == {
        "Qwen3_5ForCausalLM": ("vllm_ascend.models.qwen3_5:AscendQwen3_5ForCausalLM"),
        "Qwen3_5MoeForCausalLM": ("vllm_ascend.models.qwen3_5:AscendQwen3_5MoeForCausalLM"),
        "Qwen3_5ForConditionalGeneration": ("vllm_ascend.models.qwen3_5:AscendQwen3_5ForConditionalGeneration"),
        "Qwen3_5MoeForConditionalGeneration": ("vllm_ascend.models.qwen3_5:AscendQwen3_5MoeForConditionalGeneration"),
        "Qwen3_5MTP": "vllm_ascend.models.qwen3_5:AscendQwen3_5MTP",
        "Qwen3_5MoeMTP": "vllm_ascend.models.qwen3_5:AscendQwen3_5MoeMTP",
    }
