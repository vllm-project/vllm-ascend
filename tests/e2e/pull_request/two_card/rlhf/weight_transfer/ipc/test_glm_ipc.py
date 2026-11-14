# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""GLM IPC weight-transfer coverage, scheduled and timed independently."""

import pytest
import torch
from transformers import AutoModelForCausalLM, Glm5NextTextConfig, Glm5NextTextModel, PreTrainedModel
from vllm import envs

from tests.e2e.common.rlhf.weight_transfer.case import (
    GLM_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)


class _Glm53FlashMetaForCausalLM(PreTrainedModel):
    """Checkpoint-shape adapter; never used for inference or pretrained loading."""

    config_class = Glm5NextTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm5NextTextModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        for layer in self.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "conv1d"):
                # HF fuses the KDA convolution; checkpoint loaders expect q/k/v.
                for projection, weight in zip(("q", "k", "v"), attn.conv1d.weight.chunk(3, dim=0)):
                    conv = torch.nn.Module()
                    conv.register_parameter("weight", torch.nn.Parameter(weight))
                    setattr(attn, f"{projection}_conv1d", conv)
                del attn.conv1d
            if hasattr(layer.mlp, "gate"):
                # HF stores this learned checkpoint tensor as a floating buffer.
                # Expose it to the shared source's parameter enumeration.
                gate = layer.mlp.gate
                bias = gate.e_score_correction_bias
                del gate.e_score_correction_bias
                gate.register_parameter("e_score_correction_bias", torch.nn.Parameter(bias))


# Transformers 5.19 has a Flash text backbone but no AutoModelForCausalLM entry.
# The checkpoint mapper is pickled with the case, so trainer peers import this
# module and register the same meta-only adapter before constructing the source.
AutoModelForCausalLM.register(Glm5NextTextConfig, _Glm53FlashMetaForCausalLM, exist_ok=True)


def _glm53_flash_checkpoint_name(name: str) -> str:
    for source, target in (
        (".attn_hc.", ".hc_attn_"),
        (".ffn_hc.", ".hc_ffn_"),
        (".forget_gate.", "."),
    ):
        name = name.replace(source, target)
    return name


GLM_IPC_CASES = (
    *GLM_CASES,
    WeightUpdateModelCase(
        id="glm-5.3-flash-kda-sfa-mhc",
        # ModelScope's zai-org mirror lacks the tokenizer files.
        model="ZhipuAI/GLM-5.3-Flash" if envs.VLLM_USE_MODELSCOPE else "zai-org/GLM-5.3-Flash",
        hf_overrides={
            "architectures": ["Glm5NextForCausalLM"],
            # The generated payload is BF16, irrespective of the published FP8 config.
            "quantization_config": None,
            "text_config": {
                "num_hidden_layers": 4,
                "n_routed_experts": 8,
                "num_nextn_predict_layers": 0,
                "layer_types": ["linear_attention"] * 3 + ["indexed_attention"],
                "mlp_layer_types": ["dense"] * 3 + ["sparse"],
                "indexer_types": ["full"] * 4,
            },
        },
        meta_config_attribute="text_config",
        checkpoint_name_map=_glm53_flash_checkpoint_name,
        extra_server_args=(
            "--block-size",
            "128",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "512",
            "--no-enable-prefix-caching",
        ),
    ),
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=GLM_IPC_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
