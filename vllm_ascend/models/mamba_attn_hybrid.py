# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.config import VllmConfig
from vllm.model_executor.models.mamba_attn_hybrid import MambaAttnHybridForCausalLM

from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM, align_draft_weights


class AscendMambaAttnHybridForCausalLM(MambaAttnHybridForCausalLM):
    """Keep H-Spec's FP32 recurrent state and align target auxiliary features."""

    configure_target_aux_hidden_capture = AscendQwen3DSparkForCausalLM.configure_target_aux_hidden_capture

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers if hasattr(layer, "self_attn")]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers if hasattr(layer, "self_attn")]
