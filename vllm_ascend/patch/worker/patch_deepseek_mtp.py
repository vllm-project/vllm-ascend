import torch
import torch.nn as nn
import vllm
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP, DeepSeekMultiTokenPredictorLayer

MTP_ROT_WEIGHT_NAME = "rot.weight"


def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config | DeepseekV3Config, weight_name: str) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and config.num_nextn_predict_layers > 0:
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}.") or weight_name.startswith(MTP_ROT_WEIGHT_NAME):
                return layer_idx + i
    return None


class AscendDeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__(vllm_config, prefix)
        quant_description = getattr(vllm_config.quant_config, "quant_description", None)
        self.is_rot_used = quant_description.get("is_rot_used", False) if quant_description is not None else False
        self.target_model_type = vllm_config.speculative_config.target_model_config.hf_text_config.model_type
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            previous_hidden_states = self.rot(previous_hidden_states)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class AscendDeepSeekMTP(DeepSeekMTP):
    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if name != MTP_ROT_WEIGHT_NAME:
            return super()._rewrite_spec_layer_name(spec_layer, name)
        else:
            return f"model.layers.{spec_layer}.rot.weight"


vllm.model_executor.models.deepseek_v2.get_spec_layer_idx_from_weight_name = get_spec_layer_idx_from_weight_name
vllm.model_executor.models.deepseek_mtp.get_spec_layer_idx_from_weight_name = get_spec_layer_idx_from_weight_name
vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer = AscendDeepSeekMultiTokenPredictorLayer
vllm.model_executor.models.deepseek_mtp.DeepSeekMTP = AscendDeepSeekMTP
