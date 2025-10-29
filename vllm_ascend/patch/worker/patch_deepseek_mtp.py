import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.deepseek_mtp import \
    DeepSeekMultiTokenPredictorLayer
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.11.0"):
    from collections.abc import Iterable

    from vllm.compilation.decorators import support_torch_compile
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader, maybe_remap_kv_scale_name)
    from vllm.model_executor.models.deepseek_mtp import (
        DeepSeekMTP, DeepSeekMultiTokenPredictor)
    from vllm.model_executor.models.deepseek_v2 import \
        get_spec_layer_idx_from_weight_name
    from vllm.model_executor.models.interfaces import SupportsPP
    from vllm.sequence import IntermediateTensors


class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def predictor_init(self, vllm_config: VllmConfig, prefix: str) -> None:
    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config

    self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.eh_proj = nn.Linear(config.hidden_size * 2,
                             config.hidden_size,
                             bias=False)
    # We don't need topk_indices_buffer in Ascend
    topk_indices_buffer = None
    self.shared_head = SharedHead(config=config,
                                  prefix=prefix,
                                  quant_config=quant_config)
    self.mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix,
                                            topk_indices_buffer)


def predictor_forward(
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
    previous_hidden_states = self.hnorm(previous_hidden_states)

    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
    )

    hidden_states, residual = self.mtp_block(positions=positions,
                                             hidden_states=hidden_states,
                                             residual=None)
    hidden_states = residual + hidden_states
    return hidden_states


@support_torch_compile
class CustomDeepSeekMTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepSeekMultiTokenPredictor(vllm_config=vllm_config,
                                                 prefix=maybe_prefix(
                                                     prefix, "model"))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, hidden_states,
                                   inputs_embeds, spec_step_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    # According to DeepSeek-V3 Technical Report, MTP modules
                    # shares embedding layer. We only load the first weights.
                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


DeepSeekMultiTokenPredictorLayer.__init__ = predictor_init
DeepSeekMultiTokenPredictorLayer.forward = predictor_forward
if vllm_version_is("0.11.0"):
    DeepSeekMTP = CustomDeepSeekMTP
