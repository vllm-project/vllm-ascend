from typing import Dict, Iterable, Tuple
from vllm.model_executor.models.step3_text import Step3TextForCausalLM
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.config import VllmConfig

class CustomStep3TextForCausalLM(Step3TextForCausalLM):
    experts_ = [f"experts.{i}.{proj}" for i in range(48) for proj in ("down_proj", "gate_proj", "up_proj")]
    
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj":[
            "gate_proj",
            "up_proj",
        ],
        "experts": experts_
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        ):
        super().__init__(vllm_config=vllm_config, prefix="model")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        qkv_params_mapping = [
            # (param_name, shard_name, relative_start_idx, relative_end_idx)
            (".qkv_proj", ".q_proj", 0, self.config.share_q_dim /
             (self.config.share_q_dim + self.config.head_dim * 2)),
            (".qkv_proj", ".k_proj", self.config.share_q_dim /
             (self.config.share_q_dim + self.config.head_dim * 2),
             (self.config.share_q_dim + self.config.head_dim) /
             (self.config.share_q_dim + self.config.head_dim * 2)),
            (".qkv_proj", ".v_proj",
             (self.config.share_q_dim + self.config.head_dim) /
             (self.config.share_q_dim + self.config.head_dim * 2),
             (self.config.share_q_dim + self.config.head_dim * 2) /
             (self.config.share_q_dim + self.config.head_dim * 2)),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        if self.vllm_config.quant_config is not None:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name="gate_proj",
                    ckpt_down_proj_name="down_proj",
                    ckpt_up_proj_name="up_proj",
                    num_experts=self.model.config.moe_num_experts)
            is_fused_moe = False
        else:
            expert_params_mapping = [
                (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2")
            ]
            is_fused_moe = True

        disable_moe_stacked_params = [
            data[1] for data in expert_params_mapping
        ]

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(disable_moe_stacked_param in name
                       for disable_moe_stacked_param in
                       disable_moe_stacked_params):
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    if len(mapping) == 4:
                        param_name, weight_name, expert_id, shard_id = mapping
                    else:
                        param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if is_fused_moe:
                        for expert_id in range(loaded_weight.shape[0]):
                            loaded_weight_expert = loaded_weight[expert_id]
                            weight_loader(param,
                                loaded_weight_expert,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id)
                    else:
                        weight_loader(param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id)
                    loaded_params.add(name)
                    break
                else:
                    for (param_name, weight_name, start_idx,
                         end_idx) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        if hasattr(param, "output_dim"):
                            dim = param.shape[param.output_dim]
                            begin_idx = int(start_idx * dim)
                            end_idx = int(end_idx * dim)
                            param_slice = param.narrow(param.output_dim, begin_idx,
                                                    end_idx-begin_idx)
                            param_slice.copy_(loaded_weight)
                        else:
                            param.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
        return loaded_params
