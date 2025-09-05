# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend NPU优化的LongCat Flash模型，通过继承vLLM实现来减少代码重复。"""
from typing import Optional, Union, Iterable, Callable
import typing
import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    block_dequant)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.distributed import get_pp_group
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import (
    IntermediateTensors, PPMissingLayer, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix, is_pp_missing_parameter
)

# 继承原始的vLLM LongCat Flash实现
from vllm.model_executor.models.longcat_flash import (
    FlashConfig,
    LongcatMoe,
    LongcatRouter,
    FlashDecoderLayer,
    FlashModel,
    LongcatFlashForCausalLM
)

# 导入Ascend特定的实现
from vllm_ascend.models.deepseek_v2 import CustomDeepseekV2MLAAttention

logger = init_logger(__name__)


class CustomLongcatMoe(nn.Module):
    """Ascend NPU优化的LongcatMoe，使用AscendFusedMoE支持LongCat Flash零专家功能。
    """
    
    def __init__(
        self,
        config,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype = None,
        quant_config = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type
        self.routed_scaling_factor = config.routed_scaling_factor
        self.enable_eplb = enable_eplb
        # Gate always runs at half / full precision for now.
        self.rounter_params_dtype = params_dtype
        if config.router_dtype == "float32":
            self.rounter_params_dtype = torch.float32

        # 使用原始LongcatRouter
        self.router = LongcatRouter(
            config=config,
            zero_expert_num=self.zero_expert_num,
            rounter_params_dtype=self.rounter_params_dtype,
            prefix=f"{prefix}.gate")

        # 关键：使用vllm_ascend的AscendFusedMoE替代原始FusedMoE
        from vllm_ascend.ops.fused_moe import AscendFusedMoE
        self.experts = AscendFusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            params_dtype=params_dtype,
            e_score_correction_bias=self.router.e_score_correction_bias,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            zero_expert_num=self.zero_expert_num,
            zero_expert_type=self.zero_expert_type,
            enable_eplb=self.enable_eplb,
            routed_scaling_factor=config.routed_scaling_factor,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.router(hidden_states.to(
            self.rounter_params_dtype))
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)

        return final_hidden_states.view(num_tokens, hidden_dim)


class CustomFlashDecoderLayer(FlashDecoderLayer):
    """Ascend优化的Flash decoder layer，使用CustomDeepseekV2MLAAttention和CustomLongcatMoe"""

    def __init__(
        self,
        config: FlashConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        # 不调用父类初始化，直接初始化nn.Module
        nn.Module.__init__(self)
        self.layer_idx = int(prefix.split(sep='.')[-1])
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)

        # 动态添加缺失的配置属性，确保与CustomDeepseekV2MLAAttention兼容
        if not hasattr(config, 'first_k_dense_replace'):
            config.first_k_dense_replace = 0
        if not hasattr(config, 'moe_layer_freq'):
            config.moe_layer_freq = 1

        # Dual attention structure - 关键修改：使用CustomDeepseekV2MLAAttention
        self.self_attn = nn.ModuleList([
            CustomDeepseekV2MLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(config.q_lora_rank if hasattr(
                    config, "q_lora_rank") else None),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=None if "self_attn" in getattr(
                    config, "disable_quant_module", []) else quant_config,
                prefix=f"{prefix}.self_attn_{i}",
            ) for i in range(2)
        ])
        
        # Dual layernorm structure - 与父类保持一致
        self.input_layernorm = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for i in range(2)
        ])
        self.post_attention_layernorm = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for i in range(2)
        ])

        # Dual MLP structure - 与父类保持一致，导入FlashMLP
        from vllm.model_executor.models.longcat_flash import FlashMLP
        self.mlps = nn.ModuleList([
            FlashMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=None if "mlps" in getattr(
                    config, "disable_quant_module", []) else quant_config,
                prefix=f"{prefix}.mlps.{i}",
            ) for i in range(2)
        ])

        # MoE层 - 关键修改：使用CustomLongcatMoe替代LongcatMoe
        self.mlp = CustomLongcatMoe(
            config=config,
            num_experts=config.n_routed_experts if hasattr(
                config, "n_routed_experts") else
            config.num_experts[self.layer_idx],
            top_k=config.moe_topk
            if hasattr(config, "moe_topk") else config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=(f"{prefix}.mlp"),
        )

    # CustomFlashDecoderLayer继承父类forward方法，无需重复实现


class CustomFlashModel(FlashModel):
    """Ascend优化的Flash模型，使用CustomFlashDecoderLayer"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 不调用父类初始化，直接初始化nn.Module以避免原始FlashDecoderLayer的创建
        nn.Module.__init__(self)
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        # Pipeline Parallel支持：只在第一个rank创建embed_tokens
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        
        # 创建layers，使用CustomFlashDecoderLayer替代FlashDecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomFlashDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers")
        
        # Pipeline Parallel支持：只在最后一个rank创建norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        
        # 使用与父类相同的make_empty_intermediate_tensors工厂函数
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
    
    # CustomFlashModel继承所有父类方法，无需重复实现


class CustomLongcatFlashForCausalLM(LongcatFlashForCausalLM):
    """Ascend优化的LongCat Flash因果语言模型"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 不调用父类初始化，直接初始化nn.Module
        nn.Module.__init__(self)
        
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        config.intermediate_size = config.ffn_hidden_size if hasattr(
            config, "ffn_hidden_size") else config.intermediate_size
        self.lora_config = lora_config
        self.quant_config = quant_config

        # 使用我们的CustomFlashModel
        from vllm.model_executor.models.utils import maybe_prefix
        self.model = CustomFlashModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))

        # 初始化其他组件
        from vllm.distributed import get_pp_group
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.models.utils import PPMissingLayer
        
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        print("[DEBUG] Starting load_weights process")

        stacked_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        print(f"[DEBUG] Stacked params mapping: {stacked_params_mapping}")

        expert_params_mapping = self.get_expert_mapping()
        print(f"[DEBUG] Expert params mapping count: {len(expert_params_mapping)}")
        print(f"[DEBUG] Expert params mapping: {expert_params_mapping}")
        loaded_params: set[str] = set()

        params_dict = dict(self.named_parameters())
        print(f"[DEBUG] Total parameters in model: {len(params_dict)}")
        for name, loaded_weight in weights:
            print(f"[DEBUG] Processing weight: {name}, shape: {loaded_weight.shape}")
            if "rotary_emb.inv_freq" in name:
                print(f"[DEBUG] Skipping rotary_emb.inv_freq: {name}")
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp" in name and "mlps" not in name:
                    print(f"[DEBUG] Skipping mlp weight (not mlps): {name}")
                    continue
                name = name.replace(weight_name, param_name)
                print(f"[DEBUG] Mapping {weight_name} -> {param_name}, new name: {name}")
                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if ((param_name == "fused_qkv_a_proj")
                        and name not in params_dict):
                    print(f"[DEBUG] QKV fusion not enabled, skipping: {name}")
                    continue
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias")
                        or name.endswith("_bias")) and name not in params_dict:
                    print(f"[DEBUG] Skipping extra bias: {name}")
                    continue
                # Skip mtp
                if ".mtp." in name:
                    print(f"[DEBUG] Skipping mtp: {name}")
                    continue
                if is_pp_missing_parameter(name, self):
                    print(f"[DEBUG] Skipping PP missing parameter: {name}")
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                print(f"[DEBUG] Loading stacked param: {name}, shard_id: {shard_id}")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                print(f"[DEBUG] Processing as non-stacked weight: {name}")
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    print(f"[DEBUG] Found expert weight: {name}, expert_id: {expert_id}")
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    print(f"[DEBUG] Expert weight mapping: {name} -> {name_mapped}")
                    # Skip mtp
                    if ".mtp." in name_mapped:
                        print(f"[DEBUG] Skipping expert mtp: {name_mapped}")
                        continue
                    if (name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name not in params_dict:
                        print(f"[DEBUG] Skipping expert bias: {name_mapped}")
                        continue
                    if is_pp_missing_parameter(name, self):
                        print(f"[DEBUG] Skipping expert PP missing: {name_mapped}")
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
                    print(f"[DEBUG] Loading expert weight: {name_mapped}, expert_id: {expert_id}, shard_id: {shard_id}")
                    success = weight_loader(param,
                                            loaded_weight,
                                            name_mapped,
                                            shard_id=shard_id,
                                            expert_id=expert_id,
                                            return_success=True)
                    if success:
                        print(f"[DEBUG] Successfully loaded expert weight: {name_mapped}")
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        print(f"[DEBUG] Skipping expert weight not mapped to this rank: {name}")
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        print(f"[DEBUG] Skipping extra bias: {name}")
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        print(f"[DEBUG] Skipping kv_scale: {name}")
                        continue
                    # Skip mtp
                    if ".mtp." in name:
                        print(f"[DEBUG] Skipping mtp: {name}")
                        continue
                    if name is None:
                        print(f"[DEBUG] Skipping None name")
                        continue
                    if is_pp_missing_parameter(name, self):
                        print(f"[DEBUG] Skipping PP missing parameter: {name}")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    print(f"[DEBUG] Loading regular weight: {name}")
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
            print(f"[DEBUG] Added to loaded_params: {name}")
        print(f"[DEBUG] Starting post-processing for {self.config.num_hidden_layers} layers")
        for layer_id in range(self.config.num_hidden_layers):
            for i in range(2):
                if isinstance(self.model.layers[layer_id], PPMissingLayer):
                    print(f"[DEBUG] Skipping PP missing layer {layer_id}")
                    continue
                print(f"[DEBUG] Post-processing layer {layer_id}, attention {i}")
                self_attn = self.model.layers[layer_id].self_attn[i]
                if hasattr(self.quant_config, "weight_block_size"
                           ) and self_attn.kv_b_proj.weight.dtype in (
                               torch.float8_e4m3fn,
                               torch.float8_e4m3fnuz,
                           ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        dtype = torch.get_default_dtype()
                        w = block_dequant(self_attn.kv_b_proj.weight,
                                          self_attn.kv_b_proj.weight_scale_inv,
                                          weight_block_size).to(dtype)
                else:
                    w = self_attn.kv_b_proj.weight

                w_kc, w_vc = w.unflatten(
                    0,
                    (-1,
                     self_attn.qk_nope_head_dim + self_attn.v_head_dim)).split(
                         [self_attn.qk_nope_head_dim, self_attn.v_head_dim],
                         dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(
                    1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank)**0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size /
                        self.config.kv_lora_rank)**0.5
        print(f"[DEBUG] Load weights completed, total loaded: {len(loaded_params)}")
        return loaded_params