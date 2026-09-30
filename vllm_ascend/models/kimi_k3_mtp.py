# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MTP draft model for Ascend."""

import copy
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_pp_missing_layer_names,
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MTP as UpstreamKimiK3MTP,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MultiTokenPredictor as UpstreamKimiK3MultiTokenPredictor,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MultiTokenPredictorLayer as UpstreamKimiK3MultiTokenPredictorLayer,
)
from vllm.models.kimi_k3.amd.mtp import SharedHead

from vllm_ascend.models.kimi_k3 import (
    AscendKimiDecoderLayer,
    AscendKimiMoE,
    KimiMixtureOfExperts,
)


class AscendKimiK3MultiTokenPredictorLayer(
    UpstreamKimiK3MultiTokenPredictorLayer,
):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        # The upstream constructor hard-codes the AMD decoder layer.  Build the
        # same container with the Ascend decoder and inherit its forward path.
        nn.Module.__init__(self)
        self.config = config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=vllm_config.quant_config,
        )
        block_config = copy.copy(config)
        block_config.attn_res_block_size = None
        self.mtp_block = AscendKimiDecoderLayer(
            block_config,
            vllm_config,
            prefix=prefix,
        )


class AscendKimiK3MultiTokenPredictor(UpstreamKimiK3MultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The upstream constructor hard-codes its predictor-layer class.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): AscendKimiK3MultiTokenPredictorLayer(
                    config,
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


class AscendKimiK3MTP(UpstreamKimiK3MTP, KimiMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.vllm_config = vllm_config
        self.model = AscendKimiK3MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        # Physical expert slots appended by EPLB; load_weights needs the count
        # so the duplicated initial expert placements receive weights.
        self.n_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
            if vllm_config.parallel_config.enable_eplb
            else 0
        )
        # Set MoE hyperparameters for the EPLB registration.
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "num_expert_group", None) or 1
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, AscendKimiK3MultiTokenPredictorLayer)
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, AscendKimiMoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Mirror upstream ``KimiK3MTP.load_weights`` with one addition:
        ``n_redundant_experts`` is propagated into the expert mapping so the
        duplicated initial expert placements receive weights when EPLB is
        enabled.
        """
        # Mirror KimiLinearForCausalLM.load_weights naming: leading-dot shard
        # names, q_lora-conditional fused QKV, and w1/w2/w3 expert weights.
        kda_config = self.config.linear_attn_config
        use_full_rank_gate = bool(kda_config and kda_config.get("use_full_rank_gate", False))
        beta_shard_id = 5 if use_full_rank_gate else 3
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".in_proj_qkvgfab", ".q_proj", 0),
            (".in_proj_qkvgfab", ".k_proj", 1),
            (".in_proj_qkvgfab", ".v_proj", 2),
            (".in_proj_qkvgfab", ".b_proj", beta_shard_id),
            (".in_proj_qkvgfab", ".f_a_proj", 4),
            (".conv1d", ".q_conv1d", 0),
            (".conv1d", ".k_conv1d", 1),
            (".conv1d", ".v_conv1d", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if use_full_rank_gate:
            stacked_params_mapping.append((".in_proj_qkvgfab", ".g_proj", 3))
        if getattr(self.config, "q_lora_rank", None) is not None:
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]

        expert_params_mapping = (
            fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.n_redundant_experts,
            )
            if self.config.is_moe
            else []
        )

        pp_missing_layer_names = get_pp_missing_layer_names(self)
        params_dict = dict(self.named_parameters())
        # Under the MXFP4 quant interface the routed experts register unpacked
        # params (``w13_weight``), while the compressed-tensors checkpoint names
        # them ``.weight_packed``. Rebind so the expert mapping resolves; scales
        # already share the ``.weight_scale`` suffix.
        experts_unpacked = not any(n.endswith("w13_weight_packed") for n in params_dict)
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # The multimodal checkpoint prefixes text weights with
            # ``language_model.``; strip it so names match this draft model's
            # parameter paths (``model.layers.{i}.``). Non-text weights
            # (vision_tower, mm_projector, ...) never match a spec layer below.
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (``.experts.{i}.w1/w2/w3``) are handled by the
                # expert mapping below; skip them here. Shared experts
                # (``.shared_experts.``) use gate/up_proj and fall through.
                if ".experts." in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # Only take this mapping if the fused destination actually
                # exists (e.g. QKV fusion is only present when q_lora is used).
                if name_mapped not in params_dict:
                    continue
                if name_mapped in pp_missing_layer_names:
                    continue
                name = name_mapped
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    expert_param_name,
                    expert_weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if expert_weight_name not in name:
                        continue
                    name_mapped = name.replace(expert_weight_name, expert_param_name)
                    if name_mapped in pp_missing_layer_names:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    name = name_mapped
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name

                    # The embedding is shared across MTP layers; only the first
                    # spec layer carries the hoisted (non-".layers") copy.
                    if spec_layer != self.model.mtp_start_layer_idx and (".layers" not in name):
                        continue
                    if name in pp_missing_layer_names:
                        continue
                    # The base model uses an attn-residual scheme whose per-layer
                    # weights (self_attention_res_*, mlp_res_*) are not used by
                    # the draft block; such names have no matching parameter and
                    # are safely skipped.
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint. The checkpoint may not include "
                    f"the MTP layer weights. Use a checkpoint that includes "
                    f"MTP layer weights, or disable speculative decoding."
                )

        return loaded_params
