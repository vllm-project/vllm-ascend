#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import \
        Qwen3VLConfig
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import \
        Qwen3VLMoeConfig
except ImportError:
    pass
from vllm.config import VllmConfig
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention

try:
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3_VisionBlock, Qwen3_VisionPatchEmbed, Qwen3_VisionTransformer,
        Qwen3VLDummyInputsBuilder, Qwen3VLForConditionalGeneration,
        Qwen3VLMultiModalProcessor, Qwen3VLProcessingInfo)
    from vllm.model_executor.models.qwen3_vl_moe import (
        Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeProcessingInfo)
except ImportError:
    Qwen3_VisionBlock = object
    Qwen3_VisionPatchEmbed = object
    Qwen3_VisionTransformer = object
    Qwen3VLDummyInputsBuilder = object
    Qwen3VLForConditionalGeneration = object
    Qwen3VLMultiModalProcessor = object
    Qwen3VLProcessingInfo = object
    Qwen3VLMoeForConditionalGeneration = object
    Qwen3VLMoeProcessingInfo = object
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY


class AscendQwen3_VisionPatchEmbed(Qwen3_VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1))
        x = x + self.proj.bias
        return x


class AscendQwen3_VisionBlock(Qwen3_VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__(dim, num_heads, mlp_hidden_dim, act_fn, norm_layer,
                         quant_config, prefix, use_data_parallel)
        self.attn = Qwen2_5_VisionAttention(embed_dim=dim,
                                            num_heads=num_heads,
                                            projection_size=dim,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attn")

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens=cu_seqlens, cos=cos, sin=sin)

        x = x + self.mlp(self.norm2(x))
        return x


class AscendQwen3_VisionTransformer(Qwen3_VisionTransformer):

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix,
                         use_data_parallel)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.patch_embed = AscendQwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.blocks = nn.ModuleList([
            AscendQwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])
        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, self.num_heads)

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()  # [seqlen, rotary_dim / 2]
        sin = rotary_pos_emb.sin()
        cos_new = torch.cat((cos, cos), dim=-1)
        sin_new = torch.cat((sin, sin), dim=-1)
        cos_new = cos_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        return cos_new, sin_new

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        grid_thw_tensor = torch.tensor(grid_thw,
                                       device=self.device,
                                       dtype=torch.int32)
        cu_seqlens = torch.repeat_interleave(
            grid_thw_tensor[:, 1] * grid_thw_tensor[:, 2],
            grid_thw_tensor[:, 0]).cpu().to(torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = hidden_states.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        cos, sin = self.cal_cos_sin(rotary_pos_emb)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                cos=cos,
                                sin=sin)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(
                    layer_num)
                deepstack_feature = self.deepstack_merger_list[
                    deepstack_merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists,
            dim=1)  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
                                        info=Qwen3VLProcessingInfo,
                                        dummy_inputs=Qwen3VLDummyInputsBuilder)
class AscendQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen3VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.visual = AscendQwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
            use_data_parallel=self.use_data_parallel)


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
                                        info=Qwen3VLMoeProcessingInfo,
                                        dummy_inputs=Qwen3VLDummyInputsBuilder)
class AscendQwen3VLMoeForConditionalGeneration(
        Qwen3VLMoeForConditionalGeneration):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen3VLMoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.visual = AscendQwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
            use_data_parallel=self.use_data_parallel,
        )
