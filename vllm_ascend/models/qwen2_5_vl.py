#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_5_vl.py
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
from typing import Callable, Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from einops import rearrange
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionBlock, Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding, Qwen2_5_VisionTransformer,
    Qwen2_5_VLDummyInputsBuilder, Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor, Qwen2_5_VLProcessingInfo)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendQwen2_5_VisionAttention(Qwen2_5_VisionAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            projection_size,
            quant_config,
            prefix,
        )
        self.embed_dim = embed_dim
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
        if self.hidden_size_per_attention_head > MIN_PAD_SIZE and self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.origin_hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)

        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class AscendQwen2_5_VisionBlock(Qwen2_5_VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim, num_heads, mlp_hidden_dim, act_fn, norm_layer,
                         quant_config, prefix)
        self.attn = AscendQwen2_5_VisionAttention(embed_dim=dim,
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


class AscendQwen2_5_VisionPatchEmbed(Qwen2_5_VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1))
        return x


class AscendQwen2_5_VisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__(dim, theta)
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = inv_freq


class AscendQwen2_5_VisionTransformer(Qwen2_5_VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)
        norm_layer = partial(RMSNorm, eps=norm_eps)
        self.interleaved = interleaved
        self.enable_pad = False
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = AscendQwen2_5_VisionRotaryEmbedding(head_dim //
                                                                  2)
        self.patch_embed = AscendQwen2_5_VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        act_fn = get_act_and_mul_fn(vision_config.hidden_act)
        self.blocks = nn.ModuleList([
            AscendQwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=act_fn,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, self.num_heads)

        if self.hidden_size_per_attention_head > MIN_PAD_SIZE and self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.enable_pad = True
            self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
            self.half_origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head // 2
            self.half_pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head) // 2
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()  # [seqlen, rotary_dim / 2]
        sin = rotary_pos_emb.sin()
        if self.enable_pad:
            cos = torch.nn.functional.pad(
                cos, (0, self.half_pad_hidden_size_per_attention_head))
            sin = torch.nn.functional.pad(
                sin, (0, self.half_pad_hidden_size_per_attention_head))

        if not self.interleaved:
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        else:
            cos_new = rearrange(torch.stack((cos, cos), dim=-1),
                                "... d two -> ...(d two)",
                                two=2)
            sin_new = rearrange(torch.stack((sin, sin), dim=-1),
                                "... d two -> ...(d two)",
                                two=2)
        cos_new = cos_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1,
                                  self.hidden_size_per_attention_head)
        return cos_new, sin_new

    def pad_qkv_bias(self, bias):
        first_half = bias.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head
        )[:, :, :self.half_origin_hidden_size_per_attention_head]
        second_half = bias.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head
        )[:, :, self.half_origin_hidden_size_per_attention_head:]
        first_half_padded = torch.nn.functional.pad(
            first_half, (0, self.half_pad_hidden_size_per_attention_head))
        second_half_padded = torch.nn.functional.pad(
            second_half, (0, self.half_pad_hidden_size_per_attention_head))
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(-1)
        return bias_final

    def pad_qkv_weight(self, data):
        qkv_weight_first_half = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, self.hidden_size
        )[:, :, :self.half_origin_hidden_size_per_attention_head, :]
        qkv_weight_second_half = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, self.hidden_size
        )[:, :, self.half_origin_hidden_size_per_attention_head:, :]

        qkv_weight_first_half_padded = torch.nn.functional.pad(
            qkv_weight_first_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head))
        qkv_weight_second_half_padded = torch.nn.functional.pad(
            qkv_weight_second_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head))
        qkv_weight_padded = torch.cat(
            [qkv_weight_first_half_padded, qkv_weight_second_half_padded],
            dim=2)
        qkv_weight_final = qkv_weight_padded.reshape(-1, self.hidden_size)
        return qkv_weight_final

    def pad_proj_weight(self, data):
        out_weight = torch.nn.functional.pad(
            data.reshape(self.hidden_size, -1,
                         self.half_origin_hidden_size_per_attention_head),
            (0, self.half_pad_hidden_size_per_attention_head, 0, 0)).reshape(
                self.hidden_size, -1)
        return out_weight

    def pad_qkv_weight_scale_offset(self, data):
        reshaped_data = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, 1)
        data1 = reshaped_data[:, :, :self.
                              half_origin_hidden_size_per_attention_head, :]
        data2 = reshaped_data[:, :, self.
                              half_origin_hidden_size_per_attention_head:, :]
        data1_paded = torch.nn.functional.pad(
            data1, (0, 0, 0, self.half_pad_hidden_size_per_attention_head, 0,
                    0, 0, 0))
        data2_paded = torch.nn.functional.pad(
            data2, (0, 0, 0, self.half_pad_hidden_size_per_attention_head, 0,
                    0, 0, 0))
        res = torch.cat([data1_paded, data2_paded], dim=2)
        res = res.reshape(-1, 1)
        return res

    def pad_qkv_deq_scale_quant_bias(self, data):
        reshaped_data = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head)
        data1 = reshaped_data[:, :, :self.
                              half_origin_hidden_size_per_attention_head]
        data2 = reshaped_data[:, :,
                              self.half_origin_hidden_size_per_attention_head:]

        data1_paded = torch.nn.functional.pad(
            data1, (0, self.half_pad_hidden_size_per_attention_head))
        data2_paded = torch.nn.functional.pad(
            data2, (0, self.half_pad_hidden_size_per_attention_head))

        res = torch.cat([data1_paded, data2_paded], dim=2)
        res = res.reshape(-1)
        return res

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping: list[tuple[str, str, Union[str, int]]] = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("mlp.gate_up_proj.", "mlp.gate_proj.", 0),
            ("mlp.gate_up_proj.", "mlp.up_proj.", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                if ("attn.proj.weight_scale" in name or
                        "attn.proj.weight_offset" in name) and self.enable_pad:
                    continue
                elif ("attn.proj.deq_scale" in name
                      or "attn.proj.quant_bias" in name) and self.enable_pad:
                    continue
                elif ("attn.qkv.weight_scale" in name
                      or "attn.qkv.weight_offset" in name) and self.enable_pad:
                    param.data = self.pad_qkv_weight_scale_offset(param.data)
                elif ("attn.qkv.deq_scale" in name
                      or "attn.qkv.quant_bias" in name) and self.enable_pad:
                    param.data = self.pad_qkv_deq_scale_quant_bias(param.data)
                elif ("attn.proj.weight" in name) and self.enable_pad:
                    param.data = self.pad_proj_weight(param.data)
                elif ("attn.qkv.weight" in name) and self.enable_pad:
                    param.data = self.pad_qkv_weight(param.data)
                elif ("attn.qkv.bias" in name) and self.enable_pad:
                    param.data = self.pad_qkv_bias(param.data)
            loaded_params.add(name)
        return loaded_params

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h,
                                                vit_merger_window_size,
                                                num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
                vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(
                0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:,
                                                      0]).cpu().to(torch.int32)

        # patchify
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = torch.diff(cu_window_seqlens).cpu().to(torch.int32)
        seq_len, _ = x.size()
        x = x.reshape(seq_len // self.spatial_merge_unit,
                      self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cos, sin = self.cal_cos_sin(rotary_pos_emb)

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(x, cu_seqlens=cu_seqlens_now, cos=cos, sin=sin)

        # adapter
        x = self.merger(x)
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]
        return x


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class AscendQwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLForConditionalGeneration):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.visual = AscendQwen2_5_VisionTransformer(
            vision_config=config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(self, video_input) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return video_embeds.split(sizes.tolist())
