#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLImageInputs, Qwen2_5_VLVideoInputs)
from vllm.model_executor.models.qwen2_vl import Qwen2VisionAttention
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import set_ascend_forward_context

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight


class AscendQwen2_5_VisionAttention(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        # Split q k v.
        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        origin_shape = q.shape[-1]

        # Convert cumulative tensor to intervals and move it to cpu.
        cu_seqlens = torch.diff(cu_seqlens).to("cpu")

        cos = torch.cat((rotary_pos_emb_cos, rotary_pos_emb_cos), dim=-1)
        sin = torch.cat((rotary_pos_emb_sin, rotary_pos_emb_sin), dim=-1)
        cos = cos.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        sin = sin.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        q, k, v = [
            einops.rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        enable_pad = (envs_ascend.USE_OPTIMIZED_MODEL
                      and self.hidden_size_per_attention_head > MIN_PAD_SIZE
                      and self.hidden_size_per_attention_head < MAX_PAD_SIZE)

        if enable_pad:
            pad_len = MAX_PAD_SIZE - origin_shape
            # q/k/v: [b * s, head, head_dim] -> [b * s, head, MAX_PAD_SIZE]
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer = torch.empty_like(q)

        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer,
        )

        if enable_pad:
            context_layer = context_layer[..., :origin_shape]

        context_layer = einops.rearrange(context_layer,
                                         "(b s) h d -> s b (h d)",
                                         b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class AscendQwen2_5_VLForConditionalGeneration(nn.Module):

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_ascend_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values,
                        grid_thw_list,
                        rope_type="rope_3d")
                else:
                    image_embeds = self.visual(pixel_values,
                                               grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            with set_ascend_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values_videos,
                        grid_thw_list,
                        rope_type="rope_3d",
                    )
                else:
                    video_embeds = self.visual(pixel_values_videos,
                                               grid_thw=grid_thw_list)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)


# NOTE: This will be removed after MMEncoderAttention has been extract as a CustomOp in vllm.
Qwen2VisionAttention.forward = AscendQwen2_5_VisionAttention.forward
Qwen2_5_VisionAttention.forward = AscendQwen2_5_VisionAttention.forward

# NOTE: These will be removed after https://github.com/vllm-project/vllm/pull/29388 is merged.
Qwen2_5_VLForConditionalGeneration._process_image_input = AscendQwen2_5_VLForConditionalGeneration._process_image_input
Qwen2_5_VLForConditionalGeneration._process_video_input = AscendQwen2_5_VLForConditionalGeneration._process_video_input
