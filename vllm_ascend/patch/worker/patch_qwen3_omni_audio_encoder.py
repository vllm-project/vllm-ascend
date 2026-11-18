#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Adapted from vllm/model_executor/models/qwen3_omni_moe_thinker.py and
# vllm/model_executor/models/qwen3_asr.py.
#
# The Qwen3-Omni audio tower does its chunk bookkeeping on the device and then
# reads it back several times per call: `.item()` on the post-CNN lengths,
# `.tolist()` on the chunk lengths and on aftercnn_lens, and a boolean-mask
# gather whose output size is only known on the device. Each read drains the
# queue, and the tower runs while other requests are decoding on the same
# engine, so every one of them stalls those too.
#
# Keeping the lengths on the host also keeps `cu_seqlens` there, which matters
# on Ascend: AscendMMEncoderAttention calls `cu_seqlens.cpu()` on every
# attention call (vllm_ascend/ops/mm_encoder_attention.py), so a device
# `cu_seqlens` is copied back once per encoder layer. Built on the host it is
# already there and the copy is free. This is why the patch does not follow
# vllm-project/vllm#68099 in pinning `cu_seqlens` to the feature device: that
# is needed for the CUDA/ROCm attention backends, not for Ascend.

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.models.qwen2_5_omni_thinker import Qwen2_5OmniAudioFeatureInputs
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder
from vllm.utils.torch_utils import PIN_MEMORY


def _process_audio_input(
    self,
    audio_input: Qwen2_5OmniAudioFeatureInputs,
) -> torch.Tensor:
    input_features = audio_input["input_features"]
    # Keep the lengths on the host. The audio tower only needs them for chunk
    # bookkeeping, and reading them back from the device would synchronize once
    # per encoder call, including for the `.tolist()` in the return below.
    audio_feature_lengths = audio_input["audio_feature_lengths"].cpu()

    audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)

    audio_features = self.audio_tower(
        input_features.to(self.audio_tower.dtype),
        feature_lens=audio_feature_lengths,
        aftercnn_lens=audio_output_lengths,
    )
    return audio_features.split(audio_output_lengths.tolist())


def _audio_encoder_forward(
    self,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor,
):
    # Compute chunk information. The lengths stay on the host so that the
    # `.item()`/`.tolist()` calls below read host values, and so that
    # `cu_seqlens` is built on the host for the encoder attention.
    feature_lens = feature_lens.cpu()
    aftercnn_lens = aftercnn_lens.cpu()
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = self.n_window * 2

    # Split input features into chunks and pad
    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)

    # Compute feature lengths after CNN
    feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
    # Vectorized mask creation: avoid creating many small tensors
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    indices = torch.arange(max_len_after_cnn)
    padded_mask_after_cnn = indices.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

    # Add channel dimension for conv2d
    padded_feature = padded_feature.unsqueeze(1)

    # Apply convolutional layers (chunk if needed to avoid OOM)
    if padded_feature.size(0) <= self.conv_chunksize:
        # Fast path: no chunking needed
        padded_embed = F.gelu(self.conv2d1(padded_feature))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
    else:
        # Chunked processing to avoid OOM
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

    # (batch, channels, freq, time) -> (batch, time, channels*freq)
    b, c, f, t = padded_embed.size()
    padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    # Add positional embedding
    positional_embedding = (
        self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding

    # Extract valid hidden states. The rows are gathered with host-built
    # indices: indexing with a boolean mask has to read the output size back
    # from the device before the kernel can be issued.
    valid_rows = padded_mask_after_cnn.flatten().nonzero().squeeze(1)
    if PIN_MEMORY:
        valid_rows = valid_rows.pin_memory()
    valid_rows = valid_rows.to(padded_embed.device, non_blocking=True)
    hidden_states = padded_embed.flatten(0, 1).index_select(0, valid_rows)

    # Compute cumulative sequence lengths for chunked attention
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
    for cnn_len in aftercnn_lens.tolist():
        num_full_chunks = cnn_len // window_aftercnn
        remainder = cnn_len % window_aftercnn
        cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
        if remainder:
            cu_chunk_lens.append(remainder)
    # aftercnn_lens is on the host, so cu_seqlens is built there too and the
    # `cu_seqlens.cpu()` in the encoder attention costs nothing per layer.
    cu_seqlens = torch.tensor(cu_chunk_lens, dtype=torch.int32, device=aftercnn_lens.device).cumsum(
        -1, dtype=torch.int32
    )

    max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

    # Apply transformer layers
    for encoder_layer in self.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens,
            max_seqlen,
        )

    # Apply output layers
    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)

    return hidden_states


Qwen3ASRForConditionalGeneration._process_audio_input = _process_audio_input
Qwen3OmniMoeAudioEncoder.forward = _audio_encoder_forward
