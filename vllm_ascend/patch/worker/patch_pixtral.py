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
"""NPU compatibility patches for the Mistral-format Pixtral vision encoder.

Two Ascend-specific gaps are addressed here:

1. xformers (``xops.memory_efficient_attention`` / ``BlockDiagonalMask``) is a
   CUDA-only library and is unavailable on Ascend NPU. vLLM already ships a
   native fallback (``F.scaled_dot_product_attention`` plus the transformers
   block-diagonal mask) that is gated behind ``USE_XFORMERS_OPS``. When xformers
   happens to be importable in the environment vLLM still selects the xformers
   path on non-CUDA platforms, so we force the flag off to take the NPU-friendly
   route.

2. ``self.freqs_cis`` is a ``complex64`` tensor and the original forward indexes
   it directly. The Ascend ``aclnnIndex`` operator does not support indexing
   complex tensors, so we index on the real view and convert back to complex.
"""

import torch
import vllm.model_executor.models.pixtral as pixtral
from vllm.model_executor.models.pixtral import (
    VisionTransformer,
    position_meshgrid,
)

# xformers does not work on Ascend NPU; always take the native code paths.
pixtral.USE_XFORMERS_OPS = False


def _vision_transformer_forward_npu(
    self,
    images: list[torch.Tensor],
) -> torch.Tensor:
    # pass images through initial convolution independently
    patch_embeds_list = [self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images]

    patch_embeds = [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list]
    embed_sizes = [p.shape[1] for p in patch_embeds]

    # flatten to a single sequence
    patch_embeds = torch.cat(patch_embeds, dim=1)
    patch_embeds = self.ln_pre(patch_embeds)

    # positional embeddings
    positions = position_meshgrid(patch_embeds_list).to(self.device)
    # NPU: aclnnIndex cannot index complex64 tensors directly. Index on the
    # real view and convert the result back to complex.
    freqs_cis_real = torch.view_as_real(self.freqs_cis)
    freqs_cis = torch.view_as_complex(freqs_cis_real[positions[:, 0], positions[:, 1]].contiguous())

    # block diagonal mask delimiting images (NPU: native, non-xformers path)
    from transformers.models.pixtral.modeling_pixtral import generate_block_attention_mask

    mask = generate_block_attention_mask(
        [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
        patch_embeds,
    )

    out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

    # squeeze dim 0 and split into separate tensors for each image
    return torch.split(out.squeeze(0), embed_sizes)


VisionTransformer.forward = _vision_transformer_forward_npu
