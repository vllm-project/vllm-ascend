#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d

COMPRESSED_MASK_SEQ_LEN = 2048
SPLITFUSE_MASK_TILE_SIZE = 16


def is_compressed_mask_supported() -> bool:
    return hasattr(torch_npu, "_npu_flash_attention_v3") and hasattr(torch_npu, "_npu_paged_attention_splitfuse_v2")


class AttentionMaskBuilder310:
    def __init__(self, device: torch.device, max_seqlen: int):
        """
        Initializes the AttentionMaskBuilder for the 310P device.

        Args:
            device (torch.device): The device on which tensors will be allocated.
            max_seqlen (int): Maximum length of a sequence (including prompt and generated text).
        """
        self.max_seqlen = max_seqlen
        self.causal_attn_mask_cache = None
        self.non_causal_attn_mask_cache = None
        self.support_compressed_mask = is_compressed_mask_supported()
        self.device = device

    @staticmethod
    def gen_causal_additive_mask(max_seq_len: int, device: torch.device):
        """
        Generates a standard causal lower-triangular attention mask.

        The upper triangular part is filled with negative infinity (float("-inf"))
        to mask out future tokens, while the lower triangular part is kept as 0.

        Args:
            max_seq_len (int): The maximum sequence length for the mask.
            device (torch.device): The target device for the tensor.

        Returns:
            torch.Tensor: A float16 tensor representing the causal mask.
        """
        tril = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device).tril_()
        upper = ~tril
        mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
        mask.masked_fill_(upper, float("-inf"))
        return mask

    @staticmethod
    def get_splitfuse_mask(
        query_start_loc: torch.Tensor,
        context_lens: torch.Tensor,
        num_tokens: int,
        max_context_len: int,
        *,
        capturing: bool,
    ) -> torch.Tensor:
        """Build the per-query causal mask in the legacy SplitFuse blocked layout."""
        tile = SPLITFUSE_MASK_TILE_SIZE
        padded_tokens = (num_tokens + tile - 1) // tile * tile
        padded_context = (max_context_len + tile - 1) // tile * tile
        device = context_lens.device

        # Use the live device lengths so replay rebuilds the correct mask.
        query_ends = query_start_loc[1 : context_lens.numel() + 1].to(device)
        rows = torch.arange(padded_tokens, dtype=query_ends.dtype, device=device)
        requests = torch.searchsorted(query_ends, rows, out_int32=True, right=True)
        requests.clamp_max_(context_lens.numel() - 1)
        cutoffs = (context_lens - query_ends).index_select(0, requests) + rows
        cutoffs.masked_fill_(rows >= num_tokens, padded_context - 1)

        columns = torch.arange(padded_context, dtype=rows.dtype, device=device).view(1, -1, 1, tile)
        mask = torch.zeros((1, padded_context // tile, padded_tokens, tile), dtype=torch.float16, device=device)
        mask.masked_fill_(columns > cutoffs.view(1, 1, -1, 1), float("-inf"))
        if capturing:
            # The legacy op consumes the blocked layout directly. FormatCast
            # uses an aclop on 310P that cannot run during ACLGraph capture.
            return mask
        return torch_npu.npu_format_cast(mask, ACL_FORMAT_FRACTAL_NZ)

    def get_attention_mask(self, causal: bool, model_config) -> torch.Tensor:
        """
        Retrieves the appropriate attention mask based on the model configuration.

        When compressed mask is supported, the mask is generated as a fixed
        [2048, 2048] logical mask and converted to 4D FRACTAL_NZ.

        Args:
            causal (bool): Whether to generate a causal mask.
            model_config: Configuration object containing runner details.

        Returns:
            torch.Tensor: The causal attention mask.

        Raises:
            NotImplementedError: If the runner_type is 'pooling'.
        """
        max_seq_len = COMPRESSED_MASK_SEQ_LEN if self.support_compressed_mask else self.max_seqlen
        if getattr(model_config, "runner_type", None) == "pooling":
            if causal:
                return self._get_causal_mask(max_seq_len)
            else:
                return self._get_non_causal_mask(max_seq_len, model_config.dtype)

        return self._get_causal_mask(max_seq_len)

    def _get_causal_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        Internal method to get or update the cached causal attention mask.

        If the cache is empty, a new mask is generated and converted to the
        NPU fractal format.

        Returns:
            torch.Tensor: The cached causal mask in ACL_FORMAT_FRACTAL_NZ.
        """
        if self.causal_attn_mask_cache is None:
            attn_mask = self.gen_causal_additive_mask(max_seq_len, self.device)
            self.causal_attn_mask_cache = torch_npu.npu_format_cast(nd_to_nz_2d(attn_mask), ACL_FORMAT_FRACTAL_NZ)
        return self.causal_attn_mask_cache

    def _get_non_causal_mask(self, max_seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Internal method to get or update the cached non-causal attention mask.

        If the cache is empty, a new mask is generated and converted to the
        NPU fractal format.

        Returns:
            torch.Tensor: The cached causal mask in ACL_FORMAT_FRACTAL_NZ.
        """
        if self.non_causal_attn_mask_cache is not None:
            return self.non_causal_attn_mask_cache

        attention_mask_npu = torch.zeros(
            size=(max_seq_len, max_seq_len),
            dtype=dtype,
            device=self.device,
        )
        attention_mask_npu = nd_to_nz_2d(attention_mask_npu)
        self.non_causal_attn_mask_cache = torch_npu.npu_format_cast(
            attention_mask_npu.contiguous(), ACL_FORMAT_FRACTAL_NZ
        )

        return self.non_causal_attn_mask_cache
