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

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d, nd_to_nz_spec


class AttentionMaskBuilder310:
    chunked_prefill_attn_mask = None
    max_seqlen = 16384

    def __init__(self, device: torch.device, max_seqlen: int):
        """
        Initializes the AttentionMaskBuilder for the 310P device.

        Args:
            device (torch.device): The device on which tensors will be allocated.
            max_seqlen (int): Maximum length of a sequence (including prompt and generated text).
        """
        AttentionMaskBuilder310.max_seqlen = max_seqlen
        self.attn_mask_cache = None
        self.device = device
        self.swa_mask = None

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

    @classmethod
    def get_splitfuse_mask(cls, attn_metadata: AscendMetadata, device: torch.device):
        """
        Generates and formats the attention mask for SplitFuse (chunked prefill) decoding.

        It calculates the specific indices required based on query start locations
        and context lengths, selects the relevant parts from the global chunked
        mask, and converts the result to the NPU-specific fractal format.

        Args:
            attn_metadata (AscendMetadata): Metadata containing query start locations and sequence lengths.
            device (torch.device): The device to perform operations on.

        Returns:
            torch.Tensor: The splitfuse attention mask cast to ACL_FORMAT_FRACTAL_NZ.
        """
        if cls.chunked_prefill_attn_mask is None:
            cls.chunked_prefill_attn_mask = cls.gen_causal_additive_mask(cls.max_seqlen, device)
        qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
        qlens = qsl[1:] - qsl[:-1]
        q_list = qlens.tolist()
        context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)
        c_list = context_lens.tolist()
        pos_list = [p for ql, cl in zip(q_list, c_list) for p in range(cl - ql, cl)]
        position = torch.tensor(pos_list, dtype=torch.int32, device=device)
        splitfuse_mask = cls.chunked_prefill_attn_mask.index_select(0, position)
        splitfuse_mask_nz = torch_npu.npu_format_cast(nd_to_nz_spec(splitfuse_mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        return splitfuse_mask_nz

    def get_swa_mask(self, dtype: torch.dtype, sliding_window):
        """
        Generates or retrieves a cached Sliding Window Attention (SWA) mask.

        This mask allows attention only within a specific local window (diagonal band),
        masking out tokens that are too far in the past or in the future.

        Args:
            dtype (torch.dtype): Data type of the mask.
            sliding_window (int): The size of the sliding window.

        Returns:
            torch.Tensor: The SWA mask cast to ACL_FORMAT_FRACTAL_NZ.
        """
        assert dtype == torch.float16
        if sliding_window is not None and self.swa_mask is None:
            mask = torch.ones(self.max_seqlen, self.max_seqlen, dtype=torch.bool)
            triu_mask = torch.triu(mask, diagonal=1).to(self.device)
            tril_mask = torch.tril(mask, -sliding_window).to(self.device)
            mask = triu_mask + tril_mask
            swa_mask = torch.zeros((self.max_seqlen, self.max_seqlen), dtype=dtype, device=self.device)
            swa_mask.masked_fill_(mask, float("-inf"))
            self.swa_mask = torch_npu.npu_format_cast(nd_to_nz_2d(swa_mask), ACL_FORMAT_FRACTAL_NZ)
        return self.swa_mask

    def get_attention_mask(self, model_config) -> torch.Tensor:
        """
        Retrieves the appropriate attention mask based on the model configuration.

        It explicitly checks for 'pooling' runner types which are not supported
        on 310P hardware.

        Args:
            model_config: Configuration object containing runner details.

        Returns:
            torch.Tensor: The causal attention mask.

        Raises:
            NotImplementedError: If the runner_type is 'pooling'.
        """
        if getattr(model_config, "runner_type", None) == "pooling":
            # TODO: pooling model will be supported soon.
            raise NotImplementedError("310P does not support runner_type='pooling'")
        return self._get_causal_mask(self.max_seqlen)

    def _get_causal_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        Internal method to get or update the cached causal attention mask.

        If the cache is empty or the requested length exceeds the cached length,
        a new mask is generated and converted to the NPU fractal format.

        Args:
            max_seq_len (int): The required sequence length.

        Returns:
            torch.Tensor: The cached causal mask in ACL_FORMAT_FRACTAL_NZ.
        """
        if self.attn_mask_cache is None:
            attn_mask = self.gen_causal_additive_mask(max_seq_len, self.device)
            self.attn_mask_cache = torch_npu.npu_format_cast(nd_to_nz_2d(attn_mask), ACL_FORMAT_FRACTAL_NZ)
        return self.attn_mask_cache
