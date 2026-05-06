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

    def get_attention_mask(self, model_config) -> torch.Tensor:
        """
        Retrieves the appropriate attention mask based on the model configuration.

        Pooling models that use Qwen architecture or LAST sequence pooling,
        such as Qwen3 embedding, still use causal attention on 310P. Other
        pooling models are not supported yet.

        Args:
            model_config: Configuration object containing runner details.

        Returns:
            torch.Tensor: The causal attention mask.

        Raises:
            NotImplementedError: If the runner_type is 'pooling' without LAST
                sequence pooling.
        """
        if getattr(model_config, "runner_type", None) == "pooling":
            hf_config = getattr(model_config, "hf_config", None)
            pooler_config = getattr(model_config, "pooler_config", None)
            architectures = getattr(hf_config, "architectures", None) or []
            if isinstance(architectures, str):
                architectures = [architectures]
            if any("qwen" in str(architecture).lower() for architecture in architectures):
                return self._get_causal_mask(self.max_seqlen)
            if getattr(pooler_config, "seq_pooling_type", None) == "LAST":
                return self._get_causal_mask(self.max_seqlen)
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
