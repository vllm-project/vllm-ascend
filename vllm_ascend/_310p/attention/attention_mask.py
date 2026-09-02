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
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend._310p.dflash_full_and_piecewise import (
    is_310p_dflash_effective_piecewise,
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend._310p.dflash_full_decode_only import is_310p_dflash_full_decode_only
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d, nd_to_nz_spec

COMPRESSED_MASK_SEQ_LEN = 2048
PAGED_ATTENTION_COMPRESSED_MASK_VALUE = -10000.0


def is_compressed_mask_supported() -> bool:
    return hasattr(torch_npu, "_npu_flash_attention_v3") and hasattr(torch_npu, "_npu_paged_attention_splitfuse_v2")


class AttentionMaskBuilder310:
    chunked_prefill_attn_mask = None
    compressed_chunked_prefill_attn_mask = None
    compressed_non_causal_splitfuse_mask = None
    max_seqlen = 16384

    def __init__(self, device: torch.device, max_seqlen: int):
        """
        Initializes the AttentionMaskBuilder for the 310P device.

        Args:
            device (torch.device): The device on which tensors will be allocated.
            max_seqlen (int): Maximum length of a sequence (including prompt and generated text).
        """
        AttentionMaskBuilder310.max_seqlen = max_seqlen
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
    def _requires_graph_safe_query_positions() -> bool:
        try:
            forward_context = get_forward_context()
        except (AssertionError, RuntimeError) as exc:
            logger.debug(
                "[310p-dflash-graph/mask-route] context=unavailable error=%s",
                type(exc).__name__,
            )
            return False
        vllm_config = getattr(forward_context, "vllm_config", None)
        if vllm_config is None:
            logger.debug(
                "[310p-dflash-graph/mask-route] context=missing-vllm-config runtime=%s",
                getattr(
                    getattr(forward_context, "cudagraph_runtime_mode", None),
                    "name",
                    None,
                ),
            )
            return False
        runtime_mode = forward_context.cudagraph_runtime_mode
        piecewise_graph = is_310p_dflash_effective_piecewise(
            vllm_config,
            runtime_mode,
        )
        hybrid_full_graph = runtime_mode == CUDAGraphMode.FULL and is_310p_dflash_full_and_piecewise(vllm_config)
        full_decode_graph = runtime_mode == CUDAGraphMode.FULL and is_310p_dflash_full_decode_only(vllm_config)
        logger.debug(
            "[310p-dflash-graph/mask-route] runtime=%s configured=%s piecewise=%s hybrid_full=%s full_decode_only=%s",
            getattr(runtime_mode, "name", runtime_mode),
            getattr(
                vllm_config.compilation_config.cudagraph_mode,
                "name",
                vllm_config.compilation_config.cudagraph_mode,
            ),
            piecewise_graph,
            hybrid_full_graph,
            full_decode_graph,
        )
        return piecewise_graph or hybrid_full_graph or full_decode_graph

    @staticmethod
    def _get_graph_safe_query_positions(
        attn_metadata: AscendMetadata,
        device: torch.device,
        causal: bool,
        zero_descriptor_padding: bool = False,
    ) -> torch.Tensor:
        """Build SplitFuse rows entirely on-device for capture and replay."""
        num_query_tokens = int(attn_metadata.num_actual_tokens)
        query_start_loc = attn_metadata.query_start_loc.to(
            device=device,
            dtype=torch.int32,
        )
        seq_lens = attn_metadata.seq_lens.to(device=device, dtype=torch.int32)
        num_reqs = seq_lens.shape[0]

        token_indices = torch.arange(
            num_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        row_starts = query_start_loc[:num_reqs]
        row_ends = query_start_loc[1 : num_reqs + 1]
        request_mask = (token_indices.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
            token_indices.unsqueeze(0) < row_ends.unsqueeze(1)
        )
        valid_token_mask = request_mask.any(dim=0)
        request_indices = request_mask.to(torch.int32).argmax(dim=0)
        if not zero_descriptor_padding:
            request_indices = torch.where(
                valid_token_mask,
                request_indices,
                torch.full_like(request_indices, num_reqs - 1),
            )

        token_row_starts = row_starts.index_select(0, request_indices)
        query_lens = (row_ends - row_starts).index_select(0, request_indices)
        context_lens = seq_lens.index_select(0, request_indices)
        if causal:
            positions = context_lens - query_lens + token_indices - token_row_starts
        else:
            positions = context_lens - 1
        positions.clamp_min_(0)
        if zero_descriptor_padding:
            positions.mul_(valid_token_mask.to(dtype=positions.dtype))
        return positions

    @staticmethod
    def _requires_zero_descriptor_padding() -> bool:
        try:
            forward_context = get_forward_context()
        except (AssertionError, RuntimeError):
            return False
        vllm_config = getattr(forward_context, "vllm_config", None)
        return (
            vllm_config is not None
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and is_310p_dflash_full_decode_only(vllm_config)
        )

    @classmethod
    def _get_query_positions(
        cls,
        attn_metadata: AscendMetadata,
        device: torch.device,
        causal: bool,
    ) -> torch.Tensor:
        if cls._requires_graph_safe_query_positions():
            return cls._get_graph_safe_query_positions(
                attn_metadata,
                device,
                causal=causal,
                zero_descriptor_padding=cls._requires_zero_descriptor_padding(),
            )

        qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
        query_lens = (qsl[1:] - qsl[:-1]).tolist()
        context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32).tolist()
        if causal:
            rows = [
                row
                for query_len, context_len in zip(query_lens, context_lens)
                for row in range(context_len - query_len, context_len)
            ]
        else:
            rows = [
                context_len - 1 for query_len, context_len in zip(query_lens, context_lens) for _ in range(query_len)
            ]
        return torch.tensor(rows, dtype=torch.int32, device=device)

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
        position = cls._get_query_positions(attn_metadata, device, causal=True)
        splitfuse_mask = cls.chunked_prefill_attn_mask.index_select(0, position)
        splitfuse_mask_nz = torch_npu.npu_format_cast(nd_to_nz_spec(splitfuse_mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        return splitfuse_mask_nz

    @classmethod
    def get_non_causal_splitfuse_mask(cls, attn_metadata: AscendMetadata, device: torch.device):
        """SplitFuse mask for full / non-causal attention (dflash/dspark draft).

        Every query token must attend to the entire valid sequence ``[0, cl)``
        (full context + all query tokens), and nothing beyond ``cl``.

        A naive all-zero ``[num_query_tokens, max_seqlen]`` mask makes
        ``_npu_paged_attention_splitfuse`` emit an all-zero output (the op
        expects the same additive-mask structure as the causal path, where
        out-of-range columns carry ``-inf``). Instead, reuse the causal additive
        matrix and give every query token the row at ``cl - 1``: that row is 0
        for columns ``< cl`` and ``-inf`` for columns ``>= cl``, i.e. full
        attention over the valid sequence with the garbage tail masked. This
        mirrors ``get_splitfuse_mask`` so the operator accepts it.
        """
        if cls.chunked_prefill_attn_mask is None:
            cls.chunked_prefill_attn_mask = cls.gen_causal_additive_mask(cls.max_seqlen, device)
        position = cls._get_query_positions(attn_metadata, device, causal=False)
        splitfuse_mask = cls.chunked_prefill_attn_mask.index_select(0, position)
        splitfuse_mask_nz = torch_npu.npu_format_cast(nd_to_nz_spec(splitfuse_mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        return splitfuse_mask_nz

    @classmethod
    def get_compressed_splitfuse_mask(cls, device: torch.device):
        """
        Generates the fixed ND attention mask for compressed SplitFuse PA.

        Returns:
            torch.Tensor: A [2048, 2048] float16 ND mask on the target device.
        """
        if (
            cls.compressed_chunked_prefill_attn_mask is None
            or cls.compressed_chunked_prefill_attn_mask.device != device
        ):
            mask = torch.ones(
                size=(COMPRESSED_MASK_SEQ_LEN, COMPRESSED_MASK_SEQ_LEN),
                dtype=torch.float16,
                device=device,
            )
            mask = torch.triu(mask, diagonal=1)
            cls.compressed_chunked_prefill_attn_mask = mask.mul_(PAGED_ATTENTION_COMPRESSED_MASK_VALUE)
        return cls.compressed_chunked_prefill_attn_mask

    @classmethod
    def get_compressed_non_causal_splitfuse_mask(cls, device: torch.device):
        """Fixed [2048, 2048] all-zero mask for compressed SplitFuse PA.

        Matches ``sparse_mode=0`` / no-mask semantics on the generic path.
        """
        if (
            cls.compressed_non_causal_splitfuse_mask is None
            or cls.compressed_non_causal_splitfuse_mask.device != device
        ):
            cls.compressed_non_causal_splitfuse_mask = torch.zeros(
                size=(COMPRESSED_MASK_SEQ_LEN, COMPRESSED_MASK_SEQ_LEN),
                dtype=torch.float16,
                device=device,
            )
        return cls.compressed_non_causal_splitfuse_mask

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
