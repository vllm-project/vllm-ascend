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

import os
from typing import Any

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)

logger = init_logger(__name__)
_PA310_DEBUG_ONCE_PRINTED = False
_PA310_PREFLIGHT_ONCE_PRINTED = False
_PA310_SETUP_FAIL_ONCE_PRINTED = False


def _is_rank0_process() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass

    rank_env = os.getenv("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass

    local_rank_env = os.getenv("LOCAL_RANK")
    if local_rank_env is not None:
        try:
            return int(local_rank_env) == 0
        except ValueError:
            pass

    return True


def _pa310_debug_enabled() -> bool:
    # Default off to avoid noisy multi-rank logs.
    return os.getenv("VLLM_ASCEND_310P_DEBUG", "0") == "1"


def _tensor_meta(name: str, tensor: Any) -> str:
    if tensor is None:
        return f"{name}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=<{type(tensor).__name__}>"
    return (
        f"{name}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, stride={tuple(tensor.stride())}, "
        f"contiguous={tensor.is_contiguous()})"
    )


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend310(AscendAttentionBackend):
    def __init__(self, *args, **kwargs):
        """
        Initializes the 310P backend and sets up the device-specific mask builder.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int):
        """
        Determines the shape of the Key-Value (KV) cache tensor.

        The 310P hardware requires specific memory alignment for optimal performance.
        This method defines a 5D tensor shape where the head size dimension is
        split to ensure alignment to multiples of 16.

        Args:
            num_blocks (int): Number of memory blocks.
            block_size (int): Size of each block.
            num_kv_heads (int): Number of KV heads.
            head_size (int): Dimension size of each head.

        Returns:
            tuple: The specific 5D shape required by the hardware
                   (2, num_blocks, hidden_dim_aligned, block_size, 16).
        """
        # Align to a multiple of 16, as required by the 310P device.
        return (2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)

    @staticmethod
    def get_impl_cls():
        """
        Returns the implementation class for the attention operations.
        """
        return AscendAttentionBackendImpl310

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        """
        Returns the metadata builder class specifically for 310P.
        """
        return AscendAttentionMetadataBuilder310


class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    """
    Implementation of attention operations (Prefill, Decode, Chunked Prefill)
    optimized for the Ascend 310P architecture.
    """

    def _log_pa_inputs_once(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None,
        phase: str,
    ) -> None:
        global _PA310_DEBUG_ONCE_PRINTED
        if not _pa310_debug_enabled() or not _is_rank0_process() or _PA310_DEBUG_ONCE_PRINTED:
            return
        _PA310_DEBUG_ONCE_PRINTED = True

        state = getattr(attn_metadata, "attn_state", None)
        block_tables = getattr(attn_metadata, "block_tables", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)

        logger.warning(
            "[PA310_DEBUG_ONCE] phase=%s, state=%s, num_heads=%s, num_kv_heads=%s, "
            "head_size=%s, scale=%s, num_actual_tokens=%s, num_decode_tokens=%s, "
            "num_prefill_tokens=%s",
            phase,
            state,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.scale,
            getattr(attn_metadata, "num_actual_tokens", None),
            getattr(attn_metadata, "num_decode_tokens", None),
            getattr(attn_metadata, "num_prefill_tokens", None),
        )
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("query", query))
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("output", output))
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("key_cache", self.key_cache))
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("value_cache", self.value_cache))
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("block_tables", block_tables))
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("seq_lens", seq_lens))
        logger.warning("[PA310_DEBUG_ONCE] %s", _tensor_meta("slot_mapping", slot_mapping))

    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        """
        Executes Paged Attention (typically for the decode phase).

        Ensures that the sequence length metadata is on the correct device
        before invoking the base implementation.

        Args:
            query (Any): The query tensor.
            attn_metadata (AscendMetadata): Metadata associated with the attention request.
            output (Any | None): Optional output tensor.

        Returns:
            Any: The result of the attention operation.
        """
        global _PA310_PREFLIGHT_ONCE_PRINTED
        self._log_pa_inputs_once(
            query=query,
            attn_metadata=attn_metadata,
            output=output,
            phase="decode_paged_attention",
        )
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )
        if (
            (not _PA310_PREFLIGHT_ONCE_PRINTED)
            and self.key_cache is not None
            and self.value_cache is not None
        ):
            _PA310_PREFLIGHT_ONCE_PRINTED = True
            block_table = attn_metadata.block_tables
            seq_lens_cpu = attn_metadata.seq_lens.to("cpu")
            num_blocks = int(self.key_cache.shape[0])
            block_size = int(self.key_cache.shape[2])
            bt_rows = int(block_table.shape[0])
            bt_cols = int(block_table.shape[1])
            bt_min = int(block_table.min().item()) if block_table.numel() > 0 else -1
            bt_max = int(block_table.max().item()) if block_table.numel() > 0 else -1
            max_seq = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() > 0 else 0
            max_capacity_by_table = bt_cols * block_size
            if bt_min < 0 or bt_max >= num_blocks:
                raise RuntimeError(
                    "PagedAttention preflight failed: invalid block id range "
                    f"[{bt_min}, {bt_max}] for num_blocks={num_blocks}."
                )
            if max_seq > max_capacity_by_table:
                raise RuntimeError(
                    "PagedAttention preflight failed: sequence length exceeds "
                    f"block-table capacity ({max_seq} > {max_capacity_by_table})."
                )
            if _pa310_debug_enabled() and _is_rank0_process():
                logger.warning(
                    "[PA310_PREFLIGHT_ONCE] num_blocks=%d block_size=%d "
                    "block_table_shape=(%d,%d) block_id_range=[%d,%d] "
                    "max_seq_len=%d max_capacity_by_table=%d num_heads=%d "
                    "num_kv_heads=%d head_size=%d",
                    num_blocks,
                    block_size,
                    bt_rows,
                    bt_cols,
                    bt_min,
                    bt_max,
                    max_seq,
                    max_capacity_by_table,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_size,
                )
        try:
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
            )
            return output
        except RuntimeError as exc:
            global _PA310_SETUP_FAIL_ONCE_PRINTED
            if _is_rank0_process() and not _PA310_SETUP_FAIL_ONCE_PRINTED:
                _PA310_SETUP_FAIL_ONCE_PRINTED = True
                logger.error(
                    "[PA310_SETUP_FAIL] paged_attention setup failed: %s | %s | %s | %s | %s | %s",
                    str(exc),
                    _tensor_meta("query", query),
                    _tensor_meta("key_cache", self.key_cache),
                    _tensor_meta("value_cache", self.value_cache),
                    _tensor_meta("block_tables", attn_metadata.block_tables),
                    _tensor_meta("seq_lens", attn_metadata.seq_lens),
                )
            raise

    def forward_prefill_310(self, query, key, value, attn_metadata, output):
        """
        Executes Flash Attention for the prefill phase on 310P.

        This method handles memory alignment padding. If the query shape implies
        padding (aligned_tokens > real_tokens), it adjusts the sequence length
        of the last request to account for the delta, ensuring the NPU operator
        processes the data correctly.

        Args:
            query, key, value: Input tensors.
            attn_metadata (AscendMetadata): Attention metadata containing masks and seq_lens.
            output: Output tensor.

        Returns:
            The output tensor after flash attention.
        """
        real_tokens = int(attn_metadata.seq_lens.sum().item())
        seq_len = attn_metadata.seq_lens
        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens

        # Adjust sequence length if padding (alignment) was applied to the inputs
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        torch_npu._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output,
        )
        return output

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        """
        Executes SplitFuse (Chunked Prefill) attention on 310P.

        This handles scenarios where the prefill is split into chunks. It prepares
        the necessary metadata (query lengths, block tables) and generates the
        specific splitfuse mask before calling the NPU operator.

        Args:
            query: The query tensor.
            attn_metadata (AscendMetadata): Metadata containing start locations and block tables.
            output: The output tensor.
        """
        self._log_pa_inputs_once(
            query=query,
            attn_metadata=attn_metadata,
            output=output,
            phase="chunked_prefill_splitfuse",
        )
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]

        # Calculate query lengths from start locations
        qsl_cpu = attn_metadata.query_start_loc.cpu()
        qlens = qsl_cpu[1:] - qsl_cpu[:-1]

        context_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_tables

        # Generate the specific mask for splitfuse
        mask = AttentionMaskBuilder310.get_splitfuse_mask(attn_metadata, query.device)

        if context_lens.device != query.device:
            context_lens = context_lens.to(query.device, non_blocking=True)

        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=context_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output,
        )

        return output

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        """
        Main dispatch method for attention operations.

        Routes the execution to Decode, Prefill, or Chunked Prefill methods
        based on the current attention state found in metadata.

        Args:
            query, key, value: Input tensors (Key/Value usually empty for decode/chunked).
            kv_cache: The KV cache structure.
            attn_metadata: Metadata determining the state (Prefill vs Decode).
            output: Tensor to write results to.

        Returns:
            The output tensor.

        Raises:
            NotImplementedError: If the attention state is not supported on 310P.
        """
        state = attn_metadata.attn_state
        # Condition for PrefillNoCache: No previous tokens have been processed yet
        if state == AscendAttentionState.PrefillNoCache:
            output = self.forward_prefill_310(query, key, value, attn_metadata, output)
        # Condition for DecodeOnly: Pure decoding phase where each request generates one token
        elif state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        # Condition for ChunkedPrefill:
        # 1. During speculative decoding scenarios (except mtp)
        # 2. Processing large prefill requests in chunks
        # Condition for PrefillCacheHit: Indicates prefill with some cached tokens already processed
        elif state in [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit]:
            output = self.forward_chunked_prefill_310(query, attn_metadata, output)
        # Condition for SpecDecoding: Specified for mtp, which is not supported yet.
        else:
            raise NotImplementedError(f"AscendAttentionState: {state} is not supported for 310P currently.")
        return output
