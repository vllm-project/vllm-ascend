#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import logging
import os

import torch
from vllm.distributed import get_pcp_group

from vllm_ascend.platform import ModelConfig
from vllm_ascend.utils import singleton

logger = logging.getLogger(__name__)

# Default maximum attention mask size (in bytes): 16GB
# This can be configured via VLLM_ASCEND_MAX_ATTN_MASK_MEMORY environment variable
DEFAULT_MAX_ATTN_MASK_MEMORY_GB = 16

# Warning threshold: warn when attention mask memory exceeds this value (in GB)
ATTN_MASK_MEMORY_WARNING_THRESHOLD_GB = 4


def _get_max_attn_mask_memory_bytes() -> int:
    """Get the maximum allowed attention mask memory from environment variable.

    Returns:
        Maximum allowed memory in bytes.
    """
    env_value = os.environ.get("VLLM_ASCEND_MAX_ATTN_MASK_MEMORY")
    if env_value is not None:
        try:
            max_memory_gb = float(env_value)
            if max_memory_gb <= 0:
                logger.warning(
                    "VLLM_ASCEND_MAX_ATTN_MASK_MEMORY must be positive, using default value: %d GB",
                    DEFAULT_MAX_ATTN_MASK_MEMORY_GB,
                )
                max_memory_gb = DEFAULT_MAX_ATTN_MASK_MEMORY_GB
            return int(max_memory_gb * 1024 * 1024 * 1024)
        except ValueError:
            logger.warning(
                "Invalid VLLM_ASCEND_MAX_ATTN_MASK_MEMORY value: '%s', using default value: %d GB",
                env_value,
                DEFAULT_MAX_ATTN_MASK_MEMORY_GB,
            )
    return DEFAULT_MAX_ATTN_MASK_MEMORY_GB * 1024 * 1024 * 1024


def _estimate_attn_mask_memory(max_seq_len: int, dtype: torch.dtype) -> int:
    """Estimate the memory required for attention mask tensor.

    Args:
        max_seq_len: Maximum sequence length.
        dtype: Data type of the attention mask.

    Returns:
        Estimated memory in bytes.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    # Attention mask shape is (max_seq_len, max_seq_len)
    num_elements = max_seq_len * max_seq_len
    return num_elements * element_size


def _check_attn_mask_memory(max_seq_len: int, dtype: torch.dtype) -> None:
    """Check if the attention mask memory requirement is within limits.

    This function estimates the memory required for the attention mask and
    compares it against the configured maximum. If the estimated memory
    exceeds the limit, a RuntimeError is raised with a clear error message.

    Args:
        max_seq_len: Maximum sequence length.
        dtype: Data type of the attention mask.

    Raises:
        RuntimeError: If estimated memory exceeds the configured maximum.
    """
    estimated_memory = _estimate_attn_mask_memory(max_seq_len, dtype)
    max_allowed_memory = _get_max_attn_mask_memory_bytes()

    estimated_memory_gb = estimated_memory / (1024 * 1024 * 1024)
    max_allowed_memory_gb = max_allowed_memory / (1024 * 1024 * 1024)

    # Log warning if memory usage is high but within limits
    if estimated_memory_gb > ATTN_MASK_MEMORY_WARNING_THRESHOLD_GB:
        logger.warning(
            "Attention mask for max_seq_len=%d with dtype=%s requires %.2f GB memory. "
            "This may cause performance issues or OOM errors on devices with limited memory. "
            "Consider reducing max_model_len or setting VLLM_ASCEND_MAX_ATTN_MASK_MEMORY "
            "environment variable to adjust the limit.",
            max_seq_len,
            dtype,
            estimated_memory_gb,
        )

    if estimated_memory > max_allowed_memory:
        raise RuntimeError(
            f"Attention mask memory requirement ({estimated_memory_gb:.2f} GB) exceeds "
            f"the maximum allowed limit ({max_allowed_memory_gb:.2f} GB). "
            f"The attention mask has shape ({max_seq_len}, {max_seq_len}) with dtype={dtype}, "
            f"which requires O(max_model_len^2) memory.\n\n"
            f"Possible solutions:\n"
            f"  1. Reduce max_model_len to a smaller value (e.g., --max-model-len 4096)\n"
            f"  2. Increase the memory limit by setting environment variable:\n"
            f"     export VLLM_ASCEND_MAX_ATTN_MASK_MEMORY=<memory_in_gb>\n"
            f"  3. Use a device with more memory\n\n"
            f"Note: This error prevents a potential TBE subprocess crash that would "
            f"otherwise produce an unclear error message like "
            f"'TBE Subprocess[task_distribute] raise error[], main process disappeared!'"
        )


def _generate_attn_mask(max_seq_len, dtype):
    # Check memory requirement before allocating tensors
    _check_attn_mask_memory(max_seq_len, dtype)

    # Construct lower triangle matrix.
    mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    mask_value = float("-inf") if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(mask_flag, mask_value)
    return attn_mask


@singleton
class AttentionMaskBuilder:
    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.mla_mask = None
        self.chunked_prefill_attn_mask = None
        self.pcp_mla_mask = None
        self.swa_mask = None

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            estimated_memory = _estimate_attn_mask_memory(max_seq_len, dtype)
            logger.debug(
                "Generating attention mask: max_seq_len=%d, dtype=%s, estimated_memory=%.2f MB",
                max_seq_len,
                dtype,
                estimated_memory / (1024 * 1024),
            )
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        assert self.attn_mask_cache is not None, "Something is wrong in generate_attn_mask."
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous().to(self.device, non_blocking=True)

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        if self.chunked_prefill_attn_mask is None:
            self.chunked_prefill_attn_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        return self.chunked_prefill_attn_mask

    def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        if self.mla_mask is None or self.mla_mask.dtype != dtype:
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            prefill_mask = torch.triu(torch.ones(512, 512, device=self.device, dtype=dtype), 1)
            self.mla_mask = torch.where(prefill_mask == 1, mask_value, 0).to(dtype)
        return self.mla_mask

    def get_pcp_mla_mask(self, dtype: torch.dtype):
        if self.pcp_mla_mask is None or self.pcp_mla_mask.dtype != dtype:
            self.pcp_mla_mask = torch.triu(torch.ones(512, 512, device=self.device, dtype=dtype), 1)
        return self.pcp_mla_mask

    def get_swa_mask(self, dtype: torch.dtype, sliding_window):
        if self.swa_mask is None or self.swa_mask.dtype != dtype:
            if sliding_window is not None:
                mask = torch.ones(2048, 2048, dtype=torch.bool)
                triu_mask = torch.triu(mask, diagonal=1).to(self.device)
                tril_mask = torch.tril(mask, -sliding_window).to(self.device)
                self.swa_mask = triu_mask + tril_mask
        return self.swa_mask

    def get_attention_mask(self, model_config: ModelConfig):
        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()

    def get_final_mla_mask(self, model_config: ModelConfig):
        if get_pcp_group().world_size > 1:
            return self.get_pcp_mla_mask(model_config.dtype)
        # Prefill stages use 512x512 mask with appropriate dtype
        return self.get_mla_mask(model_config.dtype)
