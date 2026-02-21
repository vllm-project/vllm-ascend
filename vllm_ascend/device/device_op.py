# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.
#

import torch
import torch_npu
from vllm.logger import init_logger

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

logger = init_logger(__name__)
_RESHAPE_CACHE_DEBUG_ONCE_PRINTED = False
_RESHAPE_CACHE_FORCE_FALLBACK_ON_310P = True


def _tensor_meta(name, tensor):
    if tensor is None:
        return f"{name}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=<{type(tensor).__name__}>"
    return (
        f"{name}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, stride={tuple(tensor.stride())}, "
        f"contiguous={tensor.is_contiguous()})"
    )


class BaseDeviceAdaptor:
    @staticmethod
    def _reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping):
        """
        Fallback path for non-contiguous KV cache tensors.

        This is used when hybrid attention+mamba sharing makes K/V cache views
        non-contiguous and backend fused scatter ops are unavailable (e.g. 310P).
        """
        slots = slot_mapping.reshape(-1).to(dtype=torch.long)
        valid = slots >= 0
        if not torch.any(valid):
            return

        if not bool(valid.all()):
            key = key[valid]
            value = value[valid]
            slots = slots[valid]

        # 310P cache layout: [num_blocks, hidden_dim/16, block_size, 16]
        if key_cache.ndim == 4 and key_cache.shape[-1] == 16:
            block_size = key_cache.shape[2]
            block_idx = torch.div(slots, block_size, rounding_mode="floor")
            token_idx = torch.remainder(slots, block_size)

            key_flat = key.reshape(key.shape[0], -1)
            value_flat = value.reshape(value.shape[0], -1)
            if key_flat.shape[-1] % 16 != 0 or value_flat.shape[-1] % 16 != 0:
                raise RuntimeError(
                    "Fallback reshape_and_cache expects hidden size divisible by 16 "
                    f"for 310P layout, got key={key_flat.shape[-1]}, value={value_flat.shape[-1]}."
                )
            key_fmt = key_flat.view(key_flat.shape[0], -1, 16)
            value_fmt = value_flat.view(value_flat.shape[0], -1, 16)

            key_cache[block_idx, :, token_idx, :] = key_fmt
            value_cache[block_idx, :, token_idx, :] = value_fmt
            return

        # Default cache layout: [num_blocks, block_size, num_kv_heads, head_size]
        if key_cache.ndim == 4 and value_cache.ndim == 4:
            block_size = key_cache.shape[1]
            block_idx = torch.div(slots, block_size, rounding_mode="floor")
            token_idx = torch.remainder(slots, block_size)
            key_cache[block_idx, token_idx, :, :] = key
            value_cache[block_idx, token_idx, :, :] = value
            return

        raise RuntimeError(
            "Unsupported non-contiguous KV cache layout in fallback path: "
            f"key_cache shape={tuple(key_cache.shape)}, value_cache shape={tuple(value_cache.shape)}"
        )

    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        global _RESHAPE_CACHE_DEBUG_ONCE_PRINTED
        if not _RESHAPE_CACHE_DEBUG_ONCE_PRINTED:
            _RESHAPE_CACHE_DEBUG_ONCE_PRINTED = True
            logger.warning("[RAC_DEBUG_ONCE] %s", _tensor_meta("key", key))
            logger.warning("[RAC_DEBUG_ONCE] %s", _tensor_meta("value", value))
            logger.warning("[RAC_DEBUG_ONCE] %s", _tensor_meta("key_cache", key_cache))
            logger.warning("[RAC_DEBUG_ONCE] %s", _tensor_meta("value_cache", value_cache))
            logger.warning("[RAC_DEBUG_ONCE] %s", _tensor_meta("slot_mapping", slot_mapping))

        # 310P path: kv cache is typically FRACTAL-NZ-like 4D layout whose
        # setup with `_npu_reshape_and_cache` is unstable for several model
        # signatures (e.g. qwen3-next). Prefer deterministic PyTorch fallback
        # to ensure correctness and avoid worker crashes.
        if (
            _RESHAPE_CACHE_FORCE_FALLBACK_ON_310P
            and key_cache.ndim == 4
            and value_cache.ndim == 4
            and key_cache.shape[-1] == 16
            and value_cache.shape[-1] == 16
        ):
            logger.warning(
                "[RAC_DEBUG_ONCE] force fallback writer for 310P KV cache layout."
            )
            cls._reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping)
            return

        # Hybrid attention+mamba sharing can produce non-contiguous KV views.
        # `_npu_reshape_and_cache` requires contiguous cache tensors.
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            logger.warning("[RAC_DEBUG_ONCE] non-contiguous cache detected, using pytorch fallback writer.")
            cls._reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping)
            return
        try:
            torch_npu._npu_reshape_and_cache(
                key=key.contiguous(),
                value=value.contiguous(),
                key_cache=key_cache,
                value_cache=value_cache,
                slot_indices=slot_mapping,
            )
        except Exception:
            logger.warning(
                "[RAC_DEBUG_ERR] _npu_reshape_and_cache failed with: %s | %s | %s | %s | %s",
                _tensor_meta("key", key),
                _tensor_meta("value", value),
                _tensor_meta("key_cache", key_cache),
                _tensor_meta("value_cache", value_cache),
                _tensor_meta("slot_mapping", slot_mapping),
            )
            logger.warning(
                "[RAC_DEBUG_ERR] fallback to pytorch reshape_and_cache writer after setup failure."
            )
            cls._reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping)


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key, value=value.contiguous(), key_cache=key_cache, value_cache=value_cache, slot_mapping=slot_mapping
        )


def get_device_adaptor():
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] | None = get_device_adaptor()
