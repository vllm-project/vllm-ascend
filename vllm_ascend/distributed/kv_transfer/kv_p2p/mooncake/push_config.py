# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Explicit compatibility gates for the first whole-request push implementation."""

import math
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

from vllm_ascend.ascend_config import get_ascend_config

DEFAULT_PUSH_TIMEOUT = 60.0
DEFAULT_HANDSHAKE_TIMEOUT = 10.0
DEFAULT_MAX_PENDING = 128


@dataclass(frozen=True)
class PushConfig:
    timeout: float
    handshake_timeout: float
    max_pending: int

    @classmethod
    def from_config(cls, config: VllmConfig, cache: KVCacheConfig) -> "PushConfig":
        parallel = config.parallel_config
        dimensions = (
            parallel.data_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
        if any(size != 1 for size in dimensions):
            raise ValueError("MooncakePushConnector initially requires DP=PP=PCP=DCP=1")
        if config.speculative_config is not None or config.model_config.use_mla:
            raise ValueError("MooncakePushConnector initially supports dense attention without speculative decoding")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("MooncakePushConnector initially requires --no-enable-prefix-caching")
        if not config.model_config.enforce_eager or config.scheduler_config.async_scheduling:
            raise ValueError("MooncakePushConnector initially requires --enforce-eager --no-async-scheduling")
        if get_ascend_config().kvpp_config.size != 1:
            raise ValueError("MooncakePushConnector does not initially support KVPP")
        if len(cache.kv_cache_groups) != 1:
            raise ValueError("MooncakePushConnector initially requires one FullAttention cache group")
        for group in cache.kv_cache_groups:
            spec = group.kv_cache_spec
            if type(spec) is not FullAttentionSpec or spec.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("MooncakePushConnector initially requires FP16/BF16 FullAttentionSpec groups")
        transfer = config.kv_transfer_config
        assert transfer is not None
        timeout = float(transfer.get_from_extra_config("push_timeout", DEFAULT_PUSH_TIMEOUT))
        handshake_timeout = float(transfer.get_from_extra_config("push_handshake_timeout", DEFAULT_HANDSHAKE_TIMEOUT))
        max_pending = int(transfer.get_from_extra_config("push_max_pending", DEFAULT_MAX_PENDING))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("push_timeout must be finite and positive")
        if not math.isfinite(handshake_timeout) or handshake_timeout <= 0:
            raise ValueError("push_handshake_timeout must be finite and positive")
        if max_pending <= 0:
            raise ValueError("push_max_pending must be positive")
        return cls(timeout, handshake_timeout, max_pending)
