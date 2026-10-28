# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import cast

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_pool_bytes_per_block


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Declare the device capacity used by Ascend's existing cache placements."""

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        # Ascend may allocate several independent tensors for one block ID.
        # Use the same divisor as placement, including shared layer slots.
        # This builder is a main-core API. The helper has a version-gated
        # one-argument release definition that mypy sees before the main one.
        pool_bytes = cast(Callable[[VllmConfig, list[KVCacheGroupSpec]], int], _ascend_pool_bytes_per_block)
        return pool_bytes(vllm_config, kv_cache_groups)
