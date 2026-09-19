# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import resolve_request_hash_block_size
from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_resolve_kv_cache_block_sizes
# isort: on


def _hybrid_kv_cache_config() -> KVCacheConfig:
    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float16)
    mamba_spec = MambaSpec(block_size=32, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="none")
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )


def _vllm_config(*, enable_prefix_caching: bool, connector_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode="align",
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        kv_transfer_config=object() if connector_enabled else None,
    )


def test_ascend_store_uses_vllm_request_hash_granularity() -> None:
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata.kv_cache_utils.resolve_kv_cache_block_sizes",
        return_value=(128, 8),
    ) as resolver:
        vllm_config = object()
        kv_cache_config = object()
        assert resolve_request_hash_block_size(vllm_config, kv_cache_config, 128) == 8
        resolver.assert_called_once_with(kv_cache_config, vllm_config)

        resolver.reset_mock()
        assert resolve_request_hash_block_size(object(), None, 128) == 128
        resolver.assert_not_called()


def test_dcp_hybrid_hashing_stays_enabled_for_kv_connectors() -> None:
    expected_scheduler_block_size = math.lcm(16, 32) * 2
    cases = [
        (False, False, expected_scheduler_block_size),
        (False, True, math.gcd(16, 32)),
        (True, False, math.gcd(16, 32)),
    ]
    for enable_prefix_caching, connector_enabled, expected_hash_block_size in cases:
        scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
            _hybrid_kv_cache_config(),
            _vllm_config(enable_prefix_caching=enable_prefix_caching, connector_enabled=connector_enabled),
        )

        assert scheduler_block_size == expected_scheduler_block_size
        assert hash_block_size == expected_hash_block_size
