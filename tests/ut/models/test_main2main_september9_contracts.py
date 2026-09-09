# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPImpl
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl, AscendSFADSADCPImpl
from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("boundary", [0, 128, 192])
def test_upstream_coordinator_can_cache_kpool_tail_without_caching_or_pruning(boundary):
    manager = KpoolTailManager.__new__(KpoolTailManager)
    manager.block_pool = Mock()
    coordinator = SimpleNamespace(
        single_type_managers=(manager,),
        num_reprefillable_tokens=0,
        retention_interval=128,
        get_replay_boundary=lambda request: boundary,
    )
    # Exercise the real upstream caller, including #53945's new keyword.
    KVCacheCoordinator.cache_blocks(coordinator, SimpleNamespace(request_id="tail"), 200)
    manager.remove_skipped_blocks("tail", 200, 200)
    assert manager.get_num_common_prefix_blocks("tail") == 0
    assert manager.get_num_skipped_tokens(200) == 0
    assert manager.block_pool.mock_calls == []


@pytest.mark.parametrize("impl", [AscendAttentionDCPImpl, AscendMlaDCPImpl, AscendSFADCPImpl, AscendSFADSADCPImpl])
def test_upstream_backend_accepts_ascend_dcp_implementations(impl):
    assert impl.supports_dcp is True
    selected_backend = SimpleNamespace(get_impl_cls=lambda: impl)
    unsupported_backend = SimpleNamespace(get_impl_cls=lambda: SimpleNamespace(supports_dcp=False))
    if vllm_version_is("0.28.0"):
        # Release checks the implementation flag directly, not a backend method.
        assert selected_backend.get_impl_cls().supports_dcp is True
        assert unsupported_backend.get_impl_cls().supports_dcp is False
    else:
        assert AttentionBackend.supports_dcp.__func__(selected_backend) is True
        assert AttentionBackend.supports_dcp.__func__(unsupported_backend) is False
