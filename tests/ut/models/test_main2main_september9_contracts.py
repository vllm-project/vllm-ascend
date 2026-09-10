# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.v1.attention.backend import AttentionBackend

from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPImpl
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl, AscendSFADSADCPImpl
from vllm_ascend.utils import vllm_version_is


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
