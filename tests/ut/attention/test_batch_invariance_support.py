# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression: Ascend attention backends must advertise batch-invariance support.

vLLM #60271 ("Check batch-invariance support for every attention backend") added
a hard check in ``vllm/v1/worker/gpu/attn_utils.init_attn_backend`` on main:
``VLLM_BATCH_INVARIANT`` mode now rejects any attention backend whose
``supports_batch_invariance()`` returns False. The v0.31.0 lane has no such
check, so the Ascend backends gate their declaration on
``vllm_version_is("0.31.0")``.
"""

import pytest

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackend
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolTailBackend,
)
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.models.minimax_m3.msa_m3 import (
    AscendMiniMaxM3IndexerBackend,
    AscendMiniMaxM3SparseBackend,
)
from vllm_ascend.utils import vllm_version_is

_BACKENDS = [
    AscendAttentionBackend,
    AscendMLABackend,
    AscendSFABackend,
    AscendDSABackend,
    DeepseekV41CacheBackend,
    AscendSFAIndexerBackend,
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolTailBackend,
    AscendMiniMaxM3IndexerBackend,
    AscendMiniMaxM3SparseBackend,
]

try:
    from vllm_ascend.attention.fa3_v1 import AscendFABackend
except ImportError:
    # flash_attn_npu_v3 is optional; the backend is only registered when it
    # is installed.
    AscendFABackend = None
else:
    _BACKENDS.append(AscendFABackend)


@pytest.mark.parametrize("backend", _BACKENDS, ids=lambda backend: backend.__name__)
def test_ascend_attention_backend_advertises_batch_invariance(backend) -> None:
    assert backend.supports_batch_invariance() is (not vllm_version_is("0.31.0"))


@pytest.mark.parametrize(
    "backend",
    [AscendAttentionBackend310, AscendC8MXFPAttentionBackend],
    ids=lambda backend: backend.__name__,
)
def test_ascend_attention_backend_subclass_inherits_batch_invariance(backend) -> None:
    assert backend.supports_batch_invariance() is (not vllm_version_is("0.31.0"))
