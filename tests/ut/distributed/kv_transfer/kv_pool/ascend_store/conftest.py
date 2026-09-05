# SPDX-License-Identifier: Apache-2.0
"""Keep the real attention gate isolated between AscendStore tests."""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence


@pytest.fixture(autouse=True)
def isolated_attention_gate(monkeypatch):
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
