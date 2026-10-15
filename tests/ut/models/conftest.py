# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def keypool_kernels(monkeypatch):
    def metadata(common, buffers, tokens, *_, **kwargs):
        n = common.num_reqs
        keys = "positions slot_mapping seq_lens raw_seq_lens query_start_loc cum_query_lens start_pos pool_tail"
        sizes = (tokens, tokens, n, n, n + 1, n, n, n)
        return tuple(getattr(buffers, key)[:size] for key, size in zip(keys.split(), sizes))

    kernels = SimpleNamespace(
        build_keypool_metadata=MagicMock(side_effect=metadata),
        build_tail_inputs=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.glm5_next_cann_keypool", kernels)
    return kernels
