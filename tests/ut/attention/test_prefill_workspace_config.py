# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.attention.utils import ascend_chunked_prefill_workspace_size


def workspace_config(**kwargs):
    # Normally supplied by init_ascend_config; no sparse offload is needed here.
    return AscendConfig(sparse_kv_offload_config=None, **kwargs)


def test_workspace_default_and_override():
    assert workspace_config().chunked_prefill_workspace_max_tokens == 128 * 1024
    assert (
        workspace_config(chunked_prefill_workspace_max_tokens=512 * 1024).chunked_prefill_workspace_max_tokens
        == 512 * 1024
    )


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "524288", None])
def test_workspace_rejects_invalid_config(value):
    with pytest.raises(ValidationError, match="chunked_prefill_workspace_max_tokens"):
        workspace_config(chunked_prefill_workspace_max_tokens=value)


@pytest.mark.parametrize(
    "cap,max_model_len,max_num_seqs,block_size,expected",
    [
        (128 * 1024, 200000, 64, 128, 128 * 1024),
        (512 * 1024, 200000, 64, 128, 512 * 1024),
        (512 * 1024, 1024, 1, 128, 8192),
        (128 * 1024, 200000, 2048, 128, 2048 * 128),
        (4096, 200000, 64, 128, 64 * 128),
        (512 * 1024, 1024, 64, 128, 4 * 64 * 128),
    ],
)
def test_workspace_sizing_preserves_heuristic(cap, max_model_len, max_num_seqs, block_size, expected):
    config = workspace_config(chunked_prefill_workspace_max_tokens=cap)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(block_size=block_size),
    )
    with patch.dict(ascend_chunked_prefill_workspace_size.__globals__, get_ascend_config=lambda: config):
        assert ascend_chunked_prefill_workspace_size(vllm_config) == expected
