# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_ascend.ascend_config import AscendConfig


def test_engram_overlap_default_and_explicit_disable():
    assert AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False)).multistream_engram_overlap is True
    assert (
        AscendConfig(
            multistream_engram_overlap=False, sparse_kv_offload_config=SimpleNamespace(enabled=False)
        ).multistream_engram_overlap
        is False
    )
