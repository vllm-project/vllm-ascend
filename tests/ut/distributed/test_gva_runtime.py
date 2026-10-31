# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.memfabric.runtime import MemFabricRuntimeRegistry


def test_runtime_claims_are_exclusive_and_reference_counted():
    registry = MemFabricRuntimeRegistry()
    signature = ("device0", "tp8", "pool-a")
    first = registry.acquire("offload", "sparse-kv", signature)
    second = registry.acquire("offload", "sparse-kv", signature)
    other_family = registry.acquire("bm", "engram", signature)

    with pytest.raises(RuntimeError, match="already claimed by owner='sparse-kv'"):
        registry.acquire("offload", "qwen-ple", signature)
    with pytest.raises(RuntimeError, match="already claimed"):
        registry.acquire("offload", "sparse-kv", ("device0", "tp4", "pool-a"))

    first.release()
    with pytest.raises(RuntimeError, match="already claimed"):
        registry.acquire("offload", "qwen-ple", signature)

    second.release()
    with registry.acquire("offload", "qwen-ple", signature):
        pass
    other_family.release()
