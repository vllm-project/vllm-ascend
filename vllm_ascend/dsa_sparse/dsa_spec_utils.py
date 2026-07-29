# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA KV-cache plane 的集中语义判定工具。

DSA 通过 vllm-ascend 的运行时 patch 接入，而部分 KV-cache spec 类型来自
vLLM 原生模块或 vllm-ascend patch 后的 live class。这里集中判断
“Indexer dense plane / MLA resident plane”，并兼容按值导入 alias 和 patch
前后 class identity 不一致的多进程时序，避免容量规划、worker cache 绑定和
DRAM arena 创建各自维护一套类型判断。

本模块只识别 spec 语义，不创建 KV group、不计算容量，也不安装 patch。
"""

from __future__ import annotations

from typing import Any

from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import IndexerKVSpec


def _isinstance_live(spec: Any, attr_name: str) -> bool:
    import vllm.v1.kv_cache_interface as kv_cache_interface

    live_cls = getattr(kv_cache_interface, attr_name, None)
    return live_cls is not None and isinstance(spec, live_cls)


def is_dsa_indexer_spec(spec: KVCacheSpec) -> bool:
    """Return whether ``spec`` is the dense indexer-cache plane."""
    return (
        isinstance(spec, IndexerKVSpec)
        or _isinstance_live(spec, "IndexerKVSpec")
        or type(spec).__name__ == "IndexerKVSpec"
    )


def is_dsa_mla_resident_spec(spec: KVCacheSpec) -> bool:
    """Return whether ``spec`` is the GLM MLA resident-cache plane.

    Do not classify a generic ``FullAttentionSpec`` as MLA.  v0.23 may add
    separate cache groups for speculative models or other attention layers;
    taking ownership of those groups would corrupt both the DSA resident map
    and the native cache lifecycle.
    """
    if is_dsa_indexer_spec(spec):
        return False
    return (
        isinstance(spec, MLAAttentionSpec)
        or _isinstance_live(spec, "MLAAttentionSpec")
        or type(spec).__name__ == "AscendMLAAttentionSpec"
    )
