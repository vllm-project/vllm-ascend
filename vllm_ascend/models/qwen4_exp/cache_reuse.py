# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp spec eligibility for layerwise buffer reuse."""

from types import MappingProxyType

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiddenStateCacheSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules
from vllm_ascend.utils import vllm_version_is


def _is_gdn_state(spec: KVCacheSpec) -> bool:
    # GDN owns conv and recurrent state. PLE owns one dense conv component.
    return isinstance(spec, MambaSpec) and len(spec.shapes) == 2


def _is_qsa_main_attention(spec: KVCacheSpec) -> bool:
    # QSA main attention holds K and V (head_size_v > 0); the raw indexer
    # circular buffer stores keys only (head_size_v == 0).
    return isinstance(spec, AttentionSpec) and spec.head_size_v > 0


if vllm_version_is("0.34.0"):
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

    QWEN4_EXP_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
        {
            FullAttentionSpec: True,
            MambaSpec: _is_gdn_state,
            MLAAttentionSpec: True,
            CircularBufferSpec: False,
            HiddenStateCacheSpec: False,
        }
    )
else:
    QWEN4_EXP_SPEC_REUSE_RULES = MappingProxyType(
        {
            AttentionSpec: _is_qsa_main_attention,
            MambaSpec: _is_gdn_state,
            MLAAttentionSpec: True,
            HiddenStateCacheSpec: False,
        }
    )
