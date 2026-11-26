# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV cache layout facts shared by the platform, the worker and the attention backends.

A leaf module without device imports: the FULL-graph gate runs while the engine
configuration is validated, before any device is initialized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as envs_vllm
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Cache blocks that store heads outermost ([B, H, N, C]; the HND family): LBHNC, the
# canonical name of the legacy HND alias, and BLHNC, its block-outer variant. FA3
# (vllm_ascend/attention/flash_attention_v3.py) stores exactly these layouts
# head-major and shares this constant with the FULL-graph gate. The two members are
# refused for different reasons; see check_full_cudagraph_kv_layout.
HEAD_MAJOR_KV_CACHE_LAYOUTS = frozenset({KVCacheLayout.LBHNC, KVCacheLayout.BLHNC})

FULL_CUDAGRAPH_HEAD_MAJOR_KV_LAYOUT_ERROR = (
    "cudagraph_mode=FULL (token-keyed graphs replayed for "
    "prefill/mixed batches) is not supported with the "
    "head-major KV cache layout (VLLM_KV_CACHE_LAYOUT=HND, "
    "LBHNC or BLHNC) on Ascend: the first replayed prefill graph crashes "
    "the NPU stream (device error 507011 'model stream execute "
    "failed', reproduced 3/3 on 910C). Use the validated graph "
    "mode for this layout, cudagraph_mode=FULL_DECODE_ONLY "
    "(decode-only graphs; prefill/mixed batches run eager), or "
    "unset VLLM_KV_CACHE_LAYOUT to keep the default (LBNHC) "
    "layout. See the graph mode guide "
    "(docs/source/user_guide/feature_guide/graph_mode.md) for "
    "the layout/cudagraph_mode support matrix."
)


def resolved_kv_cache_layout(vllm_config: VllmConfig | None) -> KVCacheLayout | None:
    """The KV cache layout this engine runs with, or None when nothing names one yet.

    An adopted value is answered by upstream's
    CacheConfig.get_resolved_kv_cache_layout(), the same call the attention
    backends make. Before the engine core adopts one, the explicit
    VLLM_KV_CACHE_LAYOUT choice (or a value pre-set by a standalone tool) is
    normalized by upstream's own name resolver, so the legacy HND/NHD spellings
    mean exactly what upstream makes of them.

    Args:
        vllm_config: The engine configuration; a partially built one is tolerated.

    Returns:
        The upstream KVCacheLayout member, or None when no layout is chosen yet.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    adopted = getattr(cache_config, "kv_cache_layout", None)
    if adopted is not None:
        resolve = getattr(cache_config, "get_resolved_kv_cache_layout", None)
        if callable(resolve):
            return resolve()
    name = adopted if adopted is not None else envs_vllm.VLLM_KV_CACHE_LAYOUT
    if name is None:
        return None
    # Upstream's alias map and enum lookup. The name is private upstream API, so
    # it is imported where it is used and only on the not-yet-resolved path.
    from vllm.config.cache import _layout_from_name

    return _layout_from_name(name)


def check_full_cudagraph_kv_layout(vllm_config: VllmConfig) -> None:
    """Refuse cudagraph_mode=FULL graphs over a head-major KV cache layout.

    The members of HEAD_MAJOR_KV_CACHE_LAYOUTS are refused for different reasons:

    * LBHNC (the legacy HND alias) is demonstrated broken with the Attention V1
      backend on Ascend 910C: the first replayed prefill graph crashes the NPU
      stream, device error 507011. On a tree without the head-major cache read of
      vllm-ascend PR #22649 the combination fails earlier instead, at graph
      capture (FIA error 561002, "key tensor must be contiguous").
    * BLHNC is refused fail-closed. FA3 stores it head-major like LBHNC and that
      path is unmeasured with FULL graphs. Attention V1 keeps its token-major
      shape for it, so the storage silently disagrees with the resolved layout:
      the engine runs, but it is not a coherent configuration.

    The check runs twice on purpose. The platform runs it while the engine
    configuration is validated, for the layout the user asked for. The worker
    runs it again when the engine core hands over the final layout
    (set_kv_cache_layout): a connector preference or a worker vote can pick a
    head-major layout after the first check and before any graph is captured.

    Args:
        vllm_config: The engine configuration.

    Raises:
        RuntimeError: FULL graphs would run over a head-major KV cache layout.
    """
    # Imported here, not at module level: vllm.platforms resolves this plugin while
    # vllm.config is still being imported, so the module must not import it.
    from vllm.config.compilation import CUDAGraphMode

    if getattr(getattr(vllm_config, "model_config", None), "enforce_eager", False):
        # No graph is captured or replayed.
        return
    if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
        return
    if resolved_kv_cache_layout(vllm_config) in HEAD_MAJOR_KV_CACHE_LAYOUTS:
        raise RuntimeError(FULL_CUDAGRAPH_HEAD_MAJOR_KV_LAYOUT_ERROR)
