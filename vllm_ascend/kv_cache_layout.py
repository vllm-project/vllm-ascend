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

# What fails without the gate was measured on two builds; the message keeps the
# two measurements apart and presents neither as universal.
FULL_CUDAGRAPH_HEAD_MAJOR_KV_LAYOUT_ERROR = (
    "cudagraph_mode=FULL (token-keyed graphs replayed for "
    "prefill/mixed batches) is not supported with the "
    "head-major KV cache layout (VLLM_KV_CACHE_LAYOUT=HND or "
    "LBHNC) on Ascend, and the engine is refused at startup "
    "instead of failing later. What fails later depends on the "
    "build: with the head-major KV cache read of vllm-ascend "
    "PR #22649, the first replayed prefill graph crashes the NPU "
    "stream (device error 507011 'model stream execute failed', "
    "observed 3/3 on 910C); without it, the Attention V1 backend "
    "rejects the head-major cache at its first read (FIA error "
    "561002 / EZ0037, 'key tensor must be contiguous'), in every "
    "graph mode. Use the default layout (unset "
    "VLLM_KV_CACHE_LAYOUT; LBNHC, alias NHD) with "
    "cudagraph_mode=FULL. Keep the head-major layout only "
    "together with PR #22649, and use cudagraph_mode=FULL_DECODE_ONLY "
    "with it (decode-only graphs; prefill/mixed batches run eager), "
    "the graph mode validated with that layout on 910C. See the "
    "graph mode guide "
    "(docs/source/user_guide/feature_guide/graph_mode.md) for "
    "the layout/cudagraph_mode support matrix."
)

# BLHNC is refused with FULL graphs as a validation restriction, not because a
# crash was demonstrated under it: its storage family depends on the attention
# backend, and FULL graphs have not been validated under either family.
FULL_CUDAGRAPH_BLHNC_KV_LAYOUT_ERROR = (
    "cudagraph_mode=FULL (token-keyed graphs replayed for "
    "prefill/mixed batches) is refused with the KV cache layout "
    "BLHNC on Ascend as a validation restriction, not because of "
    "a demonstrated crash: BLHNC has no single storage family on "
    "Ascend. The FA3 backend stores it head-major like HND, where "
    "FULL graphs are refused, and that path is unvalidated; the "
    "Attention V1 backend keeps token-major storage for it, so "
    "the resolved layout and the physical storage disagree. Use "
    "VLLM_KV_CACHE_LAYOUT=HND or LBHNC for head-major storage, or "
    "VLLM_KV_CACHE_LAYOUT=LBNHC or NHD for the token-major storage "
    "Attention V1 would allocate for BLHNC, or unset "
    "VLLM_KV_CACHE_LAYOUT to keep the default (LBNHC) layout. See "
    "the graph mode guide "
    "(docs/source/user_guide/feature_guide/graph_mode.md) for the "
    "layout/cudagraph_mode support matrix."
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
    """Refuse cudagraph_mode=FULL graphs over a KV cache layout that cannot take them.

    The members of HEAD_MAJOR_KV_CACHE_LAYOUTS are refused for different
    reasons, each with its own message:

    * LBHNC (the legacy HND alias) cannot be used with FULL graphs on any
      build, but what fails depends on the build, and neither measurement is
      universal. With the head-major cache read of vllm-ascend PR #22649 the
      first replayed prefill graph crashes the NPU stream (device error 507011,
      observed 3/3 on 910C). On a build without it the layout does not work in
      any graph mode: the Attention V1 backend rejects the head-major cache at
      its first read (FIA error 561002, "key tensor must be contiguous").
      FULL_DECODE_ONLY is the validated graph mode for the layout only together
      with #22649; the default token-major layout (LBNHC, alias NHD) is the
      supported path for FULL graphs.
    * BLHNC is refused fail-closed as a validation restriction, not as a
      demonstrated crash. FA3 stores it head-major like LBHNC and that path is
      unmeasured with FULL graphs. Attention V1 keeps its token-major shape for
      it, so the storage silently disagrees with the resolved layout: the
      engine runs, but it is not a coherent configuration.

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
    layout = resolved_kv_cache_layout(vllm_config)
    if layout is KVCacheLayout.BLHNC:
        raise RuntimeError(FULL_CUDAGRAPH_BLHNC_KV_LAYOUT_ERROR)
    if layout in HEAD_MAJOR_KV_CACHE_LAYOUTS:
        raise RuntimeError(FULL_CUDAGRAPH_HEAD_MAJOR_KV_LAYOUT_ERROR)
