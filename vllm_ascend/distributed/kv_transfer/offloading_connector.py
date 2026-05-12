# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""NPU-aware override of vLLM's :class:`OffloadingConnector`.

Adapts the upstream OffloadingConnector for the Ascend KV cache layout
where each attention layer's KV cache is registered as a tuple
``(k_cache, v_cache[, …])`` of *separately allocated* tensors, rather
than a single tensor (or a single ``(2, num_blocks, …)`` K|V-stacked
tensor as on FlashAttention).

Upstream's :class:`OffloadingConnectorWorker.register_kv_caches`:
  - asserts each ``kv_caches[layer_name]`` is a single ``torch.Tensor``
    (fails on Ascend's tuple layout);
  - canonicalises the per-layer tensor into one or more
    ``(num_blocks, page_size_bytes)`` int8 views which become a
    :class:`CanonicalKVCaches` passed to ``spec.get_handlers``.

vllm-ascend's :class:`CpuNpuOffloadingHandler` consumes the raw
``dict[str, tuple[Tensor, ...]]`` directly and does its own per-sub-tensor
bookkeeping (``data_ptr`` / ``stride(0) * element_size`` per K/V), so the
canonicalisation step is not needed on NPU. This worker simply forwards
the original ``kv_caches`` dict (plus the per-layer attention backends
that ``NPUOffloadingSpec.get_handlers`` expects) to the spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class NPUOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Worker that handles Ascend's per-layer ``(k_cache, v_cache[, …])`` tuple."""

    def register_kv_caches(
        self,
        kv_caches: dict[
            str,
            torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        ],
    ) -> None:
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            layer_names,
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend() for layer_name in layer_names if layer_name in layers
        }

        # NPU spec keeps the dict-of-tuples layout and the
        # ``(kv_caches, attn_backends)`` ``get_handlers`` signature
        # used by the existing :class:`CpuNpuOffloadingHandler`.
        for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches, attn_backends):
            self.worker.register_handler(src_cls, dst_cls, handler)


class NPUOffloadingConnector(OffloadingConnector):
    """:class:`OffloadingConnector` wired to :class:`NPUOffloadingConnectorWorker`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # Replace the upstream worker with the NPU-aware variant; reuse
        # the spec already created by ``super().__init__`` so we do not
        # construct the (potentially expensive) OffloadingSpec twice.
        if role == KVConnectorRole.WORKER and self.connector_worker is not None:
            self.connector_worker = NPUOffloadingConnectorWorker(self.connector_worker.spec)
