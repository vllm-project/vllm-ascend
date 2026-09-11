# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_placement import build_kvpp_layer_layout, create_kvpp_cache_allocation_plan
from vllm_ascend.distributed.kvpp import BroadcastKVPPTransport
from vllm_ascend.distributed.parallel_state import get_kvpp_group
from vllm_ascend.worker.kvpp_cache import get_kvpp_cache_specs


class KVPPRuntime:
    """Bind contiguous cache storage to the shared layer prefetch scheduler."""

    def __init__(self, scheduler: KVPPScheduler | None = None) -> None:
        self.scheduler = scheduler

    @classmethod
    def create_from_kv_cache(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any] | None = None,
    ) -> KVPPRuntime:
        config = KVPPConfig.from_vllm_config(vllm_config)
        if config.size <= 1:
            return cls()
        if kv_caches is None:
            kv_caches = {
                name: static_forward_context[name].kv_cache
                for group in kv_cache_config.kv_cache_groups
                for name in group.layer_names
            }
        group = get_kvpp_group()
        plan = create_kvpp_cache_allocation_plan(
            vllm_config, get_kvpp_cache_specs(kv_cache_config), group.rank_in_group
        )
        if not plan.layer_owner_ranks:
            return cls()
        layer_buffers = {}
        for name, bundle in plan.layer_bundles.items():
            if name not in plan.layer_owner_ranks:
                continue
            _, size = build_kvpp_layer_layout(bundle, plan.tensor_sizes, kv_cache_config.num_blocks)
            first = kv_caches[name][0]
            storage = first.untyped_storage()
            base = first.storage_offset() * first.element_size()
            # The allocator binds every component to this contiguous layer span.
            raw = torch.empty(0, dtype=torch.int8, device=first.device).set_(storage, base, (size,), (1,))
            layer_buffers[name] = raw
        scheduler = KVPPScheduler(
            transport=BroadcastKVPPTransport(group, plan.layer_owner_ranks, layer_buffers),
            attention_layer_names=tuple(layer_buffers),
        )
        for name in layer_buffers:
            static_forward_context[name].impl.layerwise_kv_cache_hook = scheduler
        return cls(scheduler)

    def prepare_forward(self, has_history: bool) -> None:
        if self.scheduler is not None:
            self.scheduler.schedule_forward(has_history)

    def complete_forward(self) -> None:
        if self.scheduler is not None:
            self.scheduler.complete_forward()


class KVPPScheduler:
    """Prefetch one layer ahead; Target execution ordinal selects scratch."""

    def __init__(self, transport: BroadcastKVPPTransport, attention_layer_names: tuple[str, ...]) -> None:
        self.transport = transport
        self.attention_layer_names = attention_layer_names
        self._has_history = False
        self._next_attention_layer_index = 0
        self._prefetch_future: Future[None] | None = None
        self._npu_device_id = torch.npu.current_device()
        self._kv_transfer_stream = torch.npu.Stream()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpp-prefetch")

    def schedule_forward(self, has_history: bool) -> None:
        self._has_history = has_history
        self._next_attention_layer_index = 0
        if has_history:
            self.start_layer_prefetch(self.attention_layer_names[0])

    def start_layer_prefetch(self, layer_name: str) -> None:
        cache_ready = torch.npu.Event()
        cache_ready.record(torch.npu.current_stream())
        self._prefetch_future = self._prefetch_executor.submit(self.run_layer_prefetch, layer_name, cache_ready)

    def run_layer_prefetch(self, layer_name: str, cache_ready: Any) -> None:
        torch.npu.set_device(self._npu_device_id)
        self.transport.prefetch(layer_name, cache_ready, self._kv_transfer_stream)

    def wait_for_layer(self, layer_name: str) -> None:
        if not self._has_history:
            return
        assert self._prefetch_future is not None
        self._prefetch_future.result()
        self._prefetch_future = None
        self._next_attention_layer_index += 1
        if self._next_attention_layer_index < len(self.attention_layer_names):
            self.start_layer_prefetch(self.attention_layer_names[self._next_attention_layer_index])

    def complete_forward(self) -> None:
        self._has_history = False
        self._next_attention_layer_index = 0
