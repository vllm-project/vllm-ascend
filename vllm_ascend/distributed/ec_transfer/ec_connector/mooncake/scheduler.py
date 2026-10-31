# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side request gating for asynchronous embedding Store GET."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.async_store_get import (
    AscendECMooncakeConnectorMetadata,
    AscendECMooncakeWorkerMetadata,
)


class _StoreGetState(Enum):
    """Describe GET progress, terminal outcomes, and detached request ownership."""

    PENDING = auto()
    HIT = auto()
    MISS = auto()
    FAILED = auto()
    ABANDONED = auto()


class AscendECMooncakeScheduler(ECMooncakeScheduler):
    """Defer a request while its asynchronous Store GET runs on the worker."""

    def __init__(self, vllm_config: Any) -> None:
        """Initialize asynchronous GET admission budgets and request waiters."""
        super().__init__(vllm_config)
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        extra = ec_config.ec_connector_extra_config
        self._store_get_enabled = self._cross_encoder_cache
        configured_bytes = int(extra.get("ascend_store_get_async_bytes", 2 * 1024**3))
        self._store_get_max_bytes = configured_bytes if configured_bytes > 0 else 0
        self._store_get_bytes = 0
        self._store_get_states: dict[str, _StoreGetState] = {}
        self._store_get_sizes: dict[str, int] = {}
        self._store_get_waiters: dict[str, set[str]] = {}
        self._store_get_to_dispatch: dict[str, TensorSpec] = {}
        self._store_get_to_consume: set[str] = set()
        self._store_get_to_discard: set[str] = set()

    def _release_store_get(self, identifier: str) -> None:
        """Remove GET bookkeeping and return its admission byte budget."""
        self._store_get_bytes -= self._store_get_sizes.pop(identifier)
        del self._store_get_states[identifier]
        self._store_get_waiters.pop(identifier, None)

    def ensure_cache_available(self, request: Any, num_computed_tokens: int) -> bool:
        """Admit uncached images and gate requests until their GETs resolve."""
        upstream_ready = super().ensure_cache_available(request, num_computed_tokens)
        if not self._store_get_enabled or not self._is_producer:
            return upstream_ready

        store_ready = True
        for index, feature in enumerate(request.mm_features):
            if feature.mm_position.offset + feature.mm_position.length <= num_computed_tokens:
                continue
            identifier = feature.identifier
            if feature.modality != "image" or identifier in self._local_cache:
                continue
            state = self._store_get_states.get(identifier)
            if state in (_StoreGetState.PENDING, _StoreGetState.ABANDONED):
                # An abandoned GET is still in flight. A new waiter reuses it
                # without a second dispatch or releasing its byte budget.
                self._store_get_states[identifier] = _StoreGetState.PENDING
                self._store_get_waiters[identifier].add(request.request_id)
                store_ready = False
                continue
            if state is not None:
                self._store_get_waiters[identifier].add(request.request_id)
                continue

            spec = self._encoder_output_spec(request, index)
            if self._store_get_bytes + spec.nbytes > self._store_get_max_bytes:
                # Bypassed items retain the existing synchronous Store path.
                continue
            self._store_get_states[identifier] = _StoreGetState.PENDING
            self._store_get_sizes[identifier] = spec.nbytes
            self._store_get_waiters[identifier] = {request.request_id}
            self._store_get_bytes += spec.nbytes
            self._store_get_to_dispatch[identifier] = spec
            store_ready = False
        return upstream_ready and store_ready

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        """Mark terminal results for worker consumption after cache allocation."""
        super().update_state_after_alloc(request, index)
        identifier = request.mm_features[index].identifier
        state = self._store_get_states.get(identifier)
        if state in (
            _StoreGetState.HIT,
            _StoreGetState.MISS,
            _StoreGetState.FAILED,
        ):
            self._store_get_to_consume.add(identifier)
            self._release_store_get(identifier)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendECMooncakeConnectorMetadata:
        """Transfer queued GET commands to the worker for this scheduler step."""
        base = super().build_connector_meta(scheduler_output)
        metadata = AscendECMooncakeConnectorMetadata(
            **vars(base),
            store_get_requests=self._store_get_to_dispatch,
            store_get_consume=self._store_get_to_consume,
            store_get_discard=self._store_get_to_discard,
        )
        self._store_get_to_dispatch = {}
        self._store_get_to_consume = set()
        self._store_get_to_discard = set()
        return metadata

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        """Apply worker outcomes and discard completed GETs with no waiters."""
        super().update_connector_output(connector_output)
        metadata = connector_output.ec_connector_worker_meta
        if not isinstance(metadata, AscendECMooncakeWorkerMetadata):
            return
        outcomes = (
            (metadata.store_get_hits, _StoreGetState.HIT),
            (metadata.store_get_misses, _StoreGetState.MISS),
            (metadata.store_get_failures, _StoreGetState.FAILED),
        )
        for identifiers, state in outcomes:
            for identifier in identifiers:
                current = self._store_get_states.get(identifier)
                if current is _StoreGetState.PENDING:
                    self._store_get_states[identifier] = state
                elif current is _StoreGetState.ABANDONED:
                    self._store_get_to_discard.add(identifier)
                    self._release_store_get(identifier)

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        """Detach a request while retaining ownership of any running GET."""
        result = super().request_finished(request)
        for feature in request.mm_features:
            identifier = feature.identifier
            state = self._store_get_states.get(identifier)
            if state is None:
                continue
            waiters = self._store_get_waiters[identifier]
            waiters.discard(request.request_id)
            if waiters:
                continue
            if identifier in self._store_get_to_dispatch:
                del self._store_get_to_dispatch[identifier]
                self._release_store_get(identifier)
            elif state in (_StoreGetState.PENDING, _StoreGetState.ABANDONED):
                self._store_get_states[identifier] = _StoreGetState.ABANDONED
            else:
                self._store_get_to_discard.add(identifier)
                self._release_store_get(identifier)
        return result

    def has_pending_push_work(self) -> bool:
        """Keep scheduling while upstream pushes or asynchronous GETs need work."""
        return (
            super().has_pending_push_work()
            or bool(self._store_get_to_dispatch)
            or any(
                state in (_StoreGetState.PENDING, _StoreGetState.ABANDONED) for state in self._store_get_states.values()
            )
        )
