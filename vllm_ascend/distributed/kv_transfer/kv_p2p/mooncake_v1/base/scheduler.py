# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side contract shared by the two mooncake V1 connectors.

The pull and layerwise schedulers duplicate no implementation: their five
same-named public methods are all different bodies. Only the four methods the
facade really calls (and that both classes implement) are declared abstract, so
a subclass that forgets one fails at instantiation instead of at request time.
request_finished_all_groups exists only on the layerwise scheduler and is
therefore documented here, not declared abstract.
"""

from abc import ABC, abstractmethod

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.sched.output import SchedulerOutput


class MooncakeConnectorSchedulerBase(ABC):
    """Request/scheduler contract used by the connector facade."""

    @abstractmethod
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Return (num_external_tokens, is_async) for one request."""

    @abstractmethod
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """Remember the block allocation chosen for an external-prefix request."""

    @abstractmethod
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Translate the scheduler output into this step's connector metadata."""

    @abstractmethod
    def request_finished(self, request, block_ids):
        """Release per-request scheduler state and return (delay_free, params)."""


__all__ = ["MooncakeConnectorSchedulerBase"]
