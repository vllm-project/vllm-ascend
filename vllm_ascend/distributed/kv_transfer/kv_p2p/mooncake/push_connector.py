# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""P-side push connector for heterogeneous Mooncake KV transfer (910B NPU -> GPU).

Fills the V2 ``MooncakePushConnector`` slot: inherits ``MooncakeBaseConnector``'s
facade forwarding framework while wiring the scheduler and worker to the
heterogeneous implementations. The scheduler tracks requests to send (pure
producer); the worker performs the one-shot NHD->HND push transfer.

Pairs with upstream vLLM's V1 ``MooncakeConnector`` on the D side (P-push wire
protocol); the D side is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakeBaseConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import (
    MooncakeHeterogeneousConnectorMetadata,
    MooncakeHeterogeneousPushConnectorWorker,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
        KVConnectorMetadata,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str
TransferId = str


class MooncakeHeterogeneousPushConnectorScheduler(MooncakeBaseConnectorScheduler):
    """P-side scheduler (pure producer; does not receive KV)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # p_req_id -> (transfer_id, request, local_block_ids)
        self._reqs_need_send: dict[ReqId, tuple[TransferId, Request, list[int]]] = {}

    def on_new_request(self, request: Request) -> None:
        """P is a pure producer; no per-request scheduler bookkeeping."""

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """P does not load KV, so there is no connector output to consume."""

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        """P is the source, not a consumer, so it never matches external KV."""
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        """Record a request to send (D address comes from kv_transfer_params)."""
        kv_transfer_params = request.kv_transfer_params or {}
        transfer_id = kv_transfer_params.get("transfer_id", "")
        if transfer_id:
            self._reqs_need_send[request.request_id] = (transfer_id, request, [])

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Put requests whose block_ids are filled (computed) into reqs_to_send."""
        meta = MooncakeHeterogeneousConnectorMetadata()
        finished: list[ReqId] = []
        for req_id, (transfer_id, request, local_block_ids) in self._reqs_need_send.items():
            # request_finished already filled local_block_ids when computed.
            if local_block_ids:
                meta.reqs_to_send[req_id] = (transfer_id, local_block_ids)
                finished.append(req_id)
        for req_id in finished:
            del self._reqs_need_send[req_id]
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict | None]:
        """Mark a request computed and fill block_ids; delay block release until
        the transfer is confirmed (only when tracked and has blocks).

        Mirrors upstream MooncakeConnector.request_finished:
          - no transfer_id / not in _reqs_need_send -> release immediately
            (return False), else the scheduler keeps the block but the worker
            never reports completion -> permanent block leak.
          - empty block_ids -> release immediately (return False).
          - tracked and non-empty -> fill block_ids + return True (delayed
            release until get_finished).

        block_ids is a BlockIds tuple (per-group lists); single group for now,
        use the first group.
        """
        first_group = block_ids[0] if block_ids else []
        req_id = request.request_id
        if req_id not in self._reqs_need_send:
            return False, None
        if not first_group:
            return False, None
        transfer_id, request_obj, _ = self._reqs_need_send[req_id]
        self._reqs_need_send[req_id] = (transfer_id, request_obj, list(first_group))
        return True, None  # delayed block release until get_finished confirms transfer

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Heterogeneous P does not use the V2 handshake path; D discovers P via
        bootstrap + the adapter wire protocol, so this is a no-op."""


class MooncakeHeterogeneousPushConnector(MooncakeBaseConnector):
    """P-side push connector for heterogeneous PD KV transfer (910B NPU -> GPU).

    Inherits ``MooncakeBaseConnector``'s facade forwarding (scheduler/worker
    dispatch) and overrides the heterogeneous-specific entry points: KV cache
    layout (NHD on P, converted at transfer time) and ``start_load_kv`` (which
    uses the heterogeneous connector metadata type).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeHeterogeneousPushConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MooncakeHeterogeneousPushConnectorWorker(
                vllm_config, str(self.engine_id), kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """P NPU keeps NHD; the worker converts NHD->HND at transfer time.

        Unlike the same-vendor pull connector, the heterogeneous P side must not
        force HND on the NPU KV cache (the NPU attention backend expects NHD).
        The NHD->HND conversion happens in the worker's _do_transfer when
        building the hnd_buf for RDMA.
        """
        return None

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict) -> bool:
        """Non-layerwise (one-shot transfer); no piecewise support needed."""
        return False

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        """Trigger the one-shot push transfer using the heterogeneous metadata."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeHeterogeneousConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)


__all__ = [
    "MooncakeHeterogeneousPushConnector",
    "MooncakeHeterogeneousPushConnectorScheduler",
]
