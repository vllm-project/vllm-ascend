# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-owned Mooncake registry with explicit context-parallel topology."""

from dataclasses import dataclass

import httpx
import msgspec
from fastapi import HTTPException
from pydantic import Field, model_validator
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeTransferMetadataGroups


def register_workers_with_bootstrap(
    host: str,
    port: int,
    engine_id: str,
    dp_rank: int,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
    dcp_size: int,
    encoded_metadata: bytes,
) -> None:
    """Register one P engine's workers with the shared bootstrap server."""
    groups = msgspec.msgpack.decode(encoded_metadata, type=MooncakeTransferMetadataGroups)
    registrations = []
    for pp_rank, pp_metadata in groups.metadata_by_pp_rank.items():
        for pcp_rank, pcp_metadata in pp_metadata.metadata_by_pcp_rank.items():
            for tp_rank, worker in pcp_metadata.metadata_by_tp_rank.items():
                if not worker.local_ip or not 1 <= worker.handshake_port <= 65535:
                    raise ValueError("P worker must advertise its Ready listener address")
                registrations.append(
                    {
                        "engine_id": engine_id,
                        "dp_rank": dp_rank,
                        "pp_rank": pp_rank,
                        "pcp_rank": pcp_rank,
                        "tp_rank": tp_rank,
                        "pp_size": pp_size,
                        "pcp_size": pcp_size,
                        "tp_size": tp_size,
                        "dcp_size": dcp_size,
                        "addr": make_zmq_path("tcp", worker.local_ip, worker.handshake_port),
                    }
                )
    with httpx.Client(timeout=envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT) as client:
        for payload in registrations:
            response = client.post(make_zmq_path("http", host, port) + "/register", json=payload)
            response.raise_for_status()


class AscendRegisterWorkerPayload(RegisterWorkerPayload):
    dp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    addr: str = Field(min_length=1)
    pcp_rank: int = Field(default=0, ge=0)
    tp_size: int = Field(default=1, ge=1)
    pp_size: int = Field(default=1, ge=1)
    pcp_size: int = Field(default=1, ge=1)
    dcp_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_context_ranks(self) -> "AscendRegisterWorkerPayload":
        if self.pcp_rank >= self.pcp_size or self.tp_rank >= self.tp_size or self.pp_rank >= self.pp_size:
            raise ValueError("Context-parallel rank must be smaller than its group size")
        return self


@dataclass
class AscendEngineEntry:
    engine_id: str
    pcp_size: int
    dcp_size: int
    # PCP rank -> TP rank -> PP rank -> address. In the paired vLLM layout,
    # dcp_rank = (tp_rank * pcp_size + pcp_rank) % dcp_size.
    worker_addr: dict[int, dict[int, dict[int, str]]]
    tp_size: int = 1
    pp_size: int = 1


class AscendMooncakeBootstrapServer(MooncakeBootstrapServer):
    """Reuse upstream HTTP/thread lifecycle with Ascend registration schemas.

    The DP -> engine_id query envelope remains compatible with the proxy.
    Worker discovery clients must understand the Ascend worker_addr schema.
    Routes run on the server event loop without awaits while mutating state.
    """

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.workers: dict[int, AscendEngineEntry] = {}

    def _register_routes(self):
        # Override both routes: the upstream response model drops CP fields.
        self.app.post("/register")(self.register_worker)
        self.app.get("/query", response_model=dict[int, AscendEngineEntry])(self.query)

    async def register_worker(self, payload: AscendRegisterWorkerPayload):
        entry = self.workers.get(payload.dp_rank)
        if entry is not None:
            if entry.engine_id != payload.engine_id:
                raise HTTPException(status_code=400, detail=f"Engine ID mismatch for dp_rank={payload.dp_rank}")
            if (entry.pcp_size, entry.dcp_size, entry.tp_size, entry.pp_size) != (
                payload.pcp_size,
                payload.dcp_size,
                payload.tp_size,
                payload.pp_size,
            ):
                raise HTTPException(status_code=400, detail="Context-parallel topology mismatch")
        else:
            entry = AscendEngineEntry(
                payload.engine_id, payload.pcp_size, payload.dcp_size, {}, payload.tp_size, payload.pp_size
            )
            self.workers[payload.dp_rank] = entry

        workers = entry.worker_addr.setdefault(payload.pcp_rank, {}).setdefault(payload.tp_rank, {})
        existing = workers.get(payload.pp_rank)
        if existing is not None and existing != payload.addr:
            raise HTTPException(status_code=400, detail="Worker is already registered with a different address")
        workers[payload.pp_rank] = payload.addr
        return {"status": "ok"}

    async def query(self) -> dict[int, AscendEngineEntry]:
        return self.workers
