# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-owned Mooncake registry with explicit context-parallel topology."""

from dataclasses import dataclass

from fastapi import HTTPException
from pydantic import Field, model_validator
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)


class AscendRegisterWorkerPayload(RegisterWorkerPayload):
    dp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    addr: str = Field(min_length=1)
    pcp_rank: int = Field(default=0, ge=0)
    pcp_size: int = Field(default=1, ge=1)
    dcp_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_context_ranks(self) -> "AscendRegisterWorkerPayload":
        if self.pcp_rank >= self.pcp_size:
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
            if (entry.pcp_size, entry.dcp_size) != (payload.pcp_size, payload.dcp_size):
                raise HTTPException(status_code=400, detail="Context-parallel topology mismatch")
        else:
            entry = AscendEngineEntry(payload.engine_id, payload.pcp_size, payload.dcp_size, {})
            self.workers[payload.dp_rank] = entry

        workers = entry.worker_addr.setdefault(payload.pcp_rank, {}).setdefault(payload.tp_rank, {})
        existing = workers.get(payload.pp_rank)
        if existing is not None and existing != payload.addr:
            raise HTTPException(status_code=400, detail="Worker is already registered with a different address")
        workers[payload.pp_rank] = payload.addr
        return {"status": "ok"}

    async def query(self) -> dict[int, AscendEngineEntry]:
        return self.workers
