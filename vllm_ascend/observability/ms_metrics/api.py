# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP command that sets the MS metric level on a running server."""

from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from vllm.engine.protocol import EngineClient
from vllm.logger import logger

from vllm_ascend.observability.ms_metrics.control import apply_metrics_level_command
from vllm_ascend.observability.ms_metrics.level import canonical_metric_level

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/set_metrics_level")
async def set_metrics_level(
    raw_request: Request,
    level: Annotated[str, Query()],
) -> JSONResponse:
    """Set the process-local MS metric level on the API, every EngineCore, and every worker.

    ``level`` is ``off``, ``basic``, ``detail``, ``debug``, or ``0`` through ``3``.
    """
    try:
        canonical = canonical_metric_level(level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    canonical = await apply_metrics_level_command(engine_client(raw_request), canonical)
    logger.info("[ms_metrics] set metric level to %s", canonical)
    return JSONResponse(content={"status": "ok", "level": canonical})


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
