from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import vllm.entrypoints.launchers.api_server.entry as _entry
import vllm.v1.engine.async_llm as _async_llm_mod
import vllm.v1.engine.core_client as _core_client_mod
from vllm.logger import init_logger


logger = init_logger("vllm_ascend.patch_eplb_export_api")


router = APIRouter()
_original_entry_build_app = _entry.build_app


def _call_export_api(port: int, path: str, workload: str) -> dict[str, Any]:
    body = json.dumps({"path": path, "workload": workload}, separators=(",", ":")).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/eplb/export_placement",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode())


def _auto_export_before_shutdown(engine_client: Any) -> None:
    vllm_config = getattr(engine_client, "vllm_config", None)
    if vllm_config is None:
        return
    parallel_config = vllm_config.parallel_config
    if not parallel_config.enable_eplb:
        return
    if getattr(parallel_config, "data_parallel_size", 1) <= 1:
        return

    additional_config = getattr(vllm_config, "additional_config", None) or {}
    eplb_config = additional_config.get("eplb_config") or {}
    export_path = eplb_config.get("export_path")
    export_ports = eplb_config.get("export_api_ports") or []
    if not export_path or not export_ports:
        return
    workload = eplb_config.get("export_workload") or "MRv2 dynamic EPLB deployment"
    with ThreadPoolExecutor(max_workers=len(export_ports)) as pool:
        list(
            pool.map(
                lambda port: _call_export_api(int(port), export_path, workload),
                export_ports,
            )
        )


def _make_shutdown_with_export(original_shutdown):
    def shutdown_with_export(self, timeout=None):
        try:
            _auto_export_before_shutdown(self)
        except Exception:
            logger.exception("Failed to export EPLB deployment before API server shutdown")
        return original_shutdown(self, timeout)

    return shutdown_with_export


_original_async_llm_shutdown = _async_llm_mod.AsyncLLM.shutdown
_original_mp_client_shutdown = _core_client_mod.MPClient.shutdown
_async_llm_mod.AsyncLLM.shutdown = _make_shutdown_with_export(_original_async_llm_shutdown)
_core_client_mod.MPClient.shutdown = _make_shutdown_with_export(_original_mp_client_shutdown)


@router.post("/v1/eplb/export_placement")
async def export_placement(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    path = body.get("path")
    workload = body.get("workload", "MRv2 dynamic EPLB deployment")
    timeout = body.get("timeout", 1800)
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="path must be a non-empty string")

    engine_client = request.app.state.engine_client
    results: list[Any] | None = await engine_client.collective_rpc(
        "export_eplb_placement",
        timeout=timeout,
        args=(path, workload),
    )
    if results is None:
        return JSONResponse(content={"results": None})
    return JSONResponse(content={"results": results})


def _build_app_with_eplb_export_api(*args, **kwargs):
    app = _original_entry_build_app(*args, **kwargs)
    app.include_router(router)
    return app


_entry.build_app = _build_app_with_eplb_export_api
