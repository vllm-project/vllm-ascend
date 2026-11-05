#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
disagg_encoder_proxy.py

Proxy that routes OpenAI-compatible “/v1/chat/completions” requests to two
clusters:
  • encode  (multimodal feature extraction)
  • decode  (language-model inference)

For MM input we:
    1. Extract *every* image/audio item.
    2. Fire N concurrent requests to the encoder cluster
       (one request per item, with **all text removed**).
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import time
import uuid
from collections.abc import AsyncIterator
from enum import Enum

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("proxy")

# Default overall client timeout kept for backwards compatibility; per-stage
# timeouts override it only when the corresponding CLI value is greater than 0.
_LEGACY_TOTAL_TIMEOUT_SECONDS = 100_000.0

app = FastAPI()
encode_session: aiohttp.ClientSession | None = None
prefill_session: aiohttp.ClientSession | None = None
decode_session: aiohttp.ClientSession | None = None

###############################################################################
# Utils
###############################################################################


MM_TYPES = {"image_url", "audio_url", "input_audio"}


class EncoderDispatchMode(str, Enum):
    SINGLE = "single"
    FANOUT = "fanout"


def _record_encode_failure(url: str) -> None:
    """Record an encoder URL that failed so later routing can avoid it."""
    if not hasattr(app.state, "encode_failures"):
        app.state.encode_failures = {}
    app.state.encode_failures[url] = time.time()


def _stage_timeout(stage: str) -> aiohttp.ClientTimeout | None:
    """Return the configured per-stage timeout; None means keep the session default.

    aiohttp treats an explicit timeout=None as ClientTimeout(total=None) and
    disables the session total timeout, so callers must omit the keyword
    entirely when this returns None.
    """
    seconds = getattr(app.state, "stage_timeouts", {}).get(stage, 0.0)
    return aiohttp.ClientTimeout(total=seconds) if seconds and seconds > 0 else None


def _stage_timeout_kwargs(stage: str) -> dict:
    """Return request kwargs carrying the stage timeout only when it is configured."""
    timeout = _stage_timeout(stage)
    return {"timeout": timeout} if timeout is not None else {}


def _classify_outbound_error(stage: str, url: str, exc: BaseException) -> HTTPException:
    """Map connection or timeout failures of one stage to 502/504."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return HTTPException(status_code=504, detail={"stage": stage, "url": url, "error": "timeout"})
    if isinstance(exc, aiohttp.ClientConnectionError):
        return HTTPException(
            status_code=502,
            detail={"stage": stage, "url": url, "error": "connection", "reason": str(exc)},
        )
    if isinstance(exc, aiohttp.ClientError):
        return HTTPException(
            status_code=502,
            detail={"stage": stage, "url": url, "error": "client_error", "reason": str(exc)},
        )
    return HTTPException(status_code=502, detail={"stage": stage, "url": url, "error": "outbound", "reason": str(exc)})


def _downstream_status_error(stage: str, url: str, status: int, body: bytes) -> HTTPException:
    """Map a non-2xx downstream status: keep 4xx, collapse other failures to 502."""
    message = body.decode("utf-8", errors="replace")[:500]
    if 400 <= status < 500:
        return HTTPException(
            status_code=status,
            detail={
                "stage": stage,
                "url": url,
                "status": status,
                "error": "downstream_client_error",
                "message": message,
            },
        )
    return HTTPException(
        status_code=502,
        detail={"stage": stage, "url": url, "status": status, "error": "downstream_server_error", "message": message},
    )


async def _post_stage(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
    stage: str,
    req_id: str,
) -> tuple[int, bytes]:
    """POST one stage request; the response is always released by the context manager."""
    if session is None:
        raise HTTPException(status_code=500, detail=f"{stage} session not initialized")
    try:
        async with session.post(url, json=payload, headers=headers, **_stage_timeout_kwargs(stage)) as resp:
            status = resp.status
            body = await resp.read()
    except HTTPException:
        raise
    except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
        raise _classify_outbound_error(stage, url, exc) from exc
    logger.debug("[%s] %s stage returned status %d from %s", req_id, stage, status, url)
    return status, body


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


async def _encode_fanout(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
):
    logger.info("[%s] Processing multimodal items...", req_id)

    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        logger.info("[%s] No multimodal items, skipping encoder", req_id)
        return  # nothing to do

    if encode_session is None:
        raise HTTPException(status_code=500, detail="Encode session not initialized")

    logger.info("[%s] got %d multimodal items...", req_id, len(mm_items))

    # Round-robin over encode servers to distribute load a bit
    target_urls = [e_urls[i % len(e_urls)] for i in range(len(mm_items))]

    tasks = []
    for idx, (item, target_url) in enumerate(zip(mm_items, target_urls)):
        # Derive a *child* request id:  <parent>:<index>:<random-short>
        child_req_id = f"{req_id}:{idx}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id}

        encoder_req = {
            # You *may* need to keep additional fields
            "model": orig_request.get("model"),
            "messages": [
                {"role": "user", "content": [item]},
            ],
            # Only need 1 token so the server actually runs the encoder path
            "max_tokens": 1,
            "stream": False,
        }
        tasks.append(
            _post_stage(
                encode_session,
                f"{target_url}/v1/chat/completions",
                encoder_req,
                headers,
                "encode",
                child_req_id,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures: list[tuple[int, str, BaseException]] = []
    for idx, result in enumerate(results):
        if isinstance(result, BaseException):
            _record_encode_failure(target_urls[idx])
            failures.append((idx, target_urls[idx], result))
            continue
        status, body = result
        if not (200 <= status < 300):
            _record_encode_failure(target_urls[idx])
            error = _downstream_status_error("encode", f"{target_urls[idx]}/v1/chat/completions", status, body)
            failures.append((idx, target_urls[idx], error))

    if failures:
        logger.error("[%s] %d/%d encoder requests failed", req_id, len(failures), len(mm_items))
        statuses = [error.status_code if isinstance(error, HTTPException) else 502 for _, _, error in failures]
        client_errors = [status for status in statuses if 400 <= status < 500]
        if client_errors:
            status_code = client_errors[0]
        elif 504 in statuses:
            status_code = 504
        else:
            status_code = 502
        raise HTTPException(
            status_code=status_code,
            detail={
                "stage": "encode",
                "error": "encoder_request_failed",
                "items": [
                    {"index": idx, "url": url, "reason": str(getattr(error, "detail", error))}
                    for idx, url, error in failures
                ],
            },
        )

    logger.info("[%s] All %d encoder requests completed successfully", req_id, len(mm_items))


async def _encode_single_request(
    orig_request: dict,
    e_url: str,
    req_id: str,
) -> None:
    """Send the full multimodal request to one encoder instance."""
    logger.info("[%s] Processing multimodal items...", req_id)

    request_data = copy.deepcopy(orig_request)
    headers = {"x-request-id": req_id}
    request_data["max_tokens"] = 1
    request_data["stream"] = False
    request_data.pop("stream_options", None)
    if "max_completion_tokens" in request_data:
        request_data["max_completion_tokens"] = 1

    if encode_session is None:
        raise HTTPException(status_code=500, detail="Encode session not initialized")

    url = f"{e_url}/v1/chat/completions"
    try:
        status, body = await _post_stage(encode_session, url, request_data, headers, "encode", req_id)
        if not (200 <= status < 300):
            raise _downstream_status_error("encode", url, status, body)
    except Exception:
        _record_encode_failure(e_url)
        raise

    logger.debug("Encoder processing completed successfully for req_id: %s", req_id)


async def fanout_encoder_primer(
    orig_request: dict,
    req_id: str,
):
    mode = app.state.encoder_dispatch_mode

    if mode == EncoderDispatchMode.SINGLE:
        e_url = random.choice(app.state.e_urls)
        await _encode_single_request(orig_request, e_url, req_id)

    elif mode == EncoderDispatchMode.FANOUT:
        await _encode_fanout(orig_request, app.state.e_urls, req_id)

    else:
        raise RuntimeError(f"Unknown encoder dispatch mode: {mode}")


async def maybe_prefill(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """
    - Do prefill-only task if p_url exist;
    - Return modified request data with kv transfer params (for nixl connector)
    - Else, skip and return the original request data for decode
    """
    if p_url:
        logger.info("[%s] Processing through prefill: %s", req_id, p_url)

        prefill_response_json = await process_prefill_stage(req_data, p_url, req_id)
        # for nixl connector to facilitate kv transfer...
        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        return req_data
    else:
        return req_data


async def process_prefill_stage(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """Process request through Prefill stage and return its JSON response."""
    logger.info("[%s] Sending prefill request to: %s", req_id, p_url)

    prefill_request = req_data.copy()
    prefill_request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    if "stream_options" in prefill_request:
        del prefill_request["stream_options"]

    if prefill_session is None:
        raise HTTPException(status_code=500, detail="Prefill session not initialized")

    headers = {"x-request-id": req_id}
    url = f"{p_url}/v1/chat/completions"
    status, body = await _post_stage(prefill_session, url, prefill_request, headers, "prefill", req_id)
    if not (200 <= status < 300):
        logger.error("[%s] Prefill request failed with status %d", req_id, status)
        raise _downstream_status_error("prefill", url, status, body)
    try:
        prefill_response_json = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail={"stage": "prefill", "url": url, "error": "invalid_json", "reason": str(exc)},
        ) from exc
    logger.info("[%s] Prefill request completed successfully", req_id)
    return prefill_response_json


def has_mm_input(request_data: dict):
    if "messages" not in request_data:
        return False
    for message in request_data["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for content_item in message["content"]:
            if content_item.get("type") in ["image_url", "audio_url", "input_audio"]:
                return True
    return False


###############################################################################
# Middleware for request/response logging
###############################################################################


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and responses"""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Log incoming request
    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )

        return response
    except Exception as e:
        # Log errors
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, prefill_session, decode_session
    timeout = aiohttp.ClientTimeout(total=_LEGACY_TOTAL_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(limit=0, force_close=False, keepalive_timeout=0)
    encode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    if app.state.p_urls:
        # only setup if prefill instance(s) exist
        prefill_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def forward_non_stream(req_data: dict, req_id: str, p_url: str, d_url: str) -> dict:
    try:
        # Step 1: Process through Encoder instance (if has MM input)
        async def run_encoder():
            await fanout_encoder_primer(req_data, req_id)

        if has_mm_input(req_data):
            await non_stream_retry_wrap(run_encoder)

        # Step 2: Process through Prefill instance
        async def run_prefill():
            return await maybe_prefill(req_data, p_url, req_id)

        req_data = await non_stream_retry_wrap(run_prefill)

        async def run_decode_non_stream():
            # Step 3: Process through Decode instance
            logger.info("[%s] Forwarding to decode: %s", req_id, d_url)
            headers = {"x-request-id": req_id}
            url = f"{d_url}/v1/chat/completions"

            # Non-streaming response
            if decode_session is None:
                raise HTTPException(status_code=500, detail="Decode session not initialized")

            status, body = await _post_stage(decode_session, url, req_data, headers, "decode", req_id)
            if not (200 <= status < 300):
                raise _downstream_status_error("decode", url, status, body)
            try:
                return json.loads(body)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=502,
                    detail={"stage": "decode", "url": url, "error": "invalid_json", "reason": str(exc)},
                ) from exc

        return await non_stream_retry_wrap(run_decode_non_stream)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_non_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


async def stream_retry_wrap(forward_func, max_retries: int = 3, delay: float = 0.001):
    last_exc = None
    first_chunk_sent = False
    for attempt in range(max_retries):
        try:
            async for chunk in forward_func():
                first_chunk_sent = True
                yield chunk
            return
        except Exception as e:
            if first_chunk_sent:
                raise
            if isinstance(e, HTTPException) and e.status_code < 500:
                raise
            last_exc = e
            logger.warning(
                "attempt %s / %s failed retrying... ",
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay * (attempt + 1))

    raise RuntimeError(f"all {max_retries} retries failed.") from last_exc


async def non_stream_retry_wrap(forward_func, max_retries: int = 3, delay: float = 0.001):
    last_exc = None
    for attempt in range(max_retries):
        try:
            result = await forward_func()
            return result
        except Exception as e:
            if isinstance(e, HTTPException) and e.status_code < 500:
                raise
            last_exc = e
            logger.warning(
                "attempt %s / %s failed retrying... ",
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay * (attempt + 1))
    raise RuntimeError(f"all {max_retries} retries failed.") from last_exc


async def forward_stream(req_data: dict, req_id: str, p_url: str, d_url: str) -> AsyncIterator[str]:
    try:
        # Step 1: Process through Encoder instance (if has MM input)
        async def run_encoder():
            await fanout_encoder_primer(req_data, req_id)

        if has_mm_input(req_data):
            await non_stream_retry_wrap(run_encoder)

        # Step 2: Process through Prefill instance
        async def run_prefill():
            return await maybe_prefill(req_data, p_url, req_id)

        req_data = await non_stream_retry_wrap(run_prefill)

        async def run_decode_stream():
            # Step 3: Process through Decode instance
            logger.info("[%s] Starting streaming from decode: %s", req_id, d_url)
            headers = {"x-request-id": req_id}
            url = f"{d_url}/v1/chat/completions"

            # Streaming response
            if decode_session is None:
                raise HTTPException(status_code=500, detail="Decode session not initialized")

            try:
                async with decode_session.post(
                    url,
                    json=req_data,
                    headers=headers,
                    **_stage_timeout_kwargs("decode"),
                ) as resp:
                    if not (200 <= resp.status < 300):
                        body = await resp.read()
                        raise _downstream_status_error("decode", url, resp.status, body)
                    async for chunk in resp.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode("utf-8", errors="ignore")
            except HTTPException:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                raise _classify_outbound_error("decode", url, exc) from exc

            logger.info("[%s] Streaming completed", req_id)

        async for chunk in stream_retry_wrap(run_decode_stream):
            yield chunk

    except HTTPException:
        logger.exception("[%s] HTTPException in forward_stream", req_id)
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy streaming error: {str(e)}") from e


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
        d_url = random.choice(app.state.d_urls)

        is_streaming = req_data.get("stream", False)

        if is_streaming:
            return StreamingResponse(
                forward_stream(req_data, req_id, p_url, d_url),
                media_type="text/event-stream",
            )
        result = await forward_non_stream(req_data, req_id, p_url, d_url)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Request processing error: {str(e)}") from e


@app.get("/v1/models")
async def list_models():
    if decode_session is None:
        raise HTTPException(status_code=500, detail="Decode session not initialized")
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health_check():
    async def healthy(urls, session):
        if not urls:
            return "empty"
        for u in urls:
            try:
                if session is None:
                    return "unhealthy"
                async with session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return "unhealthy"
        return "healthy"

    e_status, p_status, d_status = await asyncio.gather(
        healthy(app.state.e_urls, encode_session),
        healthy(app.state.p_urls, prefill_session),
        healthy(app.state.d_urls, decode_session),
    )

    overall_healthy = all(status != "unhealthy" for status in (e_status, p_status, d_status))

    status_code = 200 if overall_healthy else 503

    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": e_status,
            "prefill_cluster": p_status,
            "decode_cluster": d_status,
        },
        status_code=status_code,
    )


###############################################################################
# Simple profiler fan-out (unchanged except for sessions)
###############################################################################


async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> dict | None:
    """
    POST `payload` to `url`.

    Returns
    -------
    • The decoded JSON body on success (2xx)
    • None if the endpoint does not exist (404)
    • Raises for anything else.
    """
    try:
        if session is None:
            return None
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:  # profiling disabled on that server
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        # Pass 404 through the branch above, re-raise everything else
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise
    except Exception:
        # Network errors etc.: propagate
        raise


async def _profile_cmd(cmd: str, payload: dict, e_url: str, p_url: str, d_url: str):
    """
    Fire & forget to both clusters, tolerate 404.
    """
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(encode_session, f"{e_url}/{cmd}_profile", payload, headers)
    prefill_task = (
        _post_if_available(prefill_session, f"{p_url}/{cmd}_profile", payload, headers)
        if p_url is not None
        else asyncio.sleep(0)
    )
    decode_task = _post_if_available(decode_session, f"{d_url}/{cmd}_profile", payload, headers)

    encode_res, prefill_res, decode_res = await asyncio.gather(encode_task, prefill_task, decode_task)

    # If *all* clusters said “I don’t have that route”, surface an error
    if encode_res is prefill_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,  # may be None
        "prefill": prefill_res,  # may be None
        "decode": decode_res,  # may be None
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, p_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, p_url, d_url)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the proxy CLI parser (kept separate so it can be unit tested)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-servers-urls",
        required=True,
        help='Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") to enable E->P->D, '
        'set "disable" or "none" to enable E->PD',
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=True,
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )

    parser.add_argument(
        "--encoder-dispatch-mode",
        choices=["single", "fanout"],
        default="single",
        help="Encoder dispatch mode: single (one request) or fanout (per-MM-item)",
    )
    parser.add_argument(
        "--encode-timeout-seconds",
        type=float,
        default=0.0,
        help="Encode stage timeout in seconds; 0 = shared overall client timeout (legacy behavior)",
    )
    parser.add_argument(
        "--prefill-timeout-seconds",
        type=float,
        default=0.0,
        help="Prefill stage timeout in seconds; 0 = shared overall client timeout (legacy behavior)",
    )
    parser.add_argument(
        "--decode-timeout-seconds",
        type=float,
        default=0.0,
        help="Decode stage timeout in seconds; 0 = shared overall client timeout (legacy behavior)",
    )
    return parser


def configure_app_state(args: argparse.Namespace) -> None:
    """Populate app.state from parsed CLI arguments."""
    app.state.e_urls = [u.strip() for u in args.encode_servers_urls.split(",") if u.strip()]
    app.state.d_urls = [u.strip() for u in args.decode_servers_urls.split(",") if u.strip()]
    # handle prefill instances
    if args.prefill_servers_urls.lower() in ("disable", "none", ""):
        app.state.p_urls = []
        logger.info("Disaggregated prefill phase explicitly disabled by user. Running E + PD...")
    else:
        app.state.p_urls = [u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()]
        logger.info("Disaggregated prefill phase is enabled. Running E + P + D...")

    app.state.encoder_dispatch_mode = EncoderDispatchMode(args.encoder_dispatch_mode)
    app.state.stage_timeouts = {
        "encode": args.encode_timeout_seconds,
        "prefill": args.prefill_timeout_seconds,
        "decode": args.decode_timeout_seconds,
    }
    app.state.encode_failures = {}


def main() -> None:
    """Parse CLI arguments, configure the app and serve the proxy."""
    args = build_arg_parser().parse_args()
    configure_app_state(args)

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_urls)
    logger.info("Prefill instances %s", app.state.p_urls)
    logger.info("Decode servers: %s", app.state.d_urls)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )


if __name__ == "__main__":
    main()
