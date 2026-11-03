# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Minimal 1P1D proxy for whole-request Mooncake push.

Both legs receive the original prompt concurrently. D samples every output
token; the token sampled on P is discarded. Each retry needs a new transfer ID.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


def build_requests(body: dict, transfer_id: str, host: str, port: int, engine_id: str, tp_size: int):
    if body.get("n", 1) != 1 or body.get("best_of", 1) != 1:
        raise ValueError("The initial push proxy supports n=best_of=1")
    prompt = body.get("prompt")
    if isinstance(prompt, list) and (not prompt or not all(type(token) is int for token in prompt)):
        raise ValueError("Use one text prompt or one token-ID prompt per push request")
    if "kv_transfer_params" in body:
        raise ValueError("The push proxy owns kv_transfer_params")
    producer = dict(body)
    for name in ("max_completion_tokens", "stream_options", "stop", "stop_token_ids", "min_tokens"):
        producer.pop(name, None)
    producer.update(
        max_tokens=1,
        min_tokens=1,
        stream=False,
        ignore_eos=True,
        kv_transfer_params={"do_remote_decode": True, "push_transfer_id": transfer_id},
    )
    consumer = dict(body)
    consumer["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "push_transfer_id": transfer_id,
        "remote_host": host,
        "remote_port": port,
        "remote_engine_id": engine_id,
        "remote_tp_size": tp_size,
    }
    return producer, consumer


def create_app(
    prefill_url: str,
    decode_url: str,
    prefill_kv_host: str,
    prefill_kv_port: int,
    prefill_engine_id: str,
    tp_size: int = 1,
    request_timeout: float = 300.0,
) -> FastAPI:
    """Own shared backend clients and concurrently route one P/D request pair."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    async def handle(request: Request):
        """Start both legs, then hand backend cleanup to the response stream.

        Before response headers, a disconnect watcher cancels both requests.
        After headers, the relay owns task cancellation and response closure.
        """
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Expected one JSON request object")
        key = str(uuid4())
        try:
            producer, consumer = build_requests(body, key, prefill_kv_host, prefill_kv_port, prefill_engine_id, tp_size)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        client = app.state.client
        headers = {"X-Request-Id": key}
        if authorization := request.headers.get("authorization"):
            headers["Authorization"] = authorization
        api = request.url.path
        prefill_task = asyncio.create_task(client.post(prefill_url.rstrip("/") + api, json=producer, headers=headers))
        decode_request = client.build_request("POST", decode_url.rstrip("/") + api, json=consumer, headers=headers)
        decode_task = asyncio.create_task(client.send(decode_request, stream=True))
        response = None
        disconnected = asyncio.Event()

        async def watch_disconnect():
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    disconnected.set()
                    return

        disconnect_task = asyncio.create_task(watch_disconnect())
        try:
            pending = {prefill_task, decode_task, disconnect_task}
            while True:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if disconnect_task in done:
                    raise asyncio.CancelledError
                if prefill_task in done:
                    prefill_task.result().raise_for_status()
                    pending.discard(prefill_task)
                if decode_task in done:
                    response = decode_task.result()
                    break
            if response.is_error:
                content = await response.aread()
                await response.aclose()
                prefill_task.cancel()
                with suppress(asyncio.CancelledError):
                    await prefill_task
                return Response(content, status_code=response.status_code, media_type="application/json")
        except BaseException as error:
            prefill_task.cancel()
            decode_task.cancel()
            await asyncio.gather(prefill_task, decode_task, return_exceptions=True)
            if response is not None:
                await response.aclose()
            if isinstance(error, httpx.HTTPStatusError):
                return Response(
                    error.response.content,
                    status_code=error.response.status_code,
                    media_type="application/json",
                )
            if isinstance(error, httpx.RequestError):
                raise HTTPException(status_code=502, detail="P/D backend request failed") from error
            raise
        finally:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)

        if disconnected.is_set():
            prefill_task.cancel()
            await asyncio.gather(prefill_task, return_exceptions=True)
            await response.aclose()
            raise asyncio.CancelledError

        async def relay():
            iterator = response.aiter_bytes()
            chunk_task = None
            prefill_checked = False
            try:
                while True:
                    chunk_task = asyncio.create_task(anext(iterator))
                    pending = {chunk_task}
                    if not prefill_checked:
                        pending.add(prefill_task)
                    while True:
                        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        if prefill_task in done:
                            prefill_task.result().raise_for_status()
                            prefill_checked = True
                            pending.discard(prefill_task)
                        if chunk_task in done:
                            break
                    try:
                        yield chunk_task.result()
                    except StopAsyncIteration:
                        break
                if not prefill_checked:
                    (await prefill_task).raise_for_status()
            finally:
                if chunk_task is not None:
                    chunk_task.cancel()
                    await asyncio.gather(chunk_task, return_exceptions=True)
                prefill_task.cancel()
                await asyncio.gather(prefill_task, return_exceptions=True)
                await response.aclose()

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    app.add_api_route("/v1/completions", handle, methods=["POST"])
    app.add_api_route("/v1/chat/completions", handle, methods=["POST"])
    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--prefill-kv-host", required=True)
    parser.add_argument("--prefill-kv-port", type=int, required=True)
    parser.add_argument("--prefill-engine-id", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(
            args.prefill_url,
            args.decode_url,
            args.prefill_kv_host,
            args.prefill_kv_port,
            args.prefill_engine_id,
            args.tp_size,
            args.request_timeout,
        ),
        host=args.host,
        port=args.port,
    )
