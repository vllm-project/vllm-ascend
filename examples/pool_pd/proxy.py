# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sequential P -> KV pool -> D completions proxy with streaming decode."""

import argparse
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


def create_app(prefill_url: str, decode_url: str, transport=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=300, transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def complete(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a JSON object")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) and not (
            isinstance(prompt, list) and prompt and all(type(token) is int for token in prompt)
        ):
            raise HTTPException(400, "Pool PD v1 accepts one text or token-ID prompt")
        if body.get("n", 1) != 1 or body.get("best_of", 1) != 1:
            raise HTTPException(400, "Pool PD v1 requires n=1, best_of=1")
        if "kv_transfer_params" in body:
            raise HTTPException(400, "The proxy owns kv_transfer_params")
        prefill = dict(body)
        prefill.update(max_tokens=1, min_tokens=0, stream=False, ignore_eos=True, stop=[], echo=False)
        prefill.pop("stop_token_ids", None)
        prefill.pop("stream_options", None)
        try:
            response = await app.state.client.post(prefill_url.rstrip("/") + "/v1/completions", json=prefill)
            if response.is_error:
                return Response(response.content, status_code=response.status_code, media_type="application/json")
            params = response.json().get("kv_transfer_params")
            if not isinstance(params, dict) or params.get("pool_pd", {}).get("status") != "ready":
                raise HTTPException(502, "Prefill did not publish a complete KV snapshot")
            decode = dict(body)
            decode["kv_transfer_params"] = params
            if body.get("stream"):
                upstream = await app.state.client.send(
                    app.state.client.build_request("POST", decode_url.rstrip("/") + "/v1/completions", json=decode),
                    stream=True,
                )
                if upstream.is_error:
                    content = await upstream.aread()
                    await upstream.aclose()
                    return Response(content, status_code=upstream.status_code, media_type="application/json")

                async def relay():
                    try:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk
                    finally:
                        await upstream.aclose()

                return StreamingResponse(relay(), media_type="text/event-stream")
            response = await app.state.client.post(decode_url.rstrip("/") + "/v1/completions", json=decode)
            return Response(response.content, status_code=response.status_code, media_type="application/json")
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Pool PD backend request failed") from exc

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", default="http://127.0.0.1:18080")
    parser.add_argument("--decode-url", default="http://127.0.0.1:18081")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    uvicorn.run(create_app(args.prefill_url, args.decode_url), host=args.host, port=args.port)
