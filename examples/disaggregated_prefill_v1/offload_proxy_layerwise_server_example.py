# SPDX-License-Identifier: Apache-2.0
"""Layerwise disaggregated proxy with an isolated startup warmup request.

This proxy reuses the routing implementation from
``load_balance_proxy_layerwise_server_example.py``.  After the HTTP listener
is available it sends one request for each paired Prefill/Decode instance
through ``Decode -> warmup metaserver -> Prefill``.  With P=A,B and D=C,D,
the pairs are A+C and B+D; it does not run the Cartesian product.  Normal
completion requests are held until all pairs finish.

The proxy runs ``max(num_prefillers, num_decoders)`` warmup edges using a
cyclic pairing. Each Prefill and Decode therefore appears at least once,
without running the Cartesian product. Each warmup attempt carries a fresh
random ``cache_salt``. Requests without that exact unexposed salt cannot reuse
the warmup prefix-cache entry.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import ipaddress
import uuid
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import JSONResponse

import load_balance_proxy_layerwise_server_example as base_proxy


logger = base_proxy.logger
app = base_proxy.app

_COMPLETION_PATHS = {"/v1/completions", "/v1/chat/completions"}
_WARMUP_BYPASS_HEADER = "X-Offload-Proxy-Warmup"
_WARMUP_METASERVER_PATH = "/v1/warmup-metaserver"


def _http_authority(host: str, port: int) -> str:
    """Return host:port with brackets when host is an IPv6 literal."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return f"{host}:{port}"
    if isinstance(address, ipaddress.IPv6Address):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layerwise offload proxy with an isolated startup warmup"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries used by normal proxy requests",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.001,
        help="Base delay in seconds used by normal proxy requests",
    )
    parser.add_argument(
        "--warmup-model",
        type=str,
        default=None,
        help="Model name for warmup; defaults to the first decoder /v1/models entry",
    )
    parser.add_argument(
        "--warmup-api",
        choices=("chat", "completions"),
        default="chat",
        help="OpenAI API used for warmup (default: chat)",
    )
    parser.add_argument(
        "--warmup-prompt",
        type=str,
        default="offload proxy startup warmup",
        help="Prompt used by the isolated startup request",
    )
    parser.add_argument(
        "--warmup-retries",
        type=int,
        default=60,
        help="Attempts while waiting for the proxy and backends to become ready",
    )
    parser.add_argument(
        "--warmup-retry-delay",
        type=float,
        default=1.0,
        help="Seconds between startup warmup attempts",
    )
    parser.add_argument(
        "--warmup-timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for one paired warmup attempt",
    )
    args = parser.parse_args()

    if args.host in ["0.0.0.0", "::", "0:0:0:0:0:0:0:0"]:
        raise ValueError(
            "The layerwise metaserver must use an address reachable by Decoder; "
            f"wildcard address {args.host!r} is not allowed"
        )
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    if not args.prefiller_hosts or not args.decoder_hosts:
        raise ValueError(
            "Warmup requires at least one prefiller and one decoder instance"
        )
    if args.warmup_retries < 1:
        raise ValueError("--warmup-retries must be at least 1")
    if args.warmup_retry_delay < 0:
        raise ValueError("--warmup-retry-delay must not be negative")
    if args.warmup_timeout <= 0:
        raise ValueError("--warmup-timeout must be greater than zero")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


async def _discover_model() -> str:
    if base_proxy.global_args.warmup_model:
        return base_proxy.global_args.warmup_model

    errors: list[str] = []
    for decoder in base_proxy.proxy_state.decoders:
        try:
            response = await decoder.client.get("models")
            response.raise_for_status()
            models = response.json().get("data", [])
            if models and models[0].get("id"):
                return models[0]["id"]
            errors.append(f"{decoder.url}: empty model list")
        except Exception as error:
            errors.append(f"{decoder.url}: {error}")
    raise RuntimeError("Unable to discover warmup model: " + "; ".join(errors))


async def _run_startup_warmup() -> None:
    args = base_proxy.global_args
    warmup_api = "/chat/completions" if args.warmup_api == "chat" else "/completions"
    model = await _discover_model()
    proxy_authority = _http_authority(args.host, args.port)

    prefiller_count = len(base_proxy.proxy_state.prefillers)
    decoder_count = len(base_proxy.proxy_state.decoders)
    warmup_count = max(prefiller_count, decoder_count)
    warmed_prefillers: set[int] = set()
    warmed_decoders: set[int] = set()

    async def warmup_pair(pair_index: int) -> tuple[int, int]:
        # Minimum-edge-cover construction for a complete bipartite topology:
        # pair by cycling the shorter side. Thus every P and D appears at
        # least once, using max(P, D) warmups rather than P*D combinations.
        prefiller_index = pair_index % prefiller_count
        decoder_index = pair_index % decoder_count
        prefiller = base_proxy.proxy_state.prefillers[prefiller_index]
        decoder = base_proxy.proxy_state.decoders[decoder_index]
        request_id = await base_proxy.proxy_state.next_req_id()
        request_length = len(args.warmup_prompt.encode("utf-8"))
        payload = {
            "model": model,
            "max_tokens": 1,
            "min_tokens": 1,
            "temperature": 0,
            "stream": False,
            "cache_salt": f"offload-proxy-warmup-{pair_index}-{uuid.uuid4().hex}",
            "kv_transfer_params": {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "metaserver": (
                    f"http://{proxy_authority}{_WARMUP_METASERVER_PATH}/"
                    f"{pair_index}"
                ),
            },
        }
        if args.warmup_api == "chat":
            payload["messages"] = [{"role": "user", "content": args.warmup_prompt}]
        else:
            payload["prompt"] = args.warmup_prompt
        logger.info(
            "Starting warmup pair %d: prefiller=%s decoder=%s",
            pair_index,
            prefiller.url,
            decoder.url,
        )
        try:
            for attempt in range(1, args.warmup_retries + 1):
                try:
                    # This is the paired A+C / B+D warmup: dispatch to the
                    # selected Decode, while its metaserver callback below
                    # dispatches to this exact Prefill instance.
                    payload["cache_salt"] = (
                        f"offload-proxy-warmup-{pair_index}-{uuid.uuid4().hex}"
                    )
                    request_id = await base_proxy.proxy_state.next_req_id()
                    request_id_api = base_proxy.get_api_request_id(warmup_api, request_id)
                    base_proxy.proxy_state.req_data_dict[request_id_api] = (
                        copy.deepcopy(payload), request_length, warmup_api
                    )
                    # Keep both forms because connector versions differ in
                    # whether the callback carries the API id or raw id.
                    base_proxy.proxy_state.req_data_dict[request_id] = (
                        copy.deepcopy(payload), request_length, warmup_api
                    )
                    prefill_done = asyncio.Event()
                    app.state.warmup_prefill_done[request_id_api] = prefill_done
                    app.state.warmup_prefill_done[request_id] = prefill_done

                    async def execute_pair_attempt():
                        response = await decoder.client.post(
                            warmup_api,
                            json=payload,
                            headers={"X-Request-Id": request_id},
                        )
                        response.raise_for_status()
                        if not response.content.strip():
                            raise RuntimeError("decoder returned an empty warmup response")
                        response.json()
                        logger.info(
                            "Decode warmup response received for pair %d from %s",
                            pair_index,
                            decoder.url,
                        )
                        await prefill_done.wait()

                    await asyncio.wait_for(
                        execute_pair_attempt(),
                        timeout=args.warmup_timeout,
                    )
                    logger.info(
                        "Warmup pair %d completed on attempt %d: prefiller=%s decoder=%s",
                        pair_index,
                        attempt,
                        prefiller.url,
                        decoder.url,
                    )
                    return prefiller_index, decoder_index
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning(
                        "Warmup pair %d attempt %d/%d failed: %s",
                        pair_index,
                        attempt,
                        args.warmup_retries,
                        error,
                    )
                    if attempt < args.warmup_retries:
                        await asyncio.sleep(args.warmup_retry_delay)
                finally:
                    base_proxy.proxy_state.req_data_dict.pop(request_id_api, None)
                    base_proxy.proxy_state.req_data_dict.pop(request_id, None)
                    app.state.warmup_prefill_done.pop(request_id_api, None)
                    app.state.warmup_prefill_done.pop(request_id, None)
            raise RuntimeError(
                f"Warmup pair {pair_index} failed after {args.warmup_retries} attempts"
            )
        finally:
            pass

    # Run pairs serially. The cyclic edge cover can reuse an instance when
    # P/D counts differ (for example P0+D0, then P0+D1). Concurrent execution
    # would send multiple cold-start requests through the same layerwise
    # workspace and no longer test or guarantee one completed warmup per edge.
    try:
        for pair_index in range(warmup_count):
            prefiller_index, decoder_index = await warmup_pair(pair_index)
            warmed_prefillers.add(prefiller_index)
            warmed_decoders.add(decoder_index)
            logger.info(
                "Warmup coverage after pair %d: prefillers=%s/%d decoders=%s/%d",
                pair_index,
                sorted(warmed_prefillers),
                prefiller_count,
                sorted(warmed_decoders),
                decoder_count,
            )
    except Exception as error:
        app.state.warmup_error = str(error)
        logger.error("Offload proxy startup warmup failed: %s", error)
        return

    if len(warmed_prefillers) != prefiller_count or len(warmed_decoders) != decoder_count:
        app.state.warmup_error = (
            "warmup completed but did not cover every prefiller and decoder: "
            f"prefillers={sorted(warmed_prefillers)}/{prefiller_count}, "
            f"decoders={sorted(warmed_decoders)}/{decoder_count}"
        )
        logger.error(app.state.warmup_error)
        return
    app.state.warmup_error = None
    logger.info(
        "Offload proxy startup warmup completed: %d pairs, all %d prefiller(s) "
        "and %d decoder(s) covered",
        warmup_count,
        prefiller_count,
        decoder_count,
    )


@app.post(_WARMUP_METASERVER_PATH + "/{pair_index}")
async def warmup_metaserver(pair_index: int, request: Request):
    """Route a warmup Decode callback to its paired Prefill only."""
    prefiller_count = len(base_proxy.proxy_state.prefillers)
    decoder_count = len(base_proxy.proxy_state.decoders)
    warmup_count = max(prefiller_count, decoder_count)
    if pair_index < 0 or pair_index >= warmup_count:
        return JSONResponse(status_code=400, content={"error": "invalid warmup pair"})

    transfer_params = await request.json()
    callback_request_id = transfer_params.get("request_id")
    if not callback_request_id:
        return JSONResponse(status_code=400, content={"error": "missing request_id"})

    record = base_proxy.proxy_state.req_data_dict.get(callback_request_id)
    prefill_done = app.state.warmup_prefill_done.get(callback_request_id)
    if record is None:
        for api in ("/chat/completions", "/completions"):
            candidate = base_proxy.get_api_request_id(api, callback_request_id)
            record = base_proxy.proxy_state.req_data_dict.get(candidate)
            if record is not None:
                prefill_done = app.state.warmup_prefill_done.get(candidate)
                break
    if record is None:
        logger.error("Unknown warmup request id from Decode: %s", callback_request_id)
        return JSONResponse(status_code=404, content={"error": "unknown warmup request"})

    req_data, request_length, api = record
    req_data = copy.deepcopy(req_data)
    req_data["kv_transfer_params"] = transfer_params
    origin_request_id = base_proxy.get_origin_request_id(api, callback_request_id)
    prefiller_index = pair_index % prefiller_count
    prefiller = base_proxy.proxy_state.prefillers[prefiller_index]
    try:
        logger.info(
            "Warmup pair %d metadata received; dispatching to prefiller %s",
            pair_index,
            prefiller.url,
        )
        await asyncio.wait_for(
            base_proxy.send_request_to_service(
                prefiller.client,
                prefiller_index,
                api,
                req_data,
                origin_request_id,
                max_retries=base_proxy.global_args.max_retries,
                base_delay=base_proxy.global_args.retry_delay,
            ),
            timeout=base_proxy.global_args.warmup_timeout,
        )
        if prefill_done is None:
            raise RuntimeError(
                f"missing Prefill completion event for {callback_request_id}"
            )
        prefill_done.set()
        logger.info(
            "Prefill warmup completed for pair %d on %s",
            pair_index,
            prefiller.url,
        )
        return {"status": "ok"}
    except Exception as error:
        logger.exception("Warmup metaserver pair %d failed", pair_index)
        return JSONResponse(status_code=502, content={"error": str(error)})


@asynccontextmanager
async def lifespan(app_instance):
    async with base_proxy.lifespan(app_instance):
        app_instance.state.warmup_done = asyncio.Event()
        app_instance.state.warmup_error = None
        app_instance.state.warmup_bypass_token = uuid.uuid4().hex
        app_instance.state.warmup_prefill_done = {}

        async def warmup_and_release_gate() -> None:
            try:
                await _run_startup_warmup()
            except Exception as error:
                app_instance.state.warmup_error = str(error)
                logger.exception("Offload proxy startup warmup crashed")
            finally:
                app_instance.state.warmup_done.set()

        warmup_task = asyncio.create_task(
            warmup_and_release_gate(), name="offload-proxy-startup-warmup"
        )
        try:
            yield
        finally:
            if not warmup_task.done():
                warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)


app.router.lifespan_context = lifespan


@app.middleware("http")
async def wait_for_startup_warmup(request: Request, call_next):
    """Keep user inference requests behind the one-time startup warmup."""
    if request.url.path not in _COMPLETION_PATHS:
        return await call_next(request)

    bypass_token = request.headers.get(_WARMUP_BYPASS_HEADER)
    if bypass_token != request.app.state.warmup_bypass_token:
        await request.app.state.warmup_done.wait()
        if request.app.state.warmup_error is not None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Offload proxy startup warmup failed",
                    "detail": request.app.state.warmup_error,
                },
            )
    return await call_next(request)


if __name__ == "__main__":
    base_proxy.global_args = parse_args()
    logger.info(
        "Decoder hosts must be able to access the layerwise metaserver at "
        "http://%s:%s/v1/metaserver",
        base_proxy.global_args.host,
        base_proxy.global_args.port,
    )

    import uvicorn

    uvicorn.run(
        app,
        host=base_proxy.global_args.host,
        port=base_proxy.global_args.port,
    )
