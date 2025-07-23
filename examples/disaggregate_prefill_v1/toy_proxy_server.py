# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import os
import random
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerState:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f'http://{host}:{port}/v1'
        self.client = httpx.AsyncClient(timeout=None,
                                        base_url=self.url,
                                        limits=httpx.Limits(
                                            max_connections=100000,
                                            max_keepalive_connections=100000))
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.lock = asyncio.Lock()  # Per-server lock for state updates


class ProxyState:

    def __init__(self, prefiller_instances, decoder_instances):
        self.prefillers = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        self.req_id_counter = 0

    async def next_req_id(self):
        async with self.req_id_lock:
            self.req_id_counter += 1
            return str(self.req_id_counter)

    async def select_prefiller(self, token_count):
        # Find the best prefiller (no lock needed for read)
        min_tokens = min(p.active_tokens for p in self.prefillers)
        candidates = [
            i for i, p in enumerate(self.prefillers)
            if p.active_tokens == min_tokens
        ]
        min_kv = min(self.prefillers[i].active_kv_cache for i in candidates)
        final_candidates = [
            i for i in candidates
            if self.prefillers[i].active_kv_cache == min_kv
        ]
        chosen = final_candidates[0] if len(
            final_candidates) == 1 else random.choice(final_candidates)
        # Only lock the chosen server for update
        async with self.prefillers[chosen].lock:
            self.prefillers[chosen].active_tokens += token_count
            self.prefillers[chosen].active_kv_cache += token_count
        return chosen

    async def release_prefiller(self, idx, token_count):
        async with self.prefillers[idx].lock:
            self.prefillers[idx].active_tokens -= token_count

    async def release_prefiller_kv(self, idx, token_count):
        async with self.prefillers[idx].lock:
            if self.prefillers[idx].active_kv_cache > 0:
                self.prefillers[idx].active_kv_cache -= token_count

    async def select_decoder(self, token_count):
        min_tokens = min(d.active_tokens for d in self.decoders)
        candidates = [
            i for i, d in enumerate(self.decoders)
            if d.active_tokens == min_tokens
        ]
        chosen = candidates[0] if len(candidates) == 1 else random.choice(
            candidates)
        async with self.decoders[chosen].lock:
            self.decoders[chosen].active_tokens += token_count
        return chosen

    async def release_decoder(self, idx, token_count):
        async with self.decoders[idx].lock:
            self.decoders[idx].active_tokens -= token_count


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        type=int,
                        nargs="+",
                        default=[8100])
    parser.add_argument("--decoder-hosts",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8200])
    parser.add_argument("--max-retries",
                        type=int,
                        default=3,
                        help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.2,
        help="Base delay (seconds) for exponential backoff retries")
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances,
                             global_args.decoder_instances)
    print(
        f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients."
    )
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(client: httpx.AsyncClient,
                                  endpoint: str,
                                  req_data: dict,
                                  request_id: str,
                                  max_retries: int = 3,
                                  base_delay: float = 0.2):
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint,
                                         json=req_data,
                                         headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(
                f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2**(attempt - 1)))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(client: httpx.AsyncClient,
                                             endpoint: str,
                                             req_data: dict,
                                             request_id: str,
                                             max_retries: int = 3,
                                             base_delay: float = 0.2):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST",
                                     endpoint,
                                     json=req_data,
                                     headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}"
                )
                await asyncio.sleep(base_delay * (2**(attempt - 1)))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for streaming {endpoint}."
                )
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if 'first_chunk_sent' in locals() and first_chunk_sent:
                logger.error(
                    f"Streaming to client interrupted after response started: {str(e)}"
                )
                return
            else:
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}"
                    )
                    await asyncio.sleep(base_delay * (2**(attempt - 1)))
                else:
                    logger.error(
                        f"All {max_retries} attempts failed for streaming {endpoint}."
                    )
                    raise e


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        input_tokens = int(req_data.get('input_tokens', 1))
        request_id = await proxy_state.next_req_id()
        # Select prefiller
        prefiller_idx = await proxy_state.select_prefiller(input_tokens)
        prefiller = proxy_state.prefillers[prefiller_idx]
        # Send request to prefiller
        response = await send_request_to_service(
            prefiller.client,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay)
        await proxy_state.release_prefiller(prefiller_idx, input_tokens)
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        # Select decoder
        decoder_idx = await proxy_state.select_decoder(input_tokens)
        decoder = proxy_state.decoders[decoder_idx]
        logger.debug("Using %s %s", prefiller.url, decoder.url)
        # Stream response from decoder
        released_kv = False

        async def generate_stream():
            nonlocal released_kv
            # Only one await per chunk, minimal logic in loop
            async for chunk in stream_service_response_with_retry(
                decoder.client,
                api,
                req_data,
                request_id=request_id,
                max_retries=global_args.max_retries,
                base_delay=global_args.retry_delay):
                if not released_kv and chunk:
                    await proxy_state.release_prefiller_kv(
                        prefiller_idx, input_tokens)
                    released_kv = True
                yield chunk
            # After streaming done, release tokens
            await proxy_state.release_decoder(decoder_idx, input_tokens)

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")
    except Exception as e:
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
