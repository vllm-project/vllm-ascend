# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the EPD Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "encoder", "pd", "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances multimodal requests to multiple encoder, pd, prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.8+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8101 ... # Encoder 1
#   vllm serve --host 0.0.0.0 --port 8102 ... # Encoder 2
#   vllm serve --host 0.0.0.0 --port 8201 ... # PD 1
#   vllm serve --host 0.0.0.0 --port 8202 ... # PD 2
#   vllm serve --host 0.0.0.0 --port 8301 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8301 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8401 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8402 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each instance:
#
# 2 Encoder instance + 2 PD instance:
#   python epd_load_balance_proxy_layerwise_server_example.py \
#     --encoder-hosts 127.0.0.1 127.0.0.1 \
#     --encoder-ports 81001 81002 \
#     --pd-hosts 127.0.0.1 127.0.0.1 \
#     --pd-ports 82001 82002 \
#     --host 0.0.0.0 \
#     --port 9000

# 2 Encoder instance + 2 Prefill instance + 2 Decode instance:
#   python epd_load_balance_proxy_layerwise_server_example.py \
#     --encoder-hosts 127.0.0.1 127.0.0.1 \
#     --encoder-ports 81001 81002 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 83001 83002 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 84001 84002 \
#     --host 0.0.0.0 \
#     --port 9000

# This will start the proxy on port 9000, load balancing between two encoder, tweo pd, two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user","content": [{"type": "image_url","image_url": {"url": f"file://{image_path}"}},
# 								                     {"type": "text","text": "Describe this image."}]}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of encoder, pd, prefiller
# and decoder instances.
#
# Notes:
# - You can scale the number of encoder, pd, prefiller and decoder servers as needed.
# - The proxy dispatches requests based on a least-loaded strategy,
#   using a priority queue to balance the active token workload across instances.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.


import argparse
import asyncio
import base64
import functools
import heapq
import io
import ipaddress
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
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
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0
        self.active_requests = 0
        self.aborted_requests = set()


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances, encoder_instances=None, pd_instances=None):
        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.encoders: list[ServerState] = [ServerState(h, p) for h, p in (encoder_instances or [])]
        self.pds: list[ServerState] = [ServerState(h, p) for h, p in pd_instances]

        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()

        self.prefiller_heap = [(0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0, i, server) for i, server in enumerate(self.decoders)]
        self.encoder_heap = [(0, i, server) for i, server in enumerate(self.encoders)]
        self.pd_heap = [(0, i, server) for i, server in enumerate(self.pds)]

        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        heapq.heapify(self.encoder_heap)
        heapq.heapify(self.pd_heap)

        self.req_id_future = {}
        self.req_data_dict = {}

    def _update_pd_priority(self, server_idx: int):
        server = self.pds[server_idx]
        priority = server.active_tokens + server.active_kv_cache * 0.3
        self.pd_heap = [(p, i, s) for p, i, s in self.pd_heap if i != server_idx]
        heapq.heappush(self.pd_heap, (priority, server_idx, server))  # type: ignore[misc]

    def _update_prefiller_priority(self, server_idx: int):
        server = self.prefillers[server_idx]
        priority = server.active_tokens + server.active_kv_cache * 0.3
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))  # type: ignore[misc]

    def _update_decoder_priority(self, server_idx: int):
        server = self.decoders[server_idx]
        priority = server.active_tokens
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))

    def _update_encoder_priority(self, server_idx: int):
        server = self.encoders[server_idx]
        priority = server.active_tokens
        self.encoder_heap = [(p, i, s) for p, i, s in self.encoder_heap if i != server_idx]
        heapq.heappush(self.encoder_heap, (priority, server_idx, server))

    def abort_pd_request(self, server_idx: int, request_id):
        self.pds[server_idx].aborted_requests.add(request_id)

    def aquire_aborted_pd_requests(self, server_idx: int):
        aborted_requests = self.pds[server_idx].aborted_requests.copy()
        self.pds[server_idx].aborted_requests.clear()
        return aborted_requests

    def abort_prefiller_request(self, server_idx: int, request_id):
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def aquire_aborted_prefiller_requests(self, server_idx: int):
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_pd(self, token_count):
        if not self.pd_heap:
            raise RuntimeError("No pd servers available")
        priority, chosen, server = heapq.heappop(self.pd_heap)
        self.pds[chosen].active_tokens += token_count
        self.pds[chosen].active_kv_cache += token_count
        self._update_pd_priority(chosen)
        return chosen

    def release_pd(self, idx, token_count):
        self.pds[idx].active_tokens -= token_count
        self._update_pd_priority(idx)

    def select_prefiller(self, token_count):
        if not self.prefiller_heap:
            raise RuntimeError("No prefiller servers available")
        priority, chosen, server = heapq.heappop(self.prefiller_heap)
        self.prefillers[chosen].active_tokens += token_count
        self.prefillers[chosen].active_kv_cache += token_count
        self._update_prefiller_priority(chosen)
        return chosen

    def release_prefiller(self, idx, token_count):
        self.prefillers[idx].active_tokens -= token_count
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")
        priority, chosen, server = heapq.heappop(self.decoder_heap)
        self.decoders[chosen].active_tokens += token_count
        self._update_decoder_priority(chosen)
        return chosen

    def release_decoder(self, idx, token_count):
        self.decoders[idx].active_tokens -= token_count
        self._update_decoder_priority(idx)

    def select_encoder(self, token_count):
        if not self.encoder_heap:
            raise RuntimeError("No encoder servers available")
        priority, chosen, server = heapq.heappop(self.encoder_heap)
        self.encoders[chosen].active_tokens += token_count
        self._update_encoder_priority(chosen)
        return chosen

    def release_encoder(self, idx, token_count):
        self.encoders[idx].active_tokens -= token_count
        self._update_encoder_priority(idx)

    def calculate_prefill_scores(self, text_length: int) -> float:
        length_score = text_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=[])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=[])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[])
    parser.add_argument("--encoder-hosts", type=str, nargs="+", default=[])
    parser.add_argument("--encoder-ports", type=int, nargs="+", default=[])
    parser.add_argument("--pd-hosts", type=str, nargs="+", default=[])
    parser.add_argument("--pd-ports", type=int, nargs="+", default=[])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    args = parser.parse_args()
    if len(args.pd_hosts) != len(args.pd_ports):
        raise ValueError("Number of pd hosts must match number of pd ports")
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    if len(args.encoder_hosts) != len(args.encoder_ports):
        raise ValueError("Number of encoder hosts must match number of encoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    args.encoder_instances = list(zip(args.encoder_hosts, args.encoder_ports))
    args.pd_instances = list(zip(args.pd_hosts, args.pd_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(
        global_args.prefiller_instances,
        global_args.decoder_instances,
        global_args.encoder_instances,
        global_args.pd_instances,
    )
    print(
        f"Initialized {len(proxy_state.encoders)} encode clients, {len(proxy_state.prefillers)} prefill clients \
            and \{len(proxy_state.decoders)} decode clients, {len(proxy_state.pds)} pd clients."
    )
    yield
    for e in proxy_state.encoders:
        await e.client.aclose()
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()
    for pd in proxy_state.pds:
        await pd.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_encode_service(
    client: httpx.AsyncClient,
    encoder_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    encoder_req = req_data.copy()
    encoder_req["stream"] = False
    encoder_req["max_tokens"] = 1
    encoder_req["min_tokens"] = 1
    if "stream_options" in encoder_req:
        del encoder_req["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=encoder_req, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise e


def fast_get_hw(b64_str):
    img_bytes = base64.b64decode(b64_str.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes))
    return img.width, img.height


SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def parse_jpeg_size(data: bytes):
    idx = 0
    length = len(data)

    if length < 2 or data[0:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG")

    idx = 2
    while idx + 9 < length:
        if data[idx] != 0xFF:
            idx += 1
            continue

        marker = data[idx + 1]

        # 跳过填充字节
        if marker == 0xFF:
            idx += 1
            continue

        if marker in SOF_MARKERS:
            h = (data[idx + 5] << 8) | data[idx + 6]
            w = (data[idx + 7] << 8) | data[idx + 8]
            return w, h

        if marker in (0xD9, 0xDA):
            break

        if idx + 3 >= length:
            break

        seg_len = (data[idx + 2] << 8) | data[idx + 3]
        if seg_len < 2:
            break

        idx += 2 + seg_len

    raise ValueError("JPEG SOF marker not found")


def parse_png_size(data: bytes):
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return w, h


def get_hw_from_local(path: str):
    if path.startswith("file://"):
        path = path[7:]

    with open(path, "rb") as f:
        data = f.read(65536)

    if data.startswith(b"\x89PNG"):
        return parse_png_size(data)
    return parse_jpeg_size(data)


def calculate_messages_size(ori_req_data, ori_req_body):
    messages = ori_req_data.get("messages")
    stats = {
        "text_char_count": 0,
        "mul_token": 0,
    }
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue

        for content_item in msg["content"]:
            content_type = content_item.get("type")
            if not content_type:
                continue

            if content_type == "text":
                text = content_item.get("text", "")
                stats["text_char_count"] += len(text)

            elif content_type == "image_url":
                img_url = content_item.get("image_url", {}).get("url", "")
                if img_url.startswith("data:image"):
                    h, w = fast_get_hw(img_url)
                else:
                    h, w = get_hw_from_local(img_url)
                stats["mul_token"] += math.ceil(h / 32) * math.ceil(w / 32)
            elif content_type == "video_url":
                stats["mul_token"] += len(ori_req_body) * 32
    return stats


def get_api_request_id(api, req_id):
    if api == "/completions":
        return "cmpl-" + req_id + "-0"
    elif api == "/chat/completions":
        return "chatcmpl-" + req_id


def get_origin_request_id(api, req_id):
    if api == "/completions":
        return req_id.replace("cmpl-", "")[:-2]
    elif api == "/chat/completions":
        return req_id.replace("chatcmpl-", "")


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


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_id = await proxy_state.next_req_id()
        request_id_api = get_api_request_id(api, request_id)
        mul_flag = False

        stats_info = calculate_messages_size(req_data, req_body)
        text_length = stats_info["text_char_count"]
        encoder_score = stats_info["mul_token"]
        if stats_info["mul_token"] != 0:
            mul_flag = True

        if mul_flag and proxy_state.encoders:
            encoder_idx = proxy_state.select_encoder(encoder_score)
            encoder = proxy_state.encoders[encoder_idx]
            logger.debug("Sending to encoder: %s", encoder.url)
            _ = await send_request_to_encode_service(
                encoder.client,
                encoder_idx,
                api,
                req_data,
                request_id,
                max_retries=global_args.max_retries,
                base_delay=global_args.retry_delay,
            )
            proxy_state.release_encoder(encoder_idx, encoder_score)

        token_score = encoder_score + text_length

        if proxy_state.pds:
            pd_idx = proxy_state.select_pd(token_score)
            pd = proxy_state.pds[pd_idx]

            async def generate_stream():
                try:
                    async for chunk in stream_service_response_with_retry(
                        pd.client,
                        api,
                        req_data,
                        request_id=request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(f"Error during streaming from pd {pd.url}: {str(e)}")
                    proxy_state.abort_pd_request(pd_idx, request_id)
                finally:
                    proxy_state.release_pd(pd_idx, token_score)

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            proxy_state.req_data_dict[request_id_api] = (req_data, token_score, api)
            req_data["kv_transfer_params"] = {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "metaserver": f"http://{global_args.host}:{global_args.port}/v1/metaserver",
            }

            # Select decoder
            decoder_score = proxy_state.calculate_decode_scores(token_score)
            logger.debug("Decoder score: %f", decoder_score)
            # Use the prefiller's kv_transfer_params to select decoder
            decoder_idx = proxy_state.select_decoder(decoder_score)
            print("d", decoder_idx, decoder_score)
            decoder = proxy_state.decoders[decoder_idx]
            # logger.debug("Using %s %s", prefiller.url, decoder.url)
            # Stream response from decoder
            released_kv = False

            async def generate_stream():
                nonlocal released_kv
                try:
                    async for chunk in stream_service_response_with_retry(
                        decoder.client,
                        api,
                        req_data,
                        request_id=request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(
                        f"Error during streaming from decoder {decoder.url}: {str(e)} the aborted request {request_id} \
                            will be routing to the target prefiller when new request is ready to dispatch to it"
                    )

                # After streaming done, release tokens
                proxy_state.release_decoder(decoder_idx, decoder_score)

            return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "encode_instances": len(proxy_state.encoders),
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
        "pd_instances": len(proxy_state.pds),
    }


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = req_data.copy()
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            if request_id in proxy_state.req_id_future:
                result_future = proxy_state.req_id_future[request_id]
                result_future.set_result(response.json()["kv_transfer_params"])
            return
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


@app.post("/v1/metaserver")
async def metaserver(request: Request):
    try:
        kv_transfer_params = await request.json()

        request_id = kv_transfer_params["request_id"]
        assert request_id in proxy_state.req_data_dict

        req_data, token_score, api = proxy_state.req_data_dict[request_id]
        request_id = get_origin_request_id(api, request_id)
        req_data["kv_transfer_params"] = kv_transfer_params
        logger.debug(f"Prefiller score: {token_score}")

        # Select prefiller
        prefiller_idx = proxy_state.select_prefiller(token_score)
        prefiller = proxy_state.prefillers[prefiller_idx]
        logger.debug(f"Using prefill {prefiller.url=} {req_data=}")
        # Send request to prefiller
        _ = await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
        proxy_state.release_prefiller(prefiller_idx, token_score)
        proxy_state.release_prefiller_kv(prefiller_idx, token_score)

    except Exception as e:
        logger.error(f"Post metaserver failed with: {str(e)}")
        proxy_state.release_prefiller(prefiller_idx, token_score)
        proxy_state.release_prefiller_kv(prefiller_idx, token_score)


######################################     profile     ######################################


async def _forward_profile(
    service_name: str, idx: int, client, host: str, port: int, endpoint: str, req_data: dict, headers: dict
):
    """Forward profiling request to one service and return raw response or error."""
    url = f"http://{host}:{port}{endpoint}"
    try:
        resp = await client.post(url, json=req_data, headers=headers, timeout=10.0)
        resp.raise_for_status()
        # 直接返回 httpx.Response，保持原始格式
        return service_name, idx, {"status_code": resp.status_code, "body": resp.text}
    except Exception as e:
        return service_name, idx, {"error": str(e)}


@app.post("/start_profile")
async def start_profile(request: Request):
    """
    Forward the /start_profile request to all encoder, prefiller, and decoder services (concurrently).
    """
    try:
        try:
            req_data = await request.json()
        except Exception as e:
            print(f"Error in stop_profile while waiting request data: {e}")
            req_data = {}

        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        tasks = []

        # encoder
        for idx, encoder in enumerate(proxy_state.encoders):
            tasks.append(
                _forward_profile(
                    "encoder", idx, encoder.client, encoder.host, encoder.port, "/start_profile", req_data, headers
                )
            )

        # prefiller
        for idx, prefill in enumerate(proxy_state.prefillers):
            tasks.append(
                _forward_profile(
                    "prefill", idx, prefill.client, prefill.host, prefill.port, "/start_profile", req_data, headers
                )
            )

        # decoder
        for idx, decoder in enumerate(proxy_state.decoders):
            tasks.append(
                _forward_profile(
                    "decoder", idx, decoder.client, decoder.host, decoder.port, "/start_profile", req_data, headers
                )
            )

        # pds
        for idx, pd in enumerate(proxy_state.pds):
            tasks.append(_forward_profile("pds", idx, pd.client, pd.host, pd.port, "/start_profile", req_data, headers))

        results_list = await asyncio.gather(*tasks)
        results = {f"{name}_{idx}": res for name, idx, res in results_list}

        return JSONResponse(content={"status": "done", "results": results}, status_code=200)

    except Exception as e:
        print(f"Error in start_profile: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    """
    Forward the /stop_profile request to all encoder, prefiller, and decoder services (concurrently).
    """
    try:
        try:
            req_data = await request.json()
        except Exception as e:
            print(f"Error in stop_profile while waiting request data: {e}")
            req_data = {}

        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        tasks = []

        # encoder
        for idx, encoder in enumerate(proxy_state.encoders):
            tasks.append(
                _forward_profile(
                    "encoder", idx, encoder.client, encoder.host, encoder.port, "/stop_profile", req_data, headers
                )
            )

        # prefiller
        for idx, prefill in enumerate(proxy_state.prefillers):
            tasks.append(
                _forward_profile(
                    "prefill", idx, prefill.client, prefill.host, prefill.port, "/stop_profile", req_data, headers
                )
            )

        # decoder
        for idx, decoder in enumerate(proxy_state.decoders):
            tasks.append(
                _forward_profile(
                    "decoder", idx, decoder.client, decoder.host, decoder.port, "/stop_profile", req_data, headers
                )
            )

        # pds
        for idx, pd in enumerate(proxy_state.pds):
            tasks.append(_forward_profile("pds", idx, pd.client, pd.host, pd.port, "/stop_profile", req_data, headers))

        results_list = await asyncio.gather(*tasks)
        results = {f"{name}_{idx}": res for name, idx, res in results_list}

        return JSONResponse(content={"status": "done", "results": results}, status_code=200)

    except Exception as e:
        print(f"Error in stop_profile: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
