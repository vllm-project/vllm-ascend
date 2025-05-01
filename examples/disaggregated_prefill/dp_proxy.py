# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import logging
import os
import threading
import time
import uuid

import aiohttp
import msgpack  # type: ignore
import zmq
from quart import Quart, make_response, request

DP_PROXY_HTTP_PORT = 10004
DP_PROXY_ZMQ_REG_PORT = 30006
DP_PROXY_ZMQ_NOTIFY_PORT = 30005

PD_PROXY_ADDRESS = "127.0.0.1:30002"

MY_HTTP_ADDRESS = f"127.0.0.1:{DP_PROXY_HTTP_PORT}"
MY_ZMQ_ADDRESS_PLACEHOLDER = f"127.0.0.1:{DP_PROXY_ZMQ_REG_PORT}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIME_INTERVAL_FOR_IDLE_RUN = 5e-4
DP_SIZE = 2

dp_instances: dict[str, bool] = {}
dp_cv = threading.Condition()
round_robin_index = 0
_idle_send_loop = None


def make_idle_request():
    # Same as before
    data = {
        "prompt": "hi",
        "max_tokens": 1,
        "temperature": 0,
    }
    return data


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def send_idle_token_to_client(schedule_dict):
    for key, value in schedule_dict.items():
        if value:
            continue
        request_received_id = random_uuid()
        idle_request_data = make_idle_request()
        forward_request_id = f"dp_idle_{key}_{request_received_id}"
        target_url = f'http://{key}/v1/completions'
        logger.debug(
            f"DP Decode Proxy: Sending idle token to D node {key} at {target_url}"
        )
        generator = forward_request_internal(target_url, idle_request_data,
                                             forward_request_id)
        try:
            async for response in generator:
                logger.debug(
                    f"DP Decode Proxy: Idle Request {request_received_id}: response from {key}, got response: {response}"
                )
        except Exception as e:
            logger.warning(
                f"DP Decode Proxy: Error sending idle token to {key}: {e}")


def metadata_collect_trigger(poller, router_socket):
    global dp_instances
    global dp_cv
    global _idle_send_loop
    with dp_cv:
        dp_cv.wait()
    while True:
        try:
            schedule_dict = copy.deepcopy(dp_instances)
            for key in schedule_dict.keys():
                schedule_dict[key] = False
            first_start = False
            start_time = None
            while not all(schedule_dict.values()):
                if start_time is not None:
                    time_interval = time.time() - start_time
                    logger.debug("check time interval: ", time_interval)
                    if time_interval > TIME_INTERVAL_FOR_IDLE_RUN:
                        logger.debug(
                            "exceeds max time interval send idle token to client"
                        )
                        # Send idle token to client in case of single dp rank run solo and block on the CCL part
                        asyncio.run_coroutine_threadsafe(
                            send_idle_token_to_client(schedule_dict),
                            _idle_send_loop)  # type: ignore
                        # Note: Reset start time prevent consistently send idle token to client
                        # We only reset start time here, for some of the client may loss the idle token send from this proxy
                        # and we only exit this while loop when we make sure all the client are exactly start inference in this
                        # step
                        start_time = time.time()
                socks = dict(poller.poll(timeout=500))  # timeout in 500ms
                if socks:
                    logger.debug("receive socks from moniter threads: ", socks)
                if router_socket in socks:
                    messages = router_socket.recv_multipart()
                    try:
                        # {"info": "notify_step", "http_address": ""}
                        for message in messages:
                            data = msgpack.loads(message)
                            http_addr = None
                            logger.debug(f"receive message {data}")
                            if data.get("info") == "notify_step":
                                http_addr = data.get("http_address")
                                if http_addr in schedule_dict.keys():
                                    schedule_dict[http_addr] = True
                                    logger.debug("set first time")
                                    if not first_start:
                                        logger.debug("record start time")
                                        first_start = True
                                        start_time = time.time()
                                else:
                                    logger.warning("Unrecognize http address")
                            else:
                                logger.warning(
                                    "Got unrecognize info type! We only accept notify step info yet"
                                )
                    except (msgpack.UnpackException, TypeError, KeyError) as e:
                        logger.error(
                            f"Error processing message from {http_addr}: {e}. Message: {data}"
                        )
        except zmq.ZMQError as e:  # type: ignore
            logger.error(f"ZMQ Error in monitor thread: {e}")
            if e.errno == zmq.ETERM:  # type: ignore
                logger.error(
                    "Monitor thread terminating due to context termination.")
                break
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error in monitor thread: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


def _listen_for_d_register(poller, router_socket):
    global dp_instances
    global dp_cv
    global DP_SIZE
    logger.info(
        f"DP Decode Proxy: D Node ZMQ Listener started on ROUTER port {DP_PROXY_ZMQ_REG_PORT}"
    )

    while True:
        try:
            socks = dict(poller.poll(timeout=1000))
            if router_socket in socks:
                remote_id, message = router_socket.recv_multipart()
                try:
                    data = msgpack.loads(message)
                    if data.get("type") == "DP":
                        http_addr = data.get("http_address")
                        zmq_addr = data.get("zmq_address")
                        if http_addr:
                            with dp_cv:
                                if http_addr not in dp_instances:
                                    logger.info(
                                        f"DP Decode Proxy: Registering D Node instance: http={http_addr}, zmq={zmq_addr}"
                                    )
                                    dp_instances[http_addr] = True
                                    if len(dp_instances) >= DP_SIZE:
                                        logger.info(
                                            f"DP Decode Proxy: Reached expected D Node count ({DP_SIZE}). Notifying metadata collector."
                                        )
                                        dp_cv.notify_all()
                                else:
                                    pass
                        else:
                            logger.warning(
                                f"DP Decode Proxy: Received D Node registration from {remote_id.decode()} without http_address. Data: {data}"
                            )
                    else:
                        logger.warning(
                            f"DP Decode Proxy: Received message with unexpected type from {remote_id.decode()}. Type: {data.get('type')}, Data: {data}"
                        )

                except (msgpack.UnpackException, TypeError, KeyError) as e:
                    logger.error(
                        f"DP Decode Proxy: Error processing D Node registration from {remote_id.decode()}: {e}. Message: {message}"
                    )
                except Exception as e:
                    logger.error(
                        f"DP Decode Proxy: Unexpected error processing D Node registration from {remote_id.decode()}: {e}"
                    )

        except zmq.ZMQError as e:  # type: ignore
            logger.error(
                f"DP Decode Proxy: ZMQ Error in D Node listener thread: {e}")
            if e.errno == zmq.ETERM:  # type: ignore
                logger.info(
                    "DP Decode Proxy: D Node Listener thread terminating.")
                break
            time.sleep(1)
        except Exception as e:
            logger.error(
                f"DP Decode Proxy: Unexpected error in D Node listener thread: {e}"
            )
            import traceback
            traceback.print_exc()
            time.sleep(1)


def _register_to_pd_proxy(pd_proxy_zmq_addr, my_http_addr, my_zmq_addr):
    context = None
    sock = None
    while True:
        try:
            if context is None:
                context = zmq.Context()  # type: ignore
            if sock is None:
                sock = context.socket(zmq.DEALER)  # type: ignore
                identity = f"dp_proxy_{my_http_addr}".encode('utf-8')
                sock.setsockopt(zmq.IDENTITY, identity)  # type: ignore
                sock.setsockopt(zmq.LINGER, 0)  # type: ignore
                logger.info(
                    f"DP Decode Proxy: Attempting to connect to PD Proxy at {pd_proxy_zmq_addr}..."
                )
                sock.connect(f"tcp://{pd_proxy_zmq_addr}")
                logger.info(
                    f"DP Decode Proxy: Connected to PD Proxy at {pd_proxy_zmq_addr}."
                )

            data = {
                "type": "D",
                "http_address": my_http_addr,
                "zmq_address": my_zmq_addr
            }
            logger.debug(
                f"DP Decode Proxy: Sending registration/heartbeat to PD Proxy: {data}"
            )
            sock.send(msgpack.dumps(data))
            time.sleep(5)

        except zmq.ZMQError as e:  # type: ignore
            logger.error(
                f"DP Decode Proxy: ZMQ Error connecting/sending to PD Proxy ({pd_proxy_zmq_addr}): {e}"
            )
            if sock:
                sock.close()
                sock = None
            time.sleep(10)
        except Exception as e:
            logger.error(
                f"DP Decode Proxy: Unexpected error in PD Proxy registration thread: {e}"
            )
            import traceback
            traceback.print_exc()
            if sock:
                sock.close()
                sock = None
            time.sleep(10)
        finally:
            pass


def start_zmq_thread(hostname, port, socket_type, target_func, thread_name):
    """Generic ZMQ thread starter for ROUTER or PULL."""
    if not hostname:
        hostname = "0.0.0.0"
    context = zmq.Context.instance()  # type: ignore
    socket = context.socket(socket_type)
    socket.setsockopt(zmq.LINGER, 0)  # type: ignore
    try:
        socket.bind(f"tcp://{hostname}:{port}")
    except zmq.ZMQError as e:  # type: ignore
        logger.error(
            f"DP Decode Proxy: Error binding ZMQ {socket_type} socket to tcp://{hostname}:{port}: {e}"
        )
        socket.close()
        raise

    poller = zmq.Poller()  # type: ignore
    poller.register(socket, zmq.POLLIN)  # type: ignore

    thread = threading.Thread(target=target_func,
                              args=(poller, socket),
                              daemon=True,
                              name=thread_name)
    thread.start()
    return thread, socket


def start_thread_with_event_loop():
    global _idle_send_loop
    asyncio.set_event_loop(_idle_send_loop)
    _idle_send_loop.run_forever()  # type: ignore


async def forward_request_internal(url, data, request_id):
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization":
                f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                "X-Request-Id": request_id,
                "Content-Type": "application/json"
            }
            async with session.post(url=url, json=data,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    error_content = await response.read()
                    logger.warning(
                        f"DP Decode Proxy: Error from D node {url} (status {response.status}): {error_content.decode(errors='ignore')}"
                    )
                    yield error_content

    except aiohttp.ClientError as e:
        logger.warning(
            f"DP Decode Proxy: Error forwarding request {request_id} to D node {url}: {e}"
        )
        error_msg = f"Failed to connect or communicate with D node at {url}: {e}".encode(
            'utf-8')
        yield error_msg


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    global dp_instances
    global dp_cv
    global round_robin_index

    request_received_id = request.headers.get("X-Request-Id")
    if not request_received_id:
        fallback_id = f"dp_fallback_{random_uuid()}"
        logger.warning(
            f"DP Decode Proxy: Received request without X-Request-Id header. Using fallback ID: {fallback_id}"
        )
        request_received_id = fallback_id
    else:
        logger.info(
            f"DP Decode Proxy: Received request from PD Proxy, using propagated ID: {request_received_id}"
        )

    try:
        original_request_data = await request.get_json()
        if not original_request_data:
            return await make_response("Request body must be valid JSON.", 400)

        target_addr = None
        with dp_cv:
            if not dp_instances:
                logger.warning(
                    f"DP Decode Proxy: Request {request_received_id}: No D Node instances available/registered."
                )
                return await make_response("No Decode instances available.",
                                           503)

            dp_addresses = list(dp_instances.keys())
            if not dp_addresses:
                logger.error(
                    f"DP Decode Proxy: Request {request_received_id}: Internal error - dp_instances populated but list is empty."
                )
                return await make_response("Internal Server Error", 500)

            current_selection_index = round_robin_index % len(dp_addresses)
            target_addr = dp_addresses[current_selection_index]
            round_robin_index += 1

        logger.info(
            f"DP Decode Proxy: Request {request_received_id}: Routing Decode to D Node {target_addr} (Index {current_selection_index})"
        )

        target_url = f'http://{target_addr}/v1/completions'

        generator = forward_request_internal(target_url, original_request_data,
                                             request_received_id)

        response = await make_response(generator)
        response.timeout = None

        if original_request_data.get("stream", False):
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
        else:
            response.headers['Content-Type'] = 'application/json'

        logger.debug(
            f"DP Decode Proxy: Request {request_received_id}: Streaming response from D node {target_addr}"
        )
        return response

    except Exception as e:
        logger.error(
            f"DP Decode Proxy: Error handling request {request_received_id}: {e}"
        )
        return await make_response("Internal Server Error", 500)


if __name__ == '__main__':
    d_listener_thread, d_reg_socket = start_zmq_thread(
        "0.0.0.0",
        DP_PROXY_ZMQ_REG_PORT,
        zmq.ROUTER,  # type: ignore
        _listen_for_d_register,  # type: ignore
        "DP_DNodeListenerThread")

    metadata_thread, notify_socket = start_zmq_thread(
        "0.0.0.0",
        DP_PROXY_ZMQ_NOTIFY_PORT,
        zmq.PULL,  # type: ignore
        metadata_collect_trigger,
        "DP_MetadataMonitorThread")

    _idle_send_loop = asyncio.new_event_loop()
    idle_loop_thread = threading.Thread(target=start_thread_with_event_loop,
                                        daemon=True,
                                        name="DP_IdleSendLoopThread")
    idle_loop_thread.start()

    pd_register_thread = threading.Thread(target=_register_to_pd_proxy,
                                          args=(PD_PROXY_ADDRESS,
                                                MY_HTTP_ADDRESS,
                                                MY_ZMQ_ADDRESS_PLACEHOLDER),
                                          daemon=True,
                                          name="DP_PDRegisterThread")
    pd_register_thread.start()

    logger.info(
        f"DP Decode Proxy: Starting Quart web server on http://0.0.0.0:{DP_PROXY_HTTP_PORT}"
    )
    zmq_context = zmq.Context.instance()  # type: ignore
    try:
        app.run(host='0.0.0.0', port=DP_PROXY_HTTP_PORT)
    except KeyboardInterrupt:
        logger.info("DP Decode Proxy: KeyboardInterrupt received, stopping...")
    except Exception as e:
        logger.error(f"DP Decode Proxy: Failed to run Quart server: {e}")
    finally:
        logger.info("DP Decode Proxy: Shutting down...")
        if _idle_send_loop and _idle_send_loop.is_running():
            logger.info("DP Decode Proxy: Stopping idle send loop...")
            _idle_send_loop.call_soon_threadsafe(_idle_send_loop.stop)

        if d_reg_socket:
            d_reg_socket.close()
        if notify_socket:
            notify_socket.close()
        if zmq_context:
            zmq_context.term()

        logger.info("DP Decode Proxy: Shutdown complete.")
