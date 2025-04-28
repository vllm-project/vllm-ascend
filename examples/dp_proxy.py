# SPDX-License-Identifier: Apache-2.0

import os
import random
import socket
import threading
import uuid
import itertools 
import time 

import aiohttp
import msgpack
import zmq
import asyncio
import copy
from quart import Quart, make_response, request


TIME_INTERVAL_FOR_IDLE_RUN = 2   # 2s
DP_SIZE = 2
dp_instances: dict[str, bool] = {}
dp_cv = threading.Condition()
round_robin_index = 0


def make_idle_request():
    data = {
        "prompt": "hi",
        "max_tokens": 1,
        "temperature": 0,
    }
    return data

async def send_idle_token_to_client(schedule_dict):
    for key, value in schedule_dict.items():
        if value:
            continue
        request_received_id = random_uuid()
        # make dummy run request for the idle dp node
        idle_request_data = make_idle_request()
        forward_request_id = f"dp_fwd_{key}_{request_received_id}"
        target_url = f'http://{key}/v1/completions' 
        print(f"send url {target_url}")
        print(f"original data {idle_request_data}")
        print(f"forwrad request id: {forward_request_id}")
        generator = forward_request(target_url, idle_request_data, forward_request_id)
        async for response in generator:
            print(f"Request {request_received_id}: response from {key}, got response: {response}")


def metadata_collect_trigger(poller, router_socket):
    global dp_instances
    global dp_cv
    with dp_cv:
        dp_cv.wait()
    while True:
        try:
            schedule_dict = copy.deepcopy(dp_instances)
            for key in schedule_dict.keys():
                schedule_dict[key] = False
            first_start = False
            start_time = None
            print("before into while")
            while not all(schedule_dict.values()):
                if start_time is not None:
                    time_interval = time.time() - start_time
                    print("check time interval: ", time_interval)
                    if time_interval > TIME_INTERVAL_FOR_IDLE_RUN:
                        print("exceeds max time interval send idle token to client")
                        # Send idle token to client in case of single dp rank run solo and block on the CCL part
                        response = asyncio.run(send_idle_token_to_client(schedule_dict))
                        # Note: Reset start time prevent consistently send idle token to client
                        # We only reset start time here, for some of the client may loss the idle token send from this proxy
                        # and we only exit this while loop when we make sure all the client are exactly start inference in this 
                        # step
                        start_time = time.time()
                socks = dict(poller.poll(timeout=500))  # timeout in 500ms
                print("receive socks from moniter threads: ", socks)
                if router_socket in socks:
                    messages = router_socket.recv_multipart()
                    try:
                        # {"info": "notify_step", "http_address": ""}
                        for message in messages:
                            data = msgpack.loads(message)
                            http_addr = None
                            print(f"receive message {data}")
                            if data.get("info") == "notify_step":
                                http_addr = data.get("http_address")
                                if http_addr in schedule_dict.keys():
                                    schedule_dict[http_addr] = True
                                    print("set first time")
                                    if not first_start:
                                        print("record start time")
                                        first_start = True
                                        start_time = time.time()
                                else:
                                    print(f"Unrecognize http address")
                            else:
                                print(f"Got unrecognize info type! We only accept notify step info yet")
                    except (msgpack.UnpackException, TypeError, KeyError) as e:
                        print(f"Error processing message from {http_addr}: {e}. Message: {data}")
        except zmq.ZMQError as e:
            print(f"ZMQ Error in listener thread: {e}")
            if e.errno == zmq.ETERM:
                print("Listener thread terminating due to context termination.")
                break 
            time.sleep(1) 
        except Exception as e:
            print(f"Unexpected error in listener thread: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1) 


def _listen_for_register(poller, router_socket):
    global dp_instances
    global dp_cv
    global DP_SIZE

    while True:
        try:
            socks = dict(poller.poll(timeout=1000)) # 1秒超时
            print("receive socks from listen threads: ", socks)
            if router_socket in socks:
                remote_address, message = router_socket.recv_multipart()
                try:
                    # {"type": "DP", "http_address": "ip:port"}
                    data = msgpack.loads(message)
                    if data.get("type") == "DP":
                        http_addr = data.get("http_address")
                        if http_addr:
                            with dp_cv: 
                                if http_addr not in dp_instances:
                                    print(f"Registering DP instance: http={http_addr}")
                                    dp_instances[http_addr] = True
                                    if len(dp_instances.keys()) == DP_SIZE:
                                        print("all dp group joined into the proxy, stop listen!")
                                        dp_cv.notify_all()
                                else:
                                    pass

                        else:
                            print(f"Warning: Received incomplete DP registration from {remote_address}. Missing 'http_address'. Data: {data}")

                    else:
                        print(f"Warning: Received message with unexpected type from {remote_address}. Type: {data.get('type')}, Data: {data}")

                except (msgpack.UnpackException, TypeError, KeyError) as e:
                    print(f"Error processing message from {remote_address}: {e}. Message: {message}")
                except Exception as e:
                    print(f"Unexpected error processing message from {remote_address}: {e}")


        except zmq.ZMQError as e:
            print(f"ZMQ Error in listener thread: {e}")
            if e.errno == zmq.ETERM:
                print("Listener thread terminating due to context termination.")
                break 
            time.sleep(1) 
        except Exception as e:
            print(f"Unexpected error in listener thread: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1) 


def start_thread_with_zmq(hostname, port, zmq_key, fn, fn_name):
    if not hostname:
        hostname = socket.gethostname()
    if port <= 0:
        raise ValueError("Port must be a positive integer")

    print(f"Starting service discovery listener on tcp://{hostname}:{port}")
    context = zmq.Context()
    router_socket = context.socket(zmq_key)
    # 设置 LINGER 为 0，这样关闭时不会等待未发送的消息
    router_socket.setsockopt(zmq.LINGER, 0)
    try:
        router_socket.bind(f"tcp://{hostname}:{port}")
    except zmq.ZMQError as e:
        print(f"Error binding ZMQ socket to tcp://{hostname}:{port}: {e}")
        router_socket.close()
        context.term()
        raise 

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    print(f"start fn with name {fn_name}")
    listener_thread = threading.Thread(target=fn,
                                       args=[poller, router_socket],
                                       daemon=True,
                                       name=fn_name) # 给线程命名方便调试
    listener_thread.start()
    return listener_thread, context, router_socket


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60) # 6 小时超时

app = Quart(__name__)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

async def forward_request(url, data, request_id):
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}", 
                "X-Request-Id": request_id,
                "Content-Type": "application/json" 
            }
            async with session.post(url=url, json=data, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    error_content = await response.read()
                    print(f"Error from backend {url} (status {response.status}): {error_content.decode(errors='ignore')}")
                    yield error_content 

    except aiohttp.ClientError as e:
        print(f"Error forwarding request {request_id} to {url}: {e}")
        error_msg = f"Failed to connect or communicate with backend service at {url}: {e}".encode('utf-8')
        yield error_msg 


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    global dp_instances
    global dp_cv
    global round_robin_index

    request_received_id = random_uuid() 
    # print(f"Received request {request_received_id}")

    try:
        original_request_data = await request.get_json()
        if not original_request_data:
             return await make_response("Request body must be valid JSON.", 400)

        target_addr = None
        with dp_cv: 
            if not dp_instances:
                res_str = "No DP instances available/registered. Please ensure DP vLLM instances are running and have registered."
                print(res_str)
                response = await make_response(res_str, 503)
                response.headers['Retry-After'] = '30' 
                return response
            print(f"dp instances:{dp_instances}")
            # --- 轮询选择实例 ---
            dp_addresses = list(dp_instances.keys())
            if not dp_addresses: 
                 res_str = "Internal Server Error: Instance list is empty despite registration."
                 print(res_str)
                 response = await make_response(res_str, 500)
                 return response

            # 使用 round_robin_index 进行轮询
            current_selection_index = round_robin_index % len(dp_addresses)
            target_addr = dp_addresses[current_selection_index]

            round_robin_index += 1

        print(f"Request {request_received_id}: Routing to DP instance {target_addr} (Index {current_selection_index})")

        forward_request_id = f"dp_fwd_{target_addr}_{request_received_id}"

        # 直接转发原始请求给选定的 DP 实例
        target_url = f'http://{target_addr}/v1/completions' 
        print(f"send url {target_url}")
        print(f"original data {original_request_data}")
        print(f"forwrad request id: {forward_request_id}")
        generator = forward_request(target_url,
                                    original_request_data,
                                    forward_request_id)

        response = await make_response(generator)
        response.timeout = None 

        if original_request_data.get("stream", False):
             response.headers['Content-Type'] = 'text/event-stream'
             response.headers['Cache-Control'] = 'no-cache'
        else:
            response.headers['Content-Type'] = 'application/json'


        print(f"Request {request_received_id}: response from {target_addr}")
        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print(f"Error handling request {request_received_id}: {e}")
        print("".join(traceback.format_exception(*exc_info)))
        response = await make_response(f"Internal Server Error: An unexpected error occurred.", 500)
        return response


if __name__ == '__main__':
    listener_thread = None
    zmq_context = None
    zmq_socket = None
    try:
        print("try start dp listen thread")
        listener_thread, zmq_context, zmq_socket = start_thread_with_zmq("0.0.0.0", 30003, zmq.ROUTER, _listen_for_register, "ZMQListenerThread")
    except Exception as e:
        print(f"Failed to start service discovery: {e}")
        exit(1)

    try:
        print("try start monitor thread")
        # start idle run monitor thread for all the dp rank, use diffent rank in case of conflict with thread listening one
        metadata_monitor_thread, zmq_monitor_context, zmq_monitor_socket = start_thread_with_zmq("0.0.0.0", 30002, zmq.PULL, metadata_collect_trigger, "metadata_monitor_thread")
    except Exception as e:
        print(f"Metadata collect thread failed to init: {e}")
        exit(1) 

    print(f"Starting Quart web server on http://0.0.0.0:10001 (PID: {os.getpid()})")
    try:
        app.run(host='0.0.0.0', port=10003) 
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server...")
    except Exception as e:
        print(f"Failed to start or run Quart server: {e}")
    finally:
        if zmq_socket:
            print("Closing ZMQ socket...")
            zmq_socket.close()
        if zmq_monitor_socket:
            print("Closing ZMQ monitor socket...")
            zmq_monitor_socket.close()
        if zmq_context:
            print("Terminating ZMQ context...")
            zmq_context.term() 

        if listener_thread and listener_thread.is_alive():
             print("Waiting for listener thread to join...")
             listener_thread.join(timeout=2.0)
             if listener_thread.is_alive():
                 print("Listener thread did not exit cleanly.")

        if metadata_monitor_thread and metadata_monitor_thread.is_alive():
            print("Prepare to put metadata monitor thread down")
            metadata_monitor_thread.join(timeout=2.0)
            if metadata_monitor_thread.is_alive():
                print("metadata monitor thread still alive")
