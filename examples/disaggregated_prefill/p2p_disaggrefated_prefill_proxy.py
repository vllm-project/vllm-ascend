import os
import socket
import threading
import uuid

import aiohttp
import msgpack  # type: ignore
import zmq
from quart import Quart, make_response, request

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

prefill_instances: dict[str, str] = {}  # http_address: zmq_address
decode_instances: dict[str, str] = {}  # http_address: zmq_address

prefill_cv = threading.Condition()
decode_cv = threading.Condition()


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    prefill_instances[
                        data["http_address"]] = data["zmq_address"]
                    print(
                        "Get a prefill register with http_addr %s and zmq_addr %s",
                        data["http_address"],
                        data["zmq_address"],
                    )
            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    decode_instances[
                        data["http_address"]] = data["zmq_address"]
                    print(
                        "Get a decode register with http_addr %s and zmq_addr %s",
                        data["http_address"],
                        data["zmq_address"],
                    )
            else:
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()  # type: ignore
    router_socket = context.socket(zmq.ROUTER)  # type: ignore
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()  # type: ignore
    poller.register(router_socket, zmq.POLLIN)  # type: ignore

    _listener_thread = threading.Thread(target=_listen_for_register,
                                        args=[poller, router_socket],
                                        daemon=True)
    _listener_thread.start()
    return _listener_thread


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        global prefill_instances
        global prefill_cv
        with prefill_cv:
            if len(prefill_instances) > 1:
                print(
                    "Found more than 1 Prefill instances. Currently we only support 1P1D, so only"
                    f"the first Prefill instance({list(prefill_instances.keys())[0]}) will be used!"
                )
            if len(prefill_instances) == 0:
                res_str = (
                    "No Prefill instances has been registered to proxy. Please confirm that you have successfully"
                    " and correctly started a Prefill vLLM instance.")
                print(res_str)
                response = await make_response(res_str)
                return response
            # prefill_addr, prefill_zmq_addr = random.choice(
            #     list(prefill_instances.items()))
            prefill_addr, prefill_zmq_addr = list(prefill_instances.items())[0]
            print(
                "handle_request, prefill_addr: %s, zmq_addr: %s",
                prefill_addr,
                prefill_zmq_addr,
            )

        global decode_instances
        global decode_cv
        with decode_cv:
            if len(decode_instances) > 1:
                print(
                    "Found more than 1 Decode instances. Currently we only support 1P1D, so only"
                    f"the first Decode instance({list(decode_instances.keys())[0]}) will be used!"
                )
            if len(decode_instances) == 0:
                res_str = (
                    "No Decode instances has been registered to proxy. Please confirm that you have successfully"
                    " and correctly started a Decode vLLM instance.")
                print(res_str)
                response = await make_response(res_str)
                return response
            # decode_addr, decode_zmq_addr = random.choice(
            #     list(decode_instances.items()))
            decode_addr, decode_zmq_addr = list(decode_instances.items())[0]
            print(
                "handle_request, decode_addr: %s, zmq_addr: %s",
                decode_addr,
                decode_zmq_addr,
            )

        request_id = f"___prefill_addr_{prefill_addr}___decode_addr_{decode_addr}_{random_uuid()}"

        # finish prefill
        async for _ in forward_request(f"http://{prefill_addr}/v1/completions",
                                       prefill_request, request_id):
            continue

        # return decode
        generator = forward_request(
            f"http://{decode_addr}/v1/completions",
            original_request_data,
            request_id,
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="args of disaggregated-prefill proxy")
    parser.add_argument("--http-port", type=int, default=10001)
    parser.add_argument("--register-port", type=int, default=10002)
    args = parser.parse_args()

    t = start_service_discovery("0.0.0.0", args.register_port)
    app.run(host="0.0.0.0", port=args.http_port)
    t.join()
