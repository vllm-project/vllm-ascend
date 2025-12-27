"""
This file provides a elastic proxy demo to support elastic scaling for P/D instances based on KV pool.
We can launch multiple vllm instances (2 for prefill and 2 for decode), and
launch this proxy demo through:
  python3 examples/elastic_scaling/elastic_proxy.py \
    --model $model_name \
    --prefill localhost:8100 localhost:8101 \
    --decode localhost:8200 localhost:8201 \
    --port 8000 \
    --check-interval 5 \
    --check-times 10 \
    --retry=times 3
"""

import argparse
import asyncio
import ipaddress
import itertools
import time
import json
import logging
import os
import threading
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

import aiohttp
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

DEFAULT_ADMIN_API_KEY = "DEFAULT_ADMIN_API_KEY"


class SchedulingPolicy(ABC):
    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def __init__(self):
        super().__init__()

    def schedule(self, cycler: itertools.cycle) -> str:
        return next(cycler)


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


class ProxyServer:
    def __init__(
            self,
            prefill_instances: List[str],
            decode_instances: List[str],
            model: str,
            port: str,
            scheduling_policy: SchedulingPolicy = RoundRobinSchedulingPolicy(),
    ):
        self.prefill_instances: List = []
        self.decode_instances: List = []
        self.prefill_cycler = itertools.cycle(self.prefill_instances)
        self.decode_cycler = itertools.cycle(self.decode_instances)
        self.model = model
        self.port = port
        self.scheduling_policy = scheduling_policy
        self.router = APIRouter()
        self.setup_routes()

        # server listening process
        self.waiting_nodes: Dict[str, int] = {}
        self.listening_thread = threading.Thread(target=self._node_listener)
        self.listening_thread.start()
        self.retried_nodes: Dict[str, int] = defaultdict(int)

        # start proxy
        self.add_instances(InstanceType.PREFILL, prefill_instances)
        self.add_instances(InstanceType.DECODE, decode_instances)
        self.run_server()

    def setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self.validate_json_request)]
        )(self.create_completion)
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self.validate_json_request)]
        )(self.create_chat_completion)
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.post(
            "/instances/add", dependencies=[Depends(self.api_key_authenticate)]
        )(self.add_instance_endpoint)
        self.router.post(
            "/instances/remove", dependencies=[Depends(self.api_key_authenticate)]
        )(self.remove_instance_endpoint)

    def run_server(self):
        app = FastAPI()
        app.include_router(self.router)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()

    @staticmethod
    async def forward_request(url: str, data: Dict, use_chunked: bool = True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            try:
                async with session.post(
                        url=url, json=data, headers=headers
                ) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(1024):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error(
                            f"Request failed with status {response.status}: {error_content}"
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                                   f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error(f"ClientError occurred: {e}")
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            # add params to request
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1
            kv_prepare_request["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_ip": None,
            }
            kv_prepare_request["stream"] = False
            if "stream_options" in kv_prepare_request:
                del kv_prepare_request["stream_options"]

            # prefill stage
            content = await self.retry_forward_request(
                kv_prepare_request, instance_type=InstanceType.PREFILL,
                rest_try_times=len(self.prefill_instances), is_chat=False
            )

            # Perform kv recv and decoding stage
            content = json.loads(content)
            if isinstance(content, dict):
                request["id"] = content.get("id")
            request["kv_transfer_params"] = content.get("kv_transfer_params", {})
            generator = await self.retry_forward_request(
                request, instance_type=InstanceType.DECODE,
                rest_try_times=len(self.decode_instances), is_chat=False
            )
            response = StreamingResponse(generator)
            return response
        except Exception:
            import sys
            logger.info("Error occurred in elastic proxy server.")
            logger.info(traceback.format_exc())
            return StreamingResponse(
                content=iter("Error occurred in elastic proxy server."), media_type="text/event-stream"
            )

    async def create_chat_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            # add params to request
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1
            kv_prepare_request["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_ip": None,
            }
            kv_prepare_request["stream"] = False
            if "stream_options" in kv_prepare_request:
                del kv_prepare_request["stream_options"]
            if "max_completion_tokens" in kv_prepare_request:
                kv_prepare_request["max_completion_tokens"] = 1

            # prefill stage
            content = await self.retry_forward_request(
                kv_prepare_request, instance_type=InstanceType.PREFILL,
                rest_try_times=len(self.prefill_instances), is_chat=True
            )

            # Perform kv recv and decoding stage
            content = json.loads(content)
            if isinstance(content, dict):
                request["id"] = content.get("id")
                request["kv_transfer_params"] = content.get("kv_transfer_params", {})
            generator = await self.retry_forward_request(
                request, instance_type=InstanceType.DECODE,
                rest_try_times=len(self.decode_instances), is_chat=True
            )
            response = StreamingResponse(content=generator)
            return response
        except Exception:
            logger.info("Error occurred in elastic proxy server.")
            logger.info(traceback.format_exc())
            return StreamingResponse(
                content=iter("Error occurred in elastic proxy server."), media_type="text/event-stream"
            )

    async def retry_forward_request(
            self, data: Dict, instance_type: str, rest_try_times: int, is_chat: bool = True
    ):
        route = "/v1/chat/completions" if is_chat else "/v1/completions"
        cycler = self.prefill_cycler if instance_type == InstanceType.PREFILL else self.decode_cycler
        instance = self.schedule(cycler)

        try:
            all_content = ""
            async for content in self.forward_request(f"http://{instance}{route}", data):
                all_content += content.decode("utf-8")
            return all_content
        except HTTPException as http_exc:
            self.retried_nodes[instance] += 1
            if self.retried_nodes.get(instance, 0) >= args.retry_times:
                self._remove_instance_endpoint(instance_type, instance)
                self.retried_nodes.pop(instance)
            if rest_try_times > 1:
                all_content = await self.retry_forward_request(data, instance_type, rest_try_times - 1, is_chat)
                return all_content
            else:
                raise http_exc

    @staticmethod
    async def validate_json_request(raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type: Only 'application/json' is allowed",
            )

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    def _node_listener(self):
        while True:
            for node, check_times in list(self.waiting_nodes.items()):
                instance, instance_type = node.split("_")
                is_valid = asyncio.run(self.validate_instance(instance))
                check_times += 1
                if is_valid:
                    self._add_instance_endpoint(instance_type, instance)
                    self.waiting_nodes.pop(node)
                elif check_times == args.check_times:
                    logger.info(f"Instance {instance} was not added to elastic proxy.")
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = check_times
            time.sleep(args.check_interval)

    def add_instances(self, instance_type: str, instances: Optional[List[str]] = None):
        if instances is None:
            return
        for instance in instances:
            is_valid = asyncio.run(self.validate_instance(instance))
            if not is_valid:
                logger.info(f"Waiting for {instance_type}_instance {instance} to start.")
                self.waiting_nodes[f"{instance}_{instance_type}"] = 0
            else:
                self._add_instance_endpoint(instance_type, instance)

    async def remove_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
                raise HTTPException(status_code=400, detail="Invalid instance type.")
            self.validate_instance_format(instance)

            is_valid = self._remove_instance_endpoint(instance_type, instance)

            if is_valid:
                return JSONResponse(
                    content={"message": f"Removed {instance} from {instance_type}_instances."}
                )
            return JSONResponse(
                    content={"message": f"Instance {instance} is not in the {instance_type}_instances."}
                )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Error in remove_instance_endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
                raise HTTPException(status_code=400, detail="Invalid instance type.")
            self.validate_instance_format(instance)

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                logger.info(f"Waiting for {instance_type}_instance {instance} to start.")
                self.waiting_nodes[f"{instance}_{instance_type}"] = 0

                return JSONResponse(
                    content={"message": f"Waiting for {instance_type}_instance {instance} to start."})

            self._add_instance_endpoint(instance_type, instance)

            return JSONResponse(
                content={"message": f"Added {instance} to {instance_type}_instances."}
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Error in add_instance_endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @staticmethod
    def validate_instance_format(instance: str):
        if not instance or ":" not in instance:
            raise HTTPException(status_code=400, detail="Invalid instance format.")

        host, port_str = instance.split(":")
        try:
            if host != "localhost":
                ipaddress.ip_address(host)
            port = int(port_str)
            if not (0 < port < 65536):
                raise HTTPException(status_code=400, detail="Invalid port number.")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail="Invalid instance address."
            ) from e

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as client:
                logger.info(f"Verifying {instance} ...")
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info(f"Instance: {instance} could be added.")
                                return True
                            else:
                                logger.warning(f"Mismatch model {instance}: {model_cur} != {self.model}")
                                return False
                        else:
                            return False
                    else:
                        return False
        except Exception as e:
            logger.error(str(e))
            return False

    def _remove_instance_endpoint(self, instance_type: str, instance: str):
        instance = instance.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        if instance_type == "prefill" and instance in self.prefill_instances:
            self.prefill_instances.remove(instance)
            self.prefill_cycler = itertools.cycle(self.prefill_instances)
            logger.info(f"Removed {instance} from {instance_type}_instances. "
                        f"prefill node counts: {len(self.prefill_instances)}, "
                        f"decode node counts: {len(self.decode_instances)}")
            return True
        elif instance_type == InstanceType.DECODE and instance in self.decode_instances:
            self.decode_instances.remove(instance)
            self.decode_cycler = itertools.cycle(self.decode_instances)
            logger.info(f"Removed {instance} from {instance_type}_instances. "
                        f"prefill node counts: {len(self.prefill_instances)}, "
                        f"decode node counts: {len(self.decode_instances)}")
            return True
        return False

    def _add_instance_endpoint(self, instance_type: str, instance: str):
        instance = instance.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        if instance_type == "prefill":
            if instance not in self.prefill_instances:
                self.prefill_instances.append(instance)
                self.prefill_cycler = itertools.cycle(self.prefill_instances)
                logger.info(f"Added {instance} to {instance_type}_instances. "
                            f"prefill node counts: {len(self.prefill_instances)}, "
                            f"decode node counts: {len(self.decode_instances)}")
            else:
                logger.info(f"{instance_type}_instance {instance} already exists.")
        else:
            if instance not in self.decode_instances:
                self.decode_instances.append(instance)
                self.decode_cycler = itertools.cycle(self.decode_instances)
                logger.info(f"Added {instance} to {instance_type}_instances. "
                            f"prefill node counts: {len(self.prefill_instances)}, "
                            f"decode node counts: {len(self.decode_instances)}")
            else:
                logger.info(f"{instance_type}_instance {instance} already exists.")

    @staticmethod
    def api_key_authenticate(x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY", DEFAULT_ADMIN_API_KEY)
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning(f"Unauthorized access attempt with API Key: {x_api_key}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )


def parse_args():
    parser = argparse.ArgumentParser("vLLM elastic proxy server.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=5,
        help="Check interval before a node is added to the node list",
    )

    parser.add_argument(
        "--check-times",
        type=int,
        default=10,
        help="Check times before a node is added to the node list",
    )

    parser.add_argument(
        "--retry-times",
        type=int,
        default=3,
        help="Retry times before a node is deleted from the node list",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy_server = ProxyServer(
        prefill_instances=args.prefill,
        decode_instances=args.decode,
        model=args.model,
        port=args.port,
    )
