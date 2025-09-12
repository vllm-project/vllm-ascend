#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import re
import socket
import threading
from typing import List, Optional

import torch
from vllm.logger import logger

from ..executor.elastic_load import P2PSend
from ..utils import find_free_port


class ElasticClient:

    def __init__(self, sources: list, device_id: int, model_path: str, tp: int,
                 pp: int):
        self.sources = sources
        self.s = None
        self.ack = None
        self.server_addr = None
        self.server_port = None

        for source in self.sources:
            try:
                ip, port = source.split(':')
                port = int(port)
            except Exception as e:
                logger.error(f"IP format error: {source}, detail: {e}")
                continue

            self.server_addr = ip
            self.server_port = port

            try:
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                logger.info("Start connection to server: {}:{}".format(
                    self.server_addr, self.server_port))
                self.s.connect((self.server_addr, self.server_port))
                logger.info("Finish connection to server: {}:{}".format(
                    self.server_addr, self.server_port))
                self.s.settimeout(60)

                self.ack = self.register(device_id, model_path, tp, pp)
                break
            except Exception as e:
                logger.error(f"Connect to {source} fails, detail: {e}")
                if self.s is not None:
                    self.s.close()
                self.s = None
                self.ack = None
                self.server_addr = None
                self.server_port = None

    def __del__(self):
        if self.s is not None:
            self.s.close()

    def send_str(self, data_str):
        if self.s is None:
            raise RuntimeError("Socket was not created correctly.")
        self.s.send(data_str.encode("utf-8"))

    def recv_str(self, buffer_size=1024):
        if self.s is None:
            raise RuntimeError("Socket was not created correctly.")
        data_str = self.s.recv(buffer_size).decode("utf-8")
        return data_str

    def register(self, device_id: int, model_path: str, tp: int, pp: int):
        free_port = find_free_port()
        data = {
            "label": "JOIN",
            "content": {
                'device_id': device_id,
                'model_path': model_path,
                'tp': tp,
                'pp': pp,
                'port': free_port
            }
        }

        try:
            self.send_str(json.dumps(data))
        except Exception as e:
            raise RuntimeError(
                f"Send data {data} to server fails, detail: {e}")

        try:
            ack_str = self.recv_str()
        except Exception as e:
            raise RuntimeError(f"Receive data from server fails, detail: {e}")

        try:
            ack = json.loads(ack_str)
        except Exception as e:
            raise RuntimeError(
                f"Receive data {ack_str} cannot be converted to JSON format, detail: {e}"
            )

        logger.info(f"Receive ack: {ack}")

        if "label" in ack and ack[
                "label"] == 'JOIN_ACK' and "content" in ack and ack[
                    "content"] is not None and "name" in ack["content"]:
            return (ack["content"]["name"], free_port)
        else:
            raise RuntimeError(
                f"Receive ack {ack} from server does not contain required fields"
            )


class ElasticServer:

    def __init__(self, addr: str, port: int, model, device_id: int,
                 model_path: str, tp: int, pp: int, int8_cache: str,
                 int8_cache_name: Optional[List[str]]):
        self.addr = addr
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.addr, self.port))
        self.s.listen(256)

        self.model = model
        self.device_id = device_id
        self.model_path = model_path
        self.tp = tp
        self.pp = pp

        self.original_int8 = {}
        int8_pattern = "|".join(
            map(re.escape,
                int8_cache_name)) if int8_cache_name is not None else "(?:)"
        for name, param in self.model.named_parameters():
            if param.dtype == torch.int8:
                if int8_cache == 'hbm':
                    if int8_cache_name is None or (
                            int8_cache_name is not None
                            and re.search(int8_pattern, name) is not None):
                        try:
                            self.original_int8[name] = param.data.clone(
                            ).detach()
                        except RuntimeError as e:
                            logger.error(
                                f"Failed to cache int8 tensor {name} to HBM, change to DRAM, due to {e}"
                            )
                            self.original_int8[name] = param.data.cpu()

                elif int8_cache == 'dram':
                    if int8_cache_name is None or (
                            int8_cache_name is not None
                            and re.search(int8_pattern, name) is not None):
                        self.original_int8[name] = param.data.cpu()
                elif int8_cache == 'no':
                    pass
                else:
                    logger.warning(
                        f"int8_cache should be selected in [HBM, DRAM], but got {int8_cache}, change to no cache"
                    )

        logger.info(
            f"Server {self.addr}:{self.port} starts, device id: {self.device_id}, model path: {self.model_path}, tp: {self.tp}, pp: {self.pp}, int8 params {self.original_int8.keys()} are saved to {int8_cache}"
        )

    def __del__(self):
        self.s.close()

    def start(self):
        handler_thread = threading.Thread(target=self.elastic_client_handler)
        handler_thread.daemon = True
        handler_thread.start()

    def elastic_client_handler(self):
        while True:
            conn, addr = self.s.accept()
            logger.info("Accept new connection from {}:{}...".format(*addr))
            self.register_handler(conn, addr)

    def register_handler(self, conn, addr, buffer_size=1024):
        data_str = conn.recv(buffer_size).decode("utf-8")
        if not data_str:
            return
        try:
            data = json.loads(data_str)
        except Exception:
            logger.error(f"Failed to load {data_str} as JSON string")
            conn.close()
            return

        def is_valid_data(data):
            if not isinstance(data, dict):
                return False
            if data.get("label") != "JOIN":
                return False
            content = data.get("content")
            if not isinstance(content, dict):
                return False
            required_keys = ["device_id", "model_path", "tp", "pp", "port"]
            if not all(k in content for k in required_keys):
                return False
            port = content["port"]
            if not (isinstance(port, int) or
                    (isinstance(port, str) and port.isdigit())):
                return False
            return True

        comm_name = None
        if is_valid_data(data):
            device_id = int(data["content"]["device_id"])
            model_path = data["content"]["model_path"]
            tp = int(data["content"]["tp"])
            pp = int(data["content"]["pp"])

            if int(self.device_id
                   ) == device_id and self.model_path == model_path and int(
                       self.tp) == tp and int(self.pp) == pp:
                comm_name = str(addr[0]) + ":" + str(addr[1])
                ack = {"label": "JOIN_ACK", "content": {"name": comm_name}}
            else:
                logger.warning(
                    f"Received data ({(device_id, model_path, tp, pp)}) does not consist with this server ({(int(self.device_id), self.model_path, int(self.tp), int(self.pp))}) "
                )
                ack = {"label": "JOIN_ACK", "content": {}}
        else:
            logger.warning(
                f"Received data does not contain required fields: {data}")
            ack = {"label": "JOIN_ACK", "content": {}}

        try:
            ack_str = json.dumps(ack).encode("utf-8")
        except Exception as e:
            logger.error(
                f"Failed to convert {ack} to JSON format, details: {e}")
            conn.close()
            return

        try:
            conn.send(ack_str)
        except Exception as e:
            logger.error(f"Failed to send {ack} to {addr}, details: {e}")
            conn.close()
            return

        if ack["content"] and isinstance(ack["content"],
                                         dict) and 'name' in ack["content"]:
            p2psend = P2PSend(self.addr, data["content"]["port"],
                              ack["content"]["name"])
            p2psend.send(self.model, self.original_int8)

        conn.close()
