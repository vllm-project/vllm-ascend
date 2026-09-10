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
import socket
import threading
from contextlib import suppress
from typing import Any

import regex as re
import torch
from vllm.logger import logger

from ..executor.elastic_load import (
    P2PSend,
    build_transfer_shape_manifest,
    get_cached_processed_layout_transfer_items,
    register_processed_layout_transfer_items,
)
from ..utils import find_free_port


def _recv_json_message(sock: socket.socket, max_size: int = 64 * 1024 * 1024) -> dict:
    """Receive one complete JSON object from a TCP socket."""
    buffer = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_size:
            raise RuntimeError(f"JSON message exceeds max size {max_size} bytes")
        if buffer.rstrip().endswith((b"}", b"]")):
            try:
                payload = json.loads(buffer.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object, got {type(payload)}")
            return payload

    if not buffer:
        raise RuntimeError("Incomplete JSON message received from server")
    try:
        payload = json.loads(buffer.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError("Incomplete JSON message received from server") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object, got {type(payload)}")
    return payload


def _parse_transfer_shape_manifest(raw_manifest: object) -> dict[str, tuple[int, ...]] | None:
    if raw_manifest is None:
        return None
    if not isinstance(raw_manifest, dict):
        raise RuntimeError(f"Invalid transfer_shapes type: {type(raw_manifest)}")

    manifest: dict[str, tuple[int, ...]] = {}
    for name, shape in raw_manifest.items():
        if not isinstance(name, str) or not isinstance(shape, list):
            raise RuntimeError(f"Invalid transfer shape entry for {name!r}: {shape!r}")
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise RuntimeError(f"Invalid transfer shape dims for {name}: {shape}")
        manifest[name] = tuple(shape)
    return manifest


class ElasticClient:
    """
    Class for handling the client-side logic of Netloader of models.
    """

    def __init__(
        self,
        sources: list[str],
        device_id: int,
        model_path: str,
        tp: int,
        pp: int,
        group_name: str = "netloader",
        int8_cache: str = "no",
    ):
        """
        Initializes the ElasticClient instance.

        Parameters:
        - sources: List of source addresses in the format IP:port.
        - device_id: The ID of the current device.
        - model_path: The path to the model.
        - tp: Tensor parallel size.
        - pp: Pipeline parallel size.
        - group_name: Name of the HCCL process group.
        - int8_cache: The type of caching for int8 parameters (HBM, DRAM, or no).
        """
        self.sources = sources
        self.device_id = device_id
        self.model_path = model_path
        self.tp = tp
        self.pp = pp
        self.group_name = group_name
        self.int8_cache = int8_cache

        self.s: socket.socket | None = None
        self.ack: tuple[str, int] | None = None
        self.transfer_shape_manifest: dict[str, tuple[int, ...]] | None = None
        self.server_addr: str | None = None
        self.server_port: int | None = None

        for source in self.sources:
            try:
                ip, port_str = source.split(":")
                port = int(port_str)
            except Exception as e:
                logger.info("IP format error: %s, detail: %s", source, e)
                continue

            self.server_addr = ip
            self.server_port = port

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                logger.info("Start connection to server: %s:%s", self.server_addr, self.server_port)
                sock.connect((self.server_addr, self.server_port))
                logger.info("Finish connection to server: %s:%s", self.server_addr, self.server_port)
                sock.settimeout(60)

                self.s = sock
                self.ack = self.register(device_id, model_path, tp, pp)
                break
            except Exception as e:
                logger.error("Connect to %s fails, detail: %s", source, e)
                if sock is not None:
                    with suppress(Exception):
                        sock.close()
                self.s = None
                self.ack = None
                self.server_addr = None
                self.server_port = None

        if self.s is None:
            sources_str = ", ".join(self.sources[:2])
            if len(self.sources) > 2:
                sources_str += f", ... (total {len(self.sources)})"
            logger.error(
                "All sources exhausted, no connection established for device_id=%s, model_path=%s, sources=[%s]",
                device_id,
                model_path,
                sources_str,
            )

    def close(self) -> None:
        """
        Closes the socket connection.
        """
        if self.s is not None:
            try:
                self.s.close()
            except Exception as e:
                logger.error("Error closing socket: %s", e)
            finally:
                self.s = None

    def __enter__(self) -> "ElasticClient":
        """
        Context manager enter method.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit method.
        """
        self.close()

    def __del__(self):
        """
        Destructor method to ensure socket is closed.
        """
        with suppress(Exception):
            self.close()

    def send_str(self, data_str: str) -> None:
        """
        Sends a string over the socket connection.

        Parameters:
        - data_str: The string to be sent.
        """
        if self.s is None:
            raise RuntimeError("Socket was not created correctly.")
        self.s.send(data_str.encode("utf-8"))

    def recv_json(self) -> dict:
        """Receive one JSON object over the socket connection."""
        if self.s is None:
            raise RuntimeError("Socket was not created correctly.")
        return _recv_json_message(self.s)

    def register(self, device_id: int, model_path: str, tp: int, pp: int) -> tuple[str, int]:
        """
        Registers the client with the server.

        Parameters:
        - device_id: The ID of the current device.
        - model_path: The path to the model.
        - tp: Tensor parallel size.
        - pp: Pipeline parallel size.

        Returns:
        - A tuple containing the communication name and port.
        """
        free_port = find_free_port()
        data = {
            "label": "JOIN",
            "content": {
                "device_id": device_id,
                "model_path": model_path,
                "tp": tp,
                "pp": pp,
                "port": free_port,
                "group_name": self.group_name,
                "int8_cache": self.int8_cache,
            },
        }

        try:
            self.send_str(json.dumps(data))
        except Exception as e:
            raise RuntimeError(f"Send data {data} to server fails, detail: {e}")

        try:
            ack = self.recv_json()
        except Exception as e:
            raise RuntimeError(f"Receive data from server fails, detail: {e}")

        content = ack.get("content") if isinstance(ack, dict) else None
        transfer_shapes = content.get("transfer_shapes") if isinstance(content, dict) else None
        logger.info(
            "Receive ack: label=%s name=%s transfer_shape_count=%s",
            ack.get("label") if isinstance(ack, dict) else None,
            content.get("name") if isinstance(content, dict) else None,
            len(transfer_shapes) if isinstance(transfer_shapes, dict) else 0,
        )

        if (
            "label" in ack
            and ack["label"] == "JOIN_ACK"
            and "content" in ack
            and ack["content"] is not None
            and "name" in ack["content"]
        ):
            content = ack["content"]
            self.transfer_shape_manifest = _parse_transfer_shape_manifest(content.get("transfer_shapes"))
            return (content["name"], free_port)
        elif "label" in ack and ack["label"] == "JOIN_NACK" and "content" in ack:
            raise RuntimeError(f"Receive nack from server, reason: {ack['content']}")
        else:
            raise RuntimeError(f"Receive ack {ack} from server does not contain required fields")


class ElasticServer:
    """
    Class for handling the server-side logic of Netloader of models.
    """

    def __init__(
        self,
        addr: str,
        port: int,
        model,
        device_id: int,
        model_path: str,
        tp: int,
        pp: int,
        int8_cache: str,
        int8_cache_name: list[str] | None,
        group_name: str = "netloader",
    ):
        """
        Initializes the ElasticServer instance.

        Parameters:
        - addr: The IP address to listen on.
        - port: The port number to listen on.
        - model: The model to be served.
        - device_id: The ID of the current device (i.e. global rank).
        - model_path: The path to the model.
        - tp: Tensor parallel size.
        - pp: Pipeline parallel size.
        - int8_cache: The type of caching for int8 parameters (HBM, DRAM, or no).
        - int8_cache_name: List of parameter names to be cached.
        - group_name: Name of the HCCL process group.
        """
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
        self.group_name = group_name
        self.int8_cache = int8_cache
        self._registered_transfer_items: list[tuple[str, torch.Tensor]] | None = None
        self._registered_transfer_shapes: dict[str, tuple[int, ...]] | None = None

        self.original_int8 = {}
        int8_pattern = "|".join(map(re.escape, int8_cache_name)) if int8_cache_name is not None else "(?:)"
        for name, param in self.model.named_parameters():
            if param.dtype == torch.int8:
                if int8_cache == "hbm":
                    if int8_cache_name is None or (
                        int8_cache_name is not None and re.search(int8_pattern, name) is not None
                    ):
                        try:
                            self.original_int8[name] = param.data.clone().detach()
                        except RuntimeError as e:
                            logger.error("Failed to cache int8 tensor %s to HBM, change to DRAM, due to %s", name, e)
                            self.original_int8[name] = param.data.cpu()

                elif int8_cache == "dram":
                    if int8_cache_name is None or (
                        int8_cache_name is not None and re.search(int8_pattern, name) is not None
                    ):
                        self.original_int8[name] = param.data.cpu()
                elif int8_cache == "no":
                    pass
                else:
                    logger.warning(
                        "int8_cache should be selected in [HBM, DRAM], but got %s, change to no cache", int8_cache
                    )

        logger.info(
            "Server %s:%s starts, device id: %s, model path: %s, tp: %s, pp: %s, int8 params %s are saved to %s",
            self.addr,
            self.port,
            self.device_id,
            self.model_path,
            self.tp,
            self.pp,
            list(self.original_int8),
            int8_cache,
        )

    def register_transfer_manifest(self, model) -> None:
        """Register processed-layout transfer manifest after weights are finalized."""
        if self.int8_cache != "no":
            return

        cached_items = get_cached_processed_layout_transfer_items(model)
        registered_items = register_processed_layout_transfer_items(model)
        self._registered_transfer_items = registered_items
        self._registered_transfer_shapes = build_transfer_shape_manifest(registered_items)
        logger.info(
            "[netloader_p2p] registered transfer manifest count=%s cache_hit=%s rank=%s group=%s",
            len(registered_items),
            cached_items is not None,
            self.device_id,
            self.group_name,
        )

    def __del__(self):
        """
        Destructor method to ensure socket is closed.
        """
        if self.s is not None:
            with suppress(Exception):
                self.s.close()

    def start(self):
        """
        Starts the server to handle incoming connections.
        """
        handler_thread = threading.Thread(target=self.elastic_client_handler)
        handler_thread.daemon = True
        handler_thread.start()

    def elastic_client_handler(self):
        """
        Handles incoming client connections.
        """
        while True:
            conn, addr = self.s.accept()
            logger.info("Accept new connection from %s:%s...", *addr)
            self.register_handler(conn, addr)

    def register_handler(self, conn, addr, buffer_size=1024):
        """
        Handles the registration of a client.

        Parameters:
        - conn: The connection socket.
        - addr: The address of the client.
        - buffer_size: The size of the buffer for receiving data.
        """
        data_str = conn.recv(buffer_size).decode("utf-8")
        if not data_str:
            return
        try:
            data = json.loads(data_str)
        except Exception:
            logger.error("Failed to load %s as JSON string from %s", data_str, addr)
            conn.close()
            return

        def is_valid_data(data):
            """
            Validates the received data.

            Parameters:
            - data: The data to be validated.

            Returns:
            - True if the data is valid, otherwise False.
            """
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
            int8_cache = content.get("int8_cache")
            if int8_cache is not None and int8_cache not in ["hbm", "dram", "no"]:
                return False
            return isinstance(port, int) or (isinstance(port, str) and port.isdigit())

        comm_name = None
        ack: dict[str, Any]
        if is_valid_data(data):
            device_id = int(data["content"]["device_id"])
            model_path = data["content"]["model_path"]
            tp = int(data["content"]["tp"])
            pp = int(data["content"]["pp"])
            int8_cache = data["content"].get("int8_cache")

            if (
                int(self.device_id) == device_id
                and self.model_path == model_path
                and int(self.tp) == tp
                and int(self.pp) == pp
            ):
                if int8_cache is not None and self.int8_cache != int8_cache:
                    msg = f"Received int8_cache {int8_cache} does not consist with this server {self.int8_cache}"
                    logger.warning(msg)
                    ack = {
                        "label": "JOIN_NACK",
                        "content": msg,
                    }
                else:
                    comm_name = str(addr[0]) + ":" + str(addr[1])
                    ack_content: dict[str, Any] = {"name": comm_name}
                    if (
                        self.int8_cache == "no"
                        and self._registered_transfer_items is not None
                        and self._registered_transfer_shapes is not None
                    ):
                        ack_content["transfer_shapes"] = {
                            name: list(shape) for name, shape in self._registered_transfer_shapes.items()
                        }
                    ack = {"label": "JOIN_ACK", "content": ack_content}
            else:
                server_desc = (int(self.device_id), self.model_path, int(self.tp), int(self.pp))
                client_desc = (device_id, model_path, tp, pp)
                msg = f"Received data {client_desc} does not consist with this server {server_desc}"
                logger.warning(msg)
                ack = {
                    "label": "JOIN_NACK",
                    "content": msg,
                }
        else:
            logger.warning("Received data does not contain required fields: %s", data)
            ack = {"label": "JOIN_NACK", "content": f"Received data does not contain required fields: {data}"}

        try:
            ack_str = json.dumps(ack).encode("utf-8")
        except Exception as e:
            logger.error("Failed to convert %s to JSON format, details: %s", ack, e)
            conn.close()
            return

        try:
            conn.sendall(ack_str)
        except Exception as e:
            logger.error("Failed to send %s to %s, details: %s", ack, addr, e)
            conn.close()
            return

        if ack["content"] and isinstance(ack["content"], dict) and "name" in ack["content"]:
            try:
                p2psend = P2PSend(
                    self.addr,
                    data["content"]["port"],
                    ack["content"]["name"],
                    data["content"].get("group_name", "netloader"),
                    send_processed_weights=self.int8_cache == "no",
                )
                p2psend.send(
                    self.model,
                    self.original_int8,
                    registered_transfer_items=self._registered_transfer_items,
                )
            except Exception as e:
                logger.error("P2PSend Failed to send model to %s, details: %s", self.addr, e)
        conn.close()
