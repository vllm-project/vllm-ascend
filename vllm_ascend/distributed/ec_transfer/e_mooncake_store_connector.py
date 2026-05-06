# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import queue
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnectorMetadata,
    MMMeta,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.distributed.ec_transfer.e_mooncake_backend import MooncakeStoreConfig, mooncake_engine_init

if TYPE_CHECKING:
    from vllm.v1.request import Request

ALIGNMENT = 2 * 1024 * 1024


class EMoonCakeStoreConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index
        self._mm_datas_need_loads: dict[str, int] = {}
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        # init mooncake store
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e
        self.ec_store = MooncakeDistributedStore()
        self.config = MooncakeStoreConfig.load_from_env()
        mooncake_engine_init(self.ec_store, self.config, role)

        if role != ECConnectorRole.SCHEDULER:
            # 获取P节点监听地址
            ec_extra_config = getattr(transfer_config, "ec_connector_extra_config", {})
            self.listen_ports = ec_extra_config.get("listen_ports", None)
            if not self.listen_ports:
                raise ValueError("Producer must have 'listen_ports' in config.")
            # 解析为 (host, port) 列表
            self.consumer_sock_addrs = [(transfer_config.ec_ip, addr_port) for addr_port in self.listen_ports]
            self.thread_executor = ThreadPoolExecutor(max_workers=getattr(transfer_config, "max_workers", 8) or 8)

            if transfer_config.ec_role == "ec_producer":
                self.send_queue = queue.Queue[tuple[str, torch.Tensor]]()
                self.zmq_paths = [make_zmq_path("tcp", host, port) for host, port in self.consumer_sock_addrs]
                self.thread_executor.submit(self.producer_run)
                logger.info("============ Producer init ===============")
            elif transfer_config.ec_role == "ec_consumer":
                self.handle_caches = dict[str, tuple[int, list[int], str]]
                self.recv_queue = queue.Queue[bytes]()
                self.thread_executor.submit(self.consumer_run)
                self.thread_executor.submit(self.recv_feat_async)
                logger.info("============= Consumer init ==============")

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue

            tensor_bytes, tensor_shape, tensor_dtype = self.handle_caches.get(mm_data.mm_hash, None)
            tensor = torch.empty(tensor_shape, dtype=tensor_dtype).npu()
            tensor_addr = (tensor.data_ptr() + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            self.ec_store.register_buffer(tensor_addr, tensor_bytes)
            self.ec_store.get_into(mm_data.mm_hash, tensor_addr, tensor_bytes)
            encoder_cache[mm_data.mm_hash] = tensor
            self.ec_store.unregister_buffer(tensor_addr)
            logger.info("recv tensor data shape %s", tensor.shape)

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return

        self.send_queue.put((mm_hash, encoder_cache[mm_hash]))

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """
        Check if cache exist externally for the media

        Args:
            identifier (str): the identifier of the media.

        Returns:
            Bool indicate that media exists in cache or not
        """
        return self.ec_store.is_exist(identifier)

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = request.get_num_encoder_embeds(index)
        # Insert mm_hash only if this block has not been recorded yet.
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECExampleConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def producer_run(self):
        while True:
            try:
                feat_key, tensor = self.send_queue.get()
                tensor_bytes = tensor.element_size() * tensor.numel()
                tensor_addr = (tensor.data_ptr() + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                self.ec_store.register_buffer(tensor_addr, tensor_bytes)
                self.ec_store.put_from(feat_key, tensor_addr, tensor_bytes)
                self.ec_store.unregister_buffer(tensor_addr)

                encoder = msgspec.msgpack.Encoder()
                encoded_data = encoder.encode((feat_key, tensor_bytes, tensor.shape, str(tensor.dtype).split(".")[-1]))

                for path in self.zmq_paths:
                    with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                        ensure_zmq_send(sock, encoded_data)
                        ack = sock.recv()
                        if ack != b"ACK":
                            raise ValueError(f"Unexpected ACK response: {ack}")
                logger.info("rank %s send the feat key %s", get_world_group().local_rank, feat_key)
                self.send_queue.task_done()
            except Exception:
                # 捕获所有异常，避免线程退出
                logger.error("send tensor info: {feat_key} to consumer")
                # 确保队列任务完成，避免死锁
                if "feat_key" in locals():
                    self.send_queue.task_done()
                continue

    def recv_feat_async(self):
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                payload = self.recv_queue.get()
                feat_key, tensor_bytes, tensor_shape, tensor_dtype = decoder.decode(payload)
                self.handle_caches[feat_key] = (tensor_bytes, tensor_shape, getattr(torch, tensor_dtype))
                self.recv_queue.task_done()
            except Exception:
                # 捕获所有异常，避免线程退出
                logger.error("recv tensor info: {feat_key} recv error")
                # 确保队列任务完成，避免死锁
                if "feat_key" in locals():
                    self.recv_queue.task_done()
                continue

    def consumer_run(self):
        """listening and recv the feat key"""
        local_rank = get_world_group().local_rank
        tp_size = len(self.consumer_sock_addrs)
        side_channel_host, handshake_port = self.consumer_sock_addrs[local_rank % tp_size]
        path = make_zmq_path("tcp", side_channel_host, handshake_port)

        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    sock.send_multipart((identity, b"", b"ACK"))
                    self.recv_queue.put(payload[0])
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries: {e}")
                raise RuntimeError(f"Failed to send data after {max_retries} retries: {e}")


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
