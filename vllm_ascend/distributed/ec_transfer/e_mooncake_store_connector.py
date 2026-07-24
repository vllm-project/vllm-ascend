# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import msgspec
import torch
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
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.distributed.ec_transfer.e_mooncake_backend import EMooncakeBackend

if TYPE_CHECKING:
    from vllm.v1.request import Request

ALIGNMENT = 2 * 1024 * 1024
WAIT_TIME = 1


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

        parallel_config = vllm_config.parallel_config
        self.ec_store = EMooncakeBackend(parallel_config)

        if role != ECConnectorRole.SCHEDULER:
            ec_extra_config = getattr(transfer_config, "ec_connector_extra_config", {})
            self.thread_executor = ThreadPoolExecutor(max_workers=getattr(transfer_config, "max_workers", 8) or 8)
            if ec_extra_config:
                self.aligned_tensor_size = ec_extra_config.get("aligned_tensor_size", 100)
            else:
                self.aligned_tensor_size = 100

            if transfer_config.ec_role == "ec_producer":
                self.send_queue = queue.Queue[tuple[str, torch.Tensor]]()
                self.thread_executor.submit(self.producer_run)
                self.encoder = msgspec.msgpack.Encoder()
                logger.info("============ Producer init ===============")
            elif transfer_config.ec_role == "ec_consumer":
                self.decoder = msgspec.msgpack.Decoder()
                logger.info("============= Consumer init ==============")

            self.aligned_tensor, _ = self.aligned_empty_tensor(
                [self.aligned_tensor_size, 1024, 1024], dtype=torch.bfloat16, device="npu"
            )
            tensor_bytes = self.aligned_tensor.element_size() * self.aligned_tensor.numel()
            self.ec_store.register_buffer_single(self.aligned_tensor.data_ptr(), tensor_bytes)

    def aligned_empty_tensor(self, shape, dtype=torch.float32, device="npu:0"):
        numel = torch.Size(shape).numel()
        elem_size = torch.tensor([], dtype=dtype).element_size()
        pad_elements = (ALIGNMENT * 2) // elem_size + 1024
        tensor_shape = (numel + pad_elements,)

        pad_tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
        ptr = pad_tensor.data_ptr()
        offset_bytes = (ALIGNMENT - (ptr % ALIGNMENT)) % ALIGNMENT
        offset_elements = offset_bytes // elem_size
        aligned_tensor = pad_tensor[offset_elements : offset_elements + numel].view(shape)

        logger.info(
            "original tensor addr %s, aligned tensor addr %s, aligned %s",
            hex(ptr),
            hex(aligned_tensor.data_ptr()),
            aligned_tensor.data_ptr() % ALIGNMENT == 0,
        )

        return aligned_tensor, pad_tensor

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

            start_time = time.perf_counter()
            while True:
                if time.perf_counter() - start_time > WAIT_TIME:
                    raise TimeoutError(
                        f"Can not find the mm_hash {mm_data.mm_hash} in the Mooncake store after {WAIT_TIME} seconds"
                    )
                if self.ec_store.exist_single(mm_data.mm_hash):
                    break

                time.sleep(0.005)

            try:
                tensor_info = self.ec_store.get_tensor_info(mm_data.mm_hash + "_info")
                if not isinstance(tensor_info, bytes):
                    raise ValueError(f"tensor_info must be bytes, got {type(tensor_info)}")

                tensor_shape, tensor_dtype = self.decoder.decode(tensor_info)
                tensor = torch.empty(tensor_shape, dtype=getattr(torch, tensor_dtype), device="npu")
                tensor_bytes = tensor.element_size() * tensor.numel()
                self.ec_store.get_tensor_single(mm_data.mm_hash, self.aligned_tensor.data_ptr(), tensor_bytes)
                tensor.copy_(self.aligned_tensor.view(-1)[: tensor.numel()].view(tensor_shape), non_blocking=False)
                encoder_cache[mm_data.mm_hash] = tensor
                logger.debug(
                    "Get tensor from store %s copy tensor %s",
                    self.aligned_tensor.view(-1)[: tensor.numel()].view(tensor_shape),
                    tensor,
                )

            except Exception as e:
                logger.error("Failed to get tensor %s from store: %s", mm_data.mm_hash, e)
                raise

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

        return self.ec_store.exist_single(identifier)

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
                self.aligned_tensor.view(-1)[: tensor.numel()].copy_(tensor.view(-1), non_blocking=True)
                tensor_bytes = tensor.element_size() * tensor.numel()

                encoded_data = self.encoder.encode((tensor.shape, str(tensor.dtype).split(".")[-1]))
                if not self.ec_store.exist_single(feat_key + "_info"):
                    self.ec_store.put_tensor_info(feat_key + "_info", encoded_data)
                torch.npu.current_stream().synchronize()

                if not self.ec_store.exist_single(feat_key):
                    self.ec_store.put_tensor_single(feat_key, self.aligned_tensor.data_ptr(), tensor_bytes)
                logger.debug(
                    "Send feat key %s tensor %s, aligned_tensor %s",
                    feat_key,
                    tensor,
                    self.aligned_tensor.view(-1)[: tensor.numel()].view(tensor.shape),
                )
                self.send_queue.task_done()

            except Exception as e:
                logger.error("send tensor info: %s to consumer, error code: %s", feat_key, e)
                if "feat_key" in locals():
                    self.send_queue.task_done()
                continue
