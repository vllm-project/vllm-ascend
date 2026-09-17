import torch, msgspec, time, queue
from vllm.config import VllmConfig
from vllm_ascend.ascend_config import get_ascend_config
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.logger import logger
from concurrent.futures import ThreadPoolExecutor
from vllm_ascend.embedding_offload.ec_mooncake_backend import EMooncakeBackend

ALIGNMENT = 2 * 1024 * 1024
WAIT_TIME = 1


class EMoonCakeStoreConnector():
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        if role == ECConnectorRole.SCHEDULER:
            torch.npu.set_device(0)

        parallel_config = vllm_config.parallel_config
        self.ec_store = EMooncakeBackend(parallel_config, role)

        if role != ECConnectorRole.SCHEDULER:
            self.aligned_tensor_size = get_ascend_config().encoder_caches_offload_config.aligned_tensor_size
            self.offload_queue = queue.Queue[tuple[str, torch.Tensor]]()
            self.thread_executor = ThreadPoolExecutor(max_workers=4)
            self.thread_executor.submit(self.embed_offload_run)
            self.decoder = msgspec.msgpack.Decoder(type=tuple)
            self.encoder = msgspec.msgpack.Encoder()
            
            self.recv_aligned_tensor, _ = self.aligned_empty_tensor([self.aligned_tensor_size, 1024, 1024], dtype=torch.bfloat16, device="npu")
            tensor_bytes = self.recv_aligned_tensor.element_size() * self.recv_aligned_tensor.numel()
            self.ec_store.register_buffer_single(self.recv_aligned_tensor.data_ptr(), tensor_bytes)

            self.send_aligned_tensor, _ = self.aligned_empty_tensor([self.aligned_tensor_size, 1024, 1024], dtype=torch.bfloat16, device="npu")
            self.ec_store.register_buffer_single(self.send_aligned_tensor.data_ptr(), tensor_bytes)

            self.swap_aligned_tensor, _ = self.aligned_empty_tensor([self.aligned_tensor_size, 1024, 1024], dtype=torch.bfloat16, device="npu")
            self.ec_store.register_buffer_single(self.swap_aligned_tensor.data_ptr(), tensor_bytes)


    def aligned_empty_tensor(self, shape, dtype=torch.float32, device="npu:0"):
        numel = torch.Size(shape).numel()
        elem_size = torch.tensor([], dtype=dtype).element_size()
        pad_elements = (ALIGNMENT * 2) // elem_size + 1024
        total_shape = (numel + pad_elements,)
        
        big_tensor = torch.zeros(total_shape, dtype=dtype, device=device)
        ptr = big_tensor.data_ptr()
        offset_bytes = (ALIGNMENT - (ptr % ALIGNMENT)) % ALIGNMENT
        offset_elements = offset_bytes // elem_size
        aligned_tensor = big_tensor[offset_elements : offset_elements + numel].view(shape)
        logger.info("original tensor addr %s, aligned tensor addr %s, aligned %s", hex(ptr), hex(aligned_tensor.data_ptr()), aligned_tensor.data_ptr() % ALIGNMENT == 0)
        
        return aligned_tensor, big_tensor

    def embed_offload_run(self):
        while True:
            try:
                mm_hash, embed_tensor = self.offload_queue.get()
                if not self.ec_store.exist_single(mm_hash):
                    self.save_caches(embed_tensor, mm_hash)
                self.offload_queue.task_done()
            except Exception as e:
                logger.error(f"embed offload {mm_hash} to cpu, error code: {str(e)}")
                if 'mm_hash' in locals():
                    self.send_queue.task_done()
                continue

    def offload_encoder_caches(self, tensor: torch.Tensor | None, mm_hash):
        self.offload_queue.put((mm_hash, tensor))
    
    def load_cache_from_store(self, mm_hash: str, aligned_tensor: torch.Tensor):
        """If hit, get the encode cache from Monncake Store"""
        start_time = time.perf_counter()
        while True:
            if time.perf_counter() - start_time > WAIT_TIME:
                raise TimeoutError(
                    f"Can not find the mm_hash {mm_hash} in the Mooncake store after {WAIT_TIME} seconds"
                )
            if self.ec_store.exist_single(mm_hash):
                break

            time.sleep(0.005)

        try:
            tensor_info = self.ec_store.get_tensor_info(mm_hash + "_info")
            if not isinstance(tensor_info, bytes):
                raise ValueError(f"tensor_info must be bytes, got {type(tensor_info)}")
    
            tensor_shape, tensor_dtype = self.decoder.decode(tensor_info)
            tensor = torch.empty(tensor_shape, dtype=getattr(torch, tensor_dtype), device="npu")
            tensor_bytes = tensor.element_size() * tensor.numel()

            self.ec_store.get_tensor_single(mm_hash, aligned_tensor.data_ptr(), tensor_bytes)
            tensor.copy_(aligned_tensor.view(-1)[:tensor.numel()].view(tensor_shape), non_blocking=False)
            return tensor

        except Exception as e:
                logger.error("Failed to get tensor %s from store: %s", mm_hash, e)
                raise

    def load_swap_caches(self, encoder_cache, mm_hashes, **kwargs) -> None:
        for mm_hash in mm_hashes:
            if mm_hash in encoder_cache:
                continue
            
            tensor = self.load_cache_from_store(mm_hash, self.swap_aligned_tensor)
            encoder_cache[mm_hash] = tensor
            
            logger.debug(
                "[Swap] Get tensor from store %s copy tensor %s",
                self.swap_aligned_tensor.view(-1)[: tensor.numel()].view(tensor.shape),
                encoder_cache[mm_hash],
            )
    
    def load_encoder_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        if mm_hash in encoder_cache:
            return

        tensor = self.load_cache_from_store(mm_hash, self.recv_aligned_tensor)
        encoder_cache[mm_hash] = tensor

        logger.info(
            "[Recv] Get tensor from store %s copy tensor %s",
            self.recv_aligned_tensor.view(-1)[: tensor.numel()].view(tensor.shape),
            tensor,
        )

    def swap_encoder_caches(self, scheduler_output, encoder_cache) -> None:
        mm_hashes = getattr(scheduler_output, 'swap_encoder_mm_hashes', [])
        if mm_hashes:
            self.thread_executor.submit(self.load_swap_caches, encoder_cache, mm_hashes)

    def save_caches(self, tensor: torch.Tensor | None, mm_hash, **kwargs) -> None:
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
        self.send_aligned_tensor.view(-1)[: tensor.numel()].copy_(tensor.view(-1), non_blocking=True)
        tensor_bytes = tensor.element_size() * tensor.numel()

        encoded_data = self.encoder.encode((tensor.shape, str(tensor.dtype).split(".")[-1]))
        if not self.ec_store.exist_single(mm_hash + "_info"):
            self.ec_store.put_tensor_info(mm_hash + "_info", encoded_data)

        if not self.ec_store.exist_single(mm_hash):
            self.ec_store.put_tensor_single(mm_hash, self.send_aligned_tensor.data_ptr(), tensor_bytes)
        logger.debug(
            "Send mm_hash %s tensor %s, aligned_tensor %s",
            mm_hash,
            tensor,
            self.send_aligned_tensor.view(-1)[: tensor.numel()].view(tensor.shape),
        )
    
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