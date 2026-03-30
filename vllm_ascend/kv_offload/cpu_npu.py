import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend  # type: ignore
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class CpuNpuOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        assert cpu_block_size % gpu_block_size == 0
        self.block_size_factor = cpu_block_size // gpu_block_size

        # npu streams for npu->cpu and cpu->npu
        self.d2h_stream = torch.npu.Stream()
        self.h2d_stream = torch.npu.Stream()

        # job_id -> transfer npu event
        self.transfer_events: dict[int, torch.npu.Event] = {}
        # list of npu events available for reuse
        self.events_pool: list[torch.npu.Event] = []

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        self.npu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            self.npu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor[0].shape

            num_blocks_idx = 0
            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * self.block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            self.cpu_tensors.append(
                (
                    torch.zeros(
                        cpu_shape,
                        dtype=gpu_tensor[0].dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    ),
                    torch.zeros(
                        cpu_shape,
                        dtype=gpu_tensor[0].dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    ),
                )
            )
    
        # Pre-compute base pointers and block sizes for batch copies.
        self._src_base_ptrs = np.array(
            [t.data_ptr() for t in self.src_tensors], dtype=np.int64
        )
        self._dst_base_ptrs = np.array(
            [t.data_ptr() for t in self.dst_tensors], dtype=np.int64
        )
        self._block_size_in_bytes_arr = np.array(
            self.tensor_block_size_in_bytes, dtype=np.int64
        )


    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        logger.info("start transfer_async...")
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream = self.h2d_stream
            src_tensors = self.cpu_tensors
            dst_tensors = self.npu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            stream = self.d2h_stream
            src_tensors = self.npu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor
        src_sub_block_count = src_blocks.size * src_block_size_factor

        assert src_sub_block_count == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip

        # src_to_dst = np.empty((src_sub_block_count, 2), dtype=np.int64)
        src_block_ids = np.empty(dst_sub_block_count, dtype=np.int64)
        dst_block_ids = np.empty(dst_sub_block_count, dtype=np.int64)
        expand_block_ids(src_blocks, src_block_size_factor, src_block_ids)
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_block_ids)

        # Build flat pointer arrays for all tensors × all block pairs.
        num_pairs = dst_sub_block_count
        num_tensors = len(self.src_tensors)
        total = num_pairs * num_tensors

        all_src = np.empty(total, dtype=np.int64)
        all_dst = np.empty(total, dtype=np.int64)
        all_sizes = np.empty(total, dtype=np.int64)

        for t_idx, bsz in enumerate(self._block_size_in_bytes_arr):
            start = t_idx * num_pairs
            end = start + num_pairs
            all_src[start:end] = self._src_base_ptrs[t_idx] + src_block_ids * bsz
            all_dst[start:end] = self._dst_base_ptrs[t_idx] + dst_block_ids * bsz
            all_sizes[start:end] = bsz

        batch_src = torch.from_numpy(all_src)
        batch_dst = torch.from_numpy(all_dst)
        batch_sizes = torch.from_numpy(all_sizes)

        event = self.events_pool.pop() if self.events_pool else torch.npu.Event()
        with torch.npu.stream(stream):

            torch.ops._C_ascend.swap_blocks_batch(src_key_cache, dst_key_cache, src_to_dst_tensor)
            torch.ops._C_ascend.swap_blocks_batch(src_value_cache, dst_value_cache, src_to_dst_tensor)
            
            event.record(stream)

        self.transfer_events[job_id] = event

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_job_ids = []
        for job_id, event in self.transfer_events.items():
            if event.query():
                results.append(
                    TransferResult(
                        job_id=job_id,
                        success=True,
                        transfer_size=None,
                        transfer_time=None,
                        transfer_type=None,
                    )
                )
                finished_job_ids.append(job_id)
                self.events_pool.append(event)
        for job_id in finished_job_ids:
            del self.transfer_events[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait (block) until all specified transfer jobs are completed.
        """
        for job_id in job_ids:
            event = self.transfer_events.get(job_id)
            if event is not None:
                # This will block until the NPU event is complete
                event.synchronize()
