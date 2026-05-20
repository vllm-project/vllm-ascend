from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend  # type: ignore
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec


@dataclass
class Transfer:
    job_id: int
    stream: torch.npu.Stream
    start_event: torch.npu.Event
    end_event: torch.npu.Event
    num_bytes: int


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

    # Vectorized: compute all sub-block IDs at once
    bases = block_ids * block_size_factor
    offsets = np.arange(block_size_factor)
    # shape: (num_blocks, block_size_factor) -> ravel to 1D
    all_ids = (bases[:, None] + offsets[None, :]).ravel()
    # Skip the first skip_count elements (only affects first block)
    if skip_count > 0:
        all_ids = all_ids[skip_count:]
    output[: len(all_ids)] = all_ids


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

        # Ordered queue of in-flight transfers per direction
        self._d2h_transfers: deque[Transfer] = deque()
        self._h2d_transfers: deque[Transfer] = deque()

        # Reusable event pool to avoid allocation overhead
        self._event_pool: list[torch.npu.Event] = []

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
        # In vllm-ascend, each layer's KV cache is stored as a tuple
        # (key_cache, value_cache), so we flatten them into individual
        # sub-tensors for batching: [layer0_key, layer0_value,
        # layer1_key, layer1_value, ...].
        npu_base_ptrs = []
        cpu_base_ptrs = []
        block_sizes_in_bytes = []

        for npu_tensor, cpu_tensor in zip(self.npu_tensors, self.cpu_tensors):
            for kv_idx in range(2):  # 0=key, 1=value
                npu_t = npu_tensor[kv_idx]
                cpu_t = cpu_tensor[kv_idx]
                npu_base_ptrs.append(npu_t.data_ptr())
                cpu_base_ptrs.append(cpu_t.data_ptr())
                # block size in bytes = stride of dim 0 (elements) * element size
                block_sizes_in_bytes.append(npu_t.stride(0) * npu_t.element_size())

        self._npu_base_ptrs = np.array(npu_base_ptrs, dtype=np.int64)
        self._cpu_base_ptrs = np.array(cpu_base_ptrs, dtype=np.int64)
        self._block_size_in_bytes_arr = np.array(block_sizes_in_bytes, dtype=np.int64)
        # Total bytes per block across all sub-tensors (for transfer stats)
        self._total_bytes_per_block = int(self._block_size_in_bytes_arr.sum())

    def _get_event(self) -> torch.npu.Event:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.npu.Event(enable_timing=True)

    def _recycle_event(self, event: torch.npu.Event) -> None:
        self._event_pool.append(event)

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream = self.h2d_stream
            src_base_ptrs = self._cpu_base_ptrs
            dst_base_ptrs = self._npu_base_ptrs
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
            is_d2h = False
            transfers = self._h2d_transfers
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            stream = self.d2h_stream
            src_base_ptrs = self._npu_base_ptrs
            dst_base_ptrs = self._cpu_base_ptrs
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor
            is_d2h = True
            transfers = self._d2h_transfers

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor
        src_sub_block_count = src_blocks.size * src_block_size_factor

        assert src_sub_block_count == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip

        # Expand block IDs into sub-block IDs
        src_block_ids = np.empty(src_sub_block_count, dtype=np.int64)
        dst_block_ids = np.empty(src_sub_block_count, dtype=np.int64)
        expand_block_ids(src_blocks, src_block_size_factor, src_block_ids)
        expand_block_ids(
            dst_blocks,
            dst_block_size_factor,
            dst_block_ids,
            skip_count=dst_sub_blocks_to_skip,
        )

        # Build flat pointer arrays for all sub-tensors × all block pairs.
        # sub-tensors = [layer0_key, layer0_value, layer1_key, layer1_value, ...]
        # Fully vectorized via numpy broadcasting (no Python loop).
        num_pairs = src_sub_block_count
        num_sub_tensors = len(self._block_size_in_bytes_arr)
        total = num_pairs * num_sub_tensors

        # (num_sub_tensors, 1) + (1, num_pairs) * (num_sub_tensors, 1) -> (num_sub_tensors, num_pairs)
        bsz_col = self._block_size_in_bytes_arr[:, None]  # (T, 1)
        all_src = (src_base_ptrs[:, None] + src_block_ids[None, :] * bsz_col).ravel()
        all_dst = (dst_base_ptrs[:, None] + dst_block_ids[None, :] * bsz_col).ravel()
        all_sizes = np.broadcast_to(bsz_col, (num_sub_tensors, num_pairs)).ravel().copy()

        batch_src = torch.from_numpy(all_src)
        batch_dst = torch.from_numpy(all_dst)
        batch_sizes = torch.from_numpy(all_sizes)

        start_event = self._get_event()
        end_event = self._get_event()

        if is_d2h:
            # Wait for model computation to finish before reading NPU data
            stream.wait_stream(torch.npu.current_stream())
        if transfers:
            # Ensure this transfer starts only after the previous one completes
            last_transfer = transfers[-1]
            stream.wait_event(last_transfer.end_event)

        with torch.npu.stream(stream):
            start_event.record(stream)
            if total > 0:
                direction = 0 if not is_d2h else 1
                torch.ops._C_ascend.swap_blocks_batch(batch_src, batch_dst, batch_sizes, direction)
            end_event.record(stream)

        transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=src_sub_block_count * self._total_bytes_per_block,
            )
        )

        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for transfers, transfer_type in [
            (self._d2h_transfers, ("NPU", "CPU")),
            (self._h2d_transfers, ("CPU", "NPU")),
        ]:
            while transfers and transfers[0].end_event.query():
                transfer = transfers.popleft()
                transfer_time = transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
                results.append(
                    TransferResult(
                        job_id=transfer.job_id,
                        success=True,
                        transfer_size=transfer.num_bytes,
                        transfer_time=transfer_time,
                        transfer_type=transfer_type,
                    )
                )
                self._recycle_event(transfer.start_event)
                self._recycle_event(transfer.end_event)
        return results

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait (block) until all specified transfer jobs are completed.
        """
        for transfers in (self._d2h_transfers, self._h2d_transfers):
            for transfer in transfers:
                if transfer.job_id in job_ids:
                    transfer.end_event.synchronize()
