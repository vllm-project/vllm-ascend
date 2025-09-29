from dataclasses import dataclass
import math
import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
import threading
import numpy as np
import torch
from vllm.distributed.parallel_state import get_tp_group, get_dp_group
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig, AttentionSpec
from vllm.v1.utils import bind_kv_cache
from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner
from vllm.model_executor.models.utils import extract_layer_index
from omni.models.common.layers.attention.backend.attention import AscendAttentionState


logger = init_logger("vllm.v1.omni")

SIZE_BYTES_PER_LAYER = 16 * 1024 * 1024 * 1024  # 16 GB
NUM_DIE_PER_MACH = 16                           # assume A3
NZ_DIM = 16                                     # nz dim


class BaseOmniCache(ABC):
    MEMMAP_PATH = '/dev/shm/kv_cache.bin'

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ):
        self.tp_rank = get_tp_group().rank
        self.tp_local_rank = get_tp_group().local_rank
        self.tp_world_size = get_tp_group().world_size
        self.dp_local_rank = get_dp_group().local_rank
        self.dp_world_size = get_dp_group().world_size
        self.device = runner.device

        attn_spec: AttentionSpec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.num_layers = len(kv_cache_config.kv_cache_groups[0].layer_names)
        self.block_size = attn_spec.block_size
        self.num_kv_heads = attn_spec.num_kv_heads
        self.head_size = attn_spec.head_size
        self.dtype = attn_spec.dtype

        if self.num_kv_heads != 1:
            raise ValueError(f"Currently only support MLA models, where num_kv_heads must be 1, but got {self.num_kv_heads}.")

        # Calculate shape and number of blocks
        self.shape, self.num_blocks = self.calc_cache_shape()

        logger.warning(f"**BaseOmniCache**: {self.shape=}, {self.tp_world_size=}")
        shared_pinned_kv_tensor = self.initialize_shared_memory()
        self.host_cache = shared_pinned_kv_tensor

        # block_len_dtype: how many elements of `dtype` in one block
        # dp_offset: how many blocks to start from for current rank.
        self.block_len_dtype, self.dp_offset = self.calculate_kv_xsfer_params()
        self.device_cache = self.initialize_device_cache(kv_cache_config, runner)

    @abstractmethod
    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        pass

    @abstractmethod
    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ) -> Optional[Dict[str, Tuple[torch.Tensor]]]:
        pass

    def initialize_shared_memory(self) -> torch.Tensor:
        total_numel = math.prod(self.shape)
        itemsize = self.dtype.itemsize
        try:
            fd = os.open(BaseOmniCache.MEMMAP_PATH, os.O_CREAT | os.O_RDWR, 0o777)
            os.ftruncate(fd, total_numel * itemsize)
            # TODO: do not use BFloat16Storage here
            storage = torch.BFloat16Storage.from_file(
                BaseOmniCache.MEMMAP_PATH,
                shared=True,
                size=total_numel,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to open shared memory file {BaseOmniCache.MEMMAP_PATH}: {e}")

        shared_pinned_kv_tensor = torch.tensor([], dtype=self.dtype, pin_memory=True).set_(storage, 0, self.shape)
        return shared_pinned_kv_tensor

    def __getitem__(self, index: int):
        return self.host_cache

    @abstractmethod
    def synchronize_h2d(self) -> None:
        pass

    @abstractmethod
    def synchronize_d2h(self, key_states: torch.Tensor, value_states: torch.Tensor, slot_mapping: torch.Tensor, layer_idx: int, kv_event: torch.npu.Event) -> None:
        pass


class PrefillOmniCache(BaseOmniCache):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
    ):
        super().__init__(kv_cache_config, runner)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.arange = torch.arange(max_model_len, device='cpu')

        # buffer for offloading current batch's KV
        self.batch_buffer_cpu = torch.empty(
            max_num_batched_tokens,
            self.num_nz_heads,
            NZ_DIM,
            dtype=self.dtype,
            device='cpu',
            pin_memory=True,
        )
        logger.warning(f"**PrefillOmniCache**: CPU buffer shape is {self.batch_buffer_cpu.shape}")

        self.batch_slots_cpu = torch.zeros(
            max_num_batched_tokens,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.d2h_stream = torch.npu.Stream(device=self.device)
        self.copy_thread = None
        self.copy_lock = threading.Lock()

        # buffer for prefix/chunk
        # layout: TND, where T is max possible total KV tokens for a batch
        self.prefix_buffer_npu = torch.empty(
            max_num_seqs * max_model_len,
            self.num_kv_heads,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.h2d_stream = torch.npu.Stream(device=self.device)
        self.h2d_event = torch.npu.Event(blocking=False, enable_timing=False)

    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        self.tp_node_id = self.tp_rank // NUM_DIE_PER_MACH
        self.tp_nnodes = divide_or_raise(self.tp_world_size, NUM_DIE_PER_MACH)

        # For prefill, each node only needs to store a segment of KV cache.
        # For example, if head_size = 576, and TP is across 2 nodes, then
        # node0 stores [0, 288) and node1 stores [288, 576).
        self.local_head_size = divide_or_raise(self.head_size, self.tp_nnodes)
        shape, num_blocks = PrefillOmniCache.calc_cache_shape_for_prefill(
            num_layers=self.num_layers,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.local_head_size,
            dtype=self.dtype,
        )

        total_num_nz_heads = shape[2]                               # e.g., 36
        rank_num_nz_heads = total_num_nz_heads // NUM_DIE_PER_MACH  # 2
        remainder = total_num_nz_heads % NUM_DIE_PER_MACH           # 4

        # [3, 3, 3, 3, 2, 2, ...]
        get_rank_heads = lambda rank: rank_num_nz_heads + 1 if rank < remainder else rank_num_nz_heads
        starts = [0]
        for i in range(NUM_DIE_PER_MACH):
            starts.append(starts[-1] + get_rank_heads(i))
        assert starts[-1] == total_num_nz_heads, f"{total_num_nz_heads=}, while {starts=}."

        # how many nz heads the current rank is responsible to copy
        self.nz_heads_slc = slice(starts[self.tp_local_rank], starts[self.tp_local_rank+1])
        self.num_nz_heads = self.nz_heads_slc.stop - self.nz_heads_slc.start

        return shape, num_blocks

    @staticmethod
    def calc_cache_shape_for_prefill(
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[int, ...], int]:
        itemize = dtype.itemsize
        numel_per_layer = divide_or_raise(SIZE_BYTES_PER_LAYER, itemize)
        numel_per_block = block_size * num_kv_heads * head_size
        num_blocks_prefill = numel_per_layer // numel_per_block  # floor division

        p_shape = (
            num_blocks_prefill,
            num_layers,
            num_kv_heads * divide_or_raise(head_size, NZ_DIM),  # 36 if TP16
            block_size,
            NZ_DIM,
        )

        return p_shape, num_blocks_prefill

    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        block_len_dtype = math.prod(self.shape[1:])
        dp_offset = 0
        return block_len_dtype, dp_offset

    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ) -> Optional[Dict[str, Tuple[torch.Tensor]]]:
        return None

    def get_prefill_prefix_copy_meta(
        self,
        block_size,
        kv_lens: np.ndarray,
        query_lens_list: list[int],
        block_tables: np.ndarray,
        attn_state: AscendAttentionState,
    ) -> Optional["PrefixCopyMeta"]:
        if attn_state != AscendAttentionState.ChunkedPrefill:
            return None

        bsz = block_tables.shape[0]
        all_segs, q_slots, q_slot_start = [], [], 0
        last_block_idx, remainder = (kv_lens-1) // block_size, kv_lens % block_size
        assert np.all(remainder == 0), f"For APC, remainder should be zeros, but {kv_lens=}."

        for i in range(bsz):
            m = last_block_idx[i]

            if m < 0:
                segs = []
            elif m == 0:
                single_block = block_tables[i, 0].item()
                segs = [[single_block, single_block+1]]
            else:
                bt = block_tables[i, :m+1]
                # bt[idx] - bt[idx-1] != 1
                split_idx = np.where(np.diff(bt, n=1, axis=0) != 1)[0] + 1
                start_idx = np.r_[0, split_idx]
                end_idx = np.r_[split_idx - 1, m]

                # consecutive blocks [start_blocks[j], start_blocks[j]+1, ..., end_blocks[j]-1] are occupied
                start_blocks = bt[start_idx]
                end_blocks = bt[end_idx] + 1  # exclusive
                segs = np.stack([start_blocks, end_blocks], axis=1).tolist()  # (N_seg, 2)

            all_segs.append(segs)
            # q_slot_end += (kv_lens[i] + query_lens_list[i])
            # q_slots.append(self.arange[q_slot_end-query_lens_list[i]:q_slot_end])
            q_slot_start += kv_lens[i]
            q_slots.append(self.arange[:query_lens_list[i]] + q_slot_start)
            q_slot_start += query_lens_list[i]

        q_slots = torch.cat(q_slots).to(device=self.device, non_blocking=True)
        prefix_meta = PrefixCopyMeta(consecutive_blocks=all_segs,
                                     query_lens=query_lens_list,
                                     query_slots=q_slots)
        return prefix_meta

    def synchronize_h2d(
        self,
        # key_states: torch.Tensor,
        # value_states: torch.Tensor,
        prefix_meta: "PrefixCopyMeta",
        # cum_query_lens: list[int],
        layer_idx: int,
    ) -> None:
        """When prefix is hit, load the relevant KV from CPU to device buffer.
        key_states: (Tq, N, Dk)
        values_states: (Tq, N, Dv)
        """
        if prefix_meta is None or layer_idx >= self.num_layers:
            return
        # device = self.prefix_buffer_npu.device
        copy_start = 0
        with torch.npu.stream(self.h2d_stream):
            # kv_states = torch.cat([key_states, value_states], dim=-1)
            for i, (block_ranges, q_len) in enumerate(zip(prefix_meta.consecutive_blocks, prefix_meta.query_lens)):
                for start_block, end_block in block_ranges:
                    copy_len = (end_block - start_block) * self.block_size
                    self.prefix_buffer_npu[copy_start:copy_start+copy_len].view(end_block-start_block, self.block_size, -1, NZ_DIM).copy_(
                        self.host_cache[start_block:end_block, layer_idx].transpose(-2, -3),
                        non_blocking=True,
                            # .to(device=device, non_blocking=True)
                            # .transpose(-2, -3)
                    )
                    copy_start += copy_len
                # self.prefix_buffer_npu[copy_start:copy_start+q_len].copy_(kv_states[q_start:q_end])
                copy_start += q_len
            self.h2d_event.record(self.h2d_stream)

            # return self.prefix_buffer_npu[:copy_start, :, :512], self.prefix_buffer_npu[:copy_start, :, 512:]

    def synchronize_d2h(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int,
        kv_event: torch.npu.Event,
    ) -> None:
        if self.copy_thread and self.copy_thread.is_alive():
            # wait for the buffer->host operation called last time (i.e., in the last layer)
            self.copy_thread.join()

        num_tokens = key_states.shape[0]
        d2h_event = torch.npu.Event(blocking=False, enable_timing=False)
        with torch.npu.stream(self.d2h_stream):
            self.d2h_stream.wait_event(kv_event)  # wait for main stream to compute key_states and value_states
            kv = torch.cat([key_states, value_states], dim=-1)
            kv = (
                kv[..., self.tp_node_id*self.local_head_size : (self.tp_node_id+1)*self.local_head_size]
                .view(num_tokens, -1, NZ_DIM)  # (T, 36, 16)
            )
            self.batch_buffer_cpu[:num_tokens].copy_(kv[:, self.nz_heads_slc], non_blocking=True)
            self.batch_slots_cpu[:num_tokens].copy_(slot_mapping, non_blocking=True)
            d2h_event.record(self.d2h_stream)
        key_states.record_stream(self.d2h_stream)
        value_states.record_stream(self.d2h_stream)

        self.copy_thread = threading.Thread(
            target=self._update_host_cache_thread,
            args=(num_tokens, layer_idx, d2h_event),
            daemon=True,
        )
        self.copy_thread.start()

    def _update_host_cache_thread(self, num_tokens, layer_idx, event):
        torch.npu.set_device(self.device)  # On Ascend, the device context is thread-local.
        event.synchronize()  # block current thread until event finishes
        with self.copy_lock:
            cpu_slot_mapping = self.batch_slots_cpu[:num_tokens]
            self.host_cache[cpu_slot_mapping // self.block_size,
                            layer_idx,
                            self.nz_heads_slc,
                            cpu_slot_mapping % self.block_size] = \
                self.batch_buffer_cpu[:num_tokens]


class DecodeOmniCache(BaseOmniCache):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ):
        super().__init__(kv_cache_config, runner)
        self.decode_h2d_stream = torch.npu.Stream(device=self.device)

    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        return DecodeOmniCache.calc_cache_shape_for_decode(
            num_layers=self.num_layers,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.dtype,
        )

    @staticmethod
    def calc_cache_shape_for_decode(
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[int, ...], int]:
        itemize = dtype.itemsize
        numel_per_layer = divide_or_raise(SIZE_BYTES_PER_LAYER, itemize)
        numel_per_block = block_size * num_kv_heads * head_size
        # For decode, each DP rank has an independent KV cache manager for block allocation.
        # Thus, we should divide num_blocks by the number of managers on each node.
        num_blocks_decode = (numel_per_layer // numel_per_block) // NUM_DIE_PER_MACH

        # Here we 'reshape' the cache to (num_dies, num_blocks_per_die, ...) for efficient addressing.
        d_shape = (
            NUM_DIE_PER_MACH,
            num_blocks_decode,
            num_layers,
            num_kv_heads * divide_or_raise(head_size, NZ_DIM),
            block_size,
            NZ_DIM,
        )

        return d_shape, num_blocks_decode

    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        block_len_dtype = math.prod(self.shape[2:])
        dp_offset = self.dp_local_rank * self.num_blocks
        return block_len_dtype, dp_offset

    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ) -> Optional[Dict[str, Tuple[torch.Tensor]]]:
        kv_caches = {}
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                if tensor_config.size % kv_cache_spec.page_size_bytes != 0:
                    raise RuntimeError("tensor_config.size must be divisible by kv_cache_spec.page_size_bytes")
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = runner.attn_backends[i].get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size
                    )
                    kv_caches[layer_name] = runner.attn_backends[i].init_kv_cache_each_layer(
                        kv_cache_shape,
                        runner.dtype,
                        runner.device,
                        runner.model_config,
                        runner.enable_torchair_graph_mode
                    )
                else:
                    raise ValueError("Unknown KV cache spec type.")
        return kv_caches

    def synchronize_h2d(self, local_block_ids: List[List[int]]) -> None:
        layer_indices = {
            layer_name: extract_layer_index(layer_name)
            for layer_name in self.device_cache.keys()
        }

        dp_rank = self.dp_local_rank
        first_key_name = next(iter(self.device_cache))
        device = self.device_cache[first_key_name][0].device

        block_table = torch.LongTensor(local_block_ids[0]).to(device)

        # with torch.npu.stream(self.decode_h2d_stream):
        buffer = self.host_cache[dp_rank, local_block_ids[1]].to(device, non_blocking=True)

        for layer_name, layer_idx in layer_indices.items():
            layer_data = buffer[:, layer_idx]
            key = layer_data[:, :32].view(-1, 128, 1, 512)
            value = layer_data[:, 32:].view(-1, 128, 1, 64)

            self.device_cache[layer_name][0][block_table] = key
            self.device_cache[layer_name][1][block_table] = value

        # copy_done_event = torch.npu.Event()
        # copy_done_event.record(self.decode_h2d_stream)
        # return copy_done_event

    def synchronize_d2h(self, key_states: torch.Tensor, value_states: torch.Tensor, slot_mapping: torch.Tensor, layer_idx: int, kv_event: torch.npu.Event) -> None:
        raise NotImplementedError


def divide_or_raise(a: int, b: int):
    if a % b != 0:
        raise ValueError(f"Error! Number 'a' {a} is not divisible by number 'b' {b}.")
    return a // b


def create_omni_cache(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    runner: NPUModelRunner
) -> BaseOmniCache:
    """
    Factory function to create the appropriate BaseOmniCache instance based on the is_prefill flag.

    Args:
        kv_cache_config: Configuration for the KV cache
        vllm_config: The VllmConfig object
        runner: NPU model runner instance

    Returns:
        PrefillOmniCache or DecodeOmniCache instance based on the is_prefill flag
    """
    is_prefill = vllm_config.kv_transfer_config.kv_role != "kv_consumer"
    if is_prefill:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.scheduler_config.max_model_len
        omni_cache = PrefillOmniCache(kv_cache_config, runner, max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
        runner.kv_caches = None
        for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
            vllm_config.compilation_config.static_forward_context[layer_name].kv_cache = [None]
    else:
        omni_cache = DecodeOmniCache(kv_cache_config, runner)
        assert omni_cache.device_cache is not None
        bind_kv_cache(
            omni_cache.device_cache,
            vllm_config.compilation_config.static_forward_context,
            runner.kv_caches,
        )
    return omni_cache


@dataclass
class PrefixCopyMeta:
    consecutive_blocks: list[list[Tuple[int, int]]]
    """The starts and ends of consecutive full blocks."""

    query_lens: list[int]
    """Number of tokens per sample in current batch."""

    query_slots: torch.Tensor
    """The positions to store the KV of current tokens."""

    num_actual_tokens: int = None
    """Total number of tokens in current batch."""

    last_block_id: Optional[int] = None
    """The last block which might be partially filled. In APC, it must be None."""

    last_block_len: Optional[int] = None
    """Number of tokens filled in the last block."""

    def __post_init__(self):
        if len(self.consecutive_blocks) != len(self.query_lens):
            raise RuntimeError(f"Lengths mismatch! {len(self.consecutive_blocks)=}, while {len(self.query_lens)=}.")
        self.num_actual_tokens = sum(self.query_lens)
        total_copy_ops = sum([len(segs) for segs in self.consecutive_blocks])
        logger.warning(f"!!! Totally {total_copy_ops} copy operations will be executed. ***")
