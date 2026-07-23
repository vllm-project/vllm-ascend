import os

import numpy as np
import torch
import torch_npu
from memfabric_hybrid import offload
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import KVOffloadDecodeConfig


# non-c8 case, # [k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v]
# TODO remove KV_OFFLOAD_COLOCATE_DEBUG after PD disaggregate is done:
# the npu k_cache/v_cache entries only exist for colocate debug (prefill
# staging) and are deleted together with the prefill path.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5
# c8 case, TODO


_SUBSCRIBED_COMPUTE_STREAMS: set[object] = set()


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


class KVOffloadDecodeManager:
    """
    A manager responsible to the offload KV cache.
    It enlarge the availble memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """
    _CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024

    @staticmethod
    def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    @classmethod
    def _empty_aligned_cpu_tensor(
        cls,
        shape: list[int],
        dtype: torch.dtype,
        alignment: int = _CPU_CACHE_ALIGNMENT,
    ) -> torch.Tensor:
        num_elements = int(np.prod(shape))
        extra_elements = cdiv(alignment, torch.empty((), dtype=dtype).element_size())
        tensor = offload.empty([num_elements + extra_elements], dtype=dtype, pin_memory=True)
        return cls._align_memory(tensor, alignment)[:num_elements].view(shape)

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_offload_decode_config: KVOffloadDecodeConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.kv_offload_decode_config = kv_offload_decode_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = kv_offload_decode_config.topk_buffer_size
        self.topk = kv_offload_decode_config.topk

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.block_table_cpu = torch.zeros(
            [self.max_num_reqs, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.block_table_expanded_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self._pending_d2h: list[tuple[object, tuple[torch.Tensor, ...]]] = []
        self._graph_subscribed_streams: set[object] = set()
        self._npu_runtime = torch_npu.npu

        self._build_cpp()

        logger.info(
            f"KVOffloadManager start init CPU KV pool with {kv_offload_decode_config.dram_size_per_dp_GB} "
            "GB dram per dp group, it might be time consuming, please wait."
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.size = kv_offload_decode_config.dram_size_per_dp_GB * 1024 * 1024 * 1024
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        offload.initialize(config)
        self.tp_group.barrier()

    def _build_cpp(self):
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        npu_include_path = os.path.join(ascend_home, "include")
        npu_lib_path = os.path.join(ascend_home, "lib64")
        if not os.path.exists(npu_lib_path):
            npu_lib_path = os.path.join(ascend_home, "lib")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        os.environ['CXX'] = 'clang++'
        os.environ['CC'] = 'clang'
        abs_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(abs_path, "kv_offload_decode.cpp")
        logger.info(f'KV offload decode build cpp utils from src: {src_path}')
        self.kv_offload_decode_cpp = torch.utils.cpp_extension.load(
            name="kv_offload_decode",
            sources=[src_path],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-fopenmp",
                "-march=armv8.2-a+sve+fp16+bf16",
                "-fPIC",
                f"-I{npu_include_path}",
                f"-I{torch_npu_include}",
            ],
            extra_ldflags=[
                "-fopenmp",
                f"-L{npu_lib_path}",
                "-lascendcl",
                f"-L{torch_npu_lib_path}",
                "-ltorch_npu",
            ],
            verbose=True,
        )

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig | None,
    ) -> int:
        assert len(kv_cache_config.kv_cache_groups) == 1, "Hybrid KV is not supported."
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        return kv_cache_spec.block_size

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.offload_layer_names = [
            layer_name for layer_name in kv_caches
            if 'indexer' not in layer_name
        ]
        if not self.offload_layer_names:
            raise ValueError("KV offload decode did not find SFA KV cache layers.")

        # Under offload, the attention path (sfa_v1.py / device_op.py) gates the
        # C8 indexer read on the GLOBAL use_sparse_c8_indexer flag, not per-layer.
        # Mixed five/six-tuple layers would therefore route a non-C8 layer through
        # the quant indexer (or vice versa). Forbid it here so the global gate
        # stays sound; C8 must be all-or-nothing across sparse offload layers.
        tuple_lens = {
            len(self._as_cache_tuple(kv_caches[name])) for name in self.offload_layer_names
        }
        if len(tuple_lens) > 1:
            raise ValueError(
                "KV offload decode does not support mixed C8 / non-C8 layers: "
                f"found tuple lengths {sorted(tuple_lens)} "
                f"(five-tuple and six-tuple coexist). Under offload, C8 must be "
                f"enabled uniformly across all sparse layers."
            )

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id
            for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info(
            "KV offload decode registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("KV offload decode layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "KV offload decode layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        self._register_offload_layers(kv_caches)

        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"KV offload decode layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.tp_rank == 0:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "KV offload decode only support sfa(mla)"
        assert dtype == torch.bfloat16, "c8 not supported now"
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "KV offload decode topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H uses a separate descriptor set from the shared H2D buffers below.
        # Both prefill (colocate debug only, gated by KV_OFFLOAD_COLOCATE_DEBUG)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int64, device=device
        )
        self.d2h_dst_ptrs_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int64, device=device
        )
        self.d2h_lengths_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int32, device=device
        )
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        pages_per_row = self.topk_buffer_size // self.block_size
        self.current_slots_npu = torch.empty(
            (self.max_num_topk_rows, self.topk),
            dtype=torch.int32,
            device=device,
        )
        self.resident_block_table_npu = torch.arange(
            self.max_num_topk_rows * pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(self.max_num_topk_rows, pages_per_row)
        self.resident_query_lens_npu = torch.arange(
            1, self.max_num_topk_rows + 1, dtype=torch.int32, device=device
        )
        self.resident_seq_lens_npu = torch.full(
            (self.max_num_topk_rows,),
            self.topk_buffer_size,
            dtype=torch.int32,
            device=device,
        )

        # sparse_copy related addrs and buffers
        self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
        self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device='npu')
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device='npu')
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                gvas_k_tensor[layer_id] = self.k_caches_cpu[layer_id].data_ptr()
                gvas_v_tensor[layer_id] = self.v_caches_cpu[layer_id].data_ptr()
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())

        gvas_buffer_offset = 0
        gvas_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8 # 2: k+v, 8: int64
        addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
        addr_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8
        size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
        size_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 4 # 2: k+v, 4: int32
        num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
        num_tokens_buffer_size_bytes = 4 # 1 * int32
        sparse_copy_args_buffer_size_bytes = gvas_buffer_size_bytes + addr_buffer_size_bytes + size_buffer_size_bytes + num_tokens_buffer_size_bytes
        self.sparse_copy_args_buffer_cpu = torch.zeros([sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device='cpu', pin_memory=True)
        self.sparse_copy_args_buffer_npu = torch.zeros([sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device='npu')

        self.gvas_buffer_cpu = self.sparse_copy_args_buffer_cpu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
        self.addr_buffer_cpu = self.sparse_copy_args_buffer_cpu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
        self.size_buffer_cpu = self.sparse_copy_args_buffer_cpu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
        self.num_tokens_buffer_cpu = \
            self.sparse_copy_args_buffer_cpu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
        assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

        self.gvas_buffer_npu = self.sparse_copy_args_buffer_npu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
        self.addr_buffer_npu = self.sparse_copy_args_buffer_npu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
        self.size_buffer_npu = self.sparse_copy_args_buffer_npu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
        self.num_tokens_buffer_npu = \
            self.sparse_copy_args_buffer_npu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
        assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_npu.shape == torch.Size([1])

        # topk cache reuse related
        self.lru_workspace_threads = 8
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_token_to_req_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [torch.full(
            [self.max_num_topk_rows, self.topk_buffer_size],
            -1,
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_slots_cpu_list = [torch.arange(
            self.topk_buffer_size,
            dtype=torch.int32,
            device='cpu',
        ).view(1, -1).repeat(self.max_num_topk_rows, 1).pin_memory() for _ in range(self.num_layers)]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_miss_tokens_cpu_list = [torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_miss_slots_cpu_list = [torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device='cpu', pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [torch.full(
            [self.max_num_topk_rows],
            -1,
            dtype=torch.int64,
            device='cpu',
            pin_memory=True,
        ) for _  in range(self.num_layers)]
        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list]
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list]
        self.lru_miss_tokens_ptrs = [lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list]
        self.lru_miss_slots_ptrs = [lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()

    def offload_new_kv(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        v: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        # TODO remove KV_OFFLOAD_COLOCATE_DEBUG after PD disaggregate is done:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Main K/V is replicated across TP ranks; only TP0 owns and writes
            # the shared CPU pool whose GVA was broadcast during registration.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("KV offload decode TP0 CPU cache is not registered")
        if has_prefill and not envs_ascend.KV_OFFLOAD_COLOCATE_DEBUG:
            raise RuntimeError(
                "KV offload decode prefill offload requires "
                "KV_OFFLOAD_COLOCATE_DEBUG=1; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        if has_prefill:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("prefill offload requires NPU paged K/V caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device

        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        token_count = slots.numel()
        if token_count > self.max_num_tokens:
            raise ValueError(
                "KV offload decode rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = (
            k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        )
        num_v_slots = (
            v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        )
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                "KV offload decode CPU K/V pools have incompatible token capacities: "
                f"k={num_k_slots}, v={num_v_slots}"
            )
        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
            keep_sources = (k_cache_npu, v_cache_npu)
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v
            keep_sources = (k_rows, v_rows)

        dst_k = int(k_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(
            self.token_size_bytes_v
        )
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)

        result = offload.sparse_copy(
            self.d2h_src_ptrs_npu,
            self.d2h_dst_ptrs_npu,
            self.d2h_lengths_npu,
            self.d2h_size_npu,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")
        if capturing:
            # Capture records the sparse copy in stream order. A Python event
            # here would describe capture time rather than each graph replay.
            return
        done_event = torch_npu.npu.Event()
        done_event.record(torch_npu.npu.current_stream())
        self._pending_d2h.append((done_event, keep_sources))

    def _wait_for_pending_d2h(self) -> None:
        for event, _ in self._pending_d2h:
            event.synchronize()
        self._pending_d2h.clear()

    def _drain_graph_host_callbacks(self) -> None:
        """Wait until captured host callbacks finish using CPU LRU state."""
        for stream in getattr(self, "_graph_subscribed_streams", ()):
            event = self._npu_runtime.Event()
            event.record(stream)
            event.synchronize()

    def prepare_scheduler_step(self) -> None:
        """Drain transfers and callbacks before scheduler block reuse."""
        self._wait_for_pending_d2h()
        self._drain_graph_host_callbacks()

    def onload_topk_kv(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        block_table: torch.Tensor,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        token_to_req_npu: torch.Tensor | None = None,
        capturing: bool = False,
    ):
        # original onload kv logic
        layer_id = self._get_offload_layer_id(layer_name)
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "KV offload decode topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        if not capturing:
            self._wait_for_pending_d2h()
        if token_to_req_npu is not None:
            # spec decode case, expand block_table to actual num decode tokens.
            token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
            token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
            block_table_expanded = torch.index_select(
                block_table, 0, token_to_req_npu[:num_tokens].to(torch.int64))
            block_table_cpu = self.block_table_expanded_cpu[:num_tokens]
            block_table_cpu.copy_(block_table_expanded, non_blocking=capturing)
        else:
            block_table_cpu = self.block_table_cpu[:num_reqs]
            block_table_cpu.copy_(block_table, non_blocking=capturing)
        topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
        topk_indices_cpu.copy_(topk_indices_npu[:num_tokens], non_blocking=capturing)
        req_ids_cpu = self.lru_req_ids_cpu[:num_tokens]
        req_ids_cpu.copy_(req_ids_npu[:num_tokens], non_blocking=capturing)
        stable_prefix_lens_cpu = self.lru_stable_prefix_lens_cpu[:num_tokens]
        stable_prefix_lens_cpu.copy_(
            stable_prefix_lens_npu[:num_tokens],
            non_blocking=capturing,
        )

        args = (
            num_tokens,
            self.lru_miss_count_cpu_list[layer_id][:num_tokens],
            self.lru_miss_tokens_cpu_list[layer_id][:num_tokens],
            self.lru_miss_slots_cpu_list[layer_id][:num_tokens],
            self.lru_req_ids_ptr,
            self.lru_last_req_ids_ptrs[layer_id],
            self.lru_topk_indices_ptr,
            self.lru_stable_prefix_lens_ptr,
            self.lru_slot_to_token_ptrs[layer_id],
            self.lru_slots_ptrs[layer_id],
            self.lru_current_slots_ptr,
            self.lru_miss_count_ptrs[layer_id],
            self.lru_miss_tokens_ptrs[layer_id],
            self.lru_miss_slots_ptrs[layer_id],
            block_table_cpu,
            self.block_size,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
            self.gvas_k_bases[layer_id],
            self.gvas_v_bases[layer_id],
            self.addr_k_bases[layer_id],
            self.addr_v_bases[layer_id],
            self.lru_token_mark_workspace_ptr,
            self.lru_token_pos_workspace_ptr,
            self.lru_slot_workspace_ptr,
            self.lru_miss_position_workspace_ptr,
            self.lru_epochs_ptr,
            self.gvas_buffer_cpu,
            self.addr_buffer_cpu,
            self.size_buffer_cpu,
            self.num_tokens_buffer_cpu,
            layer_id,
        )

        if capturing:
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)
            self._graph_subscribed_streams.add(current_compute_stream)
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._onload_topk_kv_cpu,
                args,
            )
        else:
            self._onload_topk_kv_cpu(args)

        self.sparse_copy_args_buffer_npu.copy_(self.sparse_copy_args_buffer_cpu, non_blocking=capturing)
        offload.sparse_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)

    def _onload_topk_kv_cpu(self, args):
        # code that is incompatible with graph mode, compute here outside graph
        (
            num_reqs,
            miss_count,
            miss_tokens,
            miss_slots,
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
            layer_id,
        ) = args
        if self.tp_size > 1:
            # In graph mode this callback is ordered after TP0's captured D2H;
            # in eager mode onload_topk_kv waited for TP0's event above.
            self.tp_group.barrier()
        self.kv_offload_decode_cpp.lru_resident_compact(
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            num_reqs,
            self.topk,
            self.topk_buffer_size,
            self.max_model_len,
            self.lru_workspace_threads,
            self.lru_workspace_threads,
        )
        self.kv_offload_decode_cpp.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            self.topk_buffer_size,
            self.lru_workspace_threads,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )


_KV_OFFLOAD_DECODE_MANAGER: KVOffloadDecodeManager = None


def init_kv_offload_decode_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_offload_decode_config: KVOffloadDecodeConfig,
):
    global _KV_OFFLOAD_DECODE_MANAGER
    if _KV_OFFLOAD_DECODE_MANAGER is None:
        _KV_OFFLOAD_DECODE_MANAGER = KVOffloadDecodeManager(
            vllm_config,
            kv_cache_config,
            kv_offload_decode_config,
        )
    return _KV_OFFLOAD_DECODE_MANAGER


def get_kv_offload_decode_manager():
    assert _KV_OFFLOAD_DECODE_MANAGER is not None, "KV offload manager is not initialized."
    return _KV_OFFLOAD_DECODE_MANAGER
