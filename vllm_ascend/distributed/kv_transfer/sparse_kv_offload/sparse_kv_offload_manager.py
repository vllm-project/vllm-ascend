import contextlib
import ctypes
import os
import typing
from zlib import adler32

import numpy as np
import torch
import torch_npu

with contextlib.suppress(ImportError):
    # we should remove this after memfabric.offload is merged to master and available in ci machine.
    from memfabric_hybrid import offload  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_config import SparseKVOffloadConfig, get_ascend_config

# Main BF16 cache:
# [k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v].
# Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5


_SUBSCRIBED_COMPUTE_STREAMS: set[object] = set()


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


def get_host_device_memory_usage_ratio(kv_cache_specs: dict[str, KVCacheSpec]) -> float:
    page_size_bytes_host = 0
    page_size_bytes_device = 0
    for kv_cache_spec in kv_cache_specs.values():
        assert isinstance(kv_cache_spec, KVCacheSpec)
        if getattr(kv_cache_spec, "store_on_host", False):
            page_size_bytes_host += kv_cache_spec.page_size_bytes
        else:
            page_size_bytes_device += kv_cache_spec.page_size_bytes

    assert page_size_bytes_device > 0, "Case of no device kv cache is not considered."
    return page_size_bytes_host / page_size_bytes_device


def allocate_kv_offload_topk_buffer_pair(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens

    topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
    num_kv_heads = 1  # sparse kv offload only support sfa(mla) now.
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    max_num_topk_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    topk_buffer_k_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * k_dim * torch.bfloat16.itemsize
    topk_buffer_v_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * v_dim * torch.bfloat16.itemsize
    # NOTE make sure to allocate k+v together and split them after allocate.
    # Refer to the comment in empty_aligned_int8_cpu_tensors for reason.
    topk_buffer_raw = torch.empty([topk_buffer_k_size_bytes + topk_buffer_v_size_bytes], dtype=torch.int8, device="npu")
    topk_buffer_k = (
        topk_buffer_raw[:topk_buffer_k_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, k_dim])
    )
    topk_buffer_v = (
        topk_buffer_raw[topk_buffer_k_size_bytes : topk_buffer_k_size_bytes + topk_buffer_v_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, v_dim])
    )
    return (topk_buffer_k, topk_buffer_v)


def allocate_kv_offload_topk_profile_buffers(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_offload_layers = sum(bool(getattr(spec, "store_on_host", False)) for spec in kv_cache_spec.values())
    if num_offload_layers == 0:
        raise RuntimeError("Sparse KV offload profile did not find any host-resident SFA KV cache layers.")

    buffers = [
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config) for _ in range(num_offload_layers)
    ]
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for buffer_pair in buffers for tensor in buffer_pair)
    logger.info_once(
        "Sparse KV offload reserved %.2f GiB of KV offload topk buffers across %d layers for profile run.",
        buffer_bytes / (1 << 30),
        num_offload_layers,
    )
    return buffers


_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024


def empty_aligned_int8_cpu_tensors(
    sizes: list[int],
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> list[torch.Tensor]:
    """
    Allocate multiple int8 tensors with specified sizes,
    each aligned to specified alignment,
    and minimize the gap between each tensor's data_ptr.
    This is used for GLM-5.2 indexer reuse optimize:
    make sure that delta_k_cache_addrs and delta_v_cache_addrs
    between each two layers are the same,
    so we only need to add one delta_addr to the addr tensor of sparse_copy
    without need to mask k and v separately.
    Same reason that we allocate topk_buffer_k & topk_buffer_v together.
    """
    chunk_nums = [cdiv(size, alignment) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = offload.empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
    base_addr = raw_tensor.data_ptr()
    if base_addr % alignment:
        base_addr = (base_addr // alignment + 1) * alignment
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset : base_offset + size])
        base_offset += chunk_num * alignment
    return allocate_tensors


def allocate_kv_cache_tensors_for_sparse_kv_offload(
    k_tensor_size: int,
    v_tensor_size: int,
    alignment: int,
    tp_rank: int,
    keep_device_kv_cache: bool,
    npu_kv_cache_allocate_func: typing.Callable,
):
    if tp_rank == 0:
        [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors(
            [k_tensor_size, v_tensor_size],
            alignment,
        )
    else:
        k_tensor_cpu = None
        v_tensor_cpu = None

    if keep_device_kv_cache:
        k_tensor = npu_kv_cache_allocate_func(
            k_tensor_size,
            alignment,
        )
        v_tensor = npu_kv_cache_allocate_func(
            v_tensor_size,
            alignment,
        )
    else:
        k_tensor = None
        v_tensor = None

    return (k_tensor, v_tensor, k_tensor_cpu, v_tensor_cpu, k_tensor_size + v_tensor_size)


def reshape_kv_cache_tensors_for_sparse_kv_offload(
    raw_cache_tensors: tuple,
    current_kv_cache_spec: AttentionSpec,
    attn_backend: type[AttentionBackend],
    tp_rank: int,
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    raw_k_tensor, raw_v_tensor, raw_k_tensor_cpu, raw_v_tensor_cpu, sum_page_size_bytes = raw_cache_tensors
    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        current_kv_cache_spec.block_size,
        current_kv_cache_spec.num_kv_heads,
        current_kv_cache_spec.head_size,
    )
    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    k_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        k_dim,
    )
    v_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        v_dim,
    )
    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype

    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape) if raw_k_tensor is not None else None
    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape) if raw_v_tensor is not None else None

    if tp_rank == 0:
        k_cache_cpu = raw_k_tensor_cpu.view(k_cache_dtype).view(k_shape)
        v_cache_cpu = raw_v_tensor_cpu.view(v_cache_dtype).view(v_shape)
    else:
        k_cache_cpu = None
        v_cache_cpu = None

    topk_buffer_k, topk_buffer_v = allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
    return (k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v)


def update_sparse_kv_offload_metadata(
    num_tokens: int,
    num_reqs: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    req_ids: list[str],
    query_start_loc: CpuGpuBuffer,
    offload_req_ids_tensor: CpuGpuBuffer,
    offload_token_to_req: CpuGpuBuffer,
) -> None:
    """Populate per-request identity tensors for the Sparse KV offload LRU.

    Request ids are adler32 hashes of the scheduler request id,
    rows are reset whenever the occupying request changes.
    """
    offload_req_ids_tensor.np[:num_reqs_padded].fill(0)
    effective_num_reqs = min(num_reqs, len(req_ids))
    req_id_values = np.asarray(
        [
            adler32(req_id.encode("utf-8")) if isinstance(req_id, str) else row + 1
            for row, req_id in enumerate(req_ids[:effective_num_reqs])
        ],
        dtype=np.int64,
    )
    offload_req_ids_tensor.np[:effective_num_reqs] = req_id_values
    offload_req_ids_tensor.copy_to_gpu(num_reqs_padded)

    query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1]
    query_lens = np.diff(query_start_loc_cpu.numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        raise RuntimeError(
            "KV offload token_to_req metadata is shorter than the scheduled token batch: "
            f"metadata={token_to_req.shape[0]}, tokens={num_tokens}"
        )
    offload_token_to_req.np[:num_tokens] = token_to_req[:num_tokens]
    if num_tokens_padded > num_tokens:
        offload_token_to_req.np[num_tokens:num_tokens_padded].fill(0)
    offload_token_to_req.copy_to_gpu(num_tokens_padded)


def prepare_sparse_kv_offload_mtp_dummy_metadata(
    num_tokens: int,
    num_reqs: int,
    query_start_loc_cpu: torch.Tensor,
    req_ids_buffer: CpuGpuBuffer,
    token_to_req_buffer: CpuGpuBuffer,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not get_ascend_config().sparse_kv_offload_config.enabled:
        return None, None
    if req_ids_buffer is None or token_to_req_buffer is None:
        raise RuntimeError("Sparse KV offload metadata buffers are not initialized")

    query_lens = np.diff(query_start_loc_cpu[: num_reqs + 1].numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        token_to_req = np.pad(token_to_req, (0, num_tokens - token_to_req.shape[0]))

    req_ids_buffer.np[:num_reqs] = np.arange(1, num_reqs + 1, dtype=np.int64)
    req_ids_buffer.copy_to_gpu(num_reqs)
    token_to_req_buffer.np[:num_tokens] = token_to_req[:num_tokens]
    token_to_req_buffer.copy_to_gpu(num_tokens)
    return (
        req_ids_buffer.gpu[:num_reqs],
        token_to_req_buffer.gpu[:num_tokens],
    )


class SparseKVOffloadManager:
    """
    A manager responsible to the Sparse KV cache Offload.
    It enlarge the available memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sparse_kv_offload_config: SparseKVOffloadConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.sparse_kv_offload_config = sparse_kv_offload_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
        self.topk = sparse_kv_offload_config.topk

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
            device="cpu",
            pin_memory=True,
        )
        self.block_table_expanded_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._npu_runtime = torch_npu.npu

        self._build_cpp()

        logger.info(
            "SparseKVOffloadManager start init CPU KV pool with %s "
            "GB dram per dp group, it might be time consuming, please wait.",
            sparse_kv_offload_config.dram_size_per_dp_GB,
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.reserve_size = sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30)
        config.alloc_size = sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30) if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = offload.Scene.SHARED
        assert offload.initialize(config) == 0, "Sparse KV offload offload.initialize failed."
        self.tp_group.barrier()

    def _build_cpp(self):
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = os.environ.get("TORCH_EXTENSIONS_ALWAYS_BUILD", "0")
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        npu_include_path = os.path.join(ascend_home, "include")
        npu_lib_path = os.path.join(ascend_home, "lib64")
        if not os.path.exists(npu_lib_path):
            npu_lib_path = os.path.join(ascend_home, "lib")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = os.environ.get("TORCH_EXTENSIONS_ALWAYS_BUILD", "0")
        os.environ["CXX"] = "g++"
        os.environ["CC"] = "gcc"
        abs_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(abs_path, "sparse_kv_offload.cpp")
        logger.info_once(f"Sparse KV offload build cpp utils from src: {src_path}")
        self.sparse_kv_offload_cpp = torch.utils.cpp_extension.load(
            name="sparse_kv_offload",
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
        kv_cache_config: KVCacheConfig,
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
        self.offload_layer_names = [layer_name for layer_name in kv_caches if "indexer" not in layer_name]
        if not self.offload_layer_names:
            raise ValueError("Sparse KV offload did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info_once(
            "Sparse KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        self.mtp_layer_id = self.num_layers - 1 if self.num_layers != self.num_target_layers else -1
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("Sparse KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "Sparse KV offload layer is not registered, "
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
                    f"Sparse KV offload layer {layer_name}: expected tuple length "
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
        assert kv_head_num == 1, "Sparse KV offload only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "Sparse KV offload requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "Sparse KV offload topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H uses a separate descriptor set from the shared H2D buffers below.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_dst_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_lengths_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int32, device=device)
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(self.max_num_tokens, dtype=torch.int64, device=device)

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
        self.resident_query_lens_npu = torch.arange(1, self.max_num_topk_rows + 1, dtype=torch.int32, device=device)
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
        self.cpu_block_lens: list[tuple[int, int]] = []
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        cpu_block_lens_tensor = torch.zeros([self.num_layers, 2], dtype=torch.int64, device="npu")
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                cpu_block_lens_tensor[layer_id, 0] = (
                    k_cpu.numel() * k_cpu.element_size() // self.kv_cache_config.num_blocks
                )
                cpu_block_lens_tensor[layer_id, 1] = (
                    v_cpu.numel() * v_cpu.element_size() // self.kv_cache_config.num_blocks
                )
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())
            self.cpu_block_lens.append(
                (
                    cpu_block_lens_tensor[layer_id, 0].item(),
                    cpu_block_lens_tensor[layer_id, 1].item(),
                )
            )

        gvas_buffer_offset = 0
        gvas_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8  # 2: k+v, 8: int64
        addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
        addr_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8
        size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
        size_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 4  # 2: k+v, 4: int32
        num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
        num_tokens_buffer_size_bytes = 4  # 1 * int32
        sparse_copy_args_buffer_size_bytes = (
            gvas_buffer_size_bytes + addr_buffer_size_bytes + size_buffer_size_bytes + num_tokens_buffer_size_bytes
        )
        self.sparse_copy_args_buffer_cpu = torch.zeros(
            [sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device="cpu", pin_memory=True
        )
        self.sparse_copy_args_buffer_npu = torch.zeros(
            [sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device="npu"
        )

        self.gvas_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            gvas_buffer_offset : gvas_buffer_offset + gvas_buffer_size_bytes
        ].view(torch.int64)
        self.addr_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            addr_buffer_offset : addr_buffer_offset + addr_buffer_size_bytes
        ].view(torch.int64)
        self.size_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            size_buffer_offset : size_buffer_offset + size_buffer_size_bytes
        ].view(torch.int32)
        self.num_tokens_buffer_cpu = self.sparse_copy_args_buffer_cpu[
            num_tokens_buffer_offset : num_tokens_buffer_offset + num_tokens_buffer_size_bytes
        ].view(torch.int32)
        assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

        self.gvas_buffer_npu = self.sparse_copy_args_buffer_npu[
            gvas_buffer_offset : gvas_buffer_offset + gvas_buffer_size_bytes
        ].view(torch.int64)
        self.addr_buffer_npu = self.sparse_copy_args_buffer_npu[
            addr_buffer_offset : addr_buffer_offset + addr_buffer_size_bytes
        ].view(torch.int64)
        self.size_buffer_npu = self.sparse_copy_args_buffer_npu[
            size_buffer_offset : size_buffer_offset + size_buffer_size_bytes
        ].view(torch.int32)
        self.num_tokens_buffer_npu = self.sparse_copy_args_buffer_npu[
            num_tokens_buffer_offset : num_tokens_buffer_offset + num_tokens_buffer_size_bytes
        ].view(torch.int32)
        assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_npu.shape == torch.Size([1])

        # topk cache reuse related
        self.lru_workspace_threads = 8
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_to_req_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [
            torch.full(
                [self.max_num_topk_rows, self.topk_buffer_size],
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_slots_cpu_list = [
            torch.arange(
                self.topk_buffer_size,
                dtype=torch.int32,
                device="cpu",
            )
            .view(1, -1)
            .repeat(self.max_num_topk_rows, 1)
            .pin_memory()
            for _ in range(self.num_layers)
        ]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_tokens_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_slots_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device="cpu", pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        # v8: new-KV local fill. The host func writes (row, flat_dst) index
        # pairs for the current step's fresh tokens; captured ops then copy
        # them from the local K/V activations into the resident buffers.
        self._new_kv_max = 8  # max fresh tokens per forward (spec tokens * safety)
        # int32 index descriptors consumed by the fused FillLocalCopy kernel;
        # valid==0 entries are skipped inside the kernel (strict no-op).
        self.new_kv_rows_cpu = torch.zeros([self._new_kv_max], dtype=torch.int32, device="cpu", pin_memory=True)
        self.new_kv_dst_cpu = torch.zeros([self._new_kv_max], dtype=torch.int32, device="cpu", pin_memory=True)
        self.new_kv_valid_cpu = torch.zeros([self._new_kv_max], dtype=torch.int32, device="cpu", pin_memory=True)
        self._new_kv_k_rows = None
        self._new_kv_v_rows = None
        self._new_kv_pending: list[tuple[int, int]] = []  # [(row_token_id, slot)] filled per step
        self.lru_last_req_ids_cpu_list = [
            torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [
            lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list
        ]
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [
            lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list
        ]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [
            lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list
        ]
        self.lru_miss_tokens_ptrs = [
            lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list
        ]
        self.lru_miss_slots_ptrs = [
            lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list
        ]
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
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        # v10: publish the new-KV activation rows on EVERY rank before the
        # tp0-only early return below - the local resident-slot fill runs
        # per rank (each rank owns its own top-k buffers), while the pool
        # dump itself stays tp0-only (decode K/V is TP-replicated).
        if not has_prefill:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            self._new_kv_k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            self._new_kv_v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
        else:
            self._new_kv_k_rows = None
            self._new_kv_v_rows = None
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError(
                "Sparse KV offload prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
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
                "Sparse KV offload rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        num_v_slots = v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                f"Sparse KV offload CPU K/V pools have incompatible token capacities: k={num_k_slots}, v={num_v_slots}"
            )
        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
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

        dst_k = int(k_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(self.token_size_bytes_v)
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
        skip_topk: bool = False,
    ):
        layer_id = self._get_offload_layer_id(layer_name)
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "Sparse KV offload topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        if layer_id in [0, self.mtp_layer_id]:
            # metadata which are same across all layers, only compute/copy once in first layer.
            # last layer (mtp layer) may have different metadata, do not skip.
            if token_to_req_npu is not None:
                # spec decode case, expand block_table to actual num decode tokens.
                token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
                token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
                block_table_expanded = torch.index_select(block_table, 0, token_to_req_npu[:num_tokens].to(torch.int64))
                self.block_table_expanded_cpu[:num_tokens].copy_(block_table_expanded, non_blocking=capturing)
            else:
                self.block_table_cpu[:num_reqs].copy_(block_table, non_blocking=capturing)
            self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu[:num_tokens], non_blocking=capturing)
            self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(
                stable_prefix_lens_npu[:num_tokens],
                non_blocking=capturing,
            )

        if skip_topk:
            assert layer_id > 0, "No previous layer to reuse."
            gvas_offset = self.gvas_k_bases[layer_id] - self.gvas_k_bases[layer_id - 1]
            addr_offset = self.addr_k_bases[layer_id] - self.addr_k_bases[layer_id - 1]
            assert self.gvas_v_bases[layer_id] - self.gvas_v_bases[layer_id - 1] == gvas_offset, (
                "k/v gvas base delta mismatch."
            )
            assert self.addr_v_bases[layer_id] - self.addr_v_bases[layer_id - 1] == addr_offset, (
                "k/v addr base delta mismatch."
            )
            self.gvas_buffer_npu += gvas_offset
            self.addr_buffer_npu += addr_offset
        else:
            if token_to_req_npu is not None:
                block_table_cpu = self.block_table_expanded_cpu[:num_tokens]
            else:
                block_table_cpu = self.block_table_cpu[:num_reqs]
            topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
            topk_indices_cpu.copy_(topk_indices_npu[:num_tokens], non_blocking=capturing)

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
                # v8: stable-prefix pointer for the engine-descriptor skip.
                # Only enabled for decode-only forwards where the local-fill
                # path is active; prefill/mixed forwards must keep the full
                # engine fetch (their entire query span is "fresh" and the
                # fill path is disabled there).
                (self.lru_stable_prefix_lens_ptr if self._new_kv_k_rows is not None else 0),
            )

            if capturing:
                current_compute_stream = torch_npu.npu.current_stream()
                subscribed_compute_streams = get_subscribed_compute_streams()
                if current_compute_stream not in subscribed_compute_streams:
                    torch_npu.npu._subscribe_report(current_compute_stream)
                    subscribed_compute_streams.add(current_compute_stream)
                torch_npu.npu._launch_host_func(
                    current_compute_stream,
                    self._onload_topk_kv_cpu,
                    args,
                )
            else:
                self._onload_topk_kv_cpu(args)

            self.sparse_copy_args_buffer_npu.copy_(self.sparse_copy_args_buffer_cpu, non_blocking=capturing)

        # v11: with the new-KV local fill active (decode-only forwards),
        # the only pool fetches left are for tokens from EARLIER forwards,
        # whose tp0 dumps are fully ordered before this forward on tp0's
        # stream. One broadcast at the first leader layer of each forward
        # therefore flushes every in-flight dump that any later fetch in
        # this forward could race with. Prefill/mixed forwards (fill
        # disabled) keep the per-layer sync.
        fresh_fill_active = self._new_kv_k_rows is not None
        first_leader = (not skip_topk) and (layer_id == 0 or layer_id == self.mtp_layer_id)
        if self.tp_size > 1 and (not fresh_fill_active or first_leader):
            # Make sure that tp0 d2h is finished before other tp's h2d.
            # NOTE we can't use barrier since it can't be captured in graph.
            self.tp_group.broadcast(torch.empty([], dtype=torch.int8, device="npu"), src=0)
        offload.sparse_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )
        # v13: the resident-slot fill for the current step's fresh tokens runs
        # as one fused AscendC kernel per layer (FillLocalCopy, bisheng-built).
        # The kernel reads the small index descriptor arrays (rows/slots/valid)
        # from device memory at execution time, so under graph replay the same
        # captured launch follows the per-forward indices refreshed through
        # the pinned->NPU copies below. Bases are kernel arguments and are
        # fixed at capture time (static tensors under FULL_DECODE_ONLY graphs).
        # Invalid entries (valid==0) are skipped inside the kernel, so each
        # descriptor slot is either filled or strictly untouched.
        if self._new_kv_k_rows is not None and self._new_kv_v_rows is not None:
            if not hasattr(self, "_fill_kernel_ready"):
                dev = self.topk_buffers_k[0].device
                self.fill_rows_npu = torch.zeros([self._new_kv_max], dtype=torch.int32, device=dev)
                self.fill_dst_npu = torch.zeros([self._new_kv_max], dtype=torch.int32, device=dev)
                self.fill_valid_npu = torch.zeros([self._new_kv_max], dtype=torch.int32, device=dev)
                # static kernel params: [maxN, kRowBytes, vRowBytes]
                self.fill_params_npu = torch.tensor(
                    [self._new_kv_max, self.token_size_bytes_k, self.token_size_bytes_v], dtype=torch.int32, device=dev
                )
                fill_so = os.environ.get(
                    "VLLM_ASCEND_FILL_KERNEL_SO",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fill_kernel", "libfill_local_copy.so"),
                )
                self._fill_kernel = ctypes.CDLL(fill_so)
                self._fill_kernel.FillLocalCopy.argtypes = [ctypes.c_uint64] * 8 + [ctypes.c_void_p]
                self._fill_kernel_ready = True
                self._fill_copy_done = -1
            # Re-upload the indices on EVERY first-leader call: under graph
            # replay this executes via the host func (which runs per replay
            # and refreshes the pinned buffers), and eager warmup must also
            # refresh them per forward - a "layer_id only" guard would skip
            # the draft->verify transition because both forwards hit the
            # same leader layers with stale indices in between.
            first_fill = self._fill_copy_done != id(self._new_kv_k_rows)
            if first_fill:
                self._fill_copy_done = id(self._new_kv_k_rows)
                self.fill_rows_npu.copy_(self.new_kv_rows_cpu, non_blocking=True)
                self.fill_dst_npu.copy_(self.new_kv_dst_cpu, non_blocking=True)
                self.fill_valid_npu.copy_(self.new_kv_valid_cpu, non_blocking=True)
            stream = torch_npu.npu.current_stream().npu_stream
            self._fill_kernel.FillLocalCopy(
                self.fill_rows_npu.data_ptr(),
                self.fill_dst_npu.data_ptr(),
                self.fill_valid_npu.data_ptr(),
                self.fill_params_npu.data_ptr(),
                self._new_kv_k_rows.data_ptr(),
                self.topk_buffers_k[layer_id].data_ptr(),
                self._new_kv_v_rows.data_ptr(),
                self.topk_buffers_v[layer_id].data_ptr(),
                ctypes.c_void_p(stream),
            )

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)

    # ------------------------------------------------------------------
    # Draft-Indexer Prefetch (see design.md).
    #
    # Idea: the MTP draft model and the target verify step share the same
    # indexer top-k (`topk_indices_buffer`). The draft's top-k highly overlaps
    # the target verify top-k. We capture the draft top-k during the draft
    # forward; then, just before the target's ``onload_topk_kv`` LRU runs, we
    # identify the target tokens that (a) are predicted by the draft and (b)
    # are NOT already resident (i.e. would-be DRAM H2D misses). For each such
    # *new* token we H2D its K/V from the CPU DRAM pool straight into the
    # resident buffer and stamp ``slot_to_token`` so the LRU reports it as a
    # resident hit (skipping onload's DRAM H2D for it). This converts onload
    # misses into onload hits -> the onload_topk_kv hit rate goes up and the
    # residual DRAM H2D traffic drops.
    # ------------------------------------------------------------------

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
            stable_prefix_lens_ptr_arg,
        ) = args
        self.sparse_kv_offload_cpp.lru_resident_compact(
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
        self.sparse_kv_offload_cpp.compute_lru_resident_addrs(
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
            stable_prefix_lens_ptr_arg,
        )
        # v8: collect this forward's fresh-token fills (miss rows whose
        # token >= stable prefix). row = token - stable (query-order index
        # into the local K/V activation rows); dst = req*capacity + slot
        # (flat index into the layer's resident top-k buffers). The captured
        # copy ops below consume these via the pinned buffers.
        try:
            mc = miss_count.numpy()
            mt = miss_tokens.numpy()
            ms = miss_slots.numpy()
            stable = self.lru_stable_prefix_lens_cpu[:num_reqs].numpy()
            rows_out = self.new_kv_rows_cpu.numpy()
            dst_out = self.new_kv_dst_cpu.numpy()
            valid_out = self.new_kv_valid_cpu.numpy()
            rows_out[:] = 0
            dst_out[:] = 0  # unused entries carry valid==0; the kernel skips them
            valid_out[:] = 0
            w = 0
            for r in range(num_reqs):
                cnt = int(mc[r])
                s = int(stable[r])
                for i in range(cnt):
                    t = int(mt[r][i])
                    sl = int(ms[r][i])
                    if t >= s and w < len(rows_out):
                        rows_out[w] = t - s
                        dst_out[w] = r * self.topk_buffer_size + sl
                        valid_out[w] = 1
                        w += 1
        except Exception as _e:
            logger.warning("new-kv fill index collection failed: %s", _e)
        # Stats live inside the host func so they fire on graph replays too
        # (during replay the Python call site in onload_topk_kv never runs).
        # Wrapped: a stats failure must never break inference.


_SPARSE_KV_OFFLOAD_MANAGER: SparseKVOffloadManager | None = None


def init_sparse_kv_offload_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    global _SPARSE_KV_OFFLOAD_MANAGER
    if _SPARSE_KV_OFFLOAD_MANAGER is None:
        _SPARSE_KV_OFFLOAD_MANAGER = SparseKVOffloadManager(
            vllm_config,
            kv_cache_config,
            sparse_kv_offload_config,
        )
    return _SPARSE_KV_OFFLOAD_MANAGER


def get_sparse_kv_offload_manager():
    assert _SPARSE_KV_OFFLOAD_MANAGER is not None, "KV offload manager is not initialized."
    return _SPARSE_KV_OFFLOAD_MANAGER
