"""KVTC offload worker: PCA projection with optional grouped quantization.

Supports one uniform BF16 cache group with TP and NeoX RoPE. Stream events
protect staging reuse and mark requests ready only after cache restoration.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, NamedTuple

import regex as re
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

from vllm_ascend.kvtc.copy_backend import (
    DIRECTION_D2H,
    DIRECTION_H2D,
    NPUDmaCopyBackend,
    build_params,
    copy_blocks,
)
from vllm_ascend.kvtc.kvtc_config import KVTCConfig
from vllm_ascend.kvtc.kvtc_ops import (
    kvtc_compress,
    kvtc_compress_rope,
    kvtc_decompress,
    kvtc_decompress_rope,
)
from vllm_ascend.kvtc.quant_layout import (
    PoolGeometry,
    checkpoint_layouts,
)
from vllm_ascend.kvtc.quantization import (
    QuantCodec,
    allocate_quant_buffers,
    buffer_prefix,
    require_npu_quantization,
)
from vllm_ascend.kvtc.rope import get_rotary_embedding
from vllm_ascend.utils import enable_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.v1.kv_cache_interface import KVCacheConfig


class _TransferPlan(NamedTuple):
    sink_src: list[int]
    sink_dst: list[int]
    comp_src: list[int]
    comp_dst: list[int]
    starts: list[int]


def _natural_layer_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _host_integer(value, name: str) -> int:
    # Never turn an NPU scalar into a Python integer in the transfer hot path.
    if isinstance(value, (torch.Tensor, bool)):
        raise ValueError(f"{name} must be a host integer, got {type(value)}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _host_ids(values, name: str) -> list[int]:
    if values is None or isinstance(values, torch.Tensor):
        raise ValueError(f"{name} must be a sequence of host integers")
    return [_host_integer(value, f"{name}[{i}]") for i, value in enumerate(values)]


class KVTCOffloadWorker(SimpleCPUOffloadWorker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        cpu_capacity_bytes: int,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, cpu_capacity_bytes)
        self._backend = NPUDmaCopyBackend()
        transfer = vllm_config.kv_transfer_config
        extra = (transfer.kv_connector_extra_config or {}) if transfer else {}
        config = KVTCConfig.from_extra(extra)
        self.kcr = config.kcr
        self.vcr = config.vcr
        self.uncompressed_ratio = config.uncompressed_ratio
        self.svd_path = config.svd_path
        self.kvtc_batch_blocks = config.kvtc_batch_blocks
        self.use_kvtc = config.compression_enabled
        self.quant_enabled = self.use_kvtc and config.kvtc_quantization
        self._codecs = {}
        self._store_index_refs = []
        self.k_quant_layout = self.v_quant_layout = None
        if cpu_capacity_bytes <= 0:
            raise ValueError("cpu_capacity_bytes must be positive")
        self._registered = False
        self._load_requests_by_event: dict[int, set[str]] = {}
        self._load_index_refs = []

    def _typed_cache(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Validate before viewing; never copy a cache into detached storage."""
        if tensor.device != self.device or tensor.dtype != torch.bfloat16:
            raise RuntimeError(f"{name}: expected BF16 on {self.device}, got {tensor.dtype} on {tensor.device}")
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"{name}: cache must be contiguous; shape={tuple(tensor.shape)}, "
                f"stride={tensor.stride()}. Calling contiguous() would detach "
                "the restore destination from the live attention cache."
            )
        shape = tuple(tensor.shape)
        if tensor.ndim == 3 and shape[2] == self.num_kv_heads * self.head_size:
            tensor = tensor.view(shape[0], shape[1], self.num_kv_heads, self.head_size)
            shape = tuple(tensor.shape)
        group = self.kv_cache_config.kv_cache_groups[0]
        block_size = group.kv_cache_spec.block_size
        if (
            tensor.ndim != 4
            or shape[1] <= 0
            or block_size % shape[1]
            or shape[0] * shape[1] != self.num_gpu_blocks * block_size
            or shape[2:] != (self.num_kv_heads, self.head_size)
        ):
            raise RuntimeError(
                f"{name}: expected contiguous cache for {self.num_gpu_blocks} blocks of {block_size} tokens "
                f"with H={self.num_kv_heads}, D={self.head_size}; got {shape}"
            )
        # V2 can split a scheduler block into adjacent attention-kernel blocks.
        return tensor.view(self.num_gpu_blocks, block_size, self.num_kv_heads, self.head_size)

    def _layer_pair(self, name: str, value) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        if isinstance(value, (tuple, list)):
            if len(value) != 2 or not all(isinstance(t, torch.Tensor) for t in value):
                raise RuntimeError(f"{name}: expected an explicit (K, V) pair")
            key, val = value
        elif (
            isinstance(value, torch.Tensor)
            and value.ndim in (4, 5)
            and value.shape[0] == 2
            and value.shape[1] == self.num_gpu_blocks
        ):
            # Rank, not shape[0] >= G, distinguishes packed KV even for G=1/2.
            key, val = value[0], value[1]
        else:
            raise RuntimeError(
                f"{name}: expected (K,V) or packed [2,G,B,H,D]; "
                "a single K/V tensor or blocks-first interleaved KV is ambiguous"
            )
        return self._typed_cache(f"{name}.K", key), self._typed_cache(f"{name}.V", val)

    def _register_cache_views(self, kv_caches: dict) -> None:
        self.k_op_caches, self.v_op_caches = [], []
        self.gpu_kv_caches = {}
        regions: dict[tuple, list[tuple[int, int, str]]] = {}
        self.layer_names = sorted(kv_caches, key=_natural_layer_key)
        for layer in self.layer_names:
            key, val = self._layer_pair(layer, kv_caches[layer])
            for suffix, tensor, op_list in (
                (".0", key, self.k_op_caches),
                (".1", val, self.v_op_caches),
            ):
                name = layer + suffix
                storage = tensor.untyped_storage()
                start = tensor.storage_offset() * tensor.element_size()
                end = start + tensor.numel() * tensor.element_size()
                storage_key = (tensor.device, storage.data_ptr())
                if end > storage.nbytes():
                    raise RuntimeError(f"{name}: cache view exceeds its storage")
                for other_start, other_end, other_name in regions.get(storage_key, []):
                    if start < other_end and other_start < end:
                        raise RuntimeError(
                            f"{name} overlaps {other_name}. Shared logical KV "
                            "layers need an explicit SVD/restore mapping."
                        )
                regions.setdefault(storage_key, []).append((start, end, name))
                op_list.append(tensor)
                # Byte views preserve each tensor's offset. No storage-pointer
                # deduplication: disjoint K/V slices both have to be offloaded.
                self.gpu_kv_caches[name] = tensor.view(torch.int8).view(self.num_gpu_blocks, -1)
        self.layer_num = len(self.k_op_caches)
        self.block_size = self.k_op_caches[0].shape[1]
        expected_shape = (
            self.num_gpu_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
        )
        if any(tuple(t.shape) != expected_shape for t in self.k_op_caches + self.v_op_caches):
            raise RuntimeError("All KVTC layers must have the same [G,B,H,D]")
        self.k = self.layer_num * self.num_kv_heads * self.head_size
        # The scheduler derives raw bytes from KVCacheConfig. Refuse
        # padded, heterogeneous or shared configurations that disagree with
        # the logical tensors; silently changing the pool boundary is unsafe.
        group = self.kv_cache_config.kv_cache_groups[0]
        configured_bytes = group.kv_cache_spec.page_size_bytes * len(group.layer_names) * self.num_gpu_blocks
        self.raw_block_bytes = 2 * self.block_size * self.k * 2
        if configured_bytes != self.raw_block_bytes * self.num_gpu_blocks:
            raise RuntimeError(
                "KVCacheConfig byte size differs from the registered K/V views: "
                f"config={configured_bytes}, views="
                f"{self.raw_block_bytes * self.num_gpu_blocks}. "
                "A shared/padded/hybrid layout needs matching manager support."
            )
        groups = self.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or groups[0].kv_cache_spec.block_size != self.block_size:
            raise RuntimeError("KVTC currently requires one uniform cache block group")

    def _prepare_rope(self, rotary_emb: RotaryEmbedding | None) -> None:
        if rotary_emb is None:
            raise RuntimeError("KVTC key compression requires rotary_emb")
        self.rotary_emb = rotary_emb
        if (
            int(rotary_emb.head_size) != self.head_size
            or int(rotary_emb.rotary_dim) != self.head_size
            or not bool(rotary_emb.is_neox_style)
        ):
            raise RuntimeError("KVTC requires full-head NeoX RoPE matching the KV head size")
        table = rotary_emb.cos_sin_cache
        if table.ndim != 2 or table.shape[0] <= 0 or table.shape[1] != self.head_size:
            raise RuntimeError("RoPE cache must be [max_position, head_size]")
        self.fwd_cos_sin_cache = table.to(device=self.device, dtype=torch.float32).contiguous()
        self.inv_cos_sin_cache = self.fwd_cos_sin_cache.clone()
        self.inv_cos_sin_cache[:, self.head_size // 2 :].mul_(-1.0)
        self.rope_rows = table.shape[0]

    def _load_projection(
        self,
        params: dict,
        kind: str,
        worker_key: str,
        dim: int,
        padded_dim: int | None = None,
    ) -> None:
        if kind not in params or worker_key not in params[kind]:
            raise RuntimeError(f"SVD file has no {kind}/{worker_key}; raw fallback is unsafe")
        item = params[kind][worker_key]
        mu, basis = item.get("mu"), item.get("basis")
        if not isinstance(mu, torch.Tensor) or mu.ndim != 1 or mu.numel() != self.k or not torch.is_floating_point(mu):
            raise RuntimeError(f"{kind}/{worker_key}: mu must be floating [{self.k}]")
        if (
            not isinstance(basis, torch.Tensor)
            or basis.ndim != 2
            or basis.shape[0] != self.k
            or basis.shape[1] < dim
            or not torch.is_floating_point(basis)
        ):
            raise RuntimeError(
                f"{kind}/{worker_key}: basis must have {self.k} rows "
                f"and at least {dim} columns; refusing silent truncation"
            )
        recorded_layers = item.get("layer_names")
        if recorded_layers is not None and list(recorded_layers) != self.layer_names:
            raise RuntimeError(f"{kind}/{worker_key}: SVD layer order differs from KV cache order")
        prefix = "kvtc_k" if kind == "keys" else "kvtc_v"
        mu_fp32 = mu.to(device=self.device, dtype=torch.float32).contiguous()
        basis_fp32 = basis[:, :dim].to(device=self.device, dtype=torch.float32).contiguous()
        padded_dim = dim if padded_dim is None else padded_dim
        if padded_dim < dim:
            raise ValueError("GEMM padding cannot truncate a calibrated feature")
        proj = torch.zeros((self.k, padded_dim), dtype=torch.bfloat16, device=self.device)
        proj[:, :dim].copy_(basis_fp32)
        # Bias must use the same quantized projection that addmm consumes.
        basis_fp32 = proj.float()
        bias = (-torch.matmul(mu_fp32, basis_fp32)).to(torch.bfloat16).contiguous()
        setattr(self, prefix + "_mu_bf16", mu_fp32.to(torch.bfloat16).contiguous())
        setattr(self, prefix + "_proj", proj)
        setattr(self, prefix + "_deproj", proj.transpose(0, 1).contiguous())
        setattr(self, prefix + "_projected_bias_bf16", bias)

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | tuple | list],
    ) -> None:
        if self._registered:
            raise RuntimeError("KVTC caches are already registered; recreate the worker")
        if not kv_caches:
            logger.warning("[KVTC WORKER] No KV caches to offload")
            return
        if self.kv_cache_config is None or self.kv_cache_config.num_blocks <= 0:
            raise RuntimeError("KVTC requires a nonempty KVCacheConfig")
        if getattr(self.vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1:
            raise RuntimeError("This SVD layout supports TP workers with pipeline_parallel_size=1")
        first = next(iter(kv_caches.values()))
        if isinstance(first, (tuple, list)):
            if not first:
                raise RuntimeError("Empty per-layer KV cache entry")
            first = first[0]
        if not isinstance(first, torch.Tensor):
            raise RuntimeError("KV cache entries must contain tensors")
        self.device = first.device
        torch.npu.set_device(self.device)
        self.num_gpu_blocks = self.kv_cache_config.num_blocks
        self.num_kv_heads = self.vllm_config.model_config.get_num_kv_heads(self.vllm_config.parallel_config)
        self.head_size = self.vllm_config.model_config.get_head_size()
        self.dtype = torch.bfloat16
        if self.vllm_config.model_config.dtype != self.dtype:
            raise RuntimeError("The scheduler and worker require model/cache BF16")
        # Match the current prepare/restore kernels' supported geometry.
        if self.head_size <= 0 or self.head_size > 128 or self.head_size % 16:
            raise RuntimeError("Current KVTC kernels require D in 16..128, divisible by 16")
        if not 1 <= self.num_kv_heads <= 255:
            raise RuntimeError("Current KVTC RoPE kernels require 1..255 local KV heads")
        self._register_cache_views(kv_caches)
        self.load_stream = torch.npu.Stream()
        self.store_stream = torch.npu.Stream()
        # Completion remains on load_stream; only compressed H2D uses this
        # additional stream. Raw backend copies still use load_stream.
        self._load_dma_stream = torch.npu.Stream()
        k_dim = v_dim = 0
        layouts = None
        params = None
        if self.use_kvtc:
            params = torch.load(self.svd_path, map_location="cpu", weights_only=True)
        if self.quant_enabled:
            require_npu_quantization()
            layouts = checkpoint_layouts(
                params,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                k=self.k,
                block_size=self.block_size,
                kcr=self.kcr,
                vcr=self.vcr,
            )
        self._layout = PoolGeometry.calculate(
            capacity=self.cpu_capacity_bytes,
            raw_block_bytes=self.raw_block_bytes,
            block_size=self.block_size,
            kcr=self.kcr,
            vcr=self.vcr,
            raw_ratio=self.uncompressed_ratio,
            quantized=self.quant_enabled,
            layouts=layouts,
        )
        self.max_comp_blocks = self._layout.max_comp_blocks
        self.max_sink_blocks = self._layout.max_sink_blocks
        if self.use_kvtc:
            enable_custom_op()
            for name in (
                "kvtc_prepare",
                "kvtc_prepare_rope",
                "kvtc_restore",
                "kvtc_restore_rope",
            ):
                if not hasattr(torch.ops._C_ascend, name):
                    raise RuntimeError(f"Missing custom operator _C_ascend.{name}")
            with set_current_vllm_config(self.vllm_config):
                rotary_emb = get_rotary_embedding(self.vllm_config.model_config.hf_text_config, dtype=self.dtype)
            self._prepare_rope(rotary_emb)
            tp_rank = get_tensor_model_parallel_rank()
            worker_key = f"tp_{tp_rank}_pp_0"
            logger.info("[KVTC WORKER] Loading SVD %s, key=%s", self.svd_path, worker_key)
            if self.quant_enabled:
                self.k_quant_layout = layouts[("keys", worker_key)]
                self.v_quant_layout = layouts[("values", worker_key)]
                k_features, v_features = (
                    self.k_quant_layout.feature_count,
                    self.v_quant_layout.feature_count,
                )
                k_dim, v_dim = (
                    self.k_quant_layout.gemm_dim,
                    self.v_quant_layout.gemm_dim,
                )
                self._load_projection(params, "keys", worker_key, k_features, k_dim)
                self._load_projection(params, "values", worker_key, v_features, v_dim)
                local_bytes = self.k_quant_layout.block_bytes + self.v_quant_layout.block_bytes
                logger.info(
                    "[KVTC QUANT] rank=%d retained=(%d,%d) GEMM padded=(%d,%d) "
                    "local bytes/block=%d TP quota=%d effective=%.3fx buckets=(%d,%d)",
                    tp_rank,
                    k_features,
                    v_features,
                    k_dim,
                    v_dim,
                    local_bytes,
                    self._layout.compressed_block_bytes,
                    self.raw_block_bytes / local_bytes,
                    len(self.k_quant_layout.buckets),
                    len(self.v_quant_layout.buckets),
                )
            else:
                k_dim, v_dim = (
                    int(self.k // self.kcr) // 16 * 16,
                    int(self.k // self.vcr) // 16 * 16,
                )
                self._load_projection(params, "keys", worker_key, k_dim)
                self._load_projection(params, "values", worker_key, v_dim)
            del params
        else:
            # Only valid with a manager configured for raw offload too.
            # Missing SVD data never silently selects this branch.
            self.rotary_emb = None
            self.max_comp_blocks = 0
            self.max_sink_blocks = self.cpu_capacity_bytes // self.raw_block_bytes
        self.num_cpu_blocks = self.max_sink_blocks + self.max_comp_blocks
        self.shift_threshold = self.max_comp_blocks
        if self.num_cpu_blocks <= 0:
            raise RuntimeError("CPU budget cannot hold one offload block")
        logger.info(
            "[KVTC WORKER] L=%d G=%d B=%d H=%d D=%d K=%d batch=%d; CPU compressed=%d sink=%d shift=%d",
            self.layer_num,
            self.num_gpu_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            self.k,
            self.kvtc_batch_blocks,
            self.max_comp_blocks,
            self.max_sink_blocks,
            self.shift_threshold,
        )
        logger.debug("[KVTC WORKER] SVD/cache layer order: %s", self.layer_names)
        pin_memory = is_pin_memory_available()
        self.cpu_sink_pool = {
            name: torch.zeros(
                (self.max_sink_blocks, t.shape[1]),
                dtype=t.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for name, t in self.gpu_kv_caches.items()
        }
        self.cpu_kv_caches = self.cpu_sink_pool
        # Some backends cannot build DMA descriptors for zero-sized pools.
        self._sink_store_params = self._sink_load_params = None
        if self.max_sink_blocks:
            self._sink_store_params = build_params(self.gpu_kv_caches, self.cpu_sink_pool, DIRECTION_D2H)
            self._sink_load_params = build_params(self.cpu_sink_pool, self.gpu_kv_caches, DIRECTION_H2D)
        if self.use_kvtc and self.max_comp_blocks:
            self._comp_cpu = {}
            self._comp_store = {}
            self._comp_load = {}
            for kind, dim in (("k", k_dim), ("v", v_dim)):
                if self.quant_enabled:
                    layout = getattr(self, kind + "_quant_layout")
                    cpu = allocate_quant_buffers(
                        layout,
                        self.max_comp_blocks,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    store = allocate_quant_buffers(layout, self.kvtc_batch_blocks, device=self.device)
                    load = allocate_quant_buffers(layout, self.kvtc_batch_blocks, device=self.device)
                    latent = torch.empty(
                        (self.kvtc_batch_blocks, self.block_size, dim),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    setattr(self, kind + "_dequant_staging", latent)
                    codec = QuantCodec(layout, self.device)
                    codec.validate_native_unpack()
                    self._codecs[kind] = codec
                    for target, tensors in (
                        (self._comp_cpu, cpu),
                        (self._comp_store, store),
                        (self._comp_load, load),
                    ):
                        target.update({kind + "_" + name: t for name, t in tensors.items()})
                else:
                    cpu = torch.empty(
                        (self.max_comp_blocks, self.block_size, dim),
                        dtype=self.dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    store = torch.empty(
                        (self.kvtc_batch_blocks, self.block_size, dim),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    load = torch.empty_like(store)
                    (
                        self._comp_cpu[kind],
                        self._comp_store[kind],
                        self._comp_load[kind],
                    ) = cpu, store, load
                setattr(self, "cpu_compressed_" + kind, cpu)
                setattr(self, kind + "_store_staging", store)
                setattr(self, kind + "_load_staging", load)
            self._comp_store_params = build_params(self._comp_store, self._comp_cpu, DIRECTION_D2H)
            self._comp_load_params = build_params(self._comp_cpu, self._comp_load, DIRECTION_H2D)
            self._load_slot_pools = [self._comp_load]
            self._load_slot_pools.append({name: torch.empty_like(t) for name, t in self._comp_load.items()})
            self._load_slot_params = [self._comp_load_params]
            for pool in self._load_slot_pools[1:]:
                self._load_slot_params.append(build_params(self._comp_cpu, pool, DIRECTION_H2D))
            # Persistent across metadata calls: DMA must not overwrite a slot
            # still consumed by the preceding call's reconstruction kernels.
            self._load_slot_done = [None] * len(self._load_slot_pools)
            logger.info(
                "[KVTC] mode=%s batch_blocks=%d load_slots=%d staging_bytes=%d",
                "pca+quant" if self.quant_enabled else "pca",
                self.kvtc_batch_blocks,
                len(self._load_slot_pools),
                sum(t.numel() * t.element_size() for pool in self._load_slot_pools for t in pool.values()),
            )
            allocated_bytes = sum(t.numel() * t.element_size() for t in self._comp_cpu.values())
            allocated_bytes += sum(t.numel() * t.element_size() for t in self.cpu_sink_pool.values())
            if allocated_bytes > self.cpu_capacity_bytes:
                raise RuntimeError("Actual CPU payload allocation exceeds budget")
        if self.max_sink_blocks:
            backend_gpu, backend_cpu = self.gpu_kv_caches, self.cpu_sink_pool
        else:
            # Initialize from a nonempty pool when there are no raw slots.
            # Every transfer below supplies its own params, so these defaults
            # are not used for H2D into the separate load staging buffer.
            backend_gpu = self._comp_store
            backend_cpu = self._comp_cpu
        self._backend.init(backend_gpu, backend_cpu, self.device, self.load_stream, self.store_stream)
        # One-time model state preparation; no device-wide sync per transfer.
        torch.npu.synchronize()
        self._registered = True

    def _transfer_plan(self, metadata, *, is_store: bool) -> _TransferPlan:
        if getattr(metadata, "kvtc_layout_fingerprint", None) != self._layout.fingerprint:
            raise RuntimeError("KVTC manager/worker layouts differ; install matching files and restart all ranks")
        prefix = "store" if is_store else "load"
        src = _host_ids(
            getattr(metadata, prefix + ("_gpu_blocks" if is_store else "_cpu_blocks")),
            prefix + "_src",
        )
        dst = _host_ids(
            getattr(metadata, prefix + ("_cpu_blocks" if is_store else "_gpu_blocks")),
            prefix + "_dst",
        )
        if len(src) != len(dst):
            raise ValueError(f"{prefix}: source/destination lengths differ")
        if src and _host_integer(getattr(metadata, prefix + "_event"), prefix + "_event") < 0:
            raise ValueError(f"{prefix}: nonempty transfer requires a completion event")
        gpu_ids, cpu_ids = (src, dst) if is_store else (dst, src)
        for name, ids, limit in (
            ("GPU", gpu_ids, self.num_gpu_blocks),
            ("CPU", cpu_ids, self.num_cpu_blocks),
        ):
            if any(i < 0 or i >= limit for i in ids):
                raise ValueError(f"{prefix}: {name} block ID outside [0,{limit})")
        compressed = self.use_kvtc and any(i < self.shift_threshold for i in cpu_ids)
        starts = []
        if compressed:
            starts = _host_ids(
                getattr(metadata, prefix + "_block_token_starts", None),
                prefix + "_block_token_starts",
            )
            if len(starts) != len(src):
                raise ValueError(f"{prefix}: token_starts must align with the full block list")
        plan = _TransferPlan([], [], [], [], [])
        seen_destinations: dict[int, tuple[int, int | None]] = {}
        for idx, (source, destination) in enumerate(zip(src, dst)):
            cpu_id = destination if is_store else source
            is_compressed = self.use_kvtc and cpu_id < self.shift_threshold
            position = starts[idx] if is_compressed else None
            if is_compressed and (position < 0 or position + self.block_size > self.rope_rows):
                raise ValueError(
                    f"{prefix}: token range [{position},{position + self.block_size}) "
                    f"outside RoPE table [0,{self.rope_rows})"
                )
            assignment = (source, position)
            if destination in seen_destinations:
                if seen_destinations[destination] != assignment:
                    raise ValueError(f"{prefix}: conflicting writes to destination {destination}")
                # Two requests can reference the same cached prefix. Schedule
                # identical copies once instead of concurrent duplicate writes.
                continue
            seen_destinations[destination] = assignment
            if not is_compressed:
                sink_id = cpu_id - self.shift_threshold
                plan.sink_src.append(source if is_store else sink_id)
                plan.sink_dst.append(sink_id if is_store else destination)
            else:
                plan.comp_src.append(source)
                plan.comp_dst.append(destination)
                plan.starts.append(position)
        return plan

    def bind_connector_metadata(self, metadata) -> None:
        if not self._registered:
            raise RuntimeError("KVTC worker received metadata before cache registration")
        # Validate both directions before either transfer can be submitted.
        self._load_plan = self._transfer_plan(metadata, is_store=False)
        self._store_plan = self._transfer_plan(metadata, is_store=True)
        super().bind_connector_metadata(metadata)
        self._load_submitted = False
        self._store_submitted = False
        # Completion may be observed after clear_connector_metadata().
        for idx in self._pending_load_event_indices:
            requests = metadata.load_event_to_reqs.get(idx)
            if requests:
                self._load_requests_by_event.setdefault(idx, set()).update(requests)

    def start_load_kv(self) -> None:
        self._backend.raise_if_failed()
        metadata = self._connector_metadata
        if metadata is not None and not self._load_submitted:
            if metadata.load_event >= 0:
                self._swap_in_blocks(metadata, self._load_plan)
            self._load_submitted = True

    def wait_for_save(self) -> None:
        self._backend.raise_if_failed()
        metadata = self._connector_metadata
        if metadata is not None and not self._store_submitted:
            if metadata.store_event >= 0:
                self._swap_out_blocks(metadata, self._store_plan)
            self._store_submitted = True

    def get_finished(self, finished_req_ids: set[str]) -> tuple[None, set[str] | None]:
        self._backend.raise_if_failed()
        finished_recving: set[str] = set()
        if self._pending_load_event_indices:
            watermark = self._poll_stream_events(is_store=False)
            for idx in tuple(self._pending_load_event_indices):
                if idx <= watermark:
                    self._pending_load_event_indices.remove(idx)
                    finished_recving.update(self._load_requests_by_event.pop(idx, ()))
        if self._pending_store_event_indices:
            watermark = self._poll_stream_events(is_store=True)
            for idx in tuple(self._pending_store_event_indices):
                if idx <= watermark:
                    self._pending_store_event_indices.remove(idx)
                    self._completed_store_events[idx] = 1
        return None, finished_recving or None

    @torch.inference_mode()
    def _swap_out_blocks(self, metadata, plan: _TransferPlan | None = None) -> None:
        plan = plan if plan is not None else self._transfer_plan(metadata, is_store=True)
        logger.debug(
            "[KVTC WORKER] STORE raw=%d compressed=%d",
            len(plan.sink_src),
            len(plan.comp_src),
        )
        # Order cache reads after previously submitted model work on the caller
        # stream. This is a stream dependency, not a device-wide host wait.
        producer = torch.npu.current_stream()
        if producer != self.store_stream:
            self.store_stream.wait_stream(producer)
        with torch.npu.stream(self.store_stream):
            self._store_index_refs = [item for item in self._store_index_refs if not item[0].query()]
            host_indices = device_indices = None
            if plan.comp_src:
                host_indices = torch.tensor(
                    [plan.comp_src, plan.starts],
                    dtype=torch.int64,
                    pin_memory=is_pin_memory_available(),
                )
                device_indices = host_indices.to(self.device, non_blocking=True)
            if plan.sink_src:
                self._backend.launch_copy(
                    plan.sink_src,
                    plan.sink_dst,
                    is_store=True,
                    params=self._sink_store_params,
                    wait_for_submit=True,
                )
            for begin in range(0, len(plan.comp_src), self.kvtc_batch_blocks):
                end = begin + self.kvtc_batch_blocks
                src, dst = plan.comp_src[begin:end], plan.comp_dst[begin:end]
                n = len(src)
                block_ids = device_indices[0, begin:end]
                starts = device_indices[1, begin:end]
                comp_k = kvtc_compress_rope(
                    self.k_op_caches,
                    block_ids,
                    starts,
                    self.inv_cos_sin_cache,
                    self.kvtc_k_projected_bias_bf16,
                    self.kvtc_k_proj,
                )
                if self.quant_enabled:
                    self._codecs["k"].quantize(comp_k, out=buffer_prefix(self.k_store_staging, n))
                else:
                    self.k_store_staging[:n].copy_(comp_k, non_blocking=True)
                del comp_k
                comp_v = kvtc_compress(
                    self.v_op_caches,
                    block_ids,
                    self.kvtc_v_projected_bias_bf16,
                    self.kvtc_v_proj,
                )
                if self.quant_enabled:
                    self._codecs["v"].quantize(comp_v, out=buffer_prefix(self.v_store_staging, n))
                else:
                    self.v_store_staging[:n].copy_(comp_v, non_blocking=True)
                del comp_v
                slots = list(range(n))
                # K + V + metadata share ONE batched submission on this stream.
                # Its ordering prevents next chunk from overwriting staging
                # until DMA has consumed it; no cross-batch overlap is claimed.
                copy_blocks(slots, dst, self._comp_store_params)
            event = torch.npu.Event()
            event.record(self.store_stream)
        self._store_events.append((metadata.store_event, event))
        if host_indices is not None:
            self._store_index_refs.append((event, host_indices, device_indices))

    @torch.inference_mode()
    def _swap_in_blocks(self, metadata, plan: _TransferPlan | None = None) -> None:
        plan = plan if plan is not None else self._transfer_plan(metadata, is_store=False)
        logger.debug(
            "[KVTC WORKER] LOAD raw=%d compressed=%d",
            len(plan.sink_src),
            len(plan.comp_src),
        )
        with torch.npu.stream(self.load_stream):
            # Upload all indices once, before the batch loop. Constructing an
            # NPU tensor from a Python list in each iteration can block host
            # submission and prevent the following H2D from overlapping GEMM.
            self._load_index_refs = [item for item in self._load_index_refs if not item[0].query()]
            host_indices = device_indices = None
            if plan.comp_src:
                host_indices = torch.tensor(
                    [plan.comp_dst, plan.starts],
                    dtype=torch.int64,
                    pin_memory=is_pin_memory_available(),
                )
                device_indices = host_indices.to(self.device, non_blocking=True)
            if plan.sink_src:
                self._backend.launch_copy(
                    plan.sink_src,
                    plan.sink_dst,
                    is_store=False,
                    params=self._sink_load_params,
                    wait_for_submit=True,
                )
            for begin in range(0, len(plan.comp_src), self.kvtc_batch_blocks):
                end = begin + self.kvtc_batch_blocks
                src = plan.comp_src[begin:end]
                n = len(src)
                slots = list(range(n))
                block_ids = device_indices[0, begin:end]
                starts = device_indices[1, begin:end]
                slot = (begin // self.kvtc_batch_blocks) % len(self._load_slot_pools)
                pool = self._load_slot_pools[slot]
                with torch.npu.stream(self._load_dma_stream):
                    previous = self._load_slot_done[slot]
                    if previous is not None:
                        self._load_dma_stream.wait_event(previous)
                    copy_blocks(src, slots, self._load_slot_params[slot])
                    ready = torch.npu.Event()
                    ready.record(self._load_dma_stream)
                self.load_stream.wait_event(ready)
                if self.quant_enabled:
                    comp_k = buffer_prefix(
                        {name[2:]: t for name, t in pool.items() if name.startswith("k_")},
                        n,
                    )
                    comp_v = buffer_prefix(
                        {name[2:]: t for name, t in pool.items() if name.startswith("v_")},
                        n,
                    )
                    comp_k = self._codecs["k"].dequantize(comp_k, out=self.k_dequant_staging[:n])
                    comp_v = self._codecs["v"].dequantize(comp_v, out=self.v_dequant_staging[:n])
                else:
                    comp_k, comp_v = pool["k"][:n], pool["v"][:n]
                kvtc_decompress_rope(
                    comp_k,
                    block_ids,
                    starts,
                    self.fwd_cos_sin_cache,
                    self.kvtc_k_mu_bf16,
                    self.kvtc_k_deproj,
                    self.k_op_caches,
                )
                kvtc_decompress(
                    comp_v,
                    block_ids,
                    self.kvtc_v_mu_bf16,
                    self.kvtc_v_deproj,
                    self.v_op_caches,
                )
                consumed = torch.npu.Event()
                consumed.record(self.load_stream)
                self._load_slot_done[slot] = consumed
            event = torch.npu.Event()
            event.record(self.load_stream)
        self._load_events.append((metadata.load_event, event))
        if host_indices is not None:
            self._load_index_refs.append((event, host_indices, device_indices))
