"""DSA worker-local DRAM hot store 的 Ascend 内存实现与创建入口。

本模块读取已经解耦的 KV cache 配置，以 Indexer block 容量和
``hot_cpu_block_multiple`` 估算固定 DRAM block 数（最终块数向上取整），为每层创建
NOPE/ROPE 两个 Ascend swapped-memory arena，并组装 ``AscendDSAHotKVStore``。
“hot”表示这些 MLA 满块在请求推理期间会被 KSC token-wise 换入，不等同于
通用 CPU offload connector 或更冷层 prefix cache。

逻辑块表、hash/refcount、请求释放和固定容量约束由
``dsa_hot_kv_store_core.py`` 管理；本模块不推进请求阶段，也不发射 KSC/dump
算子。arena 在初始化完成后地址和容量保持稳定，不支持运行期扩容。
"""

from __future__ import annotations

import math

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

from vllm_ascend.dsa_sparse.dsa_hot_kv_store_core import (
    BlockType,
    DSAHotKVStore,
    _calculate_hot_num_blocks,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)

logger = init_logger(__name__)


class AscendDSAHotKVStore(DSAHotKVStore):
    """Worker-local DRAM store for DSA sparse decode.

    This store is independent from vLLM-Ascend's CPUOffloadingConnector.  It
    owns per-rank NPU-visible swapped DRAM arenas and logical DRAM block
    tables used by DSA to stage MLA cache blocks before KSC
    materializes selected tokens back to HBM.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vllm_config = vllm_config

    @classmethod
    def _allocate_host_arena(cls, block_shape: tuple[int, ...],
                             dtype: torch.dtype,
                             capacity: int) -> torch.Tensor:
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError(
                "DSA sparse offload requires torch_npu swapped-memory "
                "DRAM arenas on Ascend") from exc

        device_index = torch.npu.current_device()
        arena = torch_npu.empty_with_swapped_memory(
            (int(capacity), *tuple(block_shape)),
            dtype=dtype,
            device=torch.device(f"npu:{device_index}"),
        )
        if not arena.is_contiguous():
            raise RuntimeError(
                "torch_npu.empty_with_swapped_memory must return a "
                "contiguous tensor for DSA DRAM arenas")
        return arena

    @staticmethod
    def _layer_id_from_name(layer_name: str) -> int | None:
        parts = layer_name.split(".")
        if len(parts) > 2:
            try:
                return int(parts[2])
            except ValueError:
                pass
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
        return None

    def initialize_hot_cache_from_kv_caches(
        self,
        kv_caches: dict,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Preallocate request-lifetime DRAM arenas for DSA sparse decode."""
        cache_config = self.vllm_config.cache_config
        if not bool(cache_config.enable_dsa_sparse_cache):
            return

        spec_by_layer = {
            layer_name: group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        }
        expected_mla_layers = {
            layer_name
            for layer_name, spec in spec_by_layer.items()
            if is_dsa_mla_resident_spec(spec)
        }
        indexer_num_blocks = 0
        for layer_name, cache in kv_caches.items():
            spec = spec_by_layer.get(layer_name)
            if is_dsa_indexer_spec(spec) and torch.is_tensor(cache):
                indexer_num_blocks = max(indexer_num_blocks,
                                         int(cache.shape[0]))
        if indexer_num_blocks <= 0:
            raise RuntimeError(
                "DSA split-cache initialization did not receive a dense "
                "Indexer KV tensor")
        if not expected_mla_layers:
            raise RuntimeError(
                "DSA split-cache initialization did not receive an MLA "
                "resident KV group")

        configured_multiple = cache_config.dsa_hot_cpu_block_multiple
        multiple = (3.0 if configured_multiple is None else
                    float(configured_multiple))
        hot_num_blocks = _calculate_hot_num_blocks(indexer_num_blocks,
                                                    multiple)
        block_size = int(cache_config.block_size)
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        max_logical_blocks = max(1, math.ceil(max_model_len / block_size) + 1)
        max_request_rows = int(cache_config.dsa_max_active_reqs or 256)

        initialized_layers: list[int] = []
        initialized_layer_names: set[str] = set()
        for layer_name, cache in kv_caches.items():
            spec = spec_by_layer.get(layer_name)
            if (not isinstance(spec, AttentionSpec)
                    or not is_dsa_mla_resident_spec(spec)):
                continue
            if not isinstance(cache, (tuple, list)) or len(cache) < 2:
                continue
            nopek_cache, ropek_cache = cache[0], cache[1]
            if not torch.is_tensor(nopek_cache) or not torch.is_tensor(
                    ropek_cache):
                continue
            layer_id = self._layer_id_from_name(layer_name)
            if layer_id is None:
                continue
            self.preallocate_layer_cache(
                layer_id=layer_id,
                blk_type=BlockType.NOPE_K,
                block_shape=tuple(nopek_cache.shape[1:]),
                dtype=nopek_cache.dtype,
                num_blocks=hot_num_blocks,
                max_request_rows=max_request_rows,
                max_logical_blocks=max_logical_blocks,
            )
            self.preallocate_layer_cache(
                layer_id=layer_id,
                blk_type=BlockType.ROPE_K,
                block_shape=tuple(ropek_cache.shape[1:]),
                dtype=ropek_cache.dtype,
                num_blocks=hot_num_blocks,
                max_request_rows=max_request_rows,
                max_logical_blocks=max_logical_blocks,
            )
            initialized_layers.append(layer_id)
            initialized_layer_names.add(layer_name)

        missing_mla_layers = sorted(
            expected_mla_layers - initialized_layer_names)
        if missing_mla_layers:
            raise RuntimeError(
                "DSA hot DRAM initialization is missing MLA cache tensors "
                f"for {len(missing_mla_layers)} layer(s): "
                f"{missing_mla_layers[:8]}")

        self.freeze_capacity()
        sample_arena = self.get_arena(initialized_layers[0],
                                      BlockType.NOPE_K)
        logger.debug(
            "Initialized DSA hot DRAM cache: layers=%d, "
            "hot_blocks_per_layer_type=%d, block_multiple=%s, "
            "request_rows=%d, "
            "logical_blocks=%d, arena_device=%s",
            len(set(initialized_layers)), hot_num_blocks, multiple,
            max_request_rows, max_logical_blocks, sample_arena.device)


def create_dsa_hot_kv_store(vllm_config: VllmConfig) -> AscendDSAHotKVStore:
    return AscendDSAHotKVStore(vllm_config)
