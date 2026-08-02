# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common worker-side logic for Mooncake KV transfer connectors."""

import math
import os
from typing import TYPE_CHECKING, Any

import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    get_transfer_timeout_value,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class MooncakeBaseConnectorWorker:
    """Worker implementation shared by Mooncake transfer modes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config
        self.block_size = vllm_config.cache_config.block_size
        self.num_blocks = kv_cache_config.num_blocks

        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        if self.pp_size > 1 and self.pcp_size > 1:
            raise ValueError("PP and PCP cannot be enabled at the same time")

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = self.kv_transfer_config.kv_role
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            self.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * self.tp_size * self.pp_size * self.pcp_size
        )
        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index

        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None
        self.engine = global_te.get_transfer_engine(
            self.side_channel_host,
            device_name=device_name,
        )
        self.te_rpc_port = self.engine.get_rpc_port()
        self.xfer_handshake_metadata: KVConnectorHandshakeMetadata | None = None
        self.xfer_stats = MooncakeKVConnectorStats()

        logger.info("Initializing Mooncake worker %s", engine_id)

    def _build_kv_cache_spec_mappings(self) -> None:
        """Flatten group specs and map each layer to its group and spec.

        A regular KV cache group contributes one spec. A uniform-type group
        contributes each distinct inner spec in layer order. Spec uniqueness is
        scoped to a group because different groups have different block tables,
        even when their specs compare equal.
        """
        self.kv_cache_specs: list[KVCacheSpec] = []
        self.layer_name_to_group_index: dict[str, int] = {}
        self.layer_name_to_spec_index: dict[str, int] = {}

        for group_index, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            group_spec_indices: list[int] = []

            for layer_name in group.layer_names:
                if layer_name in self.layer_name_to_group_index:
                    raise ValueError(f"Layer {layer_name!r} belongs to more than one KV cache group.")

                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    try:
                        layer_spec = group_spec.kv_cache_specs[layer_name]
                    except KeyError as exc:
                        raise ValueError(
                            f"Uniform KV cache group {group_index} has no spec for layer {layer_name!r}."
                        ) from exc
                else:
                    layer_spec = group_spec

                spec_index = next(
                    (index for index in group_spec_indices if self.kv_cache_specs[index] == layer_spec),
                    -1,
                )
                if spec_index < 0:
                    spec_index = len(self.kv_cache_specs)
                    self.kv_cache_specs.append(layer_spec)
                    group_spec_indices.append(spec_index)

                self.layer_name_to_group_index[layer_name] = group_index
                self.layer_name_to_spec_index[layer_name] = spec_index

    @staticmethod
    def _as_kv_cache_tensors(cache_or_caches: Any) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        if isinstance(cache_or_caches, (list, tuple)) and all(
            isinstance(cache, torch.Tensor) for cache in cache_or_caches
        ):
            return tuple(cache_or_caches)
        raise TypeError(
            f"A layer KV cache must be a tensor or a list/tuple of tensors, but got {type(cache_or_caches).__name__}."
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Collect per-layer KV cache metadata for D2D registration."""
        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        self._build_kv_cache_spec_mappings()

        # All lists below use registered_layer_names order. Each inner list
        # preserves the layer cache order, such as K/V or Mamba conv/SSM.
        self.registered_layer_names: list[str] = []
        self.registered_group_indices: list[int] = []
        self.registered_spec_indices: list[int] = []
        self.kv_caches_base_addr: list[list[int]] = []
        self.block_stride_per_addr: list[list[int]] = []
        self.block_len_per_addr: list[list[int]] = []
        self.block_shape_per_addr: list[list[tuple[int, ...]]] = []
        self.block_size_scale: list[list[int]] = []

        for layer_name in self.layer_name_to_group_index:
            cache_or_caches = kv_caches.get(layer_name)
            if cache_or_caches is None:
                logger.debug(
                    "Skipping layer %s because it has no worker KV cache tensor.",
                    layer_name,
                )
                continue

            base_addrs: list[int] = []
            block_strides: list[int] = []
            block_lens: list[int] = []
            block_shapes: list[tuple[int, ...]] = []
            block_size_scales: list[int] = []

            for cache in self._as_kv_cache_tensors(cache_or_caches):
                if cache.ndim == 0:
                    raise ValueError(f"KV cache tensor for layer {layer_name!r} must have a block dimension.")

                tensor_num_blocks = cache.shape[0]
                if self.num_blocks <= 0 or tensor_num_blocks % self.num_blocks:
                    raise ValueError(
                        f"KV cache tensor for layer {layer_name!r} has "
                        f"{tensor_num_blocks} physical blocks, which is not a "
                        f"multiple of {self.num_blocks} logical blocks."
                    )

                element_size = cache.element_size()
                block_shape = tuple(cache.shape[1:])
                base_addrs.append(cache.data_ptr())
                block_strides.append(cache.stride(0) * element_size)
                block_lens.append(math.prod(block_shape) * element_size)
                block_shapes.append(block_shape)
                block_size_scales.append(tensor_num_blocks // self.num_blocks)

            self.registered_layer_names.append(layer_name)
            self.registered_group_indices.append(self.layer_name_to_group_index[layer_name])
            self.registered_spec_indices.append(self.layer_name_to_spec_index[layer_name])
            self.kv_caches_base_addr.append(base_addrs)
            self.block_stride_per_addr.append(block_strides)
            self.block_len_per_addr.append(block_lens)
            self.block_shape_per_addr.append(block_shapes)
            self.block_size_scale.append(block_size_scales)

        unknown_layers = kv_caches.keys() - self.layer_name_to_group_index.keys()
        if unknown_layers:
            logger.debug(
                "Ignored KV cache tensors without a local KV cache group: %s",
                sorted(unknown_layers),
            )

        logger.debug(
            "Mooncake KV cache metadata: layers=%s, group_indices=%s, "
            "spec_indices=%s, base_addrs=%s, block_strides=%s, block_lens=%s, "
            "block_shapes=%s, block_size_scale=%s",
            self.registered_layer_names,
            self.registered_group_indices,
            self.registered_spec_indices,
            self.kv_caches_base_addr,
            self.block_stride_per_addr,
            self.block_len_per_addr,
            self.block_shape_per_addr,
            self.block_size_scale,
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return requests with completed receive and send operations."""
        raise NotImplementedError

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return local block IDs whose KV load failed."""
        raise NotImplementedError

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return and reset transfer statistics for the current interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Start D2D KV loading described by scheduler metadata."""
        raise NotImplementedError


__all__ = ["MooncakeBaseConnectorWorker"]
