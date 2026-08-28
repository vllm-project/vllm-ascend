# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common worker-side logic for Mooncake KV transfer connectors."""

import math
import os
from typing import TYPE_CHECKING

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
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    as_kv_cache_tensors,
    collect_configured_register_regions,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    get_transfer_timeout_value,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class MooncakeBaseConnectorWorker:
    """Worker implementation shared by Mooncake transfer modes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        assert vllm_config.kv_transfer_config is not None

        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if self.kv_transfer_config.is_kv_consumer == self.kv_transfer_config.is_kv_producer:
            raise ValueError(
                f"Mooncake worker requires exactly one KV transfer role, got {self.kv_transfer_config.kv_role!r}"
            )
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
        if self.dp_rank is None:
            raise ValueError("Mooncake worker requires a local DP rank")
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        pcp_group = get_pcp_group()
        self.pcp_rank = pcp_group.rank_in_group
        self.pcp_size = pcp_group.world_size
        assert self.pcp_size == 1, f"Mooncake temporarily requires prefill context parallel size 1, got {self.pcp_size}"
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
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
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_spec = group_spec.kv_cache_specs[layer_name]
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

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register configured KV cache allocations and publish metadata."""
        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        self._build_kv_cache_spec_mappings()
        layer_names: list[str] = []
        layer_block_sizes: list[int] = []
        group_indices: list[int] = []
        kv_caches_base_addr: list[list[int]] = []
        block_strides_per_layer: list[list[int]] = []
        block_lens_per_layer: list[list[int]] = []
        block_shapes_per_layer: list[list[tuple[int, ...]]] = []
        block_size_scales_per_layer: list[list[int]] = []
        configured_layer_names: set[str] = set()

        for tensor_config in self.kv_cache_config.kv_cache_tensors:
            for layer_name in tensor_config.shared_by:
                if layer_name in configured_layer_names:
                    raise ValueError(f"Layer {layer_name!r} is referenced by more than one configured KV cache tensor.")
                if layer_name not in self.layer_name_to_group_index:
                    raise ValueError(f"Configured KV cache layer {layer_name!r} does not belong to a KV cache group.")

                cache_or_caches = kv_caches.get(layer_name)
                if cache_or_caches is None:
                    raise ValueError(f"No KV cache was registered for configured layer {layer_name!r}.")

                base_addrs: list[int] = []
                block_strides: list[int] = []
                block_lens: list[int] = []
                block_shapes: list[tuple[int, ...]] = []
                block_size_scales: list[int] = []

                for cache in as_kv_cache_tensors(cache_or_caches):
                    tensor_num_blocks = cache.shape[0]
                    element_size = cache.element_size()
                    block_shape = tuple(cache.shape[1:])
                    base_addrs.append(cache.data_ptr())
                    block_strides.append(cache.stride(0) * element_size)
                    block_lens.append(math.prod(block_shape) * element_size)
                    block_shapes.append(block_shape)
                    block_size_scales.append(tensor_num_blocks // self.num_blocks)

                configured_layer_names.add(layer_name)
                layer_names.append(layer_name)
                spec_index = self.layer_name_to_spec_index[layer_name]
                spec = self.kv_cache_specs[spec_index]
                layer_block_size = spec.block_size
                if isinstance(spec, AscendSFAIndexerCacheSpec):
                    # The cache manager treats one SFA indexer block as a DCP
                    # virtual block, while every worker physically stores all
                    # replicated indexer blocks. Publish the virtual token span
                    # so dividing it by the tensor block scale recovers the
                    # physical kernel block size.
                    layer_block_size *= spec.sfa_dcp_replicated_indexer_size
                layer_block_sizes.append(layer_block_size)
                group_indices.append(self.layer_name_to_group_index[layer_name])
                kv_caches_base_addr.append(base_addrs)
                block_strides_per_layer.append(block_strides)
                block_lens_per_layer.append(block_lens)
                block_shapes_per_layer.append(block_shapes)
                block_size_scales_per_layer.append(block_size_scales)

        unexpected_layers = kv_caches.keys() - configured_layer_names
        if unexpected_layers:
            raise ValueError(f"KV caches contain layers absent from kv_cache_tensors: {sorted(unexpected_layers)}.")

        register_regions = collect_configured_register_regions(self.kv_cache_config, kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        transfer_metadata = MooncakeTransferMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            layer_names=layer_names,
            layer_block_sizes=layer_block_sizes,
            group_indices=group_indices,
            kv_caches_base_addr=kv_caches_base_addr,
            block_strides=block_strides_per_layer,
            block_lens=block_lens_per_layer,
            block_shapes=block_shapes_per_layer,
            block_size_scales=block_size_scales_per_layer,
            local_ip=self.side_channel_host,
            handshake_port=self.handshake_port,
        )
        self.transfer_metadata = transfer_metadata
        self.xfer_handshake_metadata = transfer_metadata

        logger.debug(
            "Mooncake KV cache transfer metadata: metadata=%s, register_ptrs=%s, register_lengths=%s",
            transfer_metadata,
            register_regions.ptrs,
            register_regions.lengths,
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
