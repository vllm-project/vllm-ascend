# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common worker-side logic for Mooncake KV transfer connectors."""

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
        self._get_prefill_decode_size()
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than "
                f"or equal to decode_tp_size: {self._decode_tp_size}"
            )

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
        self.dcp_rank = (
            get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        )
        if self.pp_size > 1 and self.pcp_size > 1:
            raise ValueError("PP and PCP cannot be enabled at the same time")

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = self.kv_transfer_config.kv_role
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            self.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * self.tp_size
            * self.pp_size
            * self.pcp_size
        )
        device_index = (
            self.pp_rank * self.pcp_size + self.pcp_rank
        ) * self.tp_size + self.tp_rank
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

    def _get_prefill_decode_size(self) -> None:
        prefill_config: dict[str, Any] = (
            self.kv_transfer_config.get_from_extra_config("prefill", {})
        )
        decode_config: dict[str, Any] = (
            self.kv_transfer_config.get_from_extra_config("decode", {})
        )
        if "tp_size" not in prefill_config or "dp_size" not in prefill_config:
            raise ValueError("Mooncake prefill config requires tp_size and dp_size")
        if "tp_size" not in decode_config or "dp_size" not in decode_config:
            raise ValueError("Mooncake decode config requires tp_size and dp_size")

        self._prefill_tp_size = prefill_config["tp_size"]
        self._prefill_dp_size = prefill_config["dp_size"]
        self._prefill_pp_size = prefill_config.get("pp_size", 1)
        self._prefill_pp_layer_partition = prefill_config.get("pp_layer_partition")
        self._decode_tp_size = decode_config["tp_size"]
        self._decode_dp_size = decode_config["dp_size"]
        self._decode_pp_size = decode_config.get("pp_size", 1)
        if self._decode_pp_size != 1:
            raise ValueError("Decode pipeline parallel size must be 1")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register model KV caches for D2D transfer."""
        raise NotImplementedError

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
