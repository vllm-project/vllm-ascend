# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common scheduler-side logic for Mooncake KV transfer connectors."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.ascend_config import (
    get_ascend_config,
    init_ascend_config,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class GroupTransferInfo:
    """Transfer-relevant properties of one KV cache group."""

    tokens_per_block: int
    blocks_per_window: int
    is_state_group: bool


class MooncakeBaseConnectorScheduler:
    """Scheduler logic shared by Mooncake transfer modes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        assert vllm_config.kv_transfer_config is not None

        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.block_size = vllm_config.cache_config.block_size

        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()

        self.local_ip = get_ip()
        self.side_channel_host = self.local_ip
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.max_device_id = (
            self.tp_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Handshake base port for this DP group.
        self.side_channel_port = (
            self.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * self.tp_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )

        # Worker metadata for a DP group that may span multiple nodes.
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}

        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [self._get_group_transfer_info(group) for group in self.kv_cache_groups]
        self.need_truncate = self.use_compress or any(info.is_state_group for info in self.group_transfer_info)

        logger.info("Initializing Mooncake Scheduler %s", engine_id)

    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        specs = self._get_group_unique_specs(group)
        first_spec = specs[0] if specs else group.kv_cache_spec
        block_size = getattr(
            group.kv_cache_spec,
            "block_size",
            getattr(first_spec, "block_size", self.block_size),
        )
        is_state_group = any(isinstance(spec, MambaSpec) for spec in specs)
        sliding_window = 0
        compress_ratio = 1
        for spec in specs:
            if isinstance(spec, SlidingWindowSpec):
                sliding_window = spec.sliding_window
            elif hasattr(spec, "compress_ratio"):
                compress_ratio = spec.compress_ratio

        return GroupTransferInfo(
            tokens_per_block=block_size * max(1, int(compress_ratio)),
            blocks_per_window=(cdiv(sliding_window, block_size) + 1 if sliding_window else 0),
            is_state_group=is_state_group,
        )

    @staticmethod
    def _get_group_unique_specs(group: Any) -> list[Any]:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            return [group.kv_cache_spec]

        specs = []
        for layer_name in group.layer_names:
            layer_spec = group.kv_cache_spec.kv_cache_specs[layer_name]
            if layer_spec not in specs:
                specs.append(layer_spec)
        return specs

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Drop non-prompt attention blocks while retaining state groups."""
        if not block_ids:
            return block_ids

        assert len(block_ids) == len(self.group_transfer_info), "Number of KV cache groups must match"

        transfer_block_ids = []
        cp_size = max(1, self.pcp_size * self.dcp_size)
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group:
                transfer_block_ids.append(blocks)
            else:
                num_prompt_blocks = cdiv(
                    prompt_len,
                    group_info.tokens_per_block * cp_size,
                )
                transfer_block_ids.append(blocks[:num_prompt_blocks])
        return tuple(transfer_block_ids)

    def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """Clip SWA groups to their window tail and drop block zero."""
        if not block_ids:
            return block_ids

        assert len(block_ids) == len(self.group_transfer_info), "Number of KV cache groups must match"

        transfer_block_ids = []
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group or group_info.blocks_per_window == 0:
                transfer_block_ids.append(blocks)
            else:
                window_blocks = blocks[-group_info.blocks_per_window :]
                transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
        return tuple(transfer_block_ids)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """Return the prompt token count transferred for stateful models."""
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """Drop the last P-side prompt token for stateful model transfer."""
        params = request.kv_transfer_params
        if params is None or params.get("_p_side_truncated") or request.num_prompt_tokens <= 1:
            return

        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return

        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_p_side_truncated"] = True

    def on_new_request(self, request: "Request") -> None:
        """Mooncake currently requires no request-arrival bookkeeping."""

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Mooncake currently requires no worker-output bookkeeping."""

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        raise NotImplementedError

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError

    def _port_offset_from_handshake_metadata(
        self,
        rank_metadata: KVConnectorHandshakeMetadata,
        metadata_key: int | tuple[int, ...],
    ) -> int:
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            return handshake_port - self.kv_transfer_config.kv_port
        if isinstance(metadata_key, int):
            return metadata_key
        raise ValueError(f"Mooncake handshake metadata is missing handshake_port for worker key {metadata_key}")

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Build the worker host mapping for a possibly multi-node DP group."""
        if not metadata:
            return

        updated_mapping: dict[str, dict[str, Any]] = {}
        kv_port = self.kv_transfer_config.kv_port
        for metadata_key, rank_metadata in metadata.items():
            port_offset = self._port_offset_from_handshake_metadata(rank_metadata, metadata_key)
            updated_mapping[str(port_offset)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
                "handshake_port": kv_port + port_offset,
            }

        self.multi_nodes_meta_mapping.update(updated_mapping)
        logger.info(
            "MooncakeConnector set_xfer_handshake_metadata: worker_count=%d, updated=%s, multi_nodes_meta_mapping=%s",
            len(metadata),
            updated_mapping,
            self.multi_nodes_meta_mapping,
        )

    def set_xfer_handshake_metadata(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Handle the legacy port-offset-keyed handshake entry point."""
        self.set_xfer_handshake_metadata_from_workers(metadata)


__all__ = ["GroupTransferInfo", "MooncakeBaseConnectorScheduler"]
