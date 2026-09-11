# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common scheduler-side logic for Mooncake KV transfer connectors."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    MambaSpec,
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
        if self.kv_transfer_config.is_kv_consumer == self.kv_transfer_config.is_kv_producer:
            raise ValueError(
                f"Mooncake scheduler requires exactly one KV transfer role, got {self.kv_transfer_config.kv_role!r}"
            )
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config is not None else 0
        )

        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()

        self.local_ip = get_ip()
        self.side_channel_host = self.local_ip
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        assert self.pcp_size == 1, f"Mooncake temporarily requires prefill context parallel size 1, got {self.pcp_size}"
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.max_device_id = (
            self.tp_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Worker handshake ports occupy
        # [kv_port, kv_port + dp_size * pp_size * pcp_size * tp_size).
        # Keep one scheduler control port per DP rank immediately after that
        # range so it cannot collide with a worker handshake socket.
        self.side_channel_port = (
            self.kv_transfer_config.kv_port + self.max_device_id + vllm_config.parallel_config.data_parallel_rank
        )

        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.group_block_size = [group.kv_cache_spec.block_size for group in self.kv_cache_groups]
        self.group_unique_specs = [self._get_group_unique_specs(group) for group in self.kv_cache_groups]
        self.need_truncate = self._needs_prefill_token_truncation()

        logger.info("Initializing Mooncake Scheduler %s", engine_id)

    def _needs_prefill_token_truncation(self) -> bool:
        """Return whether Prefill must leave the last prompt token to Decode."""
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        uses_compressed_cache = isinstance(compress_ratios, (list, tuple, dict))
        has_state_group = any(
            isinstance(spec, MambaSpec) for group_specs in self.group_unique_specs for spec in group_specs
        )
        return uses_compressed_cache or has_state_group

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
        """Return prompt blocks while retaining evicted-block padding.

        Attention groups are clipped by their block size. Mamba groups are not
        prompt-block aligned, so only their speculative tail blocks are removed.
        """
        if not block_ids:
            return block_ids

        assert len(block_ids) == len(self.group_unique_specs), "Number of KV cache groups must match"

        transfer_block_ids: list[list[int]] = []
        cp_size = max(1, self.pcp_size * self.dcp_size)
        for blocks, block_size, group_specs in zip(
            block_ids,
            self.group_block_size,
            self.group_unique_specs,
        ):
            if any(isinstance(spec, MambaSpec) for spec in group_specs):
                if self.num_speculative_tokens > 0:
                    transfer_block_ids.append(blocks[: -self.num_speculative_tokens])
                else:
                    transfer_block_ids.append(blocks)
            else:
                num_prompt_blocks = cdiv(prompt_len, block_size * cp_size)
                transfer_block_ids.append(blocks[:num_prompt_blocks])
        return tuple(transfer_block_ids)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """Return the prompt token count transferred for stateful models."""
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """Drop the last P-side prompt token for stateful model transfer."""
        params = request.kv_transfer_params
        if (
            params is None
            or not self.need_truncate
            or params.get("_p_side_truncated")
            or request.num_prompt_tokens <= 1
        ):
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
        raise NotImplementedError

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        raise NotImplementedError

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

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Set worker transfer metadata on a concrete scheduler."""
        raise NotImplementedError

    def set_xfer_handshake_metadata(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Handle the legacy port-offset-keyed handshake entry point."""
        self.set_xfer_handshake_metadata_from_workers(metadata)


__all__ = ["MooncakeBaseConnectorScheduler"]
