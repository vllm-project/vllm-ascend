from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params
