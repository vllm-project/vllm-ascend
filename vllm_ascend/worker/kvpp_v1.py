# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.worker.v2.kvpp import KVPPCacheLayout, KVPPRuntime


class KVPPV1Runtime:
    """Model Runner V1 adapter around the shared KVPP scheduler."""

    def __init__(self, runtime: KVPPRuntime | None = None) -> None:
        self._kvpp_runtime = runtime if runtime is not None else KVPPRuntime()

    @classmethod
    def create_from_kv_cache(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any],
        block_tables: Any,
    ) -> KVPPV1Runtime:
        if KVPPConfig.from_vllm_config(vllm_config).size <= 1:
            return cls()

        cache_group_count = len(kv_cache_config.kv_cache_groups)
        return cls(
            KVPPRuntime.create_from_cache_layout(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                static_forward_context=static_forward_context,
                cache_layout=KVPPCacheLayout(
                    layer_caches=kv_caches,
                    physical_blocks_per_kv_block=tuple(
                        block_tables[index].blocks_per_phys_block for index in range(cache_group_count)
                    ),
                    tokens_per_block=tuple(
                        block_tables[index].logical_block_size for index in range(cache_group_count)
                    ),
                ),
            ),
        )

    def prepare_forward(
        self,
        input_batch: Any,
        num_reqs: int,
        seq_lens: Any,
    ) -> None:
        runtime = self._kvpp_runtime
        if runtime.scheduler is None:
            return
        block_table = input_batch.block_table[runtime.managed_cache_group_index]
        runtime.scheduler.schedule_forward(
            block_table.get_device_tensor(num_reqs),
            seq_lens,
        )

    def complete_forward(self) -> None:
        self._kvpp_runtime.complete_forward()
