# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Push must fence the shared slots published by the current cache planner."""

from types import SimpleNamespace

import pytest
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector, push_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseReuseLayout,
    apply_layerwise_kv_cache_plan,
)


@pytest.mark.parametrize("stage_start", [0, 24])
@pytest.mark.parametrize("reuse", [False, True])
def test_push_registers_current_layout_and_stage_local_fences(monkeypatch, stage_start, reuse):
    names = [f"model.layers.{stage_start + i}.self_attn" for i in range(4)]
    layout = LayerwiseReuseLayout(
        layers=tuple(range(stage_start, stage_start + 4)),
        cache_layer_indices={name: stage_start + i for i, name in enumerate(names)},
        kv_cache_tensors=(),
        cache_view_sizes={},
        prefetch_layer_map={stage_start + 3: (stage_start + 1,)},
        independent_layers=[stage_start],
        num_prefetch_layers=2,
        reused_cache_names=frozenset((names[1], names[3])),
        draft_entry_names=frozenset(),
    )
    transfer = KVTransferConfig(kv_connector="MooncakePushConnector", kv_role="kv_producer")
    cache = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[])
    if reuse:
        cache = apply_layerwise_kv_cache_plan(cache, transfer, layout)
    config = SimpleNamespace(
        kv_transfer_config=transfer,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    def init_facade(self, *_args):
        self.engine_id = "producer"
        self.connector_worker = self.connector_scheduler = None

    monkeypatch.setattr(connector.MooncakeBaseConnector, "__init__", init_facade)
    monkeypatch.setattr(push_worker.MooncakeBaseConnectorWorker, "__init__", lambda *args: None)
    producer = connector.MooncakePushConnector(config, KVConnectorRole.WORKER, cache)
    assert producer.supports_layerwise_buffer_reuse is reuse
    assert producer.connector_worker._shared_layer_indices == ({names[1]: 1, names[3]: 3} if reuse else {})

    waiters = []
    sink = SimpleNamespace(set_external_slot_release_waiter=waiters.append)
    multi = AscendMultiConnector.__new__(AscendMultiConnector)
    multi._connectors = [producer, sink]
    multi._configure_layerwise_reuse_completion()
    assert multi._layerwise_slot_release_providers == ([producer] if reuse else [])
    assert multi._external_slot_release_sink_configured is reuse
