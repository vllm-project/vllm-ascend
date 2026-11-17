# SPDX-License-Identifier: Apache-2.0
"""Native schema compatibility; the lightweight vLLM mocks cannot prove this."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

msgspec = pytest.importorskip("msgspec")
pytest.importorskip("vllm")
from vllm.distributed.kv_events import BlockStored, KVEventAggregator  # noqa: E402

if not hasattr(BlockStored, "__struct_fields__"):
    pytest.skip("requires the native vLLM msgspec schema", allow_module_level=True)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.event_layout import (  # noqa: E402
    attach_deployment_layout,
    build_deployment_layout,
    deployment_announcement,
)


def test_layout_extension_preserves_native_decode_and_worker_quorum():
    event = BlockStored(
        block_hashes=[7],
        parent_block_hash=None,
        token_ids=[],
        block_size=64,
        lora_id=None,
        medium="cpu",
        lora_name=None,
        group_idx=1,
        kv_cache_spec_kind="mamba",
    )
    extended = attach_deployment_layout([event], '{"format":"ascend-store-v1"}')[0]
    encoded = msgspec.msgpack.encode(extended)
    assert msgspec.msgpack.decode(encoded)["type"] == "BlockStored"
    decoded = msgspec.msgpack.decode(encoded, type=BlockStored)
    for field in BlockStored.__struct_fields__:
        assert getattr(decoded, field) == getattr(event, field)
    quorum = KVEventAggregator(2)
    quorum.add_events([extended])
    quorum.add_events([extended])
    assert quorum.get_common_events() == [extended]


def test_layout_only_announcement_never_claims_residency():
    event = deployment_announcement('{"format":"ascend-store-v1"}', 128)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(event), type=BlockStored)
    assert decoded.block_hashes == []
    assert decoded.token_ids == []
    assert json.loads(event.kv_cache_layout)["format"] == "ascend-store-v1"


def test_worker_announces_layout_before_any_store_event():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.enable_kv_events = True
    worker.use_mamba = True
    worker.hash_block_size = 128
    worker.kv_send_thread = MagicMock()
    worker.kv_send_thread.get_kv_events.side_effect = [[], []]
    declaration = '{"format":"ascend-store-v1"}'
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.event_layout.build_deployment_layout",
        return_value=declaration,
    ) as build:
        (announcement,) = worker.get_kv_events()
        assert announcement.kv_cache_layout == declaration
        assert announcement.block_hashes == []
        assert worker.get_kv_events() == []
        build.assert_called_once_with(worker)


def test_complete_layout_covers_all_tp_checkpoint_parts():
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

    worker = SimpleNamespace(
        use_mamba=True,
        use_layerwise=False,
        backend_name="mooncake",
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        tp_size=4,
        put_step=4,
        model_name="Kimi-K3",
        hash_block_size=16,
        cache_transfer_granularity=64,
        grouped_block_size=[16, 64],
        group_uses_align_state=[False, True],
        kv_event_group_metadata=[("full_attention", None), ("mamba", None)],
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["layer.0"],
                    kv_cache_spec=FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16),
                ),
                KVCacheGroupSpec(
                    layer_names=["layer.1"],
                    kv_cache_spec=MambaSpec(
                        block_size=64, shapes=((2, 2),), dtypes=(torch.float16,), mamba_cache_mode="align"
                    ),
                ),
            ]
        ),
        token_database=SimpleNamespace(
            _get_key_prefix=lambda group: f"Kimi-K3@head_or_tp_rank:0@group:{group}@cache_role:kv@"
        ),
        metadata=[SimpleNamespace(head_or_tp_rank=0), SimpleNamespace(head_or_tp_rank=0)],
        m_store=SimpleNamespace(config=SimpleNamespace(master_server_address="127.0.0.1:45211")),
    )
    declaration = json.loads(build_deployment_layout(worker))
    assert declaration["format"] == "ascend-store-v1"
    assert declaration["hash_unit_tokens"] == 16
    assert declaration["alignment_tokens"] == 64
    assert declaration["storage_domain"] == "ascend://127.0.0.1:45211"
    assert len(declaration["groups"]) == 2
    assert len(declaration["groups"][0]["key_prefixes"]) == 1
    checkpoint = declaration["groups"][1]
    assert checkpoint["kind"] == "mamba"
    assert checkpoint["block_tokens"] == 64
    assert checkpoint["mamba_cache_mode"] == "align"
    assert len(checkpoint["key_prefixes"]) == 4
    assert "@head_or_tp_rank:3@" in checkpoint["key_prefixes"][3]


@pytest.mark.parametrize("overrides", [{"use_mamba": False}, {"use_layerwise": True}, {"pp_size": 2}])
def test_unsupported_deployment_is_not_inferred(overrides):
    worker = SimpleNamespace(use_mamba=True, use_layerwise=False, backend_name="mooncake", pp_size=1)
    worker.__dict__.update(overrides)
    assert build_deployment_layout(worker) is None
