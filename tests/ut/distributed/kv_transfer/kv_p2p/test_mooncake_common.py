"""Compatibility tests for the shared mooncake v1 common package.

These tests lock the two contracts that the protocol/tracker extraction must
preserve: the legacy import surface of the two V1 connector modules and the
msgspec wire field order/defaults of the exchanged metadata structures.
"""

import sys
import time

import msgspec
from vllm import envs

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as pull_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common import protocol, tracker

# Field order of the metadata encoded on the wire. Hardcoded on purpose: it must
# not follow the implementation, otherwise a wire regression would go unnoticed.
PULL_AGENT_METADATA_FIELDS = (
    "engine_id",
    "te_rpc_port",
    "kv_group2layeridx",
    "block_size",
    "kv_caches_base_addr",
    "block_size_scale",
    "num_blocks",
    "block_lens",
    "block_strides",
    "local_ip",
    "handshake_port",
)
PULL_AGENT_METADATA_DEFAULT_FIELDS = ("local_ip", "handshake_port")
LAYERWISE_AGENT_METADATA_FIELDS = ("te_rpc_port", "layer_metadata")
LAYER_METADATA_FIELDS = ("tensor_group_idx", "kv_caches_base_addr", "block_len", "block_size_scale")


def test_legacy_import_surface():
    """Both connectors keep exposing the migrated symbols as the same objects."""
    assert pull_connector.GET_META_MSG is protocol.GET_META_MSG
    assert pull_connector.DONE_RECVING_MSG is protocol.DONE_RECVING_MSG
    assert pull_connector.RemotePortInfo is protocol.RemotePortInfo
    assert pull_connector.MooncakeAgentMetadata is protocol.PullAgentMetadata
    assert pull_connector.KVCacheTaskTracker is tracker.KVCacheTaskTracker

    assert layerwise_connector.DONE_SENDING_MSG is protocol.DONE_SENDING_MSG
    assert layerwise_connector.FAILED_SENDING_MSG is protocol.FAILED_SENDING_MSG
    assert layerwise_connector.LayerMetadata is protocol.LayerMetadata
    assert layerwise_connector.MooncakeAgentMetadata is protocol.LayerwiseAgentMetadata


def test_agent_metadata_wire_fields_unchanged():
    assert protocol.PullAgentMetadata.__struct_fields__ == PULL_AGENT_METADATA_FIELDS
    assert protocol.LayerwiseAgentMetadata.__struct_fields__ == LAYERWISE_AGENT_METADATA_FIELDS

    pull_metadata = protocol.PullAgentMetadata(
        engine_id="engine-0",
        te_rpc_port=9090,
        kv_group2layeridx={0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])},
        block_size=128,
        kv_caches_base_addr=[[1024]],
        block_size_scale=[[1]],
        num_blocks=4,
        block_lens=[[1024]],
        block_strides=[[16]],
        local_ip="127.0.0.1",
        handshake_port=12345,
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(pull_metadata))
    assert list(decoded) == list(PULL_AGENT_METADATA_FIELDS)

    # omit_defaults=True is part of the wire contract: default fields are absent.
    minimal_metadata = protocol.PullAgentMetadata(
        engine_id="engine-0",
        te_rpc_port=9090,
        kv_group2layeridx={},
        block_size=128,
        kv_caches_base_addr=[],
        block_size_scale=[],
        num_blocks=0,
        block_lens=[],
        block_strides=[],
    )
    decoded_minimal = msgspec.msgpack.decode(msgspec.msgpack.encode(minimal_metadata))
    assert list(decoded_minimal) == [
        field for field in PULL_AGENT_METADATA_FIELDS if field not in PULL_AGENT_METADATA_DEFAULT_FIELDS
    ]

    layer_metadata = protocol.LayerMetadata(
        tensor_group_idx=[0],
        kv_caches_base_addr=[1024],
        block_len=[1024],
        block_size_scale=[1],
    )
    layerwise_metadata = protocol.LayerwiseAgentMetadata(
        te_rpc_port=9090,
        layer_metadata={"layer_0": layer_metadata},
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(layerwise_metadata))
    assert list(decoded) == list(LAYERWISE_AGENT_METADATA_FIELDS)
    assert list(decoded["layer_metadata"]["layer_0"]) == list(LAYER_METADATA_FIELDS)


def test_tracker_module_patch_surface():
    """The tracker module keeps ``time``/``envs`` module attributes for patching."""
    module = sys.modules[tracker.KVCacheTaskTracker.__module__]
    assert module is tracker
    assert module.time is time
    assert module.envs is envs


def test_tracker_behavior_unchanged():
    task_tracker = tracker.KVCacheTaskTracker()
    task_tracker.add_req_to_process("req-done")
    assert task_tracker.get_and_clear_finished_requests() == set()
    task_tracker.update_done_task_count("req-done")
    assert task_tracker.get_and_clear_finished_requests() == {"req-done"}
    assert task_tracker.get_and_clear_finished_requests() == set()

    task_tracker.add_req_to_process("req-not-transferred")
    task_tracker.add_not_transfer_request("req-not-transferred")
    assert task_tracker.get_and_clear_finished_requests() == {"req-not-transferred"}
