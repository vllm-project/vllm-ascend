"""Tests for the pull connector metadata models and ``TransferRequest``.

``TransferRequest.as_legacy_dict()`` is the compatibility seam for d2rh, which
still consumes the legacy ``trans_info`` dict; the expected dict order is the
legacy literal order, hardcoded here on purpose.
"""

import dataclasses

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as pull_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import metadata

TRANSFER_REQUEST_FIELDS = (
    "request_id",
    "remote_request_id",
    "local_block_ids",
    "remote_block_ids",
    "group_pulls",
    "remote_engine_id",
    "remote_host",
    "remote_handshake_port",
    "remote_block_size",
    "remote_port_send_num",
    "num_computed_tokens",
    "all_task_done",
    "shard_idx",
    "local_block_ids_replicate_k",
    "remote_block_ids_replicate_k",
    "awaiting_kvs",
)
LEGACY_DICT_KEYS = (
    "request_id",
    "local_block_ids",
    "remote_block_ids",
    "local_block_ids_replicate_k",
    "remote_block_ids_replicate_k",
    "group_pulls",
    "remote_engine_id",
    "remote_request_id",
    "remote_host",
    "remote_handshake_port",
    "num_computed_tokens",
    "remote_port_send_num",
    "all_task_done",
    "shard_idx",
    "remote_block_size",
    "awaiting_kvs",
)
REQ_META_FIELDS = (
    "local_block_ids",
    "num_external_tokens",
    "num_computed_tokens",
    "remote_block_ids",
    "remote_host",
    "remote_port",
    "remote_engine_id",
    "remote_request_id",
    "remote_pcp_size",
    "remote_dcp_size",
    "remote_ptp_size",
    "remote_multi_nodes_meta_mapping",
    "num_prompt_blocks",
    "remote_block_size",
    "local_full_block_ids",
    "do_virtual",
)


def _legacy_kwargs(**overrides):
    kwargs = {
        "request_id": "req-1",
        "remote_request_id": "remote-1",
        "local_block_ids": ([1, 2],),
        "remote_block_ids": ([3, 4],),
        "group_pulls": [metadata.GroupPull(group_id=0, remote_tp_offset=0, num_group_pulls=1)],
        "remote_engine_id": "engine-1",
        "remote_host": "127.0.0.1",
        "remote_handshake_port": 5555,
        "remote_block_size": 16,
        "remote_port_send_num": {5555: {"num": 1, "host": "127.0.0.1"}},
        "num_computed_tokens": 8,
        "all_task_done": True,
        "shard_idx": 1,
        "local_block_ids_replicate_k": ([1],),
        "remote_block_ids_replicate_k": ([3],),
        "awaiting_kvs": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_legacy_import_surface_after_move():
    assert pull_connector.ReqMeta is metadata.ReqMeta
    assert pull_connector.GroupPull is metadata.GroupPull
    assert pull_connector.MooncakeConnectorMetadata is metadata.MooncakeConnectorMetadata


def test_req_meta_fields_unchanged():
    assert tuple(field.name for field in dataclasses.fields(metadata.ReqMeta)) == REQ_META_FIELDS
    req_meta = metadata.ReqMeta(
        local_block_ids=([1],),
        num_external_tokens=1,
        num_computed_tokens=0,
        remote_block_ids=([2],),
        remote_host="host",
        remote_port=1,
        remote_engine_id="engine",
        remote_request_id="remote",
        remote_pcp_size=1,
        remote_dcp_size=1,
        remote_ptp_size=None,
        remote_multi_nodes_meta_mapping={},
        num_prompt_blocks=1,
        remote_block_size=16,
    )
    assert req_meta.local_full_block_ids == tuple()
    assert req_meta.do_virtual is False


def test_group_pull_frozen_and_defaults():
    group_pull = metadata.GroupPull(group_id=1, remote_tp_offset=2, num_group_pulls=3)
    assert group_pull.prefill_pp_rank == 0
    assert group_pull.is_group_transfer_end is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        group_pull.group_id = 2  # type: ignore[misc]


def test_transfer_request_field_order_and_defaults():
    assert tuple(field.name for field in dataclasses.fields(metadata.TransferRequest)) == TRANSFER_REQUEST_FIELDS
    request = metadata.TransferRequest(**_legacy_kwargs())
    assert request.remote_block_size == 16
    assert request.remote_port_send_num == {5555: {"num": 1, "host": "127.0.0.1"}}
    assert request.num_computed_tokens == 8
    assert request.all_task_done is True
    assert request.shard_idx == 1
    assert request.local_block_ids_replicate_k == ([1],)
    assert request.remote_block_ids_replicate_k == ([3],)
    assert request.awaiting_kvs is False

    minimal = metadata.TransferRequest(
        request_id="req",
        remote_request_id="remote",
        local_block_ids=(),
        remote_block_ids=(),
        group_pulls=[],
        remote_engine_id="engine",
        remote_host="host",
        remote_handshake_port=1,
    )
    assert minimal.remote_block_size is None
    assert minimal.remote_port_send_num is None
    assert minimal.num_computed_tokens == 0
    assert minimal.all_task_done is False
    assert minimal.shard_idx == 0
    assert minimal.local_block_ids_replicate_k is None
    assert minimal.remote_block_ids_replicate_k is None
    assert minimal.awaiting_kvs is True


def test_transfer_request_as_legacy_dict_keys_and_normalization():
    legacy = _legacy_kwargs()
    request = metadata.TransferRequest(**legacy)
    payload = request.as_legacy_dict()
    assert tuple(payload) == LEGACY_DICT_KEYS
    assert payload["request_id"] == legacy["request_id"]
    assert payload["local_block_ids"] is legacy["local_block_ids"]
    assert payload["remote_block_ids"] is legacy["remote_block_ids"]
    assert payload["group_pulls"] is legacy["group_pulls"]
    assert payload["remote_port_send_num"] is legacy["remote_port_send_num"]

    minimal = metadata.TransferRequest(
        request_id="req",
        remote_request_id="remote",
        local_block_ids=(),
        remote_block_ids=(),
        group_pulls=[],
        remote_engine_id="engine",
        remote_host="host",
        remote_handshake_port=1,
    )
    minimal_payload = minimal.as_legacy_dict()
    assert tuple(minimal_payload) == LEGACY_DICT_KEYS
    assert minimal_payload["remote_port_send_num"] == {}
    assert minimal_payload["local_block_ids_replicate_k"] == tuple()
    assert minimal_payload["remote_block_ids_replicate_k"] == tuple()
    assert minimal_payload["remote_block_size"] is None
    assert minimal_payload["num_computed_tokens"] == 0
    assert minimal_payload["all_task_done"] is False
    assert minimal_payload["shard_idx"] == 0
    assert minimal_payload["awaiting_kvs"] is True


def test_transfer_request_roundtrip():
    legacy = _legacy_kwargs(local_block_ids_replicate_k=None, remote_block_ids_replicate_k=None)
    expected_payload = metadata.TransferRequest(**legacy).as_legacy_dict()
    assert expected_payload["local_block_ids_replicate_k"] == tuple()
    assert expected_payload["remote_block_ids_replicate_k"] == tuple()

    from_kwargs = metadata.TransferRequest.from_legacy_kwargs(**legacy)
    assert from_kwargs.as_legacy_dict() == expected_payload

    from_dict = metadata.TransferRequest.from_legacy_dict(expected_payload)
    assert from_dict.as_legacy_dict() == expected_payload


def test_connector_metadata_add_new_req_unchanged():
    connector_metadata = metadata.MooncakeConnectorMetadata()
    kv_transfer_params = {
        "num_computed_tokens": 4,
        "remote_block_ids": ([7, 8],),
        "remote_engine_id": "engine-1",
        "remote_request_id": "remote-1",
        "remote_host": "127.0.0.1",
        "remote_port": 5555,
        "remote_pcp_size": 2,
        "remote_dcp_size": 2,
        "remote_ptp_size": 2,
        "remote_multi_nodes_meta_mapping": {"1": {"handshake_port": 5560}},
        "num_prompt_blocks": 3,
        "remote_block_size": 16,
        "do_virtual": True,
    }
    connector_metadata.add_new_req(
        request_id="req-1",
        local_block_ids=([1, 2],),
        num_external_tokens=2,
        kv_transfer_params=kv_transfer_params,
        local_full_block_ids=([1],),
    )
    req_meta = connector_metadata.requests["req-1"]
    assert isinstance(req_meta, metadata.ReqMeta)
    assert req_meta.local_block_ids == ([1, 2],)
    assert req_meta.num_external_tokens == 2
    assert req_meta.num_computed_tokens == 4
    assert req_meta.remote_block_ids == ([7, 8],)
    assert req_meta.remote_engine_id == "engine-1"
    assert req_meta.remote_request_id == "remote-1"
    assert req_meta.remote_host == "127.0.0.1"
    assert req_meta.remote_port == 5555
    assert req_meta.remote_pcp_size == 2
    assert req_meta.remote_dcp_size == 2
    assert req_meta.remote_ptp_size == 2
    assert req_meta.remote_multi_nodes_meta_mapping == {"1": {"handshake_port": 5560}}
    assert req_meta.num_prompt_blocks == 3
    assert req_meta.remote_block_size == 16
    assert req_meta.local_full_block_ids == ([1],)
    assert req_meta.do_virtual is True

    connector_metadata.add_new_req(
        request_id="req-2",
        local_block_ids=([],),
        num_external_tokens=0,
        kv_transfer_params={
            "remote_block_ids": ([],),
            "remote_engine_id": "engine-1",
            "remote_request_id": "remote-2",
            "remote_host": "127.0.0.1",
            "remote_port": 5555,
        },
    )
    defaults = connector_metadata.requests["req-2"]
    assert defaults.num_computed_tokens == 0
    assert defaults.remote_pcp_size == 1
    assert defaults.remote_dcp_size == 1
    assert defaults.remote_ptp_size is None
    assert defaults.remote_multi_nodes_meta_mapping == {}
    assert defaults.num_prompt_blocks == 0
    assert defaults.remote_block_size == 0
    assert defaults.local_full_block_ids == tuple()
    assert defaults.do_virtual is False
