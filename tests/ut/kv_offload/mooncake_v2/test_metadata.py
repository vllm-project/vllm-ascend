# SPDX-License-Identifier: Apache-2.0

import msgspec
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadataGroups,
)

from .helpers import make_metadata_groups, make_transfer_metadata


def test_transfer_metadata_accepts_aligned_layer_fields() -> None:
    metadata = make_transfer_metadata(
        layer_names=["layer.0", "layer.1"],
        base_addrs=[[1000, 2000], [3000]],
        block_strides=[[128, 128], [256]],
        block_lens=[[64, 64], [256]],
        block_shapes=[[(1, 16, 2), (1, 16, 2)], [(2, 16, 4)]],
        block_size_scales=[[1, 1], [2]],
    )

    assert metadata.layer_names == ["layer.0", "layer.1"]
    assert metadata.kv_caches_base_addr[0] == [1000, 2000]


def test_transfer_metadata_rejects_misaligned_layer_count() -> None:
    with pytest.raises(ValueError, match="group_indices.*1 layers, expected 2"):
        make_transfer_metadata(
            layer_names=["layer.0", "layer.1"],
            group_indices=[0],
        )


def test_transfer_metadata_rejects_misaligned_layer_block_sizes() -> None:
    with pytest.raises(ValueError, match="layer_block_sizes.*1 layers, expected 2"):
        make_transfer_metadata(
            layer_names=["layer.0", "layer.1"],
            layer_block_sizes=[16],
        )


def test_transfer_metadata_rejects_misaligned_tensor_count() -> None:
    with pytest.raises(ValueError, match="layer 'layer.0'.*block_strides"):
        make_transfer_metadata(
            layer_names=["layer.0"],
            base_addrs=[[1000, 2000]],
            block_strides=[[128]],
        )


def test_transfer_metadata_groups_msgpack_round_trip() -> None:
    groups = make_metadata_groups()

    encoded = msgspec.msgpack.encode(groups)
    decoded = msgspec.msgpack.decode(encoded, type=MooncakeTransferMetadataGroups)

    assert decoded == groups
    assert decoded.metadata_by_pp_rank[0].metadata_by_tp_rank[0].te_rpc_port == 9000


def test_connector_metadata_adds_complete_request() -> None:
    metadata = MooncakeConnectorMetadata()
    metadata.add_new_req(
        request_id="request-d",
        local_block_ids=([10, 11],),
        local_full_block_ids=([1, 2, 10, 11],),
        local_num_prompt_tokens=32,
        num_external_tokens=16,
        kv_transfer_params={
            "num_computed_tokens": 16,
            "remote_block_ids": ([20, 21],),
            "remote_host": "10.0.0.1",
            "remote_port": 6000,
            "remote_engine_id": "engine-p",
            "remote_request_id": "request-p",
            "remote_num_prompt_tokens": 31,
        },
    )

    request = metadata.requests["request-d"]
    assert request.local_block_ids == ([10, 11],)
    assert request.local_full_block_ids == ([1, 2, 10, 11],)
    assert request.local_num_prompt_tokens == 32
    assert request.num_computed_tokens == 16
    assert request.remote_request_id == "request-p"
