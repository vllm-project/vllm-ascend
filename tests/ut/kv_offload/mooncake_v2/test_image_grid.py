# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.multimodal.cache.base import MultiModalCacheMissError
from vllm.multimodal.cache.lru import LruKeyReplicatedReceiverCache
from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem, MultiModalSharedField

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakeBaseConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.image_grid import get_image_grids


def make_request(*features, **params):
    return SimpleNamespace(
        mm_features=list(features),
        kv_transfer_params={"do_remote_decode": True, "return_image_grids": True, **params},
    )


def make_feature(grid, *, image_hash="image", modality="image"):
    return SimpleNamespace(
        modality=modality,
        mm_hash=image_hash,
        identifier="encoder-id",
        data={"image_grid_thw": SimpleNamespace(data=grid)},
    )


@pytest.mark.parametrize("all_groups", [False, True])
def test_completion_returns_grids_in_order(all_groups):
    request = make_request(
        make_feature(torch.tensor([[1, 16, 20]]), image_hash="first"),
        make_feature(torch.tensor([1, 24, 32]), image_hash="second"),
    )
    connector = object.__new__(MooncakeBaseConnector)
    connector.connector_scheduler = MagicMock()
    original_metadata = {"do_remote_prefill": True, "remote_block_ids": [[1, 2]]}
    connector.connector_scheduler.request_finished.return_value = (True, original_metadata)
    finish = connector.request_finished_all_groups if all_groups else connector.request_finished
    blocks = ([1, 2],) if all_groups else [1, 2]
    delayed, metadata = finish(request, blocks)
    assert delayed is True
    assert metadata == {
        **original_metadata,
        "pd_image_grids": [
            {"hash": "first", "grid": [1, 16, 20]},
            {"hash": "second", "grid": [1, 24, 32]},
        ],
    }
    assert "pd_image_grids" not in original_metadata
    assert "pd_image_grids" not in request.kv_transfer_params
    connector.connector_scheduler.request_finished.assert_called_once_with(request, ([1, 2],))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1, 0, 16]),
        torch.tensor([1, -1, 16]),
        torch.tensor([1, 16]),
        torch.tensor([1, 16, 20, 24]),
        torch.tensor([1.0, 16.0, 20.0]),
        torch.tensor([True, True, True]),
        torch.empty(3, device="meta"),
        [1, 16, 20],
    ],
)
def test_invalid_or_non_cpu_grid_falls_back(tensor):
    assert get_image_grids(make_request(make_feature(tensor))) is None


@pytest.mark.parametrize("missing", [None, {}, {"shared_memory_handle": 123}])
def test_missing_or_unresolved_cache_item_falls_back(missing):
    feature = make_feature(torch.tensor([1, 16, 20]))
    feature.data = missing
    assert get_image_grids(make_request(feature)) is None


def test_text_and_mixed_media_fall_back():
    image = make_feature(torch.tensor([1, 16, 20]))
    video = make_feature(torch.tensor([1, 16, 20]), modality="video")
    assert get_image_grids(make_request()) is None
    assert get_image_grids(make_request(image, video)) is None


def test_restored_cache_data_and_identifier_fallback():
    feature = make_feature(torch.tensor([1, 16, 20]), image_hash=None)
    expected = [{"hash": "encoder-id", "grid": [1, 16, 20]}]
    assert get_image_grids(make_request(feature)) == expected
    assert get_image_grids(make_request(feature)) == expected
    feature.data = None
    assert get_image_grids(make_request(feature)) is None


def test_receiver_cache_restores_grids_and_reports_eviction():
    config = SimpleNamespace(get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=0.001))
    cache = LruKeyReplicatedReceiverCache(config)
    feature = make_feature(torch.tensor([1, 16, 20]))
    feature.data = MultiModalKwargsItem(
        {
            "image_grid_thw": MultiModalFieldElem(
                data=torch.tensor([1, 16, 20]), field=MultiModalSharedField(batch_size=1)
            )
        }
    )
    request = make_request(feature)
    expected = [{"hash": "image", "grid": [1, 16, 20]}]
    request.mm_features = cache.get_and_update_features(request.mm_features)
    assert get_image_grids(request) == expected
    feature.data = None
    request.mm_features = cache.get_and_update_features(request.mm_features)
    assert get_image_grids(request) == expected
    cache.clear_cache()
    feature.data = None
    assert get_image_grids(request) is None
    with pytest.raises(MultiModalCacheMissError):
        cache.get_and_update_features(request.mm_features)


@pytest.mark.parametrize(
    "params, metadata",
    [
        ({"return_image_grids": False}, {"do_remote_prefill": True}),
        ({"return_image_grids": "true"}, {"do_remote_prefill": True}),
        ({"do_remote_decode": False}, {"do_remote_prefill": True}),
        ({}, None),
    ],
)
def test_opt_in_and_failed_completion_do_not_export_grids(params, metadata):
    request = make_request(make_feature(torch.tensor([1, 16, 20])), **params)
    connector = object.__new__(MooncakeBaseConnector)
    connector.connector_scheduler = MagicMock()
    connector.connector_scheduler.request_finished.return_value = (False, metadata)
    assert connector.request_finished(request, []) == (False, metadata)
