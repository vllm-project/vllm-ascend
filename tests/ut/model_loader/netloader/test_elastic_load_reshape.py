#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.model_loader.netloader.executor.elastic_load import (
    _cast_packed_int32_via_int8,
    _cast_tensor_to_fractal_nd,
    _cast_tensor_to_fractal_nz,
    _ensure_hccl_recv_buffer,
    _finalize_hccl_recv_buffer,
    _hccl_transfer_tensor,
    _prepare_hccl_recv_buffer,
    build_transfer_shape_manifest,
    cache_processed_layout_transfer_manifest,
    get_cached_processed_layout_transfer_items,
    get_cached_processed_layout_transfer_shapes,
    register_processed_layout_transfer_items,
    reshape_tensor_to_manifest_shape,
    reshape_transfer_items_to_manifest,
)
from vllm_ascend.model_loader.netloader.interaction.elastic import (
    _parse_transfer_shape_manifest,
    _recv_json_message,
)


def test_build_transfer_shape_manifest():
    weight = torch.empty(4, 8)
    manifest = build_transfer_shape_manifest([("layer.weight", weight)])
    assert manifest == {"layer.weight": (4, 8)}


def test_cache_processed_layout_transfer_manifest():
    model = MagicMock()
    weight = torch.empty(2, 3)
    scale = torch.empty(4)
    transfer_items = [("layer.weight", weight), ("layer.scale", scale)]

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._collect_processed_layout_tensors",
        return_value=transfer_items,
    ):
        count = cache_processed_layout_transfer_manifest(model)

    assert count == 2
    assert get_cached_processed_layout_transfer_items(model) == transfer_items
    assert get_cached_processed_layout_transfer_shapes(model) == {
        "layer.weight": (2, 3),
        "layer.scale": (4,),
    }


def test_register_processed_layout_transfer_items_uses_cache():
    model = MagicMock()
    cached_items = [("layer.weight", torch.empty(2, 3))]

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load.get_cached_processed_layout_transfer_items",
        return_value=cached_items,
    ), patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._collect_processed_layout_tensors",
    ) as mock_collect:
        items = register_processed_layout_transfer_items(model)

    assert items == cached_items
    mock_collect.assert_not_called()


def test_ensure_hccl_recv_buffer_reuses_contiguous_tensor():
    tensor = torch.empty(2, 3)
    recv_buffer = _ensure_hccl_recv_buffer(tensor)
    assert recv_buffer is tensor
    assert tensor.is_contiguous()


def test_ensure_hccl_recv_buffer_replaces_non_contiguous_storage_in_place():
    tensor = torch.empty(4, 8)
    non_contiguous = tensor.t()
    assert not non_contiguous.is_contiguous()

    recv_buffer = _ensure_hccl_recv_buffer(non_contiguous)
    assert recv_buffer is non_contiguous
    assert non_contiguous.is_contiguous()
    assert non_contiguous.shape == (8, 4)


def test_hccl_transfer_tensor_reuses_contiguous_tensor():
    tensor = torch.empty(2, 3)
    assert _hccl_transfer_tensor(tensor) is tensor


def test_hccl_transfer_tensor_makes_non_contiguous_contiguous():
    tensor = torch.empty(4, 8)
    non_contiguous = tensor.t()
    payload = _hccl_transfer_tensor(non_contiguous)
    assert payload.is_contiguous()
    assert payload.shape == (8, 4)


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor", return_value=True)
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_hccl_transfer_tensor_casts_fractal_nz_to_nd(mock_format_cast, _mock_is_nz):
    tensor = torch.empty(2, 3)
    nd_tensor = torch.empty(2, 3)
    mock_format_cast.return_value = nd_tensor

    payload = _hccl_transfer_tensor(tensor)

    mock_format_cast.assert_called_once()
    assert payload is nd_tensor


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor", return_value=True)
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._cast_tensor_to_fractal_nd")
def test_prepare_hccl_recv_buffer_allocates_fresh_nd_for_fractal_nz(mock_cast_nd, _mock_is_nz):
    tensor = torch.empty(2, 3)

    recv_buffer, restore_fractal_nz = _prepare_hccl_recv_buffer(tensor)

    assert restore_fractal_nz is True
    assert tuple(recv_buffer.shape) == (2, 3)
    assert recv_buffer.dtype == tensor.dtype
    assert recv_buffer is not tensor
    mock_cast_nd.assert_not_called()


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor", return_value=True)
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._cast_tensor_to_fractal_nd")
def test_prepare_hccl_recv_buffer_allocates_nd_for_3d_fractal_nz(mock_cast_nd, _mock_is_nz):
    tensor = torch.empty(4, 8, 16)

    recv_buffer, restore_fractal_nz = _prepare_hccl_recv_buffer(tensor)

    assert restore_fractal_nz is True
    assert tuple(recv_buffer.shape) == (4, 8, 16)
    mock_cast_nd.assert_not_called()


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._cast_tensor_to_fractal_nz")
def test_finalize_hccl_recv_buffer_restores_fractal_nz(mock_cast_nz):
    tensor = torch.empty(2, 3)
    recv_buffer = torch.empty(2, 3)
    nz_tensor = torch.ones(2, 3)
    mock_cast_nz.return_value = nz_tensor

    _finalize_hccl_recv_buffer(tensor, recv_buffer, restore_fractal_nz=True)

    mock_cast_nz.assert_called_once_with(recv_buffer)
    assert tensor.data is nz_tensor


def test_finalize_hccl_recv_buffer_skips_when_not_fractal_nz():
    tensor = torch.empty(2, 3)
    original = tensor.clone()

    _finalize_hccl_recv_buffer(tensor, tensor, restore_fractal_nz=False)

    assert torch.equal(tensor, original)


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_tensor_roundtrip_uses_expected_formats(mock_format_cast):
    tensor = torch.empty(2, 3)
    nd_tensor = torch.empty(2, 3)
    nz_tensor = torch.empty(2, 3)
    mock_format_cast.side_effect = [nd_tensor, nz_tensor]

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor",
        side_effect=[True, False],
    ):
        assert _cast_tensor_to_fractal_nd(tensor) is nd_tensor
        assert _cast_tensor_to_fractal_nz(nd_tensor) is nz_tensor

    assert mock_format_cast.call_args_list[0].args[1] == 2
    assert mock_format_cast.call_args_list[1].args[1] == 29


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_nz_via_int8_view(mock_format_cast):
    # W4A8: int8 NZ packed as int32 via view; TransData must see int8.
    int8_nz = torch.arange(2 * 4 * 8, dtype=torch.int8).reshape(2, 4, 8)
    packed_int32 = int8_nz.view(torch.int32)
    assert packed_int32.shape == (2, 4, 2)

    mock_format_cast.side_effect = lambda t, _fmt: t.clone()

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor",
        return_value=True,
    ):
        result = _cast_tensor_to_fractal_nd(packed_int32)

    assert result.dtype == torch.int32
    assert tuple(result.shape) == (2, 4, 2)
    mock_format_cast.assert_called_once()
    assert mock_format_cast.call_args.args[0].dtype == torch.int8
    assert tuple(mock_format_cast.call_args.args[0].shape) == (2, 4, 8)
    assert mock_format_cast.call_args.args[1] == 2


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_nd_to_nz_via_int8_view(mock_format_cast):
    packed_int32_nd = torch.arange(2 * 4 * 2, dtype=torch.int32).reshape(2, 4, 2)
    mock_format_cast.side_effect = lambda t, _fmt: t.clone()

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor",
        return_value=False,
    ):
        result = _cast_tensor_to_fractal_nz(packed_int32_nd)

    assert result.dtype == torch.int32
    assert tuple(result.shape) == (2, 4, 2)
    mock_format_cast.assert_called_once()
    cast_input = mock_format_cast.call_args.args[0]
    assert cast_input.dtype == torch.int8
    assert tuple(cast_input.shape) == (2, 4, 8)
    # Materialized clone: must not share storage with the int32 recv buffer.
    assert cast_input.data_ptr() != packed_int32_nd.view(torch.int8).data_ptr()
    assert mock_format_cast.call_args.args[1] == 29


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_via_int8_preserves_logical_int32_shape(mock_format_cast):
    int8_tensor = torch.arange(16, dtype=torch.int8).reshape(2, 8)
    packed = int8_tensor.view(torch.int32)
    mock_format_cast.side_effect = lambda t, _fmt: t.clone()

    result = _cast_packed_int32_via_int8(packed, 2)

    assert result.dtype == torch.int32
    assert tuple(result.shape) == tuple(packed.shape)
    mock_format_cast.assert_called_once()
    assert mock_format_cast.call_args.args[0].dtype == torch.int8
    assert tuple(mock_format_cast.call_args.args[0].shape) == (2, 8)


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_via_int8_skips_contiguous_after_nz_cast(mock_format_cast):
    packed_int32_nd = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    nz_int8 = torch.arange(32, dtype=torch.int8).reshape(2, 16)
    mock_format_cast.return_value = nz_int8

    result = _cast_packed_int32_via_int8(packed_int32_nd, 29)

    assert result.dtype == torch.int32
    assert tuple(result.shape) == (2, 4)
    # NZ pack path must be view-only; do not force contiguous on NZ storage.
    assert result.data_ptr() == nz_int8.data_ptr()


def test_reshape_tensor_to_manifest_shape_noop():
    tensor = torch.empty(2, 3, 4)
    assert reshape_tensor_to_manifest_shape("t", tensor, (2, 3, 4))
    assert tuple(tensor.shape) == (2, 3, 4)


def test_reshape_tensor_to_manifest_shape_views_when_numel_matches():
    tensor = torch.empty(24)
    assert reshape_tensor_to_manifest_shape("t", tensor, (2, 3, 4))
    assert tuple(tensor.shape) == (2, 3, 4)


def test_reshape_tensor_to_manifest_shape_rejects_numel_mismatch():
    tensor = torch.empty(24)
    assert not reshape_tensor_to_manifest_shape("t", tensor, (2, 3, 5))


def test_reshape_transfer_items_to_manifest_by_name():
    weight = torch.empty(24)
    scale = torch.empty(8)
    items = [("layer.weight", weight), ("layer.scale", scale)]
    manifest = {"layer.weight": (2, 3, 4), "layer.scale": (8,)}
    assert reshape_transfer_items_to_manifest(items, manifest)
    assert tuple(weight.shape) == (2, 3, 4)
    assert tuple(scale.shape) == (8,)


def test_parse_transfer_shape_manifest():
    raw = {"a.weight": [2, 3], "b.weight": [4]}
    assert _parse_transfer_shape_manifest(raw) == {"a.weight": (2, 3), "b.weight": (4,)}


def test_recv_json_message_reads_large_payload():
    import socket

    payload = {
        "label": "JOIN_ACK",
        "content": {
            "name": "127.0.0.1:1234",
            "transfer_count": 2,
            "transfer_shapes": {"a": [2, 3], "b": [4, 5]},
        },
    }
    encoded = json.dumps(payload).encode("utf-8")

    sender, receiver = socket.socketpair()
    try:
        sender.sendall(encoded)
        sender.close()
        received = _recv_json_message(receiver)
    finally:
        receiver.close()

    assert received == payload


if __name__ == "__main__":
    pytest.main()
