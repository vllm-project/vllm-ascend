from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.eplb.core.eplb_device_transfer_loader as loader


@pytest.fixture
def mock_adaptor():
    adaptor = MagicMock()

    adaptor.expert_map_per_layer_cpu = {0: {10: torch.tensor(1), 20: torch.tensor(0)}}

    adaptor.expert_param_per_layer = {0: {0: [[torch.tensor([1.0])]], 1: [[torch.tensor([2.0])]]}}

    adaptor.expert_weight_key_per_layer = {0: "weight_key"}
    adaptor.buffer_tensor_list = {
        "weight_key": [[torch.tensor([3.0]), torch.tensor([4.0])], [torch.tensor([5.0]), torch.tensor([6.0])]]
    }
    return adaptor


@pytest.mark.parametrize("dtype_name", ["uint8", "float4_e2m1fn_x2"])
def test_hccl_p2p_tensor_uses_shared_int8_view(dtype_name):
    tensor = torch.arange(16, dtype=torch.uint8)[4:12]
    if dtype_name == "float4_e2m1fn_x2":
        native_fp4 = getattr(torch, dtype_name, None)
        if native_fp4 is None:
            pytest.skip("native FP4 dtype is unavailable")
        tensor = tensor.view(native_fp4)

    original_dtype = tensor.dtype
    transport_tensor = loader._as_hccl_p2p_tensor(tensor)

    assert transport_tensor.dtype == torch.int8
    assert transport_tensor.shape == tensor.shape
    assert transport_tensor.stride() == tensor.stride()
    assert transport_tensor.storage_offset() == tensor.storage_offset()
    assert transport_tensor.data_ptr() == tensor.data_ptr()
    assert tensor.dtype == original_dtype


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64])
def test_hccl_p2p_tensor_keeps_supported_dtype(dtype):
    tensor = torch.zeros(4, dtype=dtype)

    assert loader._as_hccl_p2p_tensor(tensor) is tensor


def test_generate_task_uses_int8_transport_views_for_byte_tensors():
    send_tensor = torch.arange(8, dtype=torch.uint8)
    recv_tensor = torch.empty(8, dtype=torch.uint8)
    adaptor = MagicMock()
    adaptor.expert_map_per_layer_cpu = {0: {10: torch.tensor(0)}}
    adaptor.expert_param_per_layer = {0: {0: [send_tensor]}}
    adaptor.expert_weight_key_per_layer = {0: "weight_key"}
    adaptor.buffer_tensor_list = {"weight_key": [[recv_tensor]]}

    comm_group = MagicMock()
    comm_group.ranks = {1: 10, 2: 20}
    comm_group.device_group = object()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=comm_group):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(adaptor)

    with patch("torch.distributed.P2POp") as mock_p2p:
        loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)], {20: torch.tensor(0)}, 0)

    send_transport_tensor = mock_p2p.call_args_list[0].args[1]
    recv_transport_tensor = mock_p2p.call_args_list[1].args[1]
    assert send_transport_tensor.dtype == torch.int8
    assert recv_transport_tensor.dtype == torch.int8
    assert send_transport_tensor.data_ptr() == send_tensor.data_ptr()
    assert recv_transport_tensor.data_ptr() == recv_tensor.data_ptr()
    assert send_tensor.dtype == torch.uint8
    assert recv_tensor.dtype == torch.uint8


def test_generate_task_and_state_flow(mock_adaptor):
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    with (
        patch("torch.distributed.P2POp") as mock_p2p,
        patch("torch.distributed.isend", return_value="isend_op"),
        patch("torch.distributed.irecv", return_value="irecv_op"),
    ):
        mock_p2p.side_effect = lambda op, tensor, rank: (op, tensor, rank)

        loader_obj.state = loader.ExpertWeightUpdateState.READY
        loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)], {20: torch.tensor(0)}, 0)
        assert loader_obj.comm_op_list is None
        loader_obj.state = loader.ExpertWeightUpdateState.WAITING

        loader_obj.generate_expert_d2d_transfer_task([], [], {}, 0)
        assert not loader_obj.comm_op_list
        assert loader_obj.state == loader.ExpertWeightUpdateState.READY


def test_generate_task_uses_layer_weight_key_buffer(mock_adaptor):
    comm_group = MagicMock()
    comm_group.ranks = {2: 20}
    comm_group.device_group = object()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=comm_group):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    with (
        patch("torch.distributed.P2POp") as mock_p2p,
        patch("torch.distributed.irecv", return_value="irecv_op"),
    ):
        mock_p2p.side_effect = lambda op, tensor, rank, group=None: (op, tensor, rank, group)
        loader_obj.generate_expert_d2d_transfer_task([], [(2, 20)], {20: torch.tensor(0)}, 0)

    assert mock_p2p.call_args_list[0].args[1] is mock_adaptor.buffer_tensor_list["weight_key"][0][0]
    assert mock_p2p.call_args_list[1].args[1] is mock_adaptor.buffer_tensor_list["weight_key"][0][1]


def test_asyn_transfer_and_update(mock_adaptor):
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    loader_obj.comm_op_list = ["fake_op"]
    loader_obj.state = loader.ExpertWeightUpdateState.READY

    reqs: list[MagicMock] = []

    with patch("torch.distributed.batch_isend_irecv", return_value=[MagicMock(), MagicMock()]):
        loader_obj.asyn_expert_weight_transfer(reqs)

    assert loader_obj.state == loader.ExpertWeightUpdateState.TRANSFERRING
    assert len(reqs) > 0

    mock_req = MagicMock()
    mock_req.wait.return_value = None
    reqs = [mock_req]

    loader_obj.recv_expert_list = [(0, 0)]
    loader_obj.updated_expert_map = {20: torch.tensor(0)}
    loader_obj.updated_log2phy_map = {"dummy": 1}
    loader_obj.layer_id = 0
    loader_obj.comm_op_list = ["op"]

    loader_obj.update_expert_map_and_weight(reqs)

    mock_adaptor.do_update_expert_map.assert_called_once()
    mock_adaptor.do_update_log2phy_map.assert_called_once()
    mock_adaptor.do_update_expert_weight.assert_called_once()

    assert loader_obj.state == loader.ExpertWeightUpdateState.WAITING
    assert loader_obj.recv_expert_list == []


def test_set_log2phy_map(mock_adaptor):
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)
    loader_obj.set_log2phy_map({"a": 1})
    assert loader_obj.updated_log2phy_map == {"a": 1}


def test_invalid_state_asyn_update(mock_adaptor):
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    loader_obj.state = loader.ExpertWeightUpdateState.WAITING
    reqs: list[Any] = []
    loader_obj.asyn_expert_weight_transfer(reqs)
    assert reqs == []

    loader_obj.state = loader.ExpertWeightUpdateState.READY
    loader_obj.update_expert_map_and_weight([])

    assert not mock_adaptor.do_update_expert_map.called
