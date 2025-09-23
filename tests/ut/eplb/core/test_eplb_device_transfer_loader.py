from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.eplb.core.eplb_device_transfer_loader as loader


@pytest.fixture
def mock_adaptor():
    adaptor = MagicMock()

    adaptor.expert_map_per_layer_cpu = {
        0: {
            10: torch.tensor(1),
            20: torch.tensor(0)
        }
    }

    adaptor.expert_param_per_layer = {
        0: {
            0: [[torch.tensor([1.0])]],
            1: [[torch.tensor([2.0])]]
        }
    }

    adaptor.buffer_tensor_list = [[[torch.tensor([3.0])],
                                   [torch.tensor([4.0])]]]
    return adaptor


def test_generate_task_and_state_flow(mock_adaptor):
    loader_obj = loader.D2DExpertWeightLoader(mock_adaptor)


    with patch("torch.distributed.P2POp") as mock_p2p, \
         patch("torch.distributed.isend", return_value="isend_op"), \
         patch("torch.distributed.irecv", return_value="irecv_op"):

        mock_p2p.side_effect = lambda op, tensor, rank: (op, tensor, rank)

        loader_obj.state = loader.ExpertWeightUpdateState.READY
        loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)],
                                                     {20: torch.tensor(0)}, 0)
        assert loader_obj.comm_op_list is None
        loader_obj.state = loader.ExpertWeightUpdateState.WAITING

        loader_obj.generate_expert_d2d_transfer_task([], [], {}, 0)
        assert loader_obj.comm_op_list is None

        updated_map = {20: torch.tensor(0)}
        loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)],
                                                     updated_map, 0)
        assert loader_obj.state == loader.ExpertWeightUpdateState.READY
        assert loader_obj.comm_op_list
        assert loader_obj.recv_expert_list


def test_asyn_transfer_and_update(mock_adaptor):
    loader_obj = loader.D2DExpertWeightLoader(mock_adaptor)

    loader_obj.comm_op_list = ["fake_op"]
    loader_obj.state = loader.ExpertWeightUpdateState.READY

    # 给 reqs 添加类型注解，初始为空列表
    reqs: list[MagicMock] = []

    with patch("torch.distributed.batch_isend_irecv",
               return_value=[MagicMock(), MagicMock()]):
        loader_obj.asyn_expert_weight_transfer(reqs)

    assert loader_obj.state == loader.ExpertWeightUpdateState.TRANSFERRING
    assert len(reqs) > 0  # 这里不直接判断 "req1"，因为现在用 MagicMock 代替字符串

    # 创建单个 mock 请求
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
    loader_obj = loader.D2DExpertWeightLoader(mock_adaptor)
    loader_obj.set_log2phy_map({"a": 1})
    assert loader_obj.updated_log2phy_map == {"a": 1}


def test_invalid_state_asyn_update(mock_adaptor):
    loader_obj = loader.D2DExpertWeightLoader(mock_adaptor)

    loader_obj.state = loader.ExpertWeightUpdateState.WAITING
    reqs: list[Any] = []
    loader_obj.asyn_expert_weight_transfer(reqs)
    assert reqs == []

    loader_obj.state = loader.ExpertWeightUpdateState.READY
    loader_obj.update_expert_map_and_weight([])

    assert not mock_adaptor.do_update_expert_map.called


def test_load_impl_not_implemented(mock_adaptor):
    loader_obj = loader.D2DExpertWeightLoader(mock_adaptor)
    with pytest.raises(NotImplementedError):
        loader_obj.load_impl({}, {})
