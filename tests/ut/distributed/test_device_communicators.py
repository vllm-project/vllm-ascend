from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.utils import StatelessProcessGroup

from vllm_ascend.distributed.device_communicators.pyhccl import \
    PyHcclCommunicator


class MockHcclUniqueId:

    def __init__(self, internal=None):
        self.internal = internal or [0] * 128


class MockStatelessUniqueId:

    def __init__(self, internal=None):
        self.internal = internal or [0] * 128

    def __getstate__(self):
        return {'internal': self.internal}

    def __setstate__(self, state):
        self.__dict__.update(state)


@pytest.fixture
def mock_dist():
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_backend', return_value='nccl'), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.get_world_size', return_value=2), \
         patch('torch.distributed.broadcast') as mock_broadcast:
        yield {
            'is_initialized': torch.distributed.is_initialized,
            'get_backend': torch.distributed.get_backend,
            'get_rank': torch.distributed.get_rank,
            'get_world_size': torch.distributed.get_world_size,
            'broadcast': mock_broadcast
        }


@pytest.fixture
def mock_stateless_group():
    group = MagicMock(spec=StatelessProcessGroup)
    group.rank = 0
    group.world_size = 2

    class MockBroadcastObj:

        def __init__(self):
            self.unique_id = MockStatelessUniqueId()

        def __call__(self, obj, src):
            return self.unique_id

    group.broadcast_obj.return_value = MockStatelessUniqueId()
    return group


@pytest.fixture
def mock_hccl_library():
    with patch(
            'vllm_ascend.distributed.device_communicators.pyhccl.HCCLLibrary'
    ) as mock_lib:
        instance = MagicMock()
        instance.hcclGetUniqueId.return_value = MockHcclUniqueId()
        instance.hcclCommInitRank.return_value = MagicMock()
        instance.hcclAllReduce.return_value = None
        instance.hcclBroadcast.return_value = None
        mock_lib.return_value = instance
        yield instance


@pytest.fixture
def mock_current_stream():
    with patch('vllm_ascend.utils.current_stream') as mock_stream:
        stream = MagicMock()
        stream.npu_stream = 123
        mock_stream.return_value = stream
        yield stream


patch_get_process_group_ranks = patch(
    'torch.distributed.get_process_group_ranks', return_value={
        0: 0,
        1: 1
    })


def apply_patch_with_args(*test_funcs):
    for func in test_funcs:
        func = patch_get_process_group_ranks(func)
    return test_funcs


@patch_get_process_group_ranks
def test_init_with_process_group(mock_get_pgr, mock_dist, mock_hccl_library,
                                 mock_current_stream):
    group = MagicMock()
    comm = PyHcclCommunicator(group, device=0)
    assert comm.rank == 0
    assert comm.world_size == 2
    assert not comm.disabled
    mock_hccl_library.hcclCommInitRank.assert_called_once()


@patch_get_process_group_ranks
def test_all_reduce_disabled(mock_get_pgr, mock_dist, mock_hccl_library):
    comm = PyHcclCommunicator(MagicMock(), device=0)
    comm.disabled = True
    result = comm.all_reduce(torch.rand(1))
    assert result is None


@patch_get_process_group_ranks
def test_all_reduce_device_mismatch(mock_get_pgr, mock_dist,
                                    mock_hccl_library):
    comm = PyHcclCommunicator(MagicMock(), device=0)
    tensor = torch.rand(1)
    with pytest.raises(AssertionError):
        comm.all_reduce(tensor)


@patch_get_process_group_ranks
def test_all_reduce_normal(mock_get_pgr, mock_dist, mock_hccl_library,
                           mock_current_stream):
    comm = PyHcclCommunicator(MagicMock(), device=0)
    tensor = torch.rand(1, device=comm.device)
    out_tensor = comm.all_reduce(tensor)
    assert out_tensor is not None
    mock_hccl_library.hcclAllReduce.assert_called()


@patch_get_process_group_ranks
def test_broadcast_device_mismatch(mock_get_pgr, mock_dist, mock_hccl_library):
    comm = PyHcclCommunicator(MagicMock(), device=0)
    tensor = torch.rand(1)
    with pytest.raises(AssertionError):
        comm.broadcast(tensor, src=0)


@patch_get_process_group_ranks
def test_broadcast_normal(mock_get_pgr, mock_dist, mock_hccl_library,
                          mock_current_stream):
    comm = PyHcclCommunicator(MagicMock(), device=0)
    tensor = torch.rand(1, device=comm.device)
    comm.broadcast(tensor, src=0)
    mock_hccl_library.hcclBroadcast.assert_called()


@patch_get_process_group_ranks
def test_init_with_custom_library_path(mock_get_pgr, mock_dist,
                                       mock_hccl_library):
    library_path = "/custom/path/to/hccl.so"
    comm = PyHcclCommunicator(MagicMock(), device=0, library_path=library_path)
    assert isinstance(comm.hccl, MagicMock)


@patch_get_process_group_ranks
def test_init_with_stateless_group(mock_get_pgr, mock_stateless_group,
                                   mock_hccl_library):
    comm = PyHcclCommunicator(mock_stateless_group, device="npu:1")
    assert comm.rank == 0
    assert comm.world_size == 2
    assert not comm.disabled


@patch_get_process_group_ranks
def test_init_world_size_1(mock_get_pgr, mock_dist):
    mock_dist['get_world_size'].return_value = 1
    comm = PyHcclCommunicator(MagicMock(), device=0)
    assert comm.disabled
    assert not comm.available


@patch_get_process_group_ranks
def test_init_hccl_load_fail(mock_get_pgr, mock_dist):
    with patch(
            'vllm_ascend.distributed.device_communicators.pyhccl.HCCLLibrary',
            side_effect=OSError("Load failed")):
        comm = PyHcclCommunicator(MagicMock(), device=0)
        assert comm.disabled
        assert not comm.available
