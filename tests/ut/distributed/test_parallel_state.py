from unittest.mock import MagicMock, patch

import pytest
from vllm.config import ParallelConfig

from vllm_ascend.distributed.parallel_state import (
    _LMTP, _MC2, _OTP, destroy_ascend_model_parallel, get_lmhead_tp_group,
    get_mc2_group, get_otp_group, init_ascend_model_parallel)


@pytest.fixture
def parallel_config():
    return ParallelConfig(data_parallel_size=2,
                          tensor_parallel_size=2,
                          pipeline_parallel_size=2)


@pytest.fixture
def mock_distributed():
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=8), \
         patch('torch.distributed.get_backend', return_value='nccl'), \
         patch('vllm_ascend.distributed.parallel_state.get_world_group') as mock_group:
        mock_group.return_value.local_rank = 0
        mock_group.return_value.device_group = MagicMock()
        yield


def test_init_ascend_model_parallel(mock_distributed, parallel_config):
    mock_ascend_config = MagicMock()
    mock_ascend_config.lmhead_tensor_parallel_size = 2
    mock_ascend_config.oproj_tensor_parallel_size = 2
    with patch('vllm_ascend.distributed.parallel_state.model_parallel_initialized', return_value=False), \
         patch('vllm_ascend.distributed.parallel_state.init_model_parallel_group'), \
         patch('vllm_ascend.distributed.parallel_state.get_ascend_config', return_value=mock_ascend_config):
        init_ascend_model_parallel(parallel_config)

        mc2_group = get_mc2_group()
        lmheadtp_group = get_lmhead_tp_group()
        otp_group = get_otp_group()
        assert mc2_group is not None
        assert otp_group is not None
        assert lmheadtp_group is not None

        destroy_ascend_model_parallel()
        assert _MC2 is None
        assert _LMTP is None
        assert _OTP is None
