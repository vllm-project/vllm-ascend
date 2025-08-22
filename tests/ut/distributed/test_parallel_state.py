from unittest.mock import MagicMock, patch

import pytest
from vllm.config import ParallelConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.parallel_state import (
    _MC2, _MLP_TP, _OTP, destroy_ascend_model_parallel, get_mc2_group,
    get_mlp_tp_group, get_otp_group, init_ascend_model_parallel)


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
    with patch('vllm_ascend.distributed.parallel_state.model_parallel_initialized', return_value=False), \
         patch('vllm_ascend.distributed.parallel_state.init_model_parallel_group'):
        parallel_config.oproj_tensor_parallel_size = 2
        envs_ascend.VLLM_ASCEND_ENABLE_MLP_OPTIMIZE = True
        init_ascend_model_parallel(parallel_config)

        mc2_group = get_mc2_group()
        assert mc2_group is not None
        otp_group = get_otp_group()
        assert otp_group is not None
        mlp_tp_group = get_mlp_tp_group()
        assert mlp_tp_group is not None

        destroy_ascend_model_parallel()
        assert _MC2 is None
        assert _OTP is None
        assert _MLP_TP is None
