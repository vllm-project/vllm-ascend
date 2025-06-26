import unittest
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.parallel_state import GroupCoordinator

import vllm_ascend
from vllm_ascend.distributed.parallel_state import (
    destory_ascend_model_parallel, get_ep_group, get_etp_group,
    init_ascend_model_parallel, model_parallel_initialized)


class TestParallelState(unittest.TestCase):

    @patch('vllm_ascend.distributed.parallel_state._EP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_get_ep_group_when_initialized(self, mock_ep):
        # Act
        result = get_ep_group()

        # Assert
        assert isinstance(result, GroupCoordinator)

    @patch('vllm_ascend.distributed.parallel_state._EP', None)
    def test_get_ep_group_when_not_initialized(self):
        # Act & Assert
        with pytest.raises(AssertionError) as excinfo:
            get_ep_group()
        assert "expert model parallel group is not initialized" in str(
            excinfo.value)

    @patch('vllm_ascend.distributed.parallel_state._ETP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_get_etp_group_when_initialized(self, mock_etp):
        # Act
        result = get_etp_group()

        # Assert
        assert isinstance(result, GroupCoordinator)

    @patch('vllm_ascend.distributed.parallel_state._ETP', None)
    def test_get_etp_group_when_not_initialized(self):
        # Act & Assert
        with pytest.raises(AssertionError) as excinfo:
            get_etp_group()
        assert "expert tensor parallel group is not initialized" in str(
            excinfo.value)

    @patch('vllm_ascend.distributed.parallel_state._ETP', None)
    @patch('vllm_ascend.distributed.parallel_state._EP', None)
    def test_model_parallel_initialized_when_both_none(self):
        # Act & Assert
        assert not model_parallel_initialized()

    @patch('vllm_ascend.distributed.parallel_state._ETP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm_ascend.distributed.parallel_state._EP', None)
    def test_model_parallel_initialized_when_ep_none(self, mock_etp):
        # Act & Assert
        assert not model_parallel_initialized()

    @patch('vllm_ascend.distributed.parallel_state._ETP', None)
    @patch('vllm_ascend.distributed.parallel_state._EP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_model_parallel_initialized_when_etp_none(self, mock_ep):
        # Act & Assert
        assert not model_parallel_initialized()

    @patch('vllm_ascend.distributed.parallel_state._ETP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm_ascend.distributed.parallel_state._EP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_model_parallel_initialized_when_etp_initialized(
            self, mock_ep, mock_etp):
        # Act & Assert
        assert model_parallel_initialized()

    @patch('vllm_ascend.distributed.parallel_state._ETP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm_ascend.distributed.parallel_state._EP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_destroy_when_both_exist(self, mock_ep, mock_etp):
        # Act
        destory_ascend_model_parallel()
        # Assert
        mock_ep.destroy.assert_called_once()
        mock_etp.destroy.assert_called_once()
        assert vllm_ascend.distributed.parallel_state._ETP is None
        assert vllm_ascend.distributed.parallel_state._EP is None

    @patch('vllm_ascend.distributed.parallel_state._ETP', None)
    @patch('vllm_ascend.distributed.parallel_state._EP',
           new_callable=lambda: MagicMock())
    def test_destory_ascend_model_parallel_when_etp_none(self, mock_ep):
        # Act
        destory_ascend_model_parallel()
        # Assert
        mock_ep.destroy.assert_called_once()
        assert vllm_ascend.distributed.parallel_state._EP is None
        assert vllm_ascend.distributed.parallel_state._ETP is None

    @patch('vllm_ascend.distributed.parallel_state._ETP',
           new_callable=lambda: MagicMock())
    @patch('vllm_ascend.distributed.parallel_state._EP', None)
    def test_destory_ascend_model_parallel_when_ep_none(self, mock_etp):
        # Act
        destory_ascend_model_parallel()
        # Assert
        mock_etp.destroy.assert_called_once()
        assert vllm_ascend.distributed.parallel_state._ETP is None
        assert vllm_ascend.distributed.parallel_state._EP is None

    @patch('vllm_ascend.distributed.parallel_state._ETP', None)
    @patch('vllm_ascend.distributed.parallel_state._EP', None)
    def test_destory_ascend_model_parallel_when_both_none(self):
        # Act
        destory_ascend_model_parallel()
        # Assert
        assert vllm_ascend.distributed.parallel_state._ETP is None
        assert vllm_ascend.distributed.parallel_state._EP is None

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('vllm_ascend.distributed.parallel_state.get_world_group',
           return_value=MagicMock(device_group='npu:0', local_rank=0))
    @patch('torch.distributed.get_backend', return_value='hccl')
    @patch('vllm_ascend.distributed.parallel_state.init_model_parallel_group')
    @patch('vllm_ascend.distributed.parallel_state.model_parallel_initialized',
           return_value=False)
    def test_init_ascend_model_parallel_normal_case(
            self, mock_mp_init, mock_init_group, mock_get_backend,
            mock_world_group, mock_get_world_size, mock_is_init):
        """Test normal initialization with default parameters"""
        # Act
        init_ascend_model_parallel()
        # Assert
        mock_init_group.assert_any_call([[0, 1, 2, 3, 4, 5, 6, 7]],
                                        0,
                                        'hccl',
                                        group_name="ep")
        mock_init_group.assert_any_call([[0]], 0, 'hccl', group_name="etp")
        self.assertIsNotNone(vllm_ascend.distributed.parallel_state._EP)
        self.assertIsNotNone(vllm_ascend.distributed.parallel_state._ETP)

    @patch('vllm_ascend.distributed.parallel_state.model_parallel_initialized',
           return_value=True)
    def test_init_ascend_model_parallel_skip_if_initialized(
            self, mock_mp_init):
        """Test skipping when model parallel already initialized"""
        with patch.object(vllm_ascend.distributed.parallel_state,
                          '_EP') as mock_ep, patch.object(
                              vllm_ascend.distributed.parallel_state,
                              '_ETP') as mock_etp:
            # Act
            init_ascend_model_parallel()
            # Assert
            mock_ep.assert_not_called()
            mock_etp.assert_not_called()

    @patch('torch.distributed.is_initialized', return_value=False)
    def test_init_ascend_model_parallel_assert_dist_not_init(
            self, mock_is_init):
        """Test assertion when distributed not initialized"""
        # Act & Assert
        with self.assertRaises(AssertionError):
            init_ascend_model_parallel()

    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('vllm_ascend.distributed.parallel_state.get_world_group',
           return_value=MagicMock(device_group='npu:0', local_rank=1))
    @patch('torch.distributed.get_backend', return_value='hccl')
    @patch('vllm_ascend.distributed.parallel_state.init_model_parallel_group')
    @patch('vllm_ascend.distributed.parallel_state.model_parallel_initialized',
           return_value=False)
    def test_init_ascend_model_parallel_custom_params(
            self, mock_mp_init, mock_init_group, mock_get_backend,
            mock_world_group, mock_get_world_size, mock_is_init):
        """Test initialization with custom parallel sizes"""
        # Act
        init_ascend_model_parallel(expert_parallel_size=2,
                                   expert_tensor_parallel_size=4,
                                   world_size=8,
                                   backend='hccl')
        #Assert
        mock_init_group.assert_any_call([[0, 4], [1, 5], [2, 6], [3, 7]],
                                        1,
                                        'hccl',
                                        group_name="ep")
        mock_init_group.assert_any_call([[0, 1, 2, 3], [4, 5, 6, 7]],
                                        1,
                                        'hccl',
                                        group_name="etp")
