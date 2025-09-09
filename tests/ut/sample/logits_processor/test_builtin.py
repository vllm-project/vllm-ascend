import torch
from pytest_mock import MockerFixture
from vllm.config import SchedulerConfig, VllmConfig

from tests.ut.base import PytestBase
from vllm_ascend.sample.logits_processor import AscendMinPLogitsProcessor


class TestMinPLogitsProcessorInitFunc(PytestBase):

    def test_init_func_with_decode_max_num_seqs(self, mocker: MockerFixture):
        device_cpu = torch.device("cpu")
        device_npu = torch.device("npu")
        is_pin_memory = False
        mock_vllm_config = mocker.MagicMock(spec=VllmConfig)
        mock_scheduler_config = mocker.MagicMock(spec=SchedulerConfig)
        mock_scheduler_config.decode_max_num_seqs = 0
        mock_scheduler_config.max_num_seqs = 128
        mock_vllm_config.scheduler_config = mock_scheduler_config
        # torch.zeros/torch.empty returns error on online ut machine, so mock it
        mock_tensor = torch.zeros((256, ),
                                  dtype=torch.float32,
                                  pin_memory=False)
        mocker.patch("torch.zeros", return_value=mock_tensor)
        mock_empty_tensor = torch.empty((256, ), dtype=torch.float32)
        mocker.patch("torch.empty", return_value=mock_empty_tensor)

        processor_cpu = AscendMinPLogitsProcessor(mock_vllm_config, device_cpu,
                                                  is_pin_memory)

        assert processor_cpu.min_p is not None
        assert processor_cpu.use_double_tensor is False
        assert processor_cpu.min_p_cpu.shape[0] == 256

        processor_cpu = AscendMinPLogitsProcessor(mock_vllm_config, device_npu,
                                                  is_pin_memory)

        assert processor_cpu.min_p is not None
        assert processor_cpu.use_double_tensor is True
        assert processor_cpu.min_p_cpu.shape[0] == 256
