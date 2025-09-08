from pytest_mock import MockerFixture
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.sample.logits_processor import MinPLogitsProcessor

from tests.ut.base import PytestBase
from vllm_ascend.ops.min_p_logits_processor import \
    min_p_logits_processor_init_func


class TestMinPLogitsProcessorInitFunc(PytestBase):

    def test_init_func_without_decode_max_num_seqs(self,
                                                   mocker: MockerFixture):
        mock_min_p_logits_processor = mocker.MagicMock(
            spec=MinPLogitsProcessor)

        min_p_logits_processor_init_func(mock_min_p_logits_processor,
                                         VllmConfig(), "cpu:0", True)

        assert mock_min_p_logits_processor.min_p_cpu is not None
        assert mock_min_p_logits_processor.min_p_device is not None
        assert mock_min_p_logits_processor.min_p_cpu.shape[0] == 128

    def test_init_func_with_decode_max_num_seqs_and_npu(
            self, mocker: MockerFixture):
        mock_min_p_logits_processor = mocker.MagicMock(
            spec=MinPLogitsProcessor)

        mock_vllm_config = mocker.MagicMock(spec=VllmConfig)
        mock_scheduler_config = mocker.MagicMock(spec=SchedulerConfig)
        mock_scheduler_config.decode_max_num_seqs = 256
        mock_scheduler_config.max_num_seqs = 128
        mock_vllm_config.scheduler_config = mock_scheduler_config
        mocker.patch(
            "vllm_ascend.ops.min_p_logits_processor.get_current_vllm_config",
            return_value=mock_vllm_config)

        min_p_logits_processor_init_func(mock_min_p_logits_processor,
                                         mock_vllm_config, "npu:0", True)

        assert mock_min_p_logits_processor.min_p_cpu.shape[0] == 256
        assert mock_min_p_logits_processor.use_double_tensor is True

    def test_init_func_with_decode_max_num_seqs_and_cpu(
            self, mocker: MockerFixture):
        mock_min_p_logits_processor = mocker.MagicMock(
            spec=MinPLogitsProcessor)

        mock_vllm_config = mocker.MagicMock(spec=VllmConfig)
        mock_scheduler_config = mocker.MagicMock(spec=SchedulerConfig)
        mock_scheduler_config.max_num_seqs = 128
        mock_scheduler_config.decode_max_num_seqs = 256
        mock_vllm_config.scheduler_config = mock_scheduler_config
        mocker.patch(
            "vllm_ascend.ops.min_p_logits_processor.get_current_vllm_config",
            return_value=mock_vllm_config)

        min_p_logits_processor_init_func(mock_min_p_logits_processor,
                                         mock_vllm_config, "cpu:0", True)

        assert mock_min_p_logits_processor.use_double_tensor is False
