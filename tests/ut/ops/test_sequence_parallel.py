import unittest
import torch
from unittest.mock import patch

from vllm_ascend.ops.sequence_parallel import init_metadata_for_flashcomm2


class TestInitMetadataForFlashcomm2(unittest.TestCase):

    def setUp(self):
        patcher = patch("vllm_ascend.ops.sequence_parallel.MetadataForPadding")
        self.MockMetadata = patcher.start()
        self.addCleanup(patcher.stop)

    def _run_case(self, tp_size, input_len, expected_padding_flag, expected_pad_size):
        with patch(
            "vllm_ascend.ops.sequence_parallel.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ):
            input_ids = torch.arange(input_len)

            result = init_metadata_for_flashcomm2(input_ids)

            # 验证 MetadataForPadding 调用参数
            self.MockMetadata.assert_called_once_with(
                lengths_sum_unpadding=input_len,
                lengths_sum_padding=((input_len + tp_size - 1) // tp_size) * tp_size,
                padding_flag=expected_padding_flag,
                pad_size=expected_pad_size,
                not_dummy_and_is_prefill=False,
            )

            # 验证返回值
            self.assertEqual(result, self.MockMetadata.return_value)

    def test_no_padding(self):
        self._run_case(tp_size=4, input_len=8, expected_padding_flag=False, expected_pad_size=0)

    def test_with_padding(self):
        self._run_case(tp_size=4, input_len=10, expected_padding_flag=True, expected_pad_size=2)

    def test_with_padding_non_multiple(self):
        self._run_case(tp_size=3, input_len=7, expected_padding_flag=True, expected_pad_size=2)

    def test_exact_multiple(self):
        self._run_case(tp_size=5, input_len=5, expected_padding_flag=False, expected_pad_size=0)

    def test_empty_input(self):
        self._run_case(tp_size=4, input_len=0, expected_padding_flag=False, expected_pad_size=0)