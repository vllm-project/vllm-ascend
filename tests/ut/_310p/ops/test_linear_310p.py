import math
from unittest.mock import MagicMock, patch

import torch
from vllm_ascend._310p.ops.linear import (
    AscendMergedColumnParallelLinear310,
    AscendReplicatedLinear310,
    AscendRowParallelLinear310,
    AscendUnquantizedLinearMethod310,
)

from tests.ut.base import TestBase

_MLP_ALIGN = 32


def _align_up(value: int, alignment: int = _MLP_ALIGN) -> int:
    return int(math.ceil(value / alignment) * alignment)


class TestAscendUnquantizedLinearMethod310(TestBase):
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_calls_nz(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, fmt: x

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.data = torch.randn(4, 4, dtype=torch.float16)
        layer._enable_nz = True
        layer.prefix = "mlp.gate_up_proj"

        method = AscendUnquantizedLinearMethod310()
        method.process_weights_after_loading(layer)

        mock_format_cast.assert_called_once()

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_skips_conv1d(self, mock_format_cast):
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.data = torch.randn(4, 4, dtype=torch.float16)
        layer._enable_nz = True
        layer.prefix = "conv1d"

        method = AscendUnquantizedLinearMethod310()
        method.process_weights_after_loading(layer)

        mock_format_cast.assert_not_called()


class TestAscendMergedColumnParallelLinear310(TestBase):
    def test_gate_up_output_padding_alignment(self):
        linear = AscendMergedColumnParallelLinear310(
            input_size=16,
            output_sizes=[33, 33],
            prefix="gate_up_proj",
            disable_tp=True,
        )

        parts_pad = list(map(int, linear.weight.parts_pad))
        expected_part_pad = _align_up(33, _MLP_ALIGN)

        self.assertEqual(parts_pad, [expected_part_pad, expected_part_pad])
        self.assertEqual(int(linear.weight.out_pad), expected_part_pad * 2)


class TestAscendRowParallelLinear310(TestBase):
    def test_down_proj_input_padding_alignment(self):
        linear = AscendRowParallelLinear310(
            input_size=33,
            output_size=16,
            prefix="down_proj",
            disable_tp=True,
        )

        in_real = int(linear.weight.in_real)
        in_pad = int(linear.weight.in_pad)

        self.assertEqual(in_real, 33)
        self.assertEqual(in_pad, _align_up(33, _MLP_ALIGN))

    def test_non_down_proj_no_input_padding(self):
        linear = AscendRowParallelLinear310(
            input_size=33,
            output_size=16,
            prefix="o_proj",
            disable_tp=True,
        )

        in_real = int(linear.weight.in_real)
        in_pad = int(linear.weight.in_pad)

        self.assertEqual(in_real, 33)
        self.assertEqual(in_pad, 33)


class TestAscendReplicatedLinear310(TestBase):
    def test_init_quant_method(self):
        linear = AscendReplicatedLinear310(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod310))

    def test_init_quant_method_disable_tp(self):
        linear = AscendReplicatedLinear310(
            input_size=16,
            output_size=8,
            disable_tp=True,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod310))
