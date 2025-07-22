from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.func_wrapper import (wrapper_rmsnorm_forward_oot,
                                                   wrapper_rmsnorm_init)


class MockRMSNorm:

    def __init__(self, hidden_size: int, **extra_args):
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size)
        self.input_scale = 1.0
        self.input_offset = 0.0
        self.variance_epsilon = 1e-6
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                       requires_grad=False)
        self.ignore_anti = extra_args.get('ignore_anti', True)


class TestFuncWrapper(TestBase):

    def test_wrapper_rmsnorm_init(self):

        @wrapper_rmsnorm_init
        def init(self, hidden_size: int, **extra_args) -> None:
            self.hidden_size = hidden_size

        hidden_size = 128
        extra_args = {'arg1': 'value1'}

        rms_norm = MockRMSNorm(hidden_size, **extra_args)
        init(rms_norm, hidden_size, **extra_args)

        self.assertTrue(hasattr(rms_norm, 'ignore_anti'))
        self.assertTrue(rms_norm.ignore_anti)

        self.assertTrue(hasattr(rms_norm, 'bias'))
        self.assertIsInstance(rms_norm.bias, torch.nn.Parameter)
        self.assertEqual(rms_norm.bias.shape, torch.Size([hidden_size]))
        self.assertFalse(rms_norm.bias.requires_grad)

    @patch('torch_npu._npu_quant_rms_norm')
    def test_wrapper_rmsnorm_forward_oot_with_residual(
            self, mock_npu_quant_rms_norm):
        hidden_size = 128
        x = torch.randn(hidden_size)
        residual = torch.randn(hidden_size)
        expected_out = torch.randn(hidden_size)

        mock_npu_quant_rms_norm.return_value = (expected_out, residual)

        @wrapper_rmsnorm_forward_oot
        def forward_oot(self, x: torch.Tensor, residual: torch.Tensor = None):
            return x, residual

        rms_norm = MockRMSNorm(hidden_size)
        rms_norm.ignore_anti = False

        output, res = forward_oot(rms_norm, x, residual)

        mock_npu_quant_rms_norm.assert_called_once()

        args, kwargs = mock_npu_quant_rms_norm.call_args
        self.assertTrue(torch.equal(args[1], rms_norm.weight))
        self.assertTrue(torch.equal(args[2], rms_norm.bias))
        self.assertEqual(args[3], rms_norm.input_scale)
        self.assertEqual(args[4], rms_norm.input_offset)
        self.assertEqual(args[5], rms_norm.variance_epsilon)
        self.assertTrue(torch.equal(res, residual))

    @patch('torch_npu._npu_quant_rms_norm')
    def test_wrapper_rmsnorm_forward_oot_without_residual(
            self, mock_npu_quant_rms_norm):
        hidden_size = 128
        x = torch.randn(hidden_size)
        expected_out = torch.randn(hidden_size)

        mock_npu_quant_rms_norm.return_value = expected_out

        @wrapper_rmsnorm_forward_oot
        def forward_oot(self, x: torch.Tensor, residual: torch.Tensor = None):
            return x

        rms_norm = MockRMSNorm(hidden_size)
        rms_norm.ignore_anti = False

        output = forward_oot(rms_norm, x)

        mock_npu_quant_rms_norm.assert_called_once()

        args, kwargs = mock_npu_quant_rms_norm.call_args
        self.assertTrue(torch.equal(args[0], x))
        self.assertTrue(torch.equal(args[1], rms_norm.weight))
        self.assertTrue(torch.equal(args[2], rms_norm.bias))
        self.assertEqual(args[3], rms_norm.input_scale)
        self.assertEqual(args[4], rms_norm.input_offset)
        self.assertEqual(args[5], rms_norm.variance_epsilon)

        self.assertTrue(torch.equal(output, expected_out))

    def test_wrapper_rmsnorm_forward_oot_ignore_anti_with_residual(self):
        hidden_size = 128
        x = torch.randn(hidden_size)
        residual = torch.randn(hidden_size)

        @wrapper_rmsnorm_forward_oot
        def forward_oot(self, x: torch.Tensor, residual: torch.Tensor = None):
            return x, residual

        rms_norm = MockRMSNorm(hidden_size)
        rms_norm.ignore_anti = True

        output, res = forward_oot(rms_norm, x, residual)

        self.assertTrue(torch.equal(output, x.add_(rms_norm.bias)))
        self.assertTrue(torch.equal(res, residual))

    def test_wrapper_rmsnorm_forward_oot_ignore_anti_no_residual(self):
        hidden_size = 128
        x = torch.randn(hidden_size)

        @wrapper_rmsnorm_forward_oot
        def forward_oot(self, x: torch.Tensor, residual: torch.Tensor = None):
            return x

        rms_norm = MockRMSNorm(hidden_size)
        rms_norm.ignore_anti = True

        output = forward_oot(rms_norm, x)

        self.assertTrue(torch.equal(output, x.add_(rms_norm.bias)))
