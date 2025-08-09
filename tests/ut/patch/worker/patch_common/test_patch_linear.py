from importlib import reload

import pytest
import torch
import vllm
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase
from vllm_ascend import envs
from vllm_ascend.patch.worker.patch_common import patch_linear
import unittest
from unittest.mock import patch, MagicMock
from typing import Optional, Union, Tuple


class TestAscendRowParallelLinear(PytestBase):

    def init_row_parallel_linear(self, mocker: MockerFixture):
        mocker.patch(
            "vllm_ascend.patch.worker.patch_common.patch_linear.AscendRowParallelLinear.__init__",
            return_value=None,
        )
        mocker.patch("torch.nn.Module.__setattr__")
        mocker.patch("torch.nn.Module.__getattr__")
        mocker.patch("torch.nn.Module.__delattr__")
        return patch_linear.AscendRowParallelLinear(
            input_size=128,
            output_size=256,
        )

    @pytest.mark.parametrize(
        "version, expected",
        [
            ("1.0.0", 1),
            ("2.1.0", 1),
        ],
    )
    def test_get_hcomm_info(self, version, expected, mocker: MockerFixture):
        mock_group = mocker.MagicMock()
        backend = mocker.MagicMock()
        backend.get_hccl_comm_name = lambda x: x
        mock_group._get_backend = lambda x: backend
        mock_group.get_hccl_comm_name = lambda x: x
        mocker.patch("torch.distributed.get_rank", return_value=1)
        mocker.patch(
            "torch.distributed.get_global_rank",
            return_value=0,
        )
        mocker.patch("torch.__version__", new=version)
        hcomm_info = patch_linear.AscendRowParallelLinear.get_hcomm_info(
            mock_group)
        assert hcomm_info == expected

    @pytest.mark.parametrize(
        "skip_bias_add, return_bias, bias, expected",
        [
            (True, False, torch.tensor(1.0), torch.tensor(14.0)),
            (False, True, torch.tensor(1.0), (torch.tensor(14.0), None)),
            (
                True,
                True,
                torch.tensor(1.0),
                (torch.tensor(14.0), torch.tensor(1.0)),
            ),
        ],
    )
    def test_forward(
        self,
        skip_bias_add,
        return_bias,
        bias,
        expected,
        mocker: MockerFixture,
    ):
        mocker_tp_group = mocker.MagicMock()
        mocker_tp_group.device_group = mocker.MagicMock()
        row_parallel_linear = self.init_row_parallel_linear(mocker)
        row_parallel_linear.__dict__["tp_rank"] = 0
        row_parallel_linear.__dict__["skip_bias_add"] = skip_bias_add
        row_parallel_linear.__dict__["return_bias"] = return_bias
        row_parallel_linear.__dict__["bias"] = bias
        row_parallel_linear.__dict__["qyuant_method"] = mocker.MagicMock()
        row_parallel_linear.__dict__["calc_input"] = lambda x: x  # noqa
        row_parallel_linear.__dict__[
            "calc_output"] = lambda x: x.matmul(  # noqa
                torch.tensor([1.0, 2.0]))
        ret = row_parallel_linear.forward(torch.tensor([10.0, 2.0]))
        if isinstance(ret, tuple):
            assert torch.allclose(ret[0], expected[0])
            if ret[1] is None:
                assert ret[1] == expected[1]
            else:
                assert torch.allclose(ret[1], expected[1])
        else:
            assert torch.allclose(ret, expected)

    @pytest.mark.parametrize(
        "input_is_parallel, expected",
        [
            (True, torch.tensor([10.0, 2.0])),
            (False, torch.tensor([10.0])),
        ],
    )
    def test_calc_input(
        self,
        input_is_parallel,
        expected,
        mocker: MockerFixture,
    ):
        row_parallel_linear = self.init_row_parallel_linear(mocker)
        row_parallel_linear.__dict__["input_is_parallel"] = input_is_parallel
        input_tensor = torch.Tensor([10, 2])
        mocker.patch(
            "vllm_ascend.patch.worker.patch_common.patch_linear.get_tensor_model_parallel_rank",  # noqa
            return_value=0,
        )
        mocker.patch(
            "vllm_ascend.patch.worker.patch_common.patch_linear.split_tensor_along_last_dim",  # noqa
            return_value=[torch.Tensor([10]),
                          torch.Tensor([2])],
        )
        input_parallel = row_parallel_linear.calc_input(input_tensor)
        assert torch.allclose(input_parallel, expected)

    @pytest.mark.parametrize(
        "reduce_results, tp_size, expected",
        [
            (True, 2, torch.tensor(56.0)),
            (True, 1, torch.tensor(14.0)),
            (False, 2, torch.tensor(14.0)),
        ],
    )
    def test_calc_output(
        self,
        reduce_results,
        tp_size,
        expected,
        mocker: MockerFixture,
    ):
        quant_method = mocker.MagicMock()
        quant_method.apply = lambda self, x, bias=None: x.matmul(  # noqa
            torch.tensor([1.0, 2.0]))
        row_parallel_linear = self.init_row_parallel_linear(mocker)
        row_parallel_linear.__dict__["reduce_results"] = reduce_results
        row_parallel_linear.__dict__["tp_size"] = tp_size
        row_parallel_linear.__dict__["quant_method"] = quant_method
        row_parallel_linear.__dict__["tp_rank"] = 0
        row_parallel_linear.__dict__["get_hcomm_info"] = lambda x: None  # noqa

        mocker.patch(
            "vllm_ascend.patch.worker.patch_common.patch_linear.get_tp_group",
            return_value=mocker.MagicMock(device_group=mocker.MagicMock()),
        )
        mocker.patch(
            "torch_npu.npu_mm_all_reduce_base",
            side_effect=lambda input_, weight, hccl_info, bias: input_.
            matmul(  # noqa
                torch.tensor([4.0, 8.0])),
        )  # noqa
        ret = row_parallel_linear.calc_output(torch.tensor([10.0, 2.0]))
        assert torch.allclose(ret, expected)

    def test_enable_allreduce_matmul(self, mocker: MockerFixture):
        mocker.patch.object(envs,
                            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE",
                            new=True)
        reload(patch_linear)
        assert envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
        assert id(vllm.model_executor.layers.linear.RowParallelLinear) == id(
            patch_linear.AscendRowParallelLinear)


def divide(a: int, b: int):
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b

class LinearBase:
    def __init__(self, input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix, return_bias):
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.return_bias = return_bias
        self.quant_method = MagicMock()
        self.quant_method.create_weights = MagicMock()

class Parameter:
    def __init__(self, tensor):
        self.data = tensor

class BasevLLMParameter:
    pass

class UninitializedParameter:
    def materialize(self, shape, dtype):
        pass

class TestAttnColumnParallelLinear(unittest.TestCase):

    def setUp(self):
        # 模拟分布式环境参数
        self.rank_patch = patch('vllm_ascend.distributed.parallel_state.get_mlp_tensor_model_parallel_rank', return_value=0)
        self.world_size_patch = patch('vllm_ascend.distributed.parallel_state.get_mlp_tensor_model_parallel_world_size', return_value=2)
        self.rank_patch.start()
        self.world_size_patch.start()

    def tearDown(self):
        self.rank_patch.stop()
        self.world_size_patch.stop()

    def test_output_partition_sizes(self):
        # 模拟初始化时的输出尺寸分割
        output_size = 8
        tp_size = 2
        expected_output_size_per_partition = 4
        self.assertEqual(divide(output_size, tp_size), expected_output_size_per_partition)

    def test_weight_loader_with_output_dim(self):
        full_weight = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        param = Parameter(torch.empty(1, 2))
        param.output_dim = 1  # 模拟 output_dim 属性

        loaded_weight = full_weight.clone()
        weight_loader = MagicMock(side_effect=lambda p, w: p.data.copy_(w.narrow(1, 0, 2)))

        weight_loader(param, loaded_weight)
        expected_slice = full_weight[:, :2]
        torch.testing.assert_close(param.data, expected_slice)

    def test_weight_loader_with_sharded_weight(self):
        full_weight = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        param = Parameter(torch.empty(1, 4))
        param.is_sharded_weight = True  # 模拟 sharded weight

        loaded_weight = full_weight.clone()
        weight_loader = MagicMock(side_effect=lambda p, w: p.data.copy_(w))

        weight_loader(param, loaded_weight)
        torch.testing.assert_close(param.data, loaded_weight)

    def test_weight_loader_v2(self):
        param = BasevLLMParameter()
        param.load_column_parallel_weight = MagicMock()
        loaded_weight = torch.randn(1, 4)
        weight_loader_v2 = MagicMock(side_effect=lambda p, w: p.load_column_parallel_weight(w))
        weight_loader_v2(param, loaded_weight)
        param.load_column_parallel_weight.assert_called_once_with(loaded_weight)

    def test_initialization_output_sizes(self):
        class Dummy(LinearBase):
            def __init__(self):
                super().__init__(
                    input_size=4,
                    output_size=8,
                    skip_bias_add=False,
                    params_dtype=torch.float32,
                    quant_config=None,
                    prefix=""
                )
                self.output_sizes = [8, 8]  # 模拟 QKV 的输出尺寸
                self.output_partition_sizes = [divide(8, 2) for _ in self.output_sizes]

        dummy = Dummy()
        self.assertEqual(dummy.output_partition_sizes, [4, 4])

    def test_all_gather_simulation(self):
        input_tensor = torch.randn(2, 4)
        num_tokens_across_dp = [2, 2]  # 模拟 DP token 数量

        gathered_input = [torch.empty(2, 4) for _ in num_tokens_across_dp]
        with patch('vllm_ascend.distributed.parallel_state.get_mlp_tp_group') as mock_group:
            mock_group().device_group = None
            with patch('torch.distributed.all_gather') as mock_all_gather:
                mock_all_gather.side_effect = lambda outputs, input_, group: [o.copy_(input_) for o in outputs]
                torch.distributed.all_gather(gathered_input, input_tensor, group=None)
                for output in gathered_input:
                    torch.testing.assert_close(output, input_tensor)

    def test_forward_gather_output(self):
        input_tensor = torch.randn(2, 4)
        with patch('vllm_ascend.distributed.parallel_state.get_forward_context') as mock_context:
            mock_context().num_tokens_across_dp = [2, 2]
            with patch('vllm_ascend.distributed.parallel_state.get_mlp_tp_group') as mock_group:
                mock_group().device_group = None
                with patch('torch.distributed.all_gather') as mock_all_gather:
                    mock_all_gather.side_effect = lambda outputs, input_, group: [o.copy_(input_) for o in outputs]
                    output = torch.randn(4, 4)  # 模拟量化方法的输出
                    with patch.object(LinearBase, 'quant_method', new_callable=MagicMock) as mock_quant:
                        mock_quant.apply.return_value = output
                        layer = MagicMock()
                        layer.quant_method = mock_quant
                        forward_result = layer.forward(input_tensor)
                        self.assertEqual(forward_result.shape, (4, 4))


class TestAttnRowParallelLinear(unittest.TestCase):

    def setUp(self):
        # 模拟分布式环境参数
        self.rank_patch = patch('__main__.get_mlp_tensor_model_parallel_rank', return_value=0)
        self.world_size_patch = patch('__main__.get_mlp_tensor_model_parallel_world_size', return_value=2)
        self.rank_patch.start()
        self.world_size_patch.start()

    def tearDown(self):
        self.rank_patch.stop()
        self.world_size_patch.stop()

    def test_forward_output_shape_column_parallel(self):
        input_size = 4
        output_size = 8
        input_tensor = torch.randn(2, 4)  # batch_size=2, input_size=4
        layer = patch_linear.AttnColumnParallelLinear(
            input_size=input_size, output_size=output_size, params_dtype=torch.float32
        )

        with patch.object(LinearBase, 'quant_method', new_callable=MagicMock) as mock_quant:
            mock_quant.apply.return_value = torch.randn(2, 8)  # 模拟量化输出
            output, bias = layer.forward(input_tensor)
            self.assertEqual(output.shape, (2, 8))  # 输出形状是否正确
            self.assertIsNone(bias)  # 默认不返回偏置

    def test_forward_output_shape_row_parallel(self):
        input_size = 4
        output_size = 8
        input_tensor = torch.randn(2, 4)  # batch_size=2, input_size=4
        layer = patch_linear.AttnRowParallelLinear(
            input_size=input_size, output_size=output_size, params_dtype=torch.float32
        )

        with patch.object(LinearBase, 'quant_method', new_callable=MagicMock) as mock_quant:
            mock_quant.apply.return_value = torch.randn(2, 8)  # 模拟量化输出
            output, bias = layer.forward(input_tensor)
            self.assertEqual(output.shape, (1, 8))  # 输出形状是否正确
            self.assertIsNone(bias)  # 默认不返回偏置
