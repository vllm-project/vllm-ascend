from importlib import reload

import pytest
import torch
import vllm
from pytest_mock import MockerFixture

import vllm_ascend.envs as envs_ascend
from tests.ut.base import PytestBase
from vllm_ascend.patch.worker.patch_common import patch_linear


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
        mocker.patch.object(envs_ascend,
                            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE",
                            new=True)
        reload(patch_linear)
        assert envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
        assert id(vllm.model_executor.layers.linear.RowParallelLinear) == id(
            patch_linear.AscendRowParallelLinear)
