from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.model_runner import (
    flashcomm_dispatch_wrapper,
)
from vllm_ascend.worker.v2.sp_utils import (
    _all_gather_hidden_states,
    _flashcomm_enabled,
)


def _config(tp_size=2):
    return SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp_size))


def test_flashcomm_dispatch_pads_before_graph_selection():
    config = _config(tp_size=4)
    dispatch = MagicMock(
        return_value=(
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=8,
                num_reqs=1,
            ),
            None,
        )
    )

    with (
        patch(
            "vllm_ascend.worker.v2.model_runner.enable_sp",
            return_value=True,
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_model_runner.dispatch_cg_and_sync_dp",
            dispatch,
        ),
        flashcomm_dispatch_wrapper(config),
    ):
        from vllm.v1.worker.gpu import model_runner as vllm_model_runner

        vllm_model_runner.dispatch_cg_and_sync_dp(
            None,
            1,
            5,
            None,
            1,
            0,
            need_eager=True,
        )

    assert dispatch.call_args.args[2] == 8


def test_all_gather_hidden_states_trims_flashcomm_padding():
    local_hidden_states = torch.arange(6).reshape(3, 2)
    gathered_hidden_states = torch.arange(12).reshape(6, 2)

    with patch(
        "vllm_ascend.worker.v2.sp_utils.tensor_model_parallel_all_gather",
        return_value=gathered_hidden_states,
    ):
        result = _all_gather_hidden_states(
            local_hidden_states,
            num_tokens=5,
        )

    torch.testing.assert_close(result, gathered_hidden_states[:5])


def test_flashcomm_dense_threshold_and_moe_behavior():
    config = _config()
    with (
        patch(
            "vllm_ascend.worker.v2.sp_utils.enable_sp",
            return_value=True,
        ),
        patch(
            "vllm_ascend.worker.v2.sp_utils.is_moe_model",
            return_value=False,
        ),
    ):
        assert not _flashcomm_enabled(config, 1000)
        assert _flashcomm_enabled(config, 1001)

    with (
        patch(
            "vllm_ascend.worker.v2.sp_utils.enable_sp",
            return_value=True,
        ),
        patch(
            "vllm_ascend.worker.v2.sp_utils.is_moe_model",
            return_value=True,
        ),
    ):
        assert _flashcomm_enabled(config, 1)
