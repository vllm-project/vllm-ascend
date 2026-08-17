from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.ascend_forward_context import (
    FirstLayerInputSource,
    get_first_layer_input_source,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.model_runner import (
    NPUModelRunner,
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


def test_mrv2_runner_source_accounts_for_pp_rank():
    runner = object.__new__(NPUModelRunner)
    runner.supports_mm_inputs = True
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)

    runner.is_first_pp_rank = True
    assert runner.get_first_layer_input_source() == FirstLayerInputSource.PRECOMPUTED_EMBEDDING

    runner.is_first_pp_rank = False
    assert runner.get_first_layer_input_source() == FirstLayerInputSource.NOT_APPLICABLE


def test_mrv2_execute_restores_input_source_after_exception():
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = _config()

    with (
        patch.object(
            runner,
            "get_first_layer_input_source",
            return_value=FirstLayerInputSource.MODEL_EMBEDDING,
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.flashcomm_dispatch_wrapper",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.GPUModelRunner.execute_model",
            side_effect=RuntimeError("forward failure"),
        ),
        pytest.raises(RuntimeError, match="forward failure"),
    ):
        runner.execute_model(MagicMock())

    assert get_first_layer_input_source() == FirstLayerInputSource.NOT_APPLICABLE


def test_mrv2_graph_capture_uses_runner_input_source():
    manager = object.__new__(ModelAclGraphManager)
    manager.model_runner = MagicMock()
    manager.model_runner.get_first_layer_input_source.return_value = FirstLayerInputSource.PRECOMPUTED_EMBEDDING

    def capture(*args, **kwargs):
        assert get_first_layer_input_source() == FirstLayerInputSource.PRECOMPUTED_EMBEDDING

    with (
        patch(
            "vllm_ascend.worker.v2.aclgraph_utils.communicator_switch",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.aclgraph_utils.ModelCudaGraphManager.capture",
            side_effect=capture,
        ),
    ):
        manager.capture(
            torch.nn.Identity(),
            MagicMock(),
            MagicMock(),
            None,
            MagicMock(),
            [],
            MagicMock(),
        )

    manager.model_runner.get_first_layer_input_source.assert_called_once_with()
    assert get_first_layer_input_source() == FirstLayerInputSource.NOT_APPLICABLE
