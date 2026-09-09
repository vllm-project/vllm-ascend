from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_config import EplbConfig, StairConfig
from vllm_ascend.worker.v2 import model_runner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@pytest.mark.parametrize("enable_eplb", [True, False])
@pytest.mark.parametrize("overrides", [{}, {"stair_config": {"sample_size": 16}}])
def test_init_selects_stair_when_eplb_is_enabled(monkeypatch, enable_eplb, overrides):
    eplb_config = EplbConfig(**overrides)
    config = MagicMock(parallel_config=SimpleNamespace(enable_eplb=enable_eplb))
    monkeypatch.setattr(model_runner, "get_ascend_config", lambda: SimpleNamespace(eplb_config=eplb_config))
    monkeypatch.setattr(model_runner, "set_potential_max_tokens", lambda _: None)
    monkeypatch.setattr(model_runner, "resolve_spec_pp_support", lambda _: None)
    monkeypatch.setattr(model_runner, "torch_cuda_wrapper", nullcontext)
    monkeypatch.setattr(model_runner, "bypass_upstream_spec_pp_guard", lambda *_: nullcontext(False))
    monkeypatch.setattr(
        GPUModelRunner,
        "__init__",
        lambda self, *_: setattr(self, "compilation_config", SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)),
    )

    # Stop after controller construction; the rest of initialization allocates device buffers.
    with (
        patch.object(
            model_runner, "AscendEPLBController", side_effect=RuntimeError("controller reached")
        ) as controller,
        pytest.raises(RuntimeError, match="controller reached"),
    ):
        NPUModelRunner(config, torch.device("cpu"))

    selected = controller.call_args.kwargs["stair_config"]
    if enable_eplb:
        assert isinstance(selected, StairConfig)
        assert selected == eplb_config.resolved_stair_config
    else:
        assert selected is None


def _make_runner(need_timing: bool = True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=need_timing))
    )
    runner.vllm_config = SimpleNamespace()
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    return runner


@pytest.mark.parametrize("is_vllm_0_27_1", [True, False], ids=["v0.27.1", "newer"])
def test_execute_model_records_profiling_time(is_vllm_0_27_1):
    runner = _make_runner()
    scheduler_output = SimpleNamespace(disable_profiling_timing=False)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ) as mock_execute_model,
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_version_is",
            return_value=is_vllm_0_27_1,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch(
            "vllm_ascend.core.profiling_chunk_predictor.time.perf_counter",
            side_effect=[10.0, 10.125],
        ),
    ):
        output = runner.execute_model(scheduler_output)

    assert output is None
    assert runner._cpp_execution_time_ms == pytest.approx(125.0)
    assert mock_synchronize.call_count == 2
    expected_kwargs: dict[str, object] = {
        "intermediate_tensors": None,
        "dummy_run": False,
        "skip_attn_for_dummy_run": False,
        "is_profile": False,
    }
    if not is_vllm_0_27_1:
        expected_kwargs["context_len"] = 0
    mock_execute_model.assert_called_once_with(scheduler_output, **expected_kwargs)


def test_execute_model_disables_profiling_timer_and_clears_stale_time():
    runner = _make_runner()
    runner._cpp_execution_time_ms = 123.0
    scheduler_output = SimpleNamespace(disable_profiling_timing=True)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch("vllm_ascend.core.profiling_chunk_predictor.time.perf_counter") as mock_perf_counter,
    ):
        runner.execute_model(scheduler_output)

    profiling_config = runner.ascend_config.scheduler_config.profiling_chunk_config
    assert not profiling_config.need_timing
    assert runner._cpp_execution_time_ms is None
    mock_synchronize.assert_not_called()
    mock_perf_counter.assert_not_called()


def test_full_decode_only_keeps_graph_descriptor_request_count():
    runner = _make_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    runner.decode_query_len = 1
    query_start_loc_np = np.array([0, 1, 2, 2, 2, 2], dtype=np.int32)

    actual, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=4,
        num_reqs_padded=4,
        num_reqs=2,
        query_start_loc_np=query_start_loc_np,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=4,
    )

    assert num_reqs_padded == 4
    np.testing.assert_array_equal(actual[:5], np.array([0, 1, 2, 3, 4], dtype=np.int32))
