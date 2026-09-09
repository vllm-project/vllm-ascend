import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _make_runner(need_timing: bool = True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=need_timing))
    )
    runner.vllm_config = SimpleNamespace()
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    # Dump contract from the production initializer; helpers no-op on None.
    runner.debugger = None
    runner._debugger_started = False
    return runner


def test_execute_model_records_profiling_time():
    runner = _make_runner()
    scheduler_output = SimpleNamespace(disable_profiling_timing=False, total_num_scheduled_tokens=0)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ) as mock_execute_model,
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
        "context_len": 0,
    }
    mock_execute_model.assert_called_once_with(scheduler_output, **expected_kwargs)


def test_execute_model_disables_profiling_timer_and_clears_stale_time():
    runner = _make_runner()
    runner._cpp_execution_time_ms = 123.0
    scheduler_output = SimpleNamespace(disable_profiling_timing=True, total_num_scheduled_tokens=0)

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


def test_sample_tokens_restores_replicated_draft_hidden_states():
    runner = _make_runner(need_timing=False)
    runner.is_last_pp_rank = True
    runner.speculator = SimpleNamespace(replicated_pcp=True)
    runner.use_spec_pp = False

    aux_hidden_states = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    ]
    state = Mock(aux_hidden_states=aux_hidden_states)
    restored_state = object()
    state._replace.return_value = restored_state
    runner.execute_model_state = state

    target_hidden_states = object()
    restored_aux_hidden_states = torch.ones(4, 5)
    runner.pcp_manager = SimpleNamespace(
        restore_hidden_state_buffer=Mock(),
        restore_hidden_states=Mock(
            return_value=restored_aux_hidden_states,
        ),
    )
    runner.model = SimpleNamespace(
        get_mtp_target_hidden_states=lambda: target_hidden_states,
    )
    grammar_output = object()
    expected_output = object()

    with patch.object(
        GPUModelRunner,
        "sample_tokens",
        return_value=expected_output,
    ) as parent_sample_tokens:
        actual = runner.sample_tokens(grammar_output)

    assert actual is expected_output
    parent_sample_tokens.assert_called_once_with(grammar_output)
    runner.pcp_manager.restore_hidden_state_buffer.assert_called_once_with(target_hidden_states)
    restored_input = runner.pcp_manager.restore_hidden_states.call_args.args[0]
    torch.testing.assert_close(
        restored_input,
        torch.cat(aux_hidden_states, dim=-1),
    )
    state._replace.assert_called_once_with(aux_hidden_states=[restored_aux_hidden_states])
    assert runner.execute_model_state is restored_state


def test_prepare_inputs_preserves_pcp_tokens_and_forwards_graph_padding():
    source_path = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "v2" / "model_runner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    padding_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "num_tokens_after_padding" for target in node.targets)
    ]
    partition_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "maybe_partition_pcp_batch"
    ]

    # prepare_inputs keeps the real global PCP batch when it is larger than the
    # graph descriptor, and forwards the descriptor as an explicit rank-local
    # padded extent on main (upstream vLLM #53515). v0.28.0 omits the kwarg.
    assert len(padding_assignments) == 1
    assert ast.unparse(padding_assignments[0].value) == "max(num_tokens, batch_desc.num_tokens)"

    assert len(partition_calls) == 2
    padded_call = next(
        call for call in partition_calls if any(keyword.arg == "padded_num_tokens" for keyword in call.keywords)
    )
    unpadded_call = next(
        call for call in partition_calls if not any(keyword.arg == "padded_num_tokens" for keyword in call.keywords)
    )
    assert unpadded_call is not None
    padded_num_tokens = next(keyword.value for keyword in padded_call.keywords if keyword.arg == "padded_num_tokens")
    assert isinstance(padded_num_tokens, ast.Attribute)
    assert padded_num_tokens.attr == "num_tokens"
    assert isinstance(padded_num_tokens.value, ast.Name)
    assert padded_num_tokens.value.id == "batch_desc"


def _make_dump_runner(debugger=None):
    runner = _make_runner(need_timing=False)
    runner.debugger = debugger
    runner.model = Mock()
    runner._debugger_started = False
    return runner


def test_start_dump_data_starts_debugger_and_forwards_kwargs():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)

    runner._start_dump_data(scheduled_tokens={0: 4})

    debugger.start.assert_called_once_with(runner.model, scheduled_tokens={0: 4})
    assert runner._debugger_started is True


def test_start_dump_data_noop_when_already_started():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner._debugger_started = True

    runner._start_dump_data()

    debugger.start.assert_not_called()
    debugger.step.assert_not_called()


def test_dump_helpers_noop_without_debugger():
    # No dump config: helpers must be safe no-ops (no AttributeError).
    runner = _make_dump_runner(None)

    runner._start_dump_data()
    runner._finalize_dump_data(dump=False)


def test_finalize_dump_data_stops_stop_capable_debugger():
    debugger = Mock()
    runner = _make_dump_runner(debugger)
    runner._debugger_started = True

    runner._finalize_dump_data()

    debugger.stop.assert_called_once_with()
    debugger.step.assert_called_once_with()
    assert runner._debugger_started is False


def test_finalize_dump_data_keeps_window_open_for_graph_debugger():
    # AclGraphDumper has no stop(); the window stays open across steps and
    # each step only calls step() (v1 parity).
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner._debugger_started = True

    runner._finalize_dump_data()

    debugger.step.assert_called_once_with()
    assert runner._debugger_started is True


def test_execute_model_opens_dump_window_for_real_step():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    scheduler_output = SimpleNamespace(
        disable_profiling_timing=False,
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={0: 4},
    )

    with patch.object(GPUModelRunner, "execute_model", return_value=None):
        output = runner.execute_model(scheduler_output)

    assert output is None
    debugger.start.assert_called_once_with(runner.model, scheduled_tokens={0: 4})
    # Last-PP-rank path: the cycle is closed later in sample_tokens().
    debugger.step.assert_not_called()


def test_execute_model_skips_dump_when_no_tokens_scheduled():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    scheduler_output = SimpleNamespace(
        disable_profiling_timing=False,
        total_num_scheduled_tokens=0,
        num_scheduled_tokens={},
    )

    with patch.object(GPUModelRunner, "execute_model", return_value=None):
        runner.execute_model(scheduler_output)

    debugger.start.assert_not_called()
    debugger.step.assert_not_called()


def test_execute_model_closes_dummy_run_dump_without_writing():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    scheduler_output = SimpleNamespace(
        disable_profiling_timing=False,
        total_num_scheduled_tokens=0,
        num_scheduled_tokens={},
    )

    with patch.object(GPUModelRunner, "execute_model", return_value=None):
        runner.execute_model(scheduler_output, dummy_run=True)

    debugger.start.assert_called_once_with(runner.model, scheduled_tokens={})
    debugger.step.assert_called_once_with(dump=False)


def test_execute_model_finalizes_dump_for_pp_non_last_rank():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    scheduler_output = SimpleNamespace(
        disable_profiling_timing=False,
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={0: 4},
    )
    intermediate = IntermediateTensors({"hidden": torch.zeros(1, 2)})

    with patch.object(GPUModelRunner, "execute_model", return_value=intermediate):
        output = runner.execute_model(scheduler_output)

    assert output is intermediate
    debugger.start.assert_called_once()
    # PP non-last ranks never reach sample_tokens(), so finalize here.
    debugger.step.assert_called_once_with()


def test_sample_tokens_closes_dump_cycle_on_last_rank():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner.pcp_manager = None
    runner.use_spec_pp = False
    runner._debugger_started = True

    with patch.object(GPUModelRunner, "sample_tokens", return_value="out"):
        result = runner.sample_tokens("grammar")

    assert result == "out"
    debugger.step.assert_called_once_with()


def test_load_model_starts_dump_before_capture_in_graph_mode():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)

    with patch.object(GPUModelRunner, "load_model") as parent_load_model:
        runner.load_model()

    parent_load_model.assert_called_once_with()
    debugger.start.assert_called_once_with(runner.model)


def test_load_model_skips_dump_in_eager_mode():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)

    with patch.object(GPUModelRunner, "load_model"):
        runner.load_model()

    debugger.start.assert_not_called()


def test_pool_closes_dump_cycle_for_pooling_models():
    # Pooling models never reach sample_tokens(): the worker calls pool()
    # directly, so the dump cycle must be closed there.
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner._debugger_started = True

    with patch.object(GPUModelRunner, "pool", return_value="pooled") as parent_pool:
        result = runner.pool()

    assert result == "pooled"
    parent_pool.assert_called_once_with()
    debugger.step.assert_called_once_with()


def test_dummy_run_flushes_dump_window_without_writing():
    debugger = Mock(spec=["start", "step"])
    runner = _make_dump_runner(debugger)
    runner._debugger_started = True
    outputs = (Mock(), Mock())

    with (
        patch.object(GPUModelRunner, "_dummy_run", return_value=outputs) as parent_dummy_run,
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False),
    ):
        result = runner._dummy_run(8, is_profile=True)

    assert result is outputs
    parent_dummy_run.assert_called_once_with(
        8,
        skip_attn=False,
        uniform_decode=False,
        skip_eplb=False,
        is_profile=True,
    )
    # Capture/profiling forwards must not leak into the first real step.
    debugger.step.assert_called_once_with(dump=False)
