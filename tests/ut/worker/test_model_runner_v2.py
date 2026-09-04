import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
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
        output = runner.execute_model(scheduler_output, valid_dummy_state_slots=True)

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
        expected_kwargs["valid_dummy_state_slots"] = True
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

    # prepare_inputs has one implementation for v0.27.1 and one for newer
    # vLLM snapshots. Both retain the real global PCP batch when it is larger
    # than the graph descriptor. Only the newer upstream contract from vLLM
    # #53515 accepts the descriptor as an explicit rank-local padded extent.
    assert len(padding_assignments) == 2
    for assignment in padding_assignments:
        assert ast.unparse(assignment.value) == "max(num_tokens, batch_desc.num_tokens)"

    assert len(partition_calls) == 2
    padded_num_tokens_values = [
        next(
            (keyword.value for keyword in call.keywords if keyword.arg == "padded_num_tokens"),
            None,
        )
        for call in partition_calls
    ]
    assert sum(value is None for value in padded_num_tokens_values) == 1
    padded_num_tokens = next(value for value in padded_num_tokens_values if value is not None)
    assert isinstance(padded_num_tokens, ast.Attribute)
    assert padded_num_tokens.attr == "num_tokens"
    assert isinstance(padded_num_tokens.value, ast.Name)
    assert padded_num_tokens.value.id == "batch_desc"
