import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, _is_compatible
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

import vllm_ascend.worker.v2.model_runner as ascend_model_runner
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def _make_runner(need_timing: bool = True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=need_timing))
    )
    runner.vllm_config = SimpleNamespace()
    runner.kvpp = SimpleNamespace(complete_forward=lambda: None)
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False)
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    return runner


def test_execute_model_records_profiling_time():
    runner = _make_runner()
    scheduler_output = SimpleNamespace(disable_profiling_timing=False)

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
    if not vllm_version_is("0.28.0"):
        expected_kwargs["valid_dummy_state_slots"] = False
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


def _legacy_pad_query_start_loc_for_fia(
    runner,
    num_tokens_padded,
    num_reqs_padded,
    num_reqs,
    query_start_loc_np,
    cudagraph_runtime_mode=None,
    batch_desc_num_reqs=None,
):
    descriptor_num_reqs = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs_padded
    has_uniform_decode_query_lens = np.all(np.diff(query_start_loc_np[: num_reqs + 1]) == runner.decode_query_len)
    matches_uniform_decode_graph_shape = (
        has_uniform_decode_query_lens and num_tokens_padded == descriptor_num_reqs * runner.decode_query_len
    )
    if (
        cudagraph_runtime_mode == CUDAGraphMode.FULL
        and runner.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        and not matches_uniform_decode_graph_shape
    ):
        num_reqs_padded = num_reqs
    else:
        num_reqs_padded = descriptor_num_reqs

    if has_uniform_decode_query_lens and num_tokens_padded == num_reqs_padded * runner.decode_query_len:
        assert num_reqs <= num_reqs_padded
        last_loc = query_start_loc_np[num_reqs]
        query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
            np.arange(1, num_reqs_padded + 1 - num_reqs) * runner.decode_query_len + last_loc
        )
    else:
        assert num_reqs == num_reqs_padded
        query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
        num_reqs_padded = num_reqs_padded + 1

    return query_start_loc_np, num_reqs_padded


@pytest.mark.parametrize(
    "dtype, query_lens, decode_query_len, num_tokens_padded, descriptor_num_reqs, strided",
    [
        (np.int32, [1, 1], 1, 4, 4, False),
        (np.int64, [2, 1], 1, 4, 4, False),
        (np.int32, [2, 4], 4, 8, 2, True),
        (np.int64, [], 1, 0, 0, True),
    ],
    ids=["u-none-d1-padded", "mixed-short-prefill", "strided-int32", "zero-req-int64-strided"],
)
def test_pad_query_start_loc_unproven_matches_legacy(
    dtype, query_lens, decode_query_len, num_tokens_padded, descriptor_num_reqs, strided
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.decode_query_len = decode_query_len
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    num_reqs = len(query_lens)

    size = descriptor_num_reqs + 3
    expected = np.full(size * (2 if strided else 1), -7, dtype=dtype)
    actual = expected.copy()
    expected_view = expected[::2] if strided else expected
    actual_view = actual[::2] if strided else actual
    expected_view[: num_reqs + 1] = np.cumsum([0, *query_lens], dtype=dtype)
    actual_view[: num_reqs + 1] = expected_view[: num_reqs + 1]

    legacy_array, legacy_num_reqs_padded = _legacy_pad_query_start_loc_for_fia(
        runner,
        num_tokens_padded,
        descriptor_num_reqs,
        num_reqs,
        expected_view,
        CUDAGraphMode.FULL,
        descriptor_num_reqs,
    )
    patched_array, patched_num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded,
        descriptor_num_reqs,
        num_reqs,
        actual_view,
        CUDAGraphMode.FULL,
        descriptor_num_reqs,
    )

    assert patched_array is actual_view
    assert legacy_array is expected_view
    assert patched_array.shape == legacy_array.shape
    assert patched_array.strides == legacy_array.strides
    assert patched_num_reqs_padded == legacy_num_reqs_padded
    np.testing.assert_array_equal(actual, expected)


def test_pad_query_start_loc_uniform_certificate_skips_diff_and_preserves_descriptor_shape():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.decode_query_len = 1
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    storage = np.full(12, -1, dtype=np.int32)
    query_start_loc = storage[::2]
    query_start_loc[:3] = [0, 1, 2]

    with (
        patch.object(ascend_model_runner.np, "diff", side_effect=AssertionError("diff should be skipped")),
        patch.object(ascend_model_runner.np, "all", side_effect=AssertionError("all should be skipped")),
    ):
        actual, num_reqs_padded = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=4,
            num_reqs_padded=4,
            num_reqs=2,
            query_start_loc_np=query_start_loc,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_desc_num_reqs=4,
            uniform_lengths_proven=True,
        )

    assert actual is query_start_loc
    assert num_reqs_padded == 4
    assert actual.strides == (storage.itemsize * 2,)
    np.testing.assert_array_equal(actual[:5], np.array([0, 1, 2, 3, 4], dtype=np.int32))


def test_cudagraph_compatibility_rejects_incompatible_uniform_u1():
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )

    assert _is_compatible(desc, 2, 2, 1, 0, None, 1)
    assert not _is_compatible(desc, 2, 2, None, 0, None, 1)
    assert not _is_compatible(desc, 2, 4, 2, 0, None, 1)


class _StopAtQueryStartTransfer(Exception):
    pass


def _run_prepare_inputs_until_query_start_transfer(
    query_lens,
    *,
    cg_mode=CUDAGraphMode.FULL,
    descriptor_num_tokens=4,
    descriptor_num_reqs=4,
    descriptor_uniform_token_count=1,
    runner_overrides=None,
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_reqs = 8
    runner.device = "cpu"
    runner.vllm_config = SimpleNamespace()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    runner.speculative_config = None
    runner.dp_size = 1
    runner.pcp_manager = None
    runner.use_dcp = False
    runner.decode_query_len = 1
    runner.input_buffers = SimpleNamespace(
        query_start_loc=torch.empty(runner.max_num_reqs + 2, dtype=torch.int32),
        seq_lens_np=np.zeros(runner.max_num_reqs, dtype=np.int32),
    )
    runner._update_seq_lens_cpu = Mock()
    for name, value in (runner_overrides or {}).items():
        setattr(runner, name, value)

    req_ids = list(range(len(query_lens)))
    batch_req_state = SimpleNamespace(
        num_tokens=sum(query_lens),
        req_ids=req_ids,
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        idx_mapping_np=np.arange(len(query_lens), dtype=np.int32),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
    )
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=descriptor_num_tokens,
        num_reqs=descriptor_num_reqs,
        uniform_token_count=descriptor_uniform_token_count,
    )

    captured = {}

    def fake_async_copy_to_gpu(data, device=None, out=None):
        if out is runner.input_buffers.query_start_loc:
            captured["query_start_loc_np"] = data.copy()
            captured["query_start_loc_out"] = out
            raise _StopAtQueryStartTransfer
        assert out is None
        captured["idx_mapping_np"] = data.copy()
        return torch.as_tensor(data, device=device)

    original_diff = ascend_model_runner.np.diff
    original_all = ascend_model_runner.np.all
    original_helper = runner._pad_query_start_loc_for_fia
    counts = {"diff": 0, "all": 0}
    helper_returns = []

    def counting_diff(*args, **kwargs):
        counts["diff"] += 1
        return original_diff(*args, **kwargs)

    def counting_all(*args, **kwargs):
        counts["all"] += 1
        return original_all(*args, **kwargs)

    def wrapped_helper(*args, **kwargs):
        result = original_helper(*args, **kwargs)
        helper_returns.append(result)
        return result

    with (
        patch.object(runner, "_pad_query_start_loc_for_fia", side_effect=wrapped_helper) as helper,
        patch.object(ascend_model_runner, "async_copy_to_gpu", side_effect=fake_async_copy_to_gpu),
        patch.object(ascend_model_runner, "build_attn_state", return_value=object()) as build_attn_state,
        patch.object(ascend_model_runner.np, "diff", side_effect=counting_diff),
        patch.object(ascend_model_runner.np, "all", side_effect=counting_all),
        pytest.raises(_StopAtQueryStartTransfer),
    ):
        runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc)

    return SimpleNamespace(
        runner=runner,
        batch_desc=batch_desc,
        captured=captured,
        counts=counts,
        helper=helper,
        helper_returns=helper_returns,
        build_attn_state=build_attn_state,
    )


def test_prepare_inputs_uniform_certificate_full_u1_skips_helper_predicate():
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    assert _is_compatible(desc, 2, 2, 1, 0, 1, 1)
    assert _is_compatible(desc, 4, 4, 1, 0, 1, 1)

    result = _run_prepare_inputs_until_query_start_transfer([1, 1])

    result.runner._update_seq_lens_cpu.assert_called_once()
    result.helper.assert_called_once()
    args = result.helper.call_args.args
    assert args[:3] == (4, 4, 2)
    assert args[4:] == (CUDAGraphMode.FULL, 4)
    assert result.helper.call_args.kwargs == {"uniform_lengths_proven": True}
    assert result.counts == {"diff": 0, "all": 0}
    assert result.helper_returns[0][1] == 4
    np.testing.assert_array_equal(
        result.captured["query_start_loc_np"][:5],
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
    )


def test_prepare_inputs_full_u_none_keeps_original_uniform_predicate_and_descriptor_shape():
    result = _run_prepare_inputs_until_query_start_transfer(
        [1, 1],
        descriptor_uniform_token_count=None,
    )

    result.helper.assert_called_once()
    assert result.helper.call_args.kwargs == {"uniform_lengths_proven": False}
    assert result.counts == {"diff": 1, "all": 1}
    assert result.helper_returns[0][1] == 4
    np.testing.assert_array_equal(
        result.captured["query_start_loc_np"][:5],
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
    )


def test_prepare_inputs_full_u_none_mixed_lengths_use_legacy_dummy_padding():
    result = _run_prepare_inputs_until_query_start_transfer(
        [2, 1],
        descriptor_uniform_token_count=None,
    )

    result.helper.assert_called_once()
    assert result.helper.call_args.kwargs == {"uniform_lengths_proven": False}
    assert result.counts == {"diff": 1, "all": 1}
    assert result.helper_returns[0][1] == 3
    np.testing.assert_array_equal(
        result.captured["query_start_loc_np"][:4],
        np.array([0, 2, 3, 4], dtype=np.int32),
    )


@pytest.mark.parametrize(
    "runner_overrides, descriptor_uniform_token_count",
    [
        ({"speculative_config": object()}, 1),
        ({"dp_size": 2}, 1),
        ({"pcp_manager": object()}, 1),
        ({"use_dcp": True}, 1),
        ({"decode_query_len": 2}, 1),
        ({}, 2),
    ],
    ids=[
        "speculative-config",
        "dp-size",
        "pcp-manager",
        "dcp",
        "decode-query-len",
        "descriptor-u2",
    ],
)
def test_prepare_inputs_uniform_certificate_false_for_each_gate(runner_overrides, descriptor_uniform_token_count):
    result = _run_prepare_inputs_until_query_start_transfer(
        [1, 1],
        descriptor_uniform_token_count=descriptor_uniform_token_count,
        runner_overrides=runner_overrides,
    )

    result.helper.assert_called_once()
    assert result.helper.call_args.kwargs == {"uniform_lengths_proven": False}
    assert result.counts == {"diff": 1, "all": 1}


@pytest.mark.parametrize("cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE], ids=lambda mode: mode.name)
def test_prepare_inputs_non_full_modes_skip_fia_helper_and_preserve_cumsum(cg_mode):
    result = _run_prepare_inputs_until_query_start_transfer([1, 1], cg_mode=cg_mode)

    result.helper.assert_not_called()
    assert result.counts == {"diff": 0, "all": 0}
    np.testing.assert_array_equal(
        result.captured["query_start_loc_np"][:5],
        np.array([0, 1, 2, 2, 2], dtype=np.int32),
    )


@pytest.mark.parametrize(
    "decode_query_len, query_lens, num_tokens_padded, descriptor_num_reqs, expected_query_start_loc",
    [
        (1, [4], 8, 8, [0, 4, 8]),
        (4, [4, 5], 16, 4, [0, 4, 9, 16]),
        (4, [2, 6], 16, 4, [0, 2, 8, 16]),
        (4, [2, 4], 8, 2, [0, 2, 6, 8]),
    ],
    ids=["non-mtp-prefill", "mtp-mixed", "mtp-uniform-average", "mtp-no-request-padding"],
)
def test_full_graph_non_uniform_queries_use_mixed_padding(
    decode_query_len, query_lens, num_tokens_padded, descriptor_num_reqs, expected_query_start_loc
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.decode_query_len = decode_query_len
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    num_reqs = len(query_lens)
    query_start_loc = np.full(descriptor_num_reqs + 2, sum(query_lens), dtype=np.int32)
    query_start_loc[: num_reqs + 1] = np.cumsum([0, *query_lens])

    padded_query_start_loc, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=descriptor_num_reqs,
        num_reqs=num_reqs,
        query_start_loc_np=query_start_loc,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=descriptor_num_reqs,
    )

    assert num_reqs_padded == num_reqs + 1
    np.testing.assert_array_equal(padded_query_start_loc[: num_reqs_padded + 1], expected_query_start_loc)
    assert padded_query_start_loc[num_reqs_padded] == num_tokens_padded


@pytest.mark.parametrize(
    "query_lens,num_tokens_padded,num_reqs_padded,expected,expected_num_reqs",
    [
        ([3, 1], 8, 4, [0, 3, 4, 6, 8], 4),
        ([2, 3], 8, 2, [0, 2, 5, 8], 3),
        ([2, 3], 5, 2, [0, 2, 5], 2),
    ],
    ids=["spread-padding", "extra-padding-request", "no-padding"],
)
def test_adaptive_verification_pads_fia_query_boundaries(
    query_lens, num_tokens_padded, num_reqs_padded, expected, expected_num_reqs
):
    """Device-reallocated DSpark queries still match the FULL graph shape."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    num_reqs = len(query_lens)
    query_start_loc = np.full(max(num_reqs_padded + 2, 5), sum(query_lens), dtype=np.int32)
    query_start_loc[: num_reqs + 1] = np.cumsum([0, *query_lens])

    actual, actual_num_reqs = runner._pad_adaptive_query_start_loc_for_fia(
        num_tokens_padded,
        num_reqs_padded,
        num_reqs,
        query_start_loc,
    )

    assert actual_num_reqs == expected_num_reqs
    np.testing.assert_array_equal(actual[: len(expected)], expected)


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


@pytest.mark.parametrize("num_reqs,num_tokens", [(4, 4), (2, 6)])
def test_pcp_dummy_refreshes_captured_buffers_after_real_batch(num_reqs, num_tokens):
    runner = _make_runner()
    runner.input_buffers = AscendInputBuffers(4, 8, torch.device("cpu"))
    manager = AscendPCPManager(2, 1, torch.device("cpu"), max_num_reqs=4, max_num_tokens=8)
    runner.pcp_manager = manager
    manager._local_block_tables = (torch.full((8, 2), 99, dtype=torch.int32),)
    manager._gathered_kv_slot_mappings = torch.full((1, 16), 99, dtype=torch.int64)
    input_buffers = manager._input_buffers
    assert input_buffers is not None
    captured = {
        name: getattr(input_buffers, name)
        for name in ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens")
    }
    for name, value in captured.items():
        value.fill_(False if name == "is_padding" else 99)
    input_buffers.seq_lens_np.fill(99)
    with patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"):
        dummy = AscendInputBatch.make_dummy(num_reqs, num_tokens, runner.input_buffers)

    block_tables, slots = runner.prepare_dummy_attn(dummy)

    for name, value in captured.items():
        expected = getattr(dummy, name)
        torch.testing.assert_close(value[: len(expected)], expected)
    np.testing.assert_array_equal(input_buffers.seq_lens_np[:num_reqs], dummy.seq_lens_np)
    assert block_tables[0].data_ptr() == manager._local_block_tables[0].data_ptr()
    assert torch.count_nonzero(block_tables[0]) == 0
    assert slots.data_ptr() == manager._gathered_kv_slot_mappings.data_ptr()
    assert slots.shape == (1, 2 * num_tokens)
    assert torch.all(slots == -1)


def test_prepare_dummy_attn_without_pcp_uses_upstream():
    runner = _make_runner()
    runner.pcp_manager = None
    dummy = object()
    with patch.object(GPUModelRunner, "prepare_dummy_attn", return_value=((), None)) as parent:
        assert runner.prepare_dummy_attn(dummy) == ((), None)
    if vllm_version_is("0.28.0"):
        parent.assert_called_once_with(dummy)
    else:
        parent.assert_called_once_with(dummy, valid_state_slots=False)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "computed,dummy_run,is_profile,expected",
    [
        ([0, 0, 99, 99], False, False, False),
        ([0, 4, 0, 0], False, False, True),
        ([0, 4, 0, 0], True, False, False),
        ([0, 4, 0, 0], False, True, False),
    ],
)
def test_kvpp_history_ignores_padding_and_dummy_work(monkeypatch, computed, dummy_run, is_profile, expected, enabled):
    from vllm_ascend.worker.v2.model_states import default

    runner = _make_runner(need_timing=False)
    events: list[object] = []
    runner.kvpp = SimpleNamespace(
        scheduler=object() if enabled else None,
        prepare_forward=lambda history: events.append(("prepare", history)),
        complete_forward=lambda: events.append("complete"),
    )
    state = default.AscendModelState.__new__(default.AscendModelState)
    state.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    state.max_model_len = 32
    state.kvpp_runtime = runner.kvpp
    runner.model_state = state
    batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens=2,
        num_tokens_after_padding=4,
        num_computed_tokens_np=np.array(computed),
        query_start_loc_np=np.array([0, 1, 2, 2, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2, 2, 2]),
        num_scheduled_tokens=torch.tensor([1, 1]),
        is_prefilling_np=np.array([True, False, False, False]),
        seq_lens=torch.tensor([1, 5]),
        seq_lens_np=np.array([1, 5]),
        dcp_local_seq_lens=None,
        positions=torch.arange(2),
        attn_state=None,
    )
    if not enabled:
        batch.num_computed_tokens_np = None  # Disabled KVPP must not inspect history.
    metadata = object()
    monkeypatch.setattr(default, "build_attn_metadata", lambda **_kwargs: metadata)

    def forward(_self, _scheduler_output, **_kwargs):
        assert state.kvpp_is_dummy_run is (dummy_run or is_profile)
        assert state.prepare_attn(batch, CUDAGraphMode.NONE, (), torch.empty(0), [], None) is metadata
        events.append("forward")
        return metadata

    monkeypatch.setattr(GPUModelRunner, "execute_model", forward)
    assert runner.execute_model(SimpleNamespace(), dummy_run=dummy_run, is_profile=is_profile) is metadata
    assert events == ([("prepare", expected)] if enabled else []) + ["forward", "complete"]
    assert state.kvpp_is_dummy_run is False
