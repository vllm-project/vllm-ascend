import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2 import model_runner as model_runner_module
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


class _CountingDraftTokens(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_calls = []

    def get(self, key, default=None):
        self.get_calls.append(key)
        return super().get(key, default)


def _make_prepare_inputs_runner(num_bonus_tokens: int = 1):
    max_num_reqs = 8
    max_num_tokens = 32
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = max_num_reqs
    runner.decode_query_len = 4
    runner.vllm_config = SimpleNamespace()
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=num_bonus_tokens)
    runner.use_dcp = False
    runner.use_pp = False
    runner.pcp_manager = None
    runner.eplb = SimpleNamespace(set_batch_phase=lambda _has_prefill: None)
    runner._update_seq_lens_cpu = lambda _scheduler_output, _req_ids: None
    runner.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int64),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64),
        is_padding=torch.zeros(max_num_tokens, dtype=torch.bool),
        query_start_loc=torch.zeros(max_num_reqs + 2, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32),
        seq_lens_np=np.zeros(max_num_reqs, dtype=np.int32),
    )
    runner.req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(max_num_reqs, dtype=torch.int64),
        draft_tokens=torch.zeros(max_num_tokens, dtype=torch.int64),
        all_token_ids=SimpleNamespace(gpu=torch.zeros(max_num_tokens, dtype=torch.int64)),
        prefill_len=SimpleNamespace(gpu=torch.zeros(max_num_reqs, dtype=torch.int32)),
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(max_num_reqs, dtype=torch.int32)),
        num_computed_tokens_np=np.arange(max_num_reqs, dtype=np.int32),
        prompt_len=SimpleNamespace(gpu=torch.zeros(max_num_reqs, dtype=torch.int32)),
        max_seq_len=np.zeros(max_num_reqs, dtype=np.int32),
    )
    return runner


def _patch_prepare_inputs_dependencies(monkeypatch):
    captures = {}

    def fake_async_copy_to_gpu(src, device=None, out=None):
        tensor = torch.as_tensor(src, device=device)
        if out is not None:
            out[: tensor.numel()].copy_(tensor.reshape(-1))
            return out
        return tensor

    def fake_build_attn_state(_config, _seq_lens_np, _num_reqs, _scheduled, num_valid_tokens):
        captures["num_valid_tokens"] = num_valid_tokens
        return "attn-state"

    def fake_expand_idx_mapping(idx_mapping, total_num_logits, cu_num_logits, decode_query_len):
        captures["expand_total_num_logits"] = total_num_logits
        captures["expand_cu_num_logits"] = cu_num_logits.clone()
        captures["expand_decode_query_len"] = decode_query_len
        return idx_mapping.repeat_interleave(1), torch.zeros(total_num_logits, dtype=torch.int32)

    def fake_combine_sampled_and_draft_tokens(*args):
        total_num_logits = args[8]
        captures["combine_total_num_logits"] = total_num_logits
        return torch.arange(total_num_logits, dtype=torch.int64)

    monkeypatch.setattr(model_runner_module, "async_copy_to_gpu", fake_async_copy_to_gpu)
    monkeypatch.setattr(model_runner_module, "build_attn_state", fake_build_attn_state)
    monkeypatch.setattr(model_runner_module, "expand_idx_mapping", fake_expand_idx_mapping)
    monkeypatch.setattr(model_runner_module, "combine_sampled_and_draft_tokens", fake_combine_sampled_and_draft_tokens)
    monkeypatch.setattr(model_runner_module, "prepare_pos_seq_lens", lambda *args: None)
    monkeypatch.setattr(model_runner_module, "update_cos_sin", lambda *_args: None)
    monkeypatch.setattr(
        model_runner_module.vllm_model_runner.pcp, "maybe_partition_pcp_batch", lambda *args, **kwargs: args[1]
    )
    monkeypatch.setattr(model_runner_module, "vllm_version_is", lambda _version: False)
    return captures


def _make_prepare_inputs_args(req_ids, scheduled_tokens, draft_tokens):
    num_scheduled_tokens = np.array(scheduled_tokens, dtype=np.int32)
    batch_req_state = SimpleNamespace(
        num_tokens=int(num_scheduled_tokens.sum()),
        req_ids=req_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        idx_mapping_np=np.arange(len(req_ids), dtype=np.int64),
        has_prefill=False,
        prefill_len_np=np.zeros(len(req_ids), dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(len(req_ids), dtype=np.int32),
        is_prefilling_np=np.zeros(len(req_ids), dtype=bool),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens=draft_tokens,
        has_structured_output_requests=False,
    )
    batch_desc = SimpleNamespace(
        num_tokens=0,
        num_reqs=0,
        cg_mode=CUDAGraphMode.NONE,
    )
    return scheduler_output, batch_req_state, batch_desc


def test_prepare_inputs_reuses_draft_counts_for_valid_tokens_and_logits(monkeypatch):
    runner = _make_prepare_inputs_runner(num_bonus_tokens=2)
    captures = _patch_prepare_inputs_dependencies(monkeypatch)
    req_ids = ["req2", "req0", "missing", "req1"]
    draft_tokens = _CountingDraftTokens(
        {
            "req0": [10],
            "req1": [20, 21],
            "req2": [],
        }
    )
    scheduler_output, batch_req_state, batch_desc = _make_prepare_inputs_args(
        req_ids,
        [5, 4, 3, 5],
        draft_tokens,
    )

    input_batch = runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc)

    expected_draft_counts = np.array([0, 1, 0, 2], dtype=np.int32)
    np.testing.assert_array_equal(input_batch.num_draft_tokens_per_req, expected_draft_counts)
    assert input_batch.num_draft_tokens_per_req.dtype == np.int32
    assert input_batch.num_draft_tokens_per_req.shape == (len(req_ids),)
    assert draft_tokens.get_calls == req_ids
    np.testing.assert_array_equal(
        captures["num_valid_tokens"],
        np.array([5, 3, 3, 3], dtype=np.int32),
    )
    assert input_batch.num_draft_tokens == 3
    np.testing.assert_array_equal(input_batch.cu_num_logits_np, np.array([0, 2, 5, 7, 11], dtype=np.int32))
    assert captures["combine_total_num_logits"] == 11
    assert captures["expand_total_num_logits"] == 11
    torch.testing.assert_close(captures["expand_cu_num_logits"], torch.tensor([0, 2, 5, 7, 11], dtype=torch.int32))


def test_prepare_inputs_no_draft_keeps_valid_alias_and_skips_count_materialization(monkeypatch):
    runner = _make_prepare_inputs_runner()
    captures = _patch_prepare_inputs_dependencies(monkeypatch)

    def fail_fromiter(*_args, **_kwargs):
        raise AssertionError("no-draft path must not materialize draft counts")

    monkeypatch.setattr(model_runner_module.np, "fromiter", fail_fromiter)
    scheduler_output, batch_req_state, batch_desc = _make_prepare_inputs_args(
        ["req0", "req1"],
        [3, 2],
        {},
    )

    input_batch = runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc)

    assert captures["num_valid_tokens"] is batch_req_state.num_scheduled_tokens
    assert input_batch.num_draft_tokens_per_req is None
    assert input_batch.num_draft_tokens == 0
    np.testing.assert_array_equal(input_batch.cu_num_logits_np, np.array([0, 1, 2], dtype=np.int32))
    assert input_batch.expanded_idx_mapping.data_ptr() == input_batch.idx_mapping.data_ptr()


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
