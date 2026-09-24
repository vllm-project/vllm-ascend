import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType
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
    runner.attn_groups = []
    runner.adaptive_verification = None
    runner.use_fia = False
    # Set by NPUModelRunner.__init__ on real instances.
    runner._finegrained_tp_requires_graph = False
    return runner


def _make_batch_state(computed: list[int], scheduled: list[int], prefill_lens: list[int]) -> BatchReqState:
    num_reqs = len(computed)
    is_prefilling = np.array(computed) < np.array(prefill_lens)
    return BatchReqState(
        req_ids=[f"req-{i}" for i in range(num_reqs)],
        num_scheduled_tokens=np.array(scheduled, dtype=np.int32),
        num_tokens=sum(scheduled),
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        prefill_len_np=np.array(prefill_lens, dtype=np.int32),
        num_computed_prefill_tokens_np=np.array(computed, dtype=np.int32),
        is_prefilling_np=is_prefilling,
        has_prefill=bool(is_prefilling.any()),
    )


def test_recompute_scheduler_reclassifies_pd_tail_in_mixed_decode_batch():
    runner = _make_runner()
    runner.decode_query_len = 1
    batch_state = _make_batch_state([127, 64], [1, 1], [128, 64])

    with (
        patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(batch_state, None)),
        patch(
            "vllm_ascend.worker.v2.model_runner.is_pd_decode_recompute_scheduler_enabled",
            return_value=True,
        ),
    ):
        gathered, uniform = runner.gather_batch_req_state(SimpleNamespace(), False)

    np.testing.assert_array_equal(gathered.is_prefilling_np, [False, False])
    assert gathered.has_prefill is False
    assert uniform == 1


@pytest.mark.parametrize(
    ("computed", "scheduled", "enabled"),
    [(64, 8, True), (127, 1, False)],
)
def test_recompute_scheduler_keeps_non_matching_prefill(computed, scheduled, enabled):
    runner = _make_runner()
    runner.decode_query_len = 1
    batch_state = _make_batch_state([computed, 64], [scheduled, 1], [128, 64])

    with (
        patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(batch_state, None)),
        patch(
            "vllm_ascend.worker.v2.model_runner.is_pd_decode_recompute_scheduler_enabled",
            return_value=enabled,
        ),
    ):
        gathered, uniform = runner.gather_batch_req_state(SimpleNamespace(), False)

    np.testing.assert_array_equal(gathered.is_prefilling_np, [True, False])
    assert gathered.has_prefill is True
    assert uniform is None


def test_recompute_scheduler_supports_multi_token_decode_query():
    runner = _make_runner()
    runner.decode_query_len = 2
    batch_state = _make_batch_state([126, 64], [2, 2], [128, 64])

    with (
        patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(batch_state, None)),
        patch(
            "vllm_ascend.worker.v2.model_runner.is_pd_decode_recompute_scheduler_enabled",
            return_value=True,
        ),
    ):
        gathered, uniform = runner.gather_batch_req_state(SimpleNamespace(), False)

    np.testing.assert_array_equal(gathered.is_prefilling_np, [False, False])
    assert gathered.has_prefill is False
    assert uniform == 2


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
        "valid_dummy_state_slots": False,
    }
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

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    # aux_hidden_states are restored by upstream sample_tokens (#56107);
    # the Ascend pre-restore only covers the target hidden states.
    state = Mock(aux_hidden_states=[torch.ones(2, 3)])
    state.hidden_states = hidden_states
    restored_state = object()
    state._replace.return_value = restored_state
    runner.execute_model_state = state

    target_hidden_states = object()
    restored_hidden_states = torch.ones(4, 3)
    runner.pcp_manager = SimpleNamespace(
        restore_hidden_state_buffer=Mock(),
        restore_hidden_states=Mock(
            return_value=restored_hidden_states,
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
    runner.pcp_manager.restore_hidden_states.assert_called_once_with(hidden_states)
    state._replace.assert_called_once_with(hidden_states=restored_hidden_states)
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
    # graph descriptor, and forwards the whole descriptor (upstream vLLM #53867
    # changed maybe_partition_pcp_batch from padded_num_tokens to a
    # BatchExecutionDescriptor).
    assert len(padding_assignments) == 1
    assert ast.unparse(padding_assignments[0].value) == "max(num_tokens, batch_desc.num_tokens)"

    assert len(partition_calls) == 1
    partition_call = partition_calls[0]
    batch_desc_kw = next(keyword.value for keyword in partition_call.keywords if keyword.arg == "batch_desc")
    assert isinstance(batch_desc_kw, ast.Name)
    assert batch_desc_kw.id == "batch_desc"


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


def test_pcp_manager_cls():
    assert _make_runner().pcp_manager_cls is AscendPCPManager


def _parent_init(self, vllm_config, device, *, full_graph=False, speculative=False):
    self.vllm_config = vllm_config
    self.device = device
    self.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE,
        mode=SimpleNamespace(),
        has_full_cudagraphs=lambda: full_graph,
    )
    self.model_config = SimpleNamespace(enforce_eager=not full_graph)
    self.speculative_config = object() if speculative else None
    self.is_last_pp_rank = True
    self.pp_handler = MagicMock()
    self.max_num_reqs = 2
    self.max_model_len = 32
    self.max_num_tokens = 8
    self.num_speculative_steps = 1 if speculative else 0
    self.vocab_size = 16
    self.dtype = torch.float16
    self.req_states = object()
    self.input_buffers = object()
    self.speculator = object()


def test_init_without_spec_pp():
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=False))
    ascend_config = SimpleNamespace(eplb_config=SimpleNamespace(load_collection_phase="all"))

    # Complete the fake with the fields NPUModelRunner reads (mirrors FinegrainedTPConfig).
    ascend_config.finegrained_tp_config = SimpleNamespace(
        oproj_tensor_parallel_size=0,
        lmhead_tensor_parallel_size=0,
        embedding_tensor_parallel_size=0,
        mlp_tensor_parallel_size=0,
    )
    with (
        patch("vllm_ascend.worker.v2.model_runner.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.worker.v2.model_runner.set_potential_max_tokens"),
        patch("vllm_ascend.worker.v2.model_runner.resolve_spec_pp_support", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner.torch_cuda_wrapper", return_value=nullcontext()),
        patch(
            "vllm_ascend.worker.v2.model_runner.bypass_upstream_spec_pp_guard",
            return_value=nullcontext(False),
        ),
        patch.object(GPUModelRunner, "__init__", lambda self, cfg, dev: _parent_init(self, cfg, dev)),
        patch("vllm_ascend.worker.v2.model_runner.AscendEPLBController", return_value="eplb"),
        patch("vllm_ascend.worker.v2.model_runner.AscendRequestState", return_value="req"),
        patch("vllm_ascend.worker.v2.model_runner.AscendInputBuffers", return_value="buf"),
        patch("vllm_ascend.worker.v2.model_runner.set_cos_and_sin"),
        patch("vllm_ascend.worker.v2.model_runner.set_mc2_tokens_capacity"),
        patch("vllm_ascend.worker.v2.model_runner.set_mc2_mask"),
        patch(
            "vllm_ascend.worker.v2.model_runner.breakable_cudagraph.is_breakable_cudagraph_enabled",
            return_value=False,
        ),
        patch("torch.npu.Event", return_value="event"),
        patch("torch.npu.Stream", return_value="stream"),
        patch("torch.empty", return_value=torch.zeros(2, dtype=torch.int32)),
    ):
        runner = NPUModelRunner(vllm_config, torch.device("cpu"))
    assert runner.eplb == "eplb"
    assert runner.req_states == "req"
    assert runner.input_buffers == "buf"
    assert runner.speculator is None
    assert runner.use_spec_pp is False
    assert runner.decode_query_len == 1


def test_init_spec_pp_full_graph_and_speculator():
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=True))
    ascend_config = SimpleNamespace(eplb_config=SimpleNamespace(load_collection_phase="decode"))

    # Complete the fake with the fields NPUModelRunner reads (mirrors FinegrainedTPConfig).
    ascend_config.finegrained_tp_config = SimpleNamespace(
        oproj_tensor_parallel_size=0,
        lmhead_tensor_parallel_size=0,
        embedding_tensor_parallel_size=0,
        mlp_tensor_parallel_size=0,
    )
    spec_pp = SimpleNamespace(needs_aux_hidden_states=True)
    speculator = SimpleNamespace()
    with (
        patch("vllm_ascend.worker.v2.model_runner.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.worker.v2.model_runner.set_potential_max_tokens"),
        patch("vllm_ascend.worker.v2.model_runner.resolve_spec_pp_support", return_value=spec_pp),
        patch("vllm_ascend.worker.v2.model_runner.torch_cuda_wrapper", return_value=nullcontext()),
        patch(
            "vllm_ascend.worker.v2.model_runner.bypass_upstream_spec_pp_guard",
            return_value=nullcontext(True),
        ),
        patch("vllm_ascend.worker.v2.model_runner.restore_pp_after_upstream_init") as restore_pp,
        patch.object(
            GPUModelRunner,
            "__init__",
            lambda self, cfg, dev: _parent_init(self, cfg, dev, full_graph=True, speculative=True),
        ),
        patch("vllm_ascend.worker.v2.model_runner.AscendEPLBController", return_value="eplb") as eplb_cls,
        patch("vllm_ascend.worker.v2.model_runner.init_speculator", return_value=speculator),
        patch("vllm_ascend.worker.v2.model_runner.AscendRequestState", return_value="req"),
        patch("vllm_ascend.worker.v2.model_runner.AscendInputBuffers", return_value="buf"),
        patch("vllm_ascend.worker.v2.model_runner.set_cos_and_sin"),
        patch("vllm_ascend.worker.v2.model_runner.set_mc2_tokens_capacity"),
        patch("vllm_ascend.worker.v2.model_runner.set_mc2_mask"),
        patch("vllm_ascend.patch.worker.patch_v2.patch_spec_pp.install_upstream_spec_pp_protocol") as install_pp,
        patch("torch.npu.Stream", return_value="stream"),
        patch("torch.npu.Event", return_value="event"),
        patch("torch.empty", return_value=torch.zeros(2, dtype=torch.int32)),
        patch(
            "vllm_ascend.worker.v2.model_runner.breakable_cudagraph.is_breakable_cudagraph_enabled",
            return_value=True,
        ),
    ):
        runner = NPUModelRunner(vllm_config, torch.device("cpu"))
    restore_pp.assert_called_once()
    assert eplb_cls.call_args.kwargs["load_collection_phase"] == "decode"
    assert runner.use_aclgraph is True
    assert runner.use_aux_hidden_state_outputs is True
    assert runner.speculator is speculator
    assert speculator.update_stream is runner.update_stream
    assert runner.use_spec_pp is False
    install_pp.assert_not_called()
    assert runner.update_stream is not None
    assert runner.decode_query_len == 2


def test_sample_tokens_non_last_pp_uses_global_batch():
    runner = _make_runner()
    runner.is_last_pp_rank = False
    runner.use_spec_pp = False
    runner.speculator = None
    global_batch = object()
    runner.pcp_manager = MagicMock(spec=AscendPCPManager)
    runner.pcp_manager.global_batch = global_batch
    state = Mock(aux_hidden_states=None)
    replaced = object()
    state._replace.return_value = replaced
    runner.execute_model_state = state
    with patch.object(GPUModelRunner, "sample_tokens", return_value="out") as parent:
        assert runner.sample_tokens(None) == "out"
    state._replace.assert_called_once_with(input_batch=global_batch)
    assert runner.execute_model_state is replaced
    parent.assert_called_once_with(None)


def test_sample_tokens_spec_pp_broadcasts_draft_tokens():
    runner = _make_runner()
    runner.is_last_pp_rank = True
    runner.use_spec_pp = True
    runner.speculator = None
    runner.pcp_manager = None
    runner.pp_handler = MagicMock()
    with patch.object(GPUModelRunner, "sample_tokens", return_value="out"):
        assert runner.sample_tokens("g") == "out"
    runner.pp_handler.broadcast_draft_tokens.assert_not_called()


def test_initialize_kv_cache_installs_aclgraph_factory_and_pcp():
    runner = _make_runner()
    runner.vllm_config = SimpleNamespace()
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.pcp_manager = MagicMock(spec=AscendPCPManager)
    runner.model_state = SimpleNamespace(pcp_manager=None, kvpp_runtime=None)
    runner.speculator = SimpleNamespace()
    runner.model_config = SimpleNamespace(enable_return_routed_experts=True)
    runner.init_routed_experts_capturer = MagicMock()
    kv_cache_config = KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    original = vllm_model_runner.ModelCudaGraphManager
    seen = {}
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    def _super(self, kv_cache_config, kv_cache_allocation_context=None):
        self.kv_cache_config = kv_cache_config
        self.attn_groups = []
        seen["factory"] = vllm_model_runner.ModelCudaGraphManager
        seen["cfg"] = kv_cache_config

    with (
        patch.object(GPUModelRunner, "initialize_kv_cache", _super),
        patch("vllm_ascend.worker.v2.model_runner.ModelAclGraphManager", return_value="acl") as acl_cls,
        patch(
            "vllm_ascend.worker.v2.model_runner.KVPPRuntime.create_from_kv_cache",
            return_value="kvpp",
        ) as create_kvpp,
    ):
        runner.initialize_kv_cache(kv_cache_config)
        seen["factory"](runner.vllm_config, torch.device("cpu"), CUDAGraphMode.FULL, 1)

    assert seen["cfg"] == kv_cache_config
    assert vllm_model_runner.ModelCudaGraphManager is original
    acl_cls.assert_called_once()
    create_kvpp.assert_called_once()
    assert runner.kvpp == "kvpp"
    assert runner.model_state.kvpp_runtime == "kvpp"
    assert runner.pcp_manager.vllm_config is runner.vllm_config
    assert runner.model_state.pcp_manager is runner.pcp_manager
    assert runner.speculator.pcp_manager is runner.pcp_manager
    runner.init_routed_experts_capturer.assert_called_once_with()


def test_initialize_kv_cache_forwards_allocation_context():
    runner = _make_runner()
    runner.vllm_config = SimpleNamespace()
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.pcp_manager = None
    runner.model_state = SimpleNamespace(pcp_manager=None, kvpp_runtime=None)
    runner.speculator = None
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    called = False
    captured_kwargs: dict[str, object] = {}
    allocation_context = object()
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    def _super(self, kv_cache_config, **kwargs):
        nonlocal called
        called = True
        captured_kwargs.update(kwargs)
        self.kv_cache_config = kv_cache_config
        self.attn_groups = []

    with (
        patch.object(GPUModelRunner, "initialize_kv_cache", _super),
        patch("vllm_ascend.worker.v2.model_runner.ModelAclGraphManager", return_value="acl"),
        patch(
            "vllm_ascend.worker.v2.model_runner.KVPPRuntime.create_from_kv_cache",
            return_value="kvpp",
        ),
    ):
        runner.initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=allocation_context)

    assert called is True
    assert captured_kwargs["kv_cache_allocation_context"] is allocation_context


@pytest.mark.parametrize("moe_type", [MoECommType.MC2, MoECommType.FUSED_MC2])
def test_profile_run_dummy_reserves_mc2(moe_type):
    runner = _make_runner()
    runner.max_num_tokens = 16
    runner.vllm_config = SimpleNamespace()
    runner.get_model = MagicMock(return_value="m")
    runner._dummy_run = MagicMock()
    with (
        patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=4),
        patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=moe_type),
        patch("vllm_ascend.worker.v2.model_runner.override_mrv2_in_profile_run", return_value=nullcontext()),
        patch("vllm_ascend.worker.v2.model_runner.disable_compilation", return_value=nullcontext()),
        patch.object(GPUModelRunner, "profile_run") as parent,
    ):
        runner.profile_run()
    runner._dummy_run.assert_called_once_with(4, skip_attn=True, skip_eplb=True, is_profile=True)
    parent.assert_called_once_with()


def test_profile_run_skips_mc2_dummy_without_capacity():
    runner = _make_runner()
    runner.max_num_tokens = 16
    runner._dummy_run = MagicMock()
    with (
        patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner.override_mrv2_in_profile_run", return_value=nullcontext()),
        patch.object(GPUModelRunner, "profile_run"),
    ):
        runner.profile_run()
    runner._dummy_run.assert_not_called()


def _prepare_inputs_runner(*, draft=False, full_cg=False, use_dcp=False, use_pp=False, rswa=False, speculator=False):
    runner = _make_runner()
    runner.max_num_reqs = 4
    runner.device = torch.device("cpu")
    runner.decode_query_len = 1
    runner.use_dcp = use_dcp
    runner.dcp_size = 2
    runner.dcp_rank = 0
    runner.cp_interleave = False
    runner.use_pp = use_pp
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL if full_cg else CUDAGraphMode.NONE)
    runner.model_config = SimpleNamespace(rswa_window=(4 if rswa else None))
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.eplb = SimpleNamespace(set_batch_phase=MagicMock())
    runner.pcp_manager = None
    runner.speculator = object() if speculator else None
    runner.num_computed_tokens_event = MagicMock()
    runner.num_computed_tokens_cpu = torch.tensor([3, 4, 0, 0], dtype=torch.int32)
    runner.input_buffers = AscendInputBuffers(4, 16, runner.device)
    runner.input_buffers.dcp_local_seq_lens = torch.zeros(4, dtype=torch.int32)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=torch.tensor([1, 2, 0, 0], dtype=torch.int32),
        num_computed_tokens_np=np.array([1, 2, 0, 0], dtype=np.int32),
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        next_prefill_tokens=MagicMock(),
        all_token_ids=SimpleNamespace(gpu=MagicMock()),
        prefill_len=SimpleNamespace(gpu=MagicMock()),
        last_sampled_tokens=MagicMock(),
        draft_tokens=MagicMock(),
        max_seq_len=np.array([8, 8, 0, 0], dtype=np.int32),
        prompt_len=SimpleNamespace(gpu=torch.tensor([4, 4, 0, 0], dtype=torch.int32)),
    )
    draft_tokens = {"r0": [9], "r1": [8, 7]} if draft else {}
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens=draft_tokens,
        has_structured_output_requests=False,
        num_scheduled_tokens={"r0": 2, "r1": 2},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r0"] if speculator else []),
    )
    batch_req_state = SimpleNamespace(
        num_tokens=4,
        req_ids=["r0", "r1"],
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        has_prefill=True,
        prefill_len_np=np.array([2, 2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        is_prefilling_np=np.array([True, True]),
    )
    batch_desc = SimpleNamespace(
        num_tokens=8 if full_cg else 4,
        num_reqs=2,
        cg_mode=CUDAGraphMode.FULL if full_cg else CUDAGraphMode.NONE,
    )
    return runner, scheduler_output, batch_req_state, batch_desc


def _fake_async_copy(src, device=None, out=None):
    tensor = torch.as_tensor(src, dtype=torch.int32)
    if out is not None:
        n = min(out.numel(), tensor.numel())
        out.view(-1)[:n].copy_(tensor.view(-1)[:n])
        return out
    return tensor


def _run_prepare_inputs(
    runner,
    scheduler_output,
    batch_req_state,
    batch_desc,
    *,
    prefill_inputs=None,
    combine_tokens=None,
):
    batch = SimpleNamespace(positions=torch.zeros(4, dtype=torch.int32))

    def _partition(_pcp_manager, input_batch, **_kwargs):
        return input_batch

    with (
        patch("vllm_ascend.worker.v2.model_runner.async_copy_to_gpu", side_effect=_fake_async_copy),
        patch("vllm_ascend.worker.v2.model_runner.build_attn_state", return_value="attn"),
        patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs", side_effect=prefill_inputs),
        patch("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens"),
        patch("vllm_ascend.worker.v2.model_runner.prepare_dcp_local_seq_lens", create=True),
        patch(
            "vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens",
            return_value=torch.tensor([0, 1], dtype=torch.int32),
            side_effect=combine_tokens,
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.expand_idx_mapping",
            return_value=(torch.tensor([0, 1], dtype=torch.int32), torch.zeros(2, dtype=torch.int32)),
        ),
        patch("vllm_ascend.worker.v2.model_runner.AscendInputBatch", return_value=batch),
        patch.object(
            vllm_model_runner,
            "pcp",
            SimpleNamespace(maybe_partition_pcp_batch=_partition),
        ),
        patch("vllm_ascend.worker.v2.model_runner.update_cos_sin"),
    ):
        return runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc), batch


def test_prepare_inputs_common_path():
    runner, scheduler_output, batch_req_state, batch_desc = _prepare_inputs_runner()
    out, partitioned = _run_prepare_inputs(runner, scheduler_output, batch_req_state, batch_desc)
    assert out is partitioned
    runner.eplb.set_batch_phase.assert_called_once_with(True)
    np.testing.assert_array_equal(runner.input_buffers.seq_lens_cpu[:2], np.array([3, 4], dtype=np.int32))


@pytest.mark.parametrize("num_spec_tokens", [0, 1, 5])
@pytest.mark.parametrize("full_cg", [False, True])
def test_pd_tail_input_survives_decode_reclassification(num_spec_tokens, full_cg):
    runner, scheduler_output, _, batch_desc = _prepare_inputs_runner(full_cg=full_cg)
    query_len = 1 + num_spec_tokens
    runner.decode_query_len = query_len
    batch_state = _make_batch_state([127, 128], [query_len, query_len], [128, 128])
    batch_state = batch_state._replace(req_ids=["r0", "r1"])
    scheduler_output.num_scheduled_tokens = dict.fromkeys(batch_state.req_ids, query_len)
    scheduler_output.scheduled_spec_decode_tokens = (
        {req_id: [-1] * num_spec_tokens for req_id in batch_state.req_ids} if num_spec_tokens else {}
    )
    batch_desc.num_tokens = 2 * query_len
    runner.input_buffers.input_ids.fill_(-999)
    runner.req_states.num_computed_tokens.gpu = torch.tensor([127, 128], dtype=torch.int32)
    runner.req_states.prefill_len.gpu = torch.tensor([128, 128], dtype=torch.int32)
    runner.req_states.all_token_ids.gpu = torch.arange(2 * 144, dtype=torch.int32).reshape(2, 144)
    runner.req_states.last_sampled_tokens = torch.tensor([700, 701], dtype=torch.int32)
    runner.req_states.draft_tokens = torch.full((2, num_spec_tokens), 702, dtype=torch.int32)
    runner.req_states.next_prefill_tokens = torch.full((1, 2), -999, dtype=torch.int32)

    with (
        patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(batch_state, None)),
        patch("vllm_ascend.worker.v2.model_runner.is_pd_decode_recompute_scheduler_enabled", return_value=True),
    ):
        gathered, uniform = runner.gather_batch_req_state(scheduler_output, False)
    assert not gathered.has_prefill
    assert uniform == query_len

    def prepare_prefill(input_ids, next_tokens, idx_mapping, query_start, all_tokens, prefill_len, computed):
        # CPU reference for the upstream kernel: prepare actual prompt tails,
        # independently of the graph-dispatch classification.
        for row, idx in enumerate(idx_mapping.tolist()):
            pos = int(computed[idx])
            if pos >= int(prefill_len[idx]):
                continue
            start, end = query_start[row : row + 2].tolist()
            input_ids[start:end] = all_tokens[idx, pos : pos + end - start]
            next_tokens[:, idx] = 0  # This step reaches the end of the prompt.

    def combine(input_ids, idx_mapping, sampled, query_start, seq_lens, prefill_len, drafts, *args):
        # The upstream combine kernel preserves the prompt-tail position.
        # It must already contain the real token, never the stale buffer value
        # or last_sampled_tokens for this newly admitted request.
        assert input_ids[0].item() == 127
        assert runner.req_states.next_prefill_tokens[0, 0].item() == 0
        input_ids[query_len] = sampled[1]
        for row in range(2):
            start = row * query_len + 1
            input_ids[start : start + num_spec_tokens] = drafts[row]
        return torch.arange(2 * query_len, dtype=torch.int32)

    _run_prepare_inputs(
        runner, scheduler_output, gathered, batch_desc, prefill_inputs=prepare_prefill, combine_tokens=combine
    )
    expected = [127] + [702] * num_spec_tokens + [701] + [702] * num_spec_tokens
    assert runner.input_buffers.input_ids[: 2 * query_len].tolist() == expected
    assert not gathered.has_prefill  # Keep decode graph eligibility.


def test_decode_without_prompt_tail_skips_prefill_preparation():
    runner, scheduler_output, batch_state, batch_desc = _prepare_inputs_runner()
    batch_state.has_prefill = False
    batch_state.num_computed_prefill_tokens_np = batch_state.prefill_len_np.copy()
    prepare = Mock()
    _run_prepare_inputs(runner, scheduler_output, batch_state, batch_desc, prefill_inputs=prepare)
    prepare.assert_not_called()


def test_prepare_inputs_covers_draft_full_dcp_pp_and_rswa():
    runner, scheduler_output, batch_req_state, batch_desc = _prepare_inputs_runner(
        draft=True, full_cg=True, use_dcp=True, use_pp=True, rswa=True, speculator=True
    )
    out, partitioned = _run_prepare_inputs(runner, scheduler_output, batch_req_state, batch_desc)
    assert out is partitioned
    runner.num_computed_tokens_event.synchronize.assert_called_once_with()
    assert runner.req_states.num_computed_tokens_cpu[0] == 3


def test_postprocess_sampled_copies_only_with_speculator():
    runner = _make_runner()
    runner.speculator = object()
    runner._copy_num_computed_tokens_to_cpu = MagicMock()
    with patch.object(GPUModelRunner, "postprocess_sampled") as parent:
        runner.postprocess_sampled("idx", "tok", 1, 0, query_start_loc="q")
    parent.assert_called_once_with("idx", "tok", 1, 0, "q")
    runner._copy_num_computed_tokens_to_cpu.assert_called_once_with()

    runner.speculator = None
    runner._copy_num_computed_tokens_to_cpu.reset_mock()
    with patch.object(GPUModelRunner, "postprocess_sampled"):
        runner.postprocess_sampled("idx", "tok", 1, 0)
    runner._copy_num_computed_tokens_to_cpu.assert_not_called()


def test_copy_num_computed_tokens_to_cpu_records_event():
    import vllm_ascend.worker.v2.model_runner as model_runner_mod

    runner = _make_runner()
    stream = MagicMock()
    runner.num_computed_tokens_stream = stream
    runner.num_computed_tokens_cpu = MagicMock()
    runner.num_computed_tokens_event = MagicMock()
    runner.req_states = SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)))
    npu_cm = MagicMock()
    npu_cm.__enter__.return_value = None
    npu_cm.__exit__.return_value = False
    default_stream = MagicMock()
    with (
        patch.object(model_runner_mod.torch.cuda, "current_stream", return_value=default_stream),
        patch.object(model_runner_mod.torch.npu, "stream", return_value=npu_cm) as npu_stream,
    ):
        runner._copy_num_computed_tokens_to_cpu()
    npu_stream.assert_called_once_with(stream)
    stream.wait_stream.assert_called_once_with(default_stream)
    runner.num_computed_tokens_cpu.copy_.assert_called_once()
    runner.num_computed_tokens_event.record.assert_called_once_with()
