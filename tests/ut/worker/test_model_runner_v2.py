import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

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
    captured = {
        name: getattr(manager.input_buffers, name)
        for name in ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens")
    }
    for name, value in captured.items():
        value.fill_(False if name == "is_padding" else 99)
    manager.input_buffers.seq_lens_np.fill(99)
    with patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"):
        dummy = AscendInputBatch.make_dummy(num_reqs, num_tokens, runner.input_buffers)

    block_tables, slots = runner.prepare_dummy_attn(dummy)

    for name, value in captured.items():
        expected = getattr(dummy, name)
        torch.testing.assert_close(value[: len(expected)], expected)
    np.testing.assert_array_equal(manager.input_buffers.seq_lens_np[:num_reqs], dummy.seq_lens_np)
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
    parent.assert_called_once_with(dummy)


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
        patch("vllm_ascend.patch.worker.patch_v2.patch_spec_pp.install_spec_pp_token_broadcast") as install_pp,
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
    assert runner.use_spec_pp is True
    assert runner.use_aux_hidden_state_outputs is True
    assert runner.speculator is speculator
    assert speculator.update_stream == "stream"
    install_pp.assert_called_once()
    assert runner.update_stream == "stream"
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
    runner.pp_handler.broadcast_draft_tokens.assert_called_once_with()


def test_initialize_kv_cache_installs_aclgraph_factory_and_pcp():
    runner = _make_runner()
    runner.vllm_config = SimpleNamespace()
    runner.pcp_manager = MagicMock(spec=AscendPCPManager)
    runner.model_state = SimpleNamespace(pcp_manager=None)
    runner.speculator = SimpleNamespace()
    runner.model_config = SimpleNamespace(enable_return_routed_experts=True)
    runner.init_routed_experts_capturer = MagicMock()
    original = vllm_model_runner.ModelCudaGraphManager
    seen = {}

    def _super(self, kv_cache_config):
        seen["factory"] = vllm_model_runner.ModelCudaGraphManager
        seen["cfg"] = kv_cache_config

    with (
        patch.object(GPUModelRunner, "initialize_kv_cache", _super),
        patch("vllm_ascend.worker.v2.model_runner.ModelAclGraphManager", return_value="acl") as acl_cls,
    ):
        runner.initialize_kv_cache("kv")
        seen["factory"](runner.vllm_config, torch.device("cpu"), CUDAGraphMode.FULL, 1)

    assert seen["cfg"] == "kv"
    assert vllm_model_runner.ModelCudaGraphManager is original
    acl_cls.assert_called_once()
    assert runner.pcp_manager.vllm_config is runner.vllm_config
    assert runner.model_state.pcp_manager is runner.pcp_manager
    assert runner.speculator.pcp_manager is runner.pcp_manager
    runner.init_routed_experts_capturer.assert_called_once_with()


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


def _run_prepare_inputs(runner, scheduler_output, batch_req_state, batch_desc, *, version_028=False):
    batch = object()
    partitioned = object()
    with (
        patch("vllm_ascend.worker.v2.model_runner.async_copy_to_gpu", side_effect=_fake_async_copy),
        patch("vllm_ascend.worker.v2.model_runner.build_attn_state", return_value="attn"),
        patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs"),
        patch("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens"),
        patch("vllm_ascend.worker.v2.model_runner.prepare_dcp_local_seq_lens"),
        patch(
            "vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens",
            return_value=torch.tensor([0, 1], dtype=torch.int32),
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.expand_idx_mapping",
            return_value=(torch.tensor([0, 1], dtype=torch.int32), torch.zeros(2, dtype=torch.int32)),
        ),
        patch("vllm_ascend.worker.v2.model_runner.AscendInputBatch", return_value=batch),
        patch.object(vllm_model_runner.pcp, "maybe_partition_pcp_batch", return_value=partitioned),
        patch("vllm_ascend.worker.v2.model_runner.update_cos_sin"),
        patch("vllm_ascend.worker.v2.model_runner.vllm_version_is", return_value=version_028),
    ):
        return runner.prepare_inputs(scheduler_output, batch_req_state, batch_desc), partitioned


def test_prepare_inputs_common_path():
    runner, scheduler_output, batch_req_state, batch_desc = _prepare_inputs_runner()
    out, partitioned = _run_prepare_inputs(runner, scheduler_output, batch_req_state, batch_desc)
    assert out is partitioned
    runner.eplb.set_batch_phase.assert_called_once_with(True)
    np.testing.assert_array_equal(runner.input_buffers.seq_lens_cpu[:2], np.array([3, 4], dtype=np.int32))


def test_prepare_inputs_covers_draft_full_dcp_pp_and_rswa():
    runner, scheduler_output, batch_req_state, batch_desc = _prepare_inputs_runner(
        draft=True, full_cg=True, use_dcp=True, use_pp=True, rswa=True, speculator=True
    )
    out, partitioned = _run_prepare_inputs(
        runner, scheduler_output, batch_req_state, batch_desc, version_028=True
    )
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
        patch("torch.cuda.current_stream", return_value=default_stream),
        patch("torch.npu.stream", return_value=npu_cm) as npu_stream,
    ):
        runner._copy_num_computed_tokens_to_cpu()
    npu_stream.assert_called_once_with(stream)
    stream.wait_stream.assert_called_once_with(default_stream)
    runner.num_computed_tokens_cpu.copy_.assert_called_once()
    runner.num_computed_tokens_event.record.assert_called_once_with()
