from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import set_dummy_context
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.utils import get_uniform_decode_token_count

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.fixture
def buffers():
    return AscendInputBuffers(16, 64, torch.device("cpu"))


@pytest.fixture
def runner(buffers):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_buffers = buffers
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    runner.decode_query_len = 2
    runner.pcp_manager = None
    runner.ubatch_runner = None
    runner.model_config = SimpleNamespace(is_hybrid=False)
    runner.adaptive_verification = None
    runner.max_num_reqs = 16
    runner.max_num_tokens = 64
    runner.speculator = None
    runner.eplb = Mock()
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    return runner


def _make_graph_manager(buffers, mode, uniform, capture_size):
    manager = ModelAclGraphManager.__new__(ModelAclGraphManager)
    manager.vllm_config = SimpleNamespace(
        speculative_config=None, cache_config=SimpleNamespace(use_kda_recoverssm=False)
    )
    manager.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[capture_size], max_cudagraph_capture_size=64)
    manager.cudagraph_mode = mode
    manager.decode_query_len = uniform
    manager.max_num_reqs = len(buffers.seq_lens_cpu)
    manager.varlen_decode = False
    manager.ubatch_runner = None
    manager.lora_capture_cases = [0]
    manager._capture_descs = {}
    manager._candidates = {}
    manager._lora_dispatch_map = {}
    manager._graphs_captured = True
    manager._init_candidates()
    return manager


@pytest.mark.parametrize("num_reqs,valid_tokens,padded_tokens", [(1, 2, 16), (3, 6, 8), (3, 7, 16), (2, 4, 4)])
def test_piecewise_dummy_keeps_queries_and_pads_only_inputs(buffers, num_reqs, valid_tokens, padded_tokens):
    buffers.input_ids.fill_(99)
    buffers.positions.fill_(99)
    buffers.is_padding.fill_(False)
    with (
        patch.object(buffers, "dummy_num_tokens", valid_tokens),
        patch("vllm_ascend.worker.v2.input_batch.update_cos_sin") as update_rotary,
    ):
        batch = AscendInputBatch.make_dummy(num_reqs, padded_tokens, buffers)

    assert batch.num_tokens == valid_tokens
    assert batch.num_tokens_after_padding == padded_tokens
    assert batch.num_reqs == batch.num_reqs_after_padding == num_reqs
    assert batch.query_start_loc_np[-1] == valid_tokens
    np.testing.assert_array_equal(np.diff(batch.query_start_loc_np), batch.num_scheduled_tokens)
    np.testing.assert_array_equal(batch.seq_lens_np, batch.num_scheduled_tokens)
    np.testing.assert_array_equal(batch.seq_lens.numpy(), batch.num_scheduled_tokens)
    np.testing.assert_array_equal(batch.seq_lens_cpu_upper_bound.numpy(), batch.num_scheduled_tokens)
    torch.testing.assert_close(batch.query_start_loc, torch.from_numpy(batch.query_start_loc_np))
    torch.testing.assert_close(batch.logits_indices, batch.query_start_loc[1:] - 1)
    for name in ("input_ids", "positions", "is_padding"):
        assert len(getattr(batch, name)) == padded_tokens
        assert getattr(batch, name).data_ptr() == getattr(buffers, name).data_ptr()
    assert torch.count_nonzero(batch.input_ids) == torch.count_nonzero(batch.positions) == 0
    assert batch.is_padding.all()
    assert torch.all(buffers.input_ids[padded_tokens:] == 99)
    assert torch.all(buffers.positions[padded_tokens:] == 99)
    assert not buffers.is_padding[padded_tokens:].any()
    update_rotary.assert_called_once_with(batch.positions)
    assert buffers.dummy_num_tokens is None


def test_capture_without_dummy_scope_keeps_existing_layout(buffers):
    with patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"):
        batch = AscendInputBatch.make_dummy(1, 16, buffers)
    assert batch.num_tokens == batch.num_tokens_after_padding == 16
    assert batch.num_scheduled_tokens.tolist() == [16]


@pytest.mark.parametrize("draft_mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE])
@pytest.mark.parametrize("capture_size,uniform", [(16, 1), (16, 2), (16, 4), (32, 2), (32, 4)])
def test_mixed_dp_dummy_keeps_graphs_and_reuses_sync(buffers, runner, draft_mode, uniform, capture_size):
    runner.decode_query_len = uniform
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"dummy": uniform}, total_num_scheduled_tokens=uniform)
    _, original_uniform = runner.gather_batch_req_state(scheduler_output, dummy_run=True)
    target = _make_graph_manager(buffers, CUDAGraphMode.PIECEWISE, uniform, capture_size)
    draft = _make_graph_manager(buffers, draft_mode, uniform, capture_size)

    def other_rank_votes(votes, group):
        assert group is cpu_group
        # The other ranks decode real requests with the same query width.
        votes[0, 1:] = torch.tensor([2 * uniform, 3 * uniform, uniform])
        votes[1, 1:] = CUDAGraphMode.PIECEWISE.value
        votes[2, 1:] = uniform
        votes[3, 1:] = uniform
        votes[5, 1:] = torch.tensor([2, 3, 1])

    def execute_dummy(num_tokens, **kwargs):
        assert buffers.dummy_num_tokens == uniform
        desc, sync = dispatch_cg_and_sync_dp(target, 1, uniform, original_uniform, 4, 0, max_query_len=uniform)
        assert sync is not None
        assert sync.uniform_token_count == uniform
        assert not sync.eager
        assert desc.cg_mode == CUDAGraphMode.PIECEWISE
        assert desc.num_tokens == capture_size
        batch = AscendInputBatch.make_dummy(1, desc.num_tokens, buffers, desc.max_query_len)
        actual_uniform = get_uniform_decode_token_count(
            batch.num_reqs, batch.num_tokens, int(batch.num_scheduled_tokens.max()), batch.has_prefill
        )
        assert actual_uniform == uniform
        draft_desc, reused_sync = dispatch_cg_and_sync_dp(
            draft, batch.num_reqs, batch.num_tokens_after_padding, actual_uniform, 4, 0, dp_sync=sync
        )
        assert draft_desc.cg_mode == draft_mode.decode_mode()
        assert draft_desc.num_tokens == capture_size
        assert reused_sync is sync
        return None, None

    cpu_group = object()
    with (
        patch.object(GPUModelRunner, "_dummy_run", side_effect=execute_dummy),
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False),
        patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"),
        patch("vllm.v1.worker.gpu.dp_utils.get_dp_group", return_value=SimpleNamespace(cpu_group=cpu_group)),
        patch("vllm.v1.worker.gpu.dp_utils.dist.all_reduce", side_effect=other_rank_votes) as collective,
    ):
        runner._dummy_run(uniform, uniform_decode=True)
    assert collective.call_count == 1
    assert buffers.dummy_num_tokens is None


@pytest.mark.parametrize(
    "kwargs,mode,pcp,is_hybrid,expected",
    [
        ({}, CUDAGraphMode.PIECEWISE, None, False, 1),
        ({"uniform_decode": True}, CUDAGraphMode.PIECEWISE, None, False, 2),
        ({"context_len": 1}, CUDAGraphMode.PIECEWISE, None, False, 1),
        ({"is_profile": True}, CUDAGraphMode.PIECEWISE, None, False, 1),
        ({}, CUDAGraphMode.PIECEWISE, object(), False, None),
        ({}, CUDAGraphMode.PIECEWISE, None, True, None),
        ({}, CUDAGraphMode.FULL_AND_PIECEWISE, None, False, None),
        ({}, CUDAGraphMode.NONE, None, False, None),
    ],
)
def test_dummy_scope_restores_state_on_exception(buffers, runner, kwargs, mode, pcp, is_hybrid, expected):
    runner.compilation_config.cudagraph_mode = mode
    runner.pcp_manager = pcp
    runner.model_config.is_hybrid = is_hybrid
    buffers.dummy_num_tokens = 7

    def execute_dummy(*args, **kwargs):
        assert buffers.dummy_num_tokens == expected
        raise RuntimeError("test")

    with (
        patch.object(GPUModelRunner, "_dummy_run", side_effect=execute_dummy),
        pytest.raises(RuntimeError, match="test"),
    ):
        runner._dummy_run(1, **kwargs)
    assert buffers.dummy_num_tokens == 7


@pytest.mark.parametrize("inner_mode,expected", [(CUDAGraphMode.PIECEWISE, 6), (CUDAGraphMode.NONE, None)])
def test_nested_dummy_runs_restore_outer_query_tokens(buffers, runner, inner_mode, expected):
    buffers.dummy_num_tokens = 7

    def execute_dummy(num_tokens, **kwargs):
        if num_tokens == 2:
            assert buffers.dummy_num_tokens == 2
            with (
                patch.object(runner.compilation_config, "cudagraph_mode", inner_mode),
                pytest.raises(RuntimeError, match="inner dummy"),
            ):
                runner._dummy_run(6)
            assert buffers.dummy_num_tokens == 2
            return None, None
        assert buffers.dummy_num_tokens == expected
        raise RuntimeError("inner dummy")

    with patch.object(GPUModelRunner, "_dummy_run", side_effect=execute_dummy):
        runner._dummy_run(2)
    assert buffers.dummy_num_tokens == 7


@pytest.mark.parametrize("num_tokens,uniform_decode", [(1, False), (1, True), (6, True), (16, False)])
def test_profile_dummy_keeps_eager_layout(buffers, runner, num_tokens, uniform_decode):
    target = _make_graph_manager(buffers, CUDAGraphMode.PIECEWISE, runner.decode_query_len, 16)

    def execute_dummy(num_tokens, **kwargs):
        assert kwargs["is_profile"]
        query_tokens = max(num_tokens, runner.decode_query_len) if uniform_decode else num_tokens
        num_reqs = query_tokens // runner.decode_query_len if uniform_decode else min(query_tokens, runner.max_num_reqs)
        desc, sync = dispatch_cg_and_sync_dp(
            target,
            num_reqs,
            query_tokens,
            runner.decode_query_len if uniform_decode else None,
            1,
            0,
            need_eager=kwargs["is_profile"],
        )
        assert desc.cg_mode == CUDAGraphMode.NONE
        assert sync is None
        batch = AscendInputBatch.make_dummy(num_reqs, desc.num_tokens, buffers)
        assert batch.num_tokens == batch.num_tokens_after_padding == query_tokens
        assert len(batch.input_ids) == len(batch.positions) == query_tokens
        assert batch.query_start_loc_np[-1] == query_tokens
        np.testing.assert_array_equal(batch.seq_lens_np, batch.num_scheduled_tokens)
        return None, None

    with (
        patch.object(GPUModelRunner, "_dummy_run", side_effect=execute_dummy),
        patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"),
    ):
        runner._dummy_run(num_tokens, uniform_decode=uniform_decode, is_profile=True)
    assert buffers.dummy_num_tokens is None


@pytest.mark.parametrize("num_tokens,expected_mode", [(16, CUDAGraphMode.PIECEWISE), (32, CUDAGraphMode.NONE)])
def test_context_dummy_keeps_capture_and_tail_layouts(buffers, runner, num_tokens, expected_mode):
    target = _make_graph_manager(buffers, CUDAGraphMode.PIECEWISE, runner.decode_query_len, 16)
    context_len = 8

    def execute_dummy(num_tokens, **kwargs):
        assert buffers.dummy_num_tokens == num_tokens
        num_reqs = min(num_tokens, runner.max_num_reqs)
        desc, _ = dispatch_cg_and_sync_dp(target, num_reqs, num_tokens, None, 1, 0)
        assert desc.cg_mode == expected_mode
        batch = AscendInputBatch.make_dummy(num_reqs, desc.num_tokens, buffers, desc.max_query_len)
        assert batch.num_tokens == batch.num_tokens_after_padding == num_tokens
        blocks = SimpleNamespace(
            input_block_tables=[torch.zeros((num_reqs, 8), dtype=torch.int32)],
            kernel_block_sizes=[16],
            blocks_per_kv_block=[1],
        )
        set_dummy_context(batch, blocks, kwargs["context_len"], 64, 128)
        expected_positions = context_len + np.tile(np.arange(num_tokens // num_reqs), num_reqs)
        np.testing.assert_array_equal(batch.positions.numpy(), expected_positions)
        np.testing.assert_array_equal(batch.seq_lens.numpy(), batch.num_scheduled_tokens + context_len)
        return None, None

    with (
        patch.object(GPUModelRunner, "_dummy_run", side_effect=execute_dummy),
        patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"),
    ):
        runner._dummy_run(num_tokens, context_len=context_len)
    assert buffers.dummy_num_tokens is None


def test_dummy_slot_mapping_uses_padded_persistent_buffer(buffers, runner):
    slots = torch.full((1, 32), 99, dtype=torch.int64)

    def get_slots(num_tokens):
        slots.fill_(-1)
        return slots[:, :num_tokens]

    runner.block_tables = SimpleNamespace(
        slot_mappings=slots,
        get_dummy_block_tables=Mock(return_value=()),
        get_dummy_slot_mappings=Mock(side_effect=get_slots),
    )
    batch = SimpleNamespace(num_reqs=1, num_tokens=2, num_tokens_after_padding=16)
    block_tables, actual_slots = runner.prepare_dummy_attn(batch)
    assert block_tables == ()
    assert actual_slots.shape == (1, 16)
    assert actual_slots.data_ptr() == slots.data_ptr()
    assert torch.all(actual_slots == -1)
    runner.block_tables.get_dummy_slot_mappings.assert_called_once_with(2)


@pytest.mark.parametrize(
    "mode,is_dummy,expected_input_tokens",
    [(CUDAGraphMode.PIECEWISE, True, 16), (CUDAGraphMode.PIECEWISE, False, 2), (CUDAGraphMode.NONE, True, 2)],
)
def test_attention_keeps_actual_queries_and_padded_dummy_inputs(buffers, mode, is_dummy, expected_input_tokens):
    with patch.object(buffers, "dummy_num_tokens", 2), patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"):
        batch = AscendInputBatch.make_dummy(1, 16, buffers)
    batch.is_dummy = is_dummy
    state = AscendModelState.__new__(AscendModelState)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1), num_speculative_tokens=0
    )
    state.max_model_len = 128
    state.pcp_manager = None
    state.kvpp_runtime = None
    state.kvpp_is_dummy_run = False
    state.device_metadata = None
    with patch("vllm_ascend.worker.v2.model_states.default.build_attn_metadata", return_value={}) as build:
        assert state.prepare_attn(batch, mode, (), torch.full((1, 16), -1), [], None) == {}
    kwargs = build.call_args.kwargs
    assert kwargs["num_input_tokens"] == expected_input_tokens
    assert kwargs.get("num_actual_tokens", kwargs["num_tokens"]) == 2
    assert kwargs["max_query_len"] == 2
    assert kwargs["query_start_loc_cpu"].tolist() == [0, 2]
    assert kwargs["seq_lens_np"].tolist() == [2]
