# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import BatchDescriptor

import vllm_ascend.spec_decode.deepseek_v4_dspark_proposer as dspark_module
import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.spec_decode.deepseek_v4_dspark_proposer import (
    AscendDeepSeekV4DSparkProposer,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@dataclass
class _DraftConfig:
    compilation_config: object
    parallel_config: object
    scheduler_config: object
    speculative_config: object
    lora_config: object | None = None
    model_config: object | None = None


class _CompilationConfig:
    def __init__(self):
        self.cudagraph_capture_sizes = [6, 12, 18]
        self.max_cudagraph_capture_size = 18

    def __copy__(self):
        result = type(self)()
        result.cudagraph_capture_sizes = copy(self.cudagraph_capture_sizes)
        result.max_cudagraph_capture_size = self.max_cudagraph_capture_size
        return result

    def adjust_cudagraph_sizes_for_spec_decode(self, query_len, tp_size):
        del tp_size
        self.cudagraph_capture_sizes = [query_len, query_len * 2]
        self.max_cudagraph_capture_size = query_len * 2


@pytest.mark.parametrize("block_size", [5, 7])
def test_draft_dispatcher_uses_target_request_rows(monkeypatch, block_size):
    created = []

    class FakeDispatcher:
        def __init__(self, config):
            self.config = config
            self.mode = None
            self.query_len = None
            self.uniform_decode_query_len = 6
            created.append(self)

        def initialize_cudagraph_keys(self, mode, query_len):
            self.mode = mode
            self.query_len = query_len

        def get_capture_descs(self):
            sizes = self.config.compilation_config.cudagraph_capture_sizes
            return [
                (
                    CUDAGraphMode.FULL,
                    [BatchDescriptor(size, size // self.query_len, True) for size in sizes],
                )
            ]

    monkeypatch.setattr(dspark_module, "CudagraphDispatcher", FakeDispatcher)
    target_dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [
            (
                CUDAGraphMode.FULL,
                [
                    BatchDescriptor(6, 1, True),
                    BatchDescriptor(18, 3, True),
                    BatchDescriptor(24, 4, True),
                ],
            )
        ]
    )
    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer.block_size = block_size
    proposer.use_cuda_graph = True
    proposer.runner = SimpleNamespace(cudagraph_dispatcher=target_dispatcher)
    proposer.vllm_config = _DraftConfig(
        compilation_config=_CompilationConfig(),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=SimpleNamespace(num_speculative_tokens=block_size),
    )

    proposer.initialize_cudagraph_keys(CUDAGraphMode.FULL)

    assert created[0].mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert created[0].query_len == block_size
    assert created[0].uniform_decode_query_len == block_size
    assert created[0].config.compilation_config.cudagraph_capture_sizes == [
        block_size,
        block_size * 3,
        block_size * 4,
    ]
    assert proposer.get_cudagraph_capture_sizes() == [block_size, block_size * 3, block_size * 4]


def test_load_model_uses_dspark_graph_runnable(monkeypatch):
    base_graph_flags = []
    wrappers = []

    class FakeWrapper:
        def __init__(self, runnable, config, runtime_mode, *, use_eagle, enable_enpu):
            wrappers.append((runnable, config, runtime_mode, use_eagle, enable_enpu))

    def fake_load_model(self, model):
        del model
        base_graph_flags.append(self.use_cuda_graph)

    monkeypatch.setattr(dspark_module.AscendDsparkProposer, "load_model", fake_load_model)
    monkeypatch.setattr(dspark_module, "ACLGraphWrapper", FakeWrapper)
    monkeypatch.setattr(
        dspark_module.torch,
        "npu",
        SimpleNamespace(Stream=lambda: "stream"),
        raising=False,
    )

    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer._draft_aclgraph_enabled = True
    proposer.use_cuda_graph = True
    proposer.vllm_config = SimpleNamespace()
    proposer.use_eagle = False
    proposer.enable_enpu = True

    proposer.load_model(object())

    assert base_graph_flags == [False]
    assert proposer.use_cuda_graph is True
    assert proposer.update_stream == "stream"
    assert len(wrappers) == 1
    assert wrappers[0][0].__func__ is AscendDeepSeekV4DSparkProposer._run_model_from_graph_buffers
    assert wrappers[0][2:] == (CUDAGraphMode.FULL, False, True)


def test_eager_drafter_config_does_not_mutate_target(monkeypatch):
    target_model_config = SimpleNamespace(enforce_eager=False)
    target_compilation_config = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    target_config = _DraftConfig(
        compilation_config=target_compilation_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(),
        speculative_config=SimpleNamespace(),
        model_config=target_model_config,
    )
    monkeypatch.setattr(
        dspark_module.AscendDsparkProposer,
        "_create_draft_vllm_config",
        lambda self: target_config,
    )
    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer._draft_aclgraph_enabled = False

    draft_config = proposer._create_draft_vllm_config()
    draft_config.model_config.enforce_eager = True
    draft_config.compilation_config.mode = CompilationMode.NONE
    draft_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    assert target_model_config.enforce_eager is False
    assert target_compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert target_compilation_config.cudagraph_mode == CUDAGraphMode.FULL


def test_padding_uses_complete_inactive_request_rows_and_stable_buffers():
    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer.block_size = 3
    proposer.max_batch_size = 4
    proposer.max_query_tokens = 12
    proposer.parallel_drafting_token_id = 99
    proposer.input_ids = torch.tensor([1, 2, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7])
    proposer.positions = torch.arange(12, dtype=torch.int32)
    proposer._token_to_req_buffer = torch.tensor([0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32)
    proposer._seed_buffer = torch.tensor([11, 22, 33, 44])
    proposer._query_start_loc_buffer = torch.tensor([0, 3, 9, 9, 9], dtype=torch.int32)
    proposer._query_start_loc_cpu_buffer = proposer._query_start_loc_buffer.cpu().clone()
    proposer._query_start_loc_cpu_base = torch.arange(5, dtype=torch.int32) * proposer.block_size
    proposer.arange_dspark = torch.arange(12, dtype=torch.int32)
    proposer._seq_lens_buffer = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    proposer._seq_lens_cpu_buffer = proposer._seq_lens_buffer.cpu().clone()
    proposer._query_slot_buffers = {0: torch.tensor([100, 101, 102, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32)}
    proposer._context_slot_buffers = {}
    proposer._query_slots_by_gid = {0: proposer._query_slot_buffers[0][:3]}
    proposer._block_table_buffers = {0: torch.tensor([[10, 11], [9, 9], [9, 9], [9, 9]], dtype=torch.int32)}
    proposer._block_tables_by_gid = {0: proposer._block_table_buffers[0][:1]}
    proposer.draft_attn_groups = [SimpleNamespace(kv_cache_group_id=0)]
    cad = SimpleNamespace(actual_seq_lengths_q=[3])
    pointers = (
        proposer._query_start_loc_buffer.data_ptr(),
        proposer._seq_lens_buffer.data_ptr(),
        proposer._query_slot_buffers[0].data_ptr(),
        proposer._block_table_buffers[0].data_ptr(),
    )

    model_tokens = proposer._pad_request_rows(cad, actual_num_reqs=1, model_num_reqs=2)

    assert model_tokens == 6
    torch.testing.assert_close(cad.query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(cad.seq_lens, torch.tensor([10, 3], dtype=torch.int32))
    torch.testing.assert_close(proposer.input_ids[:6], torch.tensor([1, 2, 3, 99, 99, 99]))
    torch.testing.assert_close(proposer.positions[:6], torch.tensor([0, 1, 2, 0, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(cad.token_to_req_indices, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(cad.slot_mapping, torch.tensor([100, 101, 102, -1, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(proposer._block_tables_by_gid[0], torch.tensor([[10, 11], [0, 0]], dtype=torch.int32))
    assert cad.actual_seq_lengths_q == [3, 3]
    assert pointers == (
        proposer._query_start_loc_buffer.data_ptr(),
        proposer._seq_lens_buffer.data_ptr(),
        proposer._query_slot_buffers[0].data_ptr(),
        proposer._block_table_buffers[0].data_ptr(),
    )


def test_dsa_metadata_tensors_keep_addresses_between_steps():
    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer._stable_attn_buffers = {}
    first = SimpleNamespace(
        prefill=SimpleNamespace(
            dspark_swa_indices=torch.ones((2, 1, 4), dtype=torch.int32),
            dspark_swa_lens=torch.ones(2, dtype=torch.int32),
            sas_metadata=torch.ones(3, dtype=torch.int32),
        )
    )
    proposer._stabilize_attn_metadata({"draft.swa": first})
    pointers = (
        first.prefill.dspark_swa_indices.data_ptr(),
        first.prefill.dspark_swa_lens.data_ptr(),
        first.prefill.sas_metadata.data_ptr(),
    )
    second = SimpleNamespace(
        prefill=SimpleNamespace(
            dspark_swa_indices=torch.full((2, 1, 4), 2, dtype=torch.int32),
            dspark_swa_lens=torch.full((2,), 2, dtype=torch.int32),
            sas_metadata=torch.full((3,), 2, dtype=torch.int32),
        )
    )

    proposer._stabilize_attn_metadata({"draft.swa": second})

    assert pointers == (
        second.prefill.dspark_swa_indices.data_ptr(),
        second.prefill.dspark_swa_lens.data_ptr(),
        second.prefill.sas_metadata.data_ptr(),
    )
    torch.testing.assert_close(second.prefill.dspark_swa_indices, torch.full((2, 1, 4), 2, dtype=torch.int32))


def test_graph_output_is_sliced_before_sampling(monkeypatch):
    context_calls = []
    graph_updates = []
    sample_calls = []

    def fake_context(metadata, config, **kwargs):
        del metadata, config
        context_calls.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(dspark_module, "set_ascend_forward_context", fake_context)
    monkeypatch.setattr(dspark_module, "get_forward_context", lambda: SimpleNamespace(moe_layer_index=-1))
    monkeypatch.setattr(dspark_module, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer.block_size = 5
    proposer.use_cuda_graph = True
    proposer.dp_rank = 0
    proposer.vllm_config = SimpleNamespace()
    proposer.input_ids = torch.arange(10)
    proposer.positions = torch.arange(10, dtype=torch.int32)
    proposer.draft_attn_groups = [SimpleNamespace()]
    proposer._graph_model_inputs = None
    proposer.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda **kwargs: (
            CUDAGraphMode.FULL,
            BatchDescriptor(10, 2, True),
        )
    )
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        dp_rank=0,
        _num_reqs_across_dp=torch.tensor([1, 2], dtype=torch.int32),
        _sync_metadata_across_dp=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("draft metadata must reuse target request counts")
        ),
    )
    cad = SimpleNamespace(num_reqs=1, num_actual_tokens=5)

    def pad_request_rows(metadata, actual, model):
        del actual
        metadata.num_reqs = model
        return model * proposer.block_size

    proposer.set_inputs_first_pass = lambda *args, **kwargs: (5, None, cad, None)
    proposer._pad_request_rows = pad_request_rows
    proposer._build_attn_metadata = lambda metadata: [{"draft": metadata}]
    proposer._precompute_context_kv = lambda: None
    proposer._update_full_graph_params = lambda *args: graph_updates.append(args)
    proposer._runnable = lambda: torch.arange(20, dtype=torch.float32).view(10, 2)

    def sample(hidden_states, num_reqs, metadata):
        del metadata
        sample_calls.append((hidden_states.clone(), num_reqs))
        return torch.tensor([[1, 2, 3, 4, 5]])

    proposer._sample_sequential = sample

    result = AscendDeepSeekV4DSparkProposer._propose(
        proposer,
        target_token_ids=torch.arange(1),
        target_positions=torch.arange(1),
        target_hidden_states=torch.zeros((1, 2)),
        next_token_ids=torch.tensor([7]),
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        target_model_batch_desc=BatchDescriptor(1),
        sampling_metadata=SimpleNamespace(),
    )

    torch.testing.assert_close(result, torch.tensor([[1, 2, 3, 4, 5]]))
    assert sample_calls[0][0].shape == (5, 2)
    assert sample_calls[0][1] == 1
    graph_model_inputs = proposer._graph_model_inputs
    assert graph_model_inputs is not None
    assert graph_model_inputs["input_ids"].shape[0] == 10
    assert context_calls[0]["aclgraph_runtime_mode"] == CUDAGraphMode.FULL
    assert len(graph_updates) == 1


@pytest.mark.parametrize("dp_rank", range(4))
@pytest.mark.parametrize(
    ("graph_modes", "expected_mode"),
    [
        (
            [
                CUDAGraphMode.PIECEWISE,
                CUDAGraphMode.FULL,
                CUDAGraphMode.PIECEWISE,
                CUDAGraphMode.FULL,
            ],
            CUDAGraphMode.NONE,
        ),
        ([CUDAGraphMode.FULL] * 4, CUDAGraphMode.FULL),
    ],
)
def test_uneven_dspark_dp_batch_uses_only_safe_graph_modes(monkeypatch, dp_rank, graph_modes, expected_mode):
    runner = cast(Any, object.__new__(NPUModelRunner))
    runner.dp_size = 4
    runner.dp_rank = dp_rank
    runner.input_batch = SimpleNamespace(num_reqs=[2, 1, 0, 1][dp_rank])
    runner.ascend_config = SimpleNamespace(dp_allreduce_on_npu=False)
    runner.vllm_config = SimpleNamespace()
    runner.drafter = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    runner._num_reqs_across_dp = None

    monkeypatch.setattr(
        model_runner_module,
        "should_skip_allreduce_across_dp_group",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        model_runner_module,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group="cpu-group", device_group="device-group"),
    )

    def fake_all_reduce(packed, group):
        assert group == "cpu-group"
        packed[0] = torch.tensor([48, 6, 0, 6], dtype=torch.int32)
        packed[1] = torch.tensor([mode.value for mode in graph_modes], dtype=torch.int32)
        packed[2] = torch.tensor([2, 1, 0, 1], dtype=torch.int32)

    monkeypatch.setattr(model_runner_module.dist, "all_reduce", fake_all_reduce)

    max_tokens, num_tokens_across_dp, graph_mode = NPUModelRunner._sync_metadata_across_dp(
        runner,
        num_tokens=[48, 6, 0, 6][dp_rank],
        cudagraph_mode=graph_modes[dp_rank],
    )

    assert max_tokens == 48
    assert graph_mode == expected_mode
    torch.testing.assert_close(num_tokens_across_dp, torch.full((4,), 48, dtype=torch.int32))
    torch.testing.assert_close(runner._num_reqs_across_dp, torch.tensor([2, 1, 0, 1], dtype=torch.int32))


def test_eager_dp_padding_adds_a_dummy_query_range():
    runner = cast(Any, object.__new__(NPUModelRunner))
    runner.use_compress = False
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.uniform_decode_query_len = 6
    runner.arange_np = np.arange(8, dtype=np.int32)
    query_start_loc = SimpleNamespace(
        np=np.array([0, 6, 0], dtype=np.int32),
        copy_to_gpu=lambda: None,
    )

    num_reqs_padded = runner._pad_query_start_loc_for_fia(
        query_start_loc,
        num_tokens_padded=116,
        num_reqs_padded=1,
        num_reqs=1,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert num_reqs_padded == 2
    np.testing.assert_array_equal(query_start_loc.np, np.array([0, 6, 116], dtype=np.int32))


def test_dsa_dp_padding_keeps_the_active_query_range():
    runner = cast(Any, object.__new__(NPUModelRunner))
    runner.use_compress = True
    query_start_loc = SimpleNamespace(
        np=np.array([0, 6, 0], dtype=np.int32),
        copy_to_gpu=lambda: None,
    )

    num_reqs_padded = runner._pad_query_start_loc_for_fia(
        query_start_loc,
        num_tokens_padded=116,
        num_reqs_padded=1,
        num_reqs=1,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert num_reqs_padded == 1
    np.testing.assert_array_equal(query_start_loc.np, np.array([0, 6, 0], dtype=np.int32))


def test_idle_dspark_dummy_run_marks_rank_inactive(monkeypatch):
    context_calls = []

    def capture_forward_context(metadata, config, **kwargs):
        del metadata, config
        context_calls.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(
        model_runner_module,
        "set_ascend_forward_context",
        capture_forward_context,
    )
    monkeypatch.setattr(model_runner_module, "update_cos_sin", lambda positions: None)
    monkeypatch.setattr(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    monkeypatch.setattr(model_runner_module, "lmhead_tp_enable", lambda: False)

    runner = cast(Any, object.__new__(NPUModelRunner))
    runner.uniform_decode_query_len = 6
    runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4)
    runner.max_num_tokens = 2048
    runner.dynamic_eplb = False
    runner.pcp_size = 1
    runner.dcp_size = 1
    runner._determine_batch_execution_and_padding = lambda **kwargs: (
        CUDAGraphMode.NONE,
        BatchDescriptor(116, 2, False),
        False,
        torch.tensor([116, 116], dtype=torch.int32),
        None,
    )
    runner._should_build_dummy_attn_metadata = lambda *args, **kwargs: False
    runner.maybe_dummy_run_with_lora = lambda *args, **kwargs: nullcontext()
    runner.lora_config = None
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(116, dtype=torch.int64))
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.positions = torch.zeros(116, dtype=torch.int64)
    runner.vllm_config = SimpleNamespace()
    runner.model = SimpleNamespace()
    runner._has_sinks = False
    runner._model_forward = lambda *args, **kwargs: torch.zeros((116, 4))
    runner.use_aux_hidden_state_outputs = False
    runner.drafter = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    runner.drafter.dummy_run = lambda *args, **kwargs: None
    runner.dp_rank = 2
    runner._num_reqs_across_dp = torch.tensor([2, 1, 0, 0], dtype=torch.int32)
    runner._finalize_dump_data = lambda **kwargs: None
    runner.use_compress = False

    runner._dummy_run(num_tokens=6, uniform_decode=True)

    assert context_calls[0]["num_tokens"] == 116
    assert context_calls[0]["num_actual_tokens"] == 0


def test_runtime_dummy_run_reuses_target_request_shape(monkeypatch):
    context_calls = []
    dispatcher_calls = []
    runnable_calls: list[dict[str, Any]] = []

    def capture_forward_context(metadata, config, **kwargs):
        del metadata, config
        context_calls.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(
        dspark_module,
        "set_ascend_forward_context",
        capture_forward_context,
    )
    monkeypatch.setattr(dspark_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(dspark_module, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    proposer = cast(Any, object.__new__(AscendDeepSeekV4DSparkProposer))
    proposer.block_size = 5
    proposer.max_query_tokens = 20
    proposer.use_cuda_graph = True
    proposer.dp_rank = 0
    proposer.device = torch.device("cpu")
    proposer.vllm_config = SimpleNamespace()
    proposer.parallel_drafting_token_id = 99
    proposer.input_ids = torch.zeros(20, dtype=torch.int64)
    proposer.positions = torch.zeros(20, dtype=torch.int32)
    proposer._token_to_req_buffer = torch.zeros(20, dtype=torch.int32)
    proposer._query_start_loc_buffer = torch.zeros(5, dtype=torch.int32)
    proposer._query_start_loc_cpu_buffer = torch.zeros(5, dtype=torch.int32)
    proposer._query_start_loc_cpu_base = torch.arange(5, dtype=torch.int32) * 5
    proposer._seq_lens_buffer = torch.zeros(4, dtype=torch.int32)
    proposer._seq_lens_cpu_buffer = torch.zeros(4, dtype=torch.int32)
    proposer.arange_dspark = torch.arange(32, dtype=torch.int32)
    proposer._query_slot_buffers = {}
    proposer._context_slot_buffers = {}
    proposer._block_tables_by_gid = {}
    proposer._graph_model_inputs = None
    proposer.draft_attn_groups = [SimpleNamespace(kv_cache_group_id=0)]
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        dp_rank=0,
        _num_reqs_across_dp=torch.tensor([2, 1, 0, 0], dtype=torch.int32),
        _sync_metadata_across_dp=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("target dummy metadata is already synchronized")
        ),
    )

    def dispatch(num_tokens, **kwargs):
        dispatcher_calls.append((num_tokens, kwargs))
        return CUDAGraphMode.FULL, BatchDescriptor(num_tokens, num_tokens // 5, True)

    proposer.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)
    proposer._get_block_table = lambda gid, cad, reqs: torch.zeros((reqs, 2), dtype=torch.int32)
    proposer._build_attn_metadata = lambda cad: [{"draft": cad}]
    proposer._update_full_graph_params = lambda *args: None

    def run_graph():
        graph_model_inputs = proposer._graph_model_inputs
        assert graph_model_inputs is not None
        runnable_calls.append(graph_model_inputs)

    proposer._runnable = run_graph

    proposer.dummy_run(
        12,
        num_reqs=1,
        num_tokens_across_dp=torch.tensor([12, 12], dtype=torch.int32),
        aclgraph_runtime_mode=CUDAGraphMode.FULL,
        is_graph_capturing=False,
    )

    assert dispatcher_calls[0][0] == 10
    assert runnable_calls[0]["input_ids"].shape[0] == 10
    assert context_calls[0]["batch_descriptor"] == BatchDescriptor(10, 2, True)
    torch.testing.assert_close(
        context_calls[0]["num_tokens_across_dp"],
        torch.tensor([10, 10], dtype=torch.int32),
    )


def test_model_runner_merges_dspark_capture_sizes(monkeypatch):
    graph_params = []
    draft_graph_params = []
    drafter_modes = []

    class FakeDrafter:
        def initialize_cudagraph_keys(self, mode):
            drafter_modes.append(mode)

        def get_cudagraph_capture_sizes(self):
            return [5, 10]

    class FakeCompilationConfig:
        def resolve_cudagraph_mode_and_sizes(self, **kwargs):
            del kwargs
            return CUDAGraphMode.FULL

    monkeypatch.setattr(model_runner_module, "AscendDflashProposer", FakeDrafter)
    monkeypatch.setattr(model_runner_module, "AscendDeepSeekV4DSparkProposer", FakeDrafter)
    monkeypatch.setattr(model_runner_module, "update_pass_config", lambda runner: nullcontext())
    monkeypatch.setattr(model_runner_module, "set_graph_params", lambda sizes: graph_params.append(sizes))
    monkeypatch.setattr(
        model_runner_module,
        "set_draft_graph_params",
        lambda sizes: draft_graph_params.append(sizes),
    )
    target_dispatcher = SimpleNamespace(
        initialize_cudagraph_keys=lambda *args: None,
        get_capture_descs=lambda: [
            (
                CUDAGraphMode.FULL,
                [BatchDescriptor(6, 1, True), BatchDescriptor(12, 2, True)],
            )
        ],
    )
    runner = cast(
        Any,
        SimpleNamespace(
            compilation_config=FakeCompilationConfig(),
            uniform_decode_query_len=6,
            parallel_config=SimpleNamespace(tensor_parallel_size=4),
            kv_cache_config=SimpleNamespace(),
            max_num_reqs=8,
            cudagraph_dispatcher=target_dispatcher,
            speculative_config=SimpleNamespace(
                use_eagle=lambda: False,
                uses_extract_hidden_states=lambda: False,
            ),
            drafter=FakeDrafter(),
            use_aclgraph=True,
        ),
    )

    NPUModelRunner._check_and_update_cudagraph_mode(runner, [], [])

    assert drafter_modes == [CUDAGraphMode.FULL]
    assert graph_params == [[6, 12]]
    assert draft_graph_params == [[5, 6, 10, 12]]
