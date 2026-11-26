# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.models.hy_v4_flash import dfly_prepare, dfly_speculator


@pytest.mark.parametrize("supported", [False, True])
def test_query_preparation_selects_fusion_by_capability(monkeypatch, supported):
    spec = object.__new__(dfly_speculator.FlashDFlySpeculator)
    spec.device = SimpleNamespace(type="npu")
    spec._prepare_query_reference = Mock()
    prepare = Mock(return_value=supported)
    monkeypatch.setattr(dfly_prepare, "try_prepare_query", prepare)
    inputs = object()
    spec._prepare_query(inputs, 2, 3)
    prepare.assert_called_once_with(spec, inputs, 2, 3)
    if supported:
        spec._prepare_query_reference.assert_not_called()
    else:
        spec._prepare_query_reference.assert_called_once_with(inputs, 2, 3)


def make_speculator(capacity=3):
    spec = object.__new__(dfly_speculator.FlashDFlySpeculator)
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=4,
        is_moe_model=True,
    )
    spec.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=256),
        additional_config={"hy4flash_dfly_dp_graph": True},
        parallel_config=parallel,
        compilation_config=SimpleNamespace(decode_cudagraph_capture_request_sizes=[1, 3]),
    )
    spec.speculative_config = SimpleNamespace(enforce_eager=False, draft_parallel_config=copy(parallel))
    spec.dp_size = 4
    spec.max_num_reqs = capacity
    spec.max_num_tokens = capacity * 6
    spec.num_query_per_req = 6
    spec.num_speculative_steps = 5
    spec.hidden_size = 4
    spec.device = torch.device("cpu")
    spec.dtype = torch.bfloat16
    spec.cudagraph_manager = None
    spec.model_state = SimpleNamespace(rope_state=None)
    spec.offsets = torch.arange(6)
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(spec.max_num_tokens, dtype=torch.int32),
        positions=torch.zeros(spec.max_num_tokens, dtype=torch.long),
        is_padding=torch.zeros(spec.max_num_tokens, dtype=torch.bool),
        query_start_loc=torch.zeros(capacity + 1, dtype=torch.int32),
        seq_lens=torch.zeros(capacity, dtype=torch.int32),
    )
    spec.draft_tokens = torch.zeros(capacity, 5, dtype=torch.long)
    spec._draft_vllm_config = copy(spec.vllm_config)
    spec._draft_vllm_config.parallel_config = copy(parallel)
    spec._draft_vllm_config.parallel_config.is_moe_model = False
    spec.target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(tp_size=1, token_exchange_group=None)),
        lm_head=SimpleNamespace(tp_size=1),
    )
    spec.model = torch.nn.Identity()
    spec.model.config = SimpleNamespace(mask_token_id=99, target_index_topk=4, context_local_window=2)
    spec.model.layers = [SimpleNamespace(self_attn=SimpleNamespace(layer_name="draft.0", context=None))]
    spec.kv_layers = (SimpleNamespace(layer_name="target.0", kv_cache=torch.zeros(1, 1, 16, 576)),)
    spec.index_layers = (SimpleNamespace(topk_snapshot=torch.tensor([1, 2, -1, 500]).repeat(spec.max_num_tokens, 1)),)
    spec._init_query_buffers()
    return spec


def test_local_tp1_draft_can_capture_without_changing_target(monkeypatch):
    spec = make_speculator()
    monkeypatch.setattr(dfly_speculator, "CudaGraphManager", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: "private-pool")
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is not None
    assert spec.cudagraph_manager.pool == "private-pool"
    assert spec._draft_vllm_config.parallel_config.is_moe_model is False
    assert spec.vllm_config.parallel_config.is_moe_model is True
    assert spec._draft_vllm_config.parallel_config is not spec.vllm_config.parallel_config
    assert spec.target_hidden_states.shape == (3, 4)


def test_draft_config_isolated_from_target_for_loading_and_execution(monkeypatch):
    target = make_speculator()
    draft_model_config = SimpleNamespace(name="draft")

    def base_init(spec, config, device):
        spec.vllm_config = config
        spec.speculative_config = target.speculative_config
        spec.draft_model_config = draft_model_config
        spec.num_speculative_steps = 5
        spec.max_num_reqs = 3
        spec.max_num_tokens = 18
        spec.hidden_size = 4
        spec.dtype = torch.bfloat16
        spec.device = device

    monkeypatch.setattr(dfly_speculator.DraftModelSpeculator, "__init__", base_init)
    spec = dfly_speculator.FlashDFlySpeculator(target.vllm_config, torch.device("cpu"))
    assert spec._draft_vllm_config.model_config is draft_model_config
    assert spec._draft_vllm_config.parallel_config is not target.speculative_config.draft_parallel_config
    assert spec._draft_vllm_config.parallel_config.is_moe_model is False
    assert spec._draft_vllm_config.parallel_config.enable_expert_parallel is False
    assert target.vllm_config.parallel_config.is_moe_model is True
    assert target.speculative_config.draft_parallel_config.is_moe_model is True


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
@pytest.mark.parametrize("draft", [False, True])
def test_parallel_draft_stays_eager(field, draft):
    spec = make_speculator()
    parallel = spec.speculative_config.draft_parallel_config if draft else spec.vllm_config.parallel_config
    setattr(parallel, field, 2)
    assert not spec._dp_graph_is_local()
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is None


@pytest.mark.parametrize("case", ["head", "embedding", "exchange", "reduce", "draft_linear"])
def test_dp_axis_collectives_stay_eager(case):
    spec = make_speculator()
    if case == "head":
        spec.target.lm_head.tp_size = 4
    elif case == "embedding":
        spec.target.model.embed_tokens.tp_size = 4
    elif case == "exchange":
        spec.target.model.embed_tokens.token_exchange_group = SimpleNamespace(world_size=4)
    elif case == "reduce":
        spec.target.model.embed_tokens.output_reduce_group = SimpleNamespace(world_size=4)
    else:
        spec.model.tp_size = 2
    assert not spec._dp_graph_is_local()
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is None


@pytest.mark.parametrize("case", ["disabled", "eager", "no_graphs", "no_sizes"])
def test_graph_opt_in_and_runtime_modes(case):
    spec = make_speculator()
    mode = CUDAGraphMode.FULL_DECODE_ONLY
    if case == "disabled":
        spec.vllm_config.additional_config.clear()
    elif case == "eager":
        spec.speculative_config.enforce_eager = True
    elif case == "no_graphs":
        mode = CUDAGraphMode.NONE
    else:
        spec.vllm_config.compilation_config.decode_cudagraph_capture_request_sizes = []
    spec.init_cudagraph_manager(mode)
    assert spec.cudagraph_manager is None


def test_graph_option_rejects_string_boolean():
    spec = make_speculator()
    spec.vllm_config.additional_config["hy4flash_dfly_dp_graph"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)


def test_single_dp_replica_retains_existing_tp_graph_support(monkeypatch):
    spec = make_speculator()
    spec.dp_size = spec.vllm_config.parallel_config.data_parallel_size = 1
    spec.vllm_config.parallel_config.tensor_parallel_size = 4
    spec.vllm_config.additional_config.clear()
    monkeypatch.setattr(dfly_speculator, "CudaGraphManager", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: "private-pool")
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is not None


def target_inputs(spec, batch):
    ends = torch.arange(batch + 1, dtype=torch.int32) * 6
    positions = torch.arange(batch * 6) + 10
    return dfly_speculator.DFlyProposeInputs(
        ends,
        positions,
        torch.arange(batch - 1, -1, -1),
        torch.ones(batch, dtype=torch.int32),
        torch.arange(batch, dtype=torch.int32) % 6,
        torch.arange(spec.max_num_reqs * 2).view(-1, 2) + 1000,
        torch.arange(batch * 6 * spec.hidden_size).view(-1, spec.hidden_size).to(spec.dtype),
    )


def test_nonuniform_target_queries_use_accepted_rows_and_request_state_mapping():
    spec = make_speculator(8)
    inputs = dfly_speculator.DFlyProposeInputs(
        torch.tensor([0, 4, 8, 9], dtype=torch.int32),
        torch.tensor([10, 11, 12, 13, 250, 251, 252, 253, 50]),
        torch.tensor([5, 1, 6, 999, 999]),
        torch.tensor([2, 4, 0]),
        torch.tensor([1, 0, 0]),
        torch.arange(16).view(8, 2) + 1000,
        torch.arange(36).view(9, 4).to(torch.bfloat16),
    )
    spec._prepare_query(inputs, 3, 5)
    assert spec.source_rows[:5].tolist() == [2, 7, 8, 0, 0]
    assert spec.anchors[:3].tolist() == [13, 254, 51]
    assert spec.active_requests[:5].tolist() == [True, False, False, False, False]
    assert spec.input_buffers.input_ids[:30].view(5, 6)[:, 0].tolist() == [1010, 0, 0, 0, 0]
    assert spec.input_buffers.query_start_loc[:6].tolist() == [0, 6, 12, 18, 24, 30]
    assert spec.input_buffers.seq_lens[:5].tolist() == [19, 6, 6, 6, 6]
    assert spec.target_hidden_states[:3].tolist() == inputs.last_hidden_states[[2, 7, 8]].tolist()
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 30, 5)
    target = {"target.0": SimpleNamespace(block_table=torch.ones(3, 16, dtype=torch.int32))}
    metadata = spec._query_attention_state(desc, 3, target)["draft.0"]
    assert metadata.indices[1:].eq(-1).all()
    assert metadata.block_table[3:].eq(0).all()


@pytest.mark.parametrize("batch,padded", [(1, 3), (9, 16), (17, 32), (33, 64)])
@pytest.mark.parametrize("rope_dims", [0, 3])
def test_preparation_matches_upstream_contract(batch, padded, rope_dims):
    spec = make_speculator(padded)
    if rope_dims:
        state = SimpleNamespace(
            rope_state=SimpleNamespace(num_dims=rope_dims, prefill_delta=SimpleNamespace(gpu=torch.arange(padded) * 7))
        )
        spec.set_attn(state, None, None, None, None)
    inputs = target_inputs(spec, batch)
    # Cover mid-prefill and sequence-limit exclusion in addition to padding.
    inputs.num_sampled[0] = 0
    if batch > 1:
        row = int(inputs.query_start_loc[batch] - inputs.num_rejected[batch - 1] - 1)
        inputs.positions[row] = spec.max_model_len - 2
    spec._prepare_query(inputs, batch, padded)
    ids = spec.input_buffers.input_ids.view(padded, 6)
    pos = spec.input_buffers.positions.view(padded, 6)
    padding = spec.input_buffers.is_padding.view(padded, 6)
    # Independent per-request oracle, including official inactive-row values.
    for request in range(padded):
        active = False
        expected_pos = list(range(6))
        expected_id = 0
        if request < batch:
            source = int(inputs.query_start_loc[request + 1] - inputs.num_rejected[request] - 1)
            anchor = int(inputs.positions[source]) + 1
            active = bool(inputs.num_sampled[request]) and anchor + 6 <= spec.max_model_len
            assert spec.source_rows[request] == source
            assert torch.equal(spec.target_hidden_states[request], inputs.last_hidden_states[source])
            if active:
                expected_pos = list(range(anchor, anchor + 6))
                expected_id = int(inputs.last_sampled[inputs.idx_mapping[request], 0])
        else:
            assert spec.target_hidden_states[request].eq(0).all()
        assert ids[request].tolist() == [expected_id, 99, 99, 99, 99, 99]
        assert pos[request].tolist() == expected_pos
        assert spec.active_requests[request] == active
        assert padding[request].eq(not active).all()
        assert spec.input_buffers.seq_lens[request] == (expected_pos[-1] + 1)
        if rope_dims:
            expected_rope = [0] * 6
            if active:
                delta = int(spec.model_state.rope_state.prefill_delta.gpu[inputs.idx_mapping[request]])
                expected_rope = [p + delta for p in expected_pos]
            assert spec.mrope_positions[:, request * 6 : (request + 1) * 6].tolist() == [expected_rope] * rope_dims
    assert spec.input_buffers.query_start_loc[: padded + 1].tolist() == list(range(0, padded * 6 + 1, 6))


def test_c8_metadata_buffers_survive_batch_shrink_and_keep_padding_invalid():
    spec = make_speculator(4)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 24, 4)
    captured = spec._query_attention_state(desc, 0, None)["draft.0"]
    pointers = (captured.block_table.data_ptr(), captured.indices.data_ptr(), captured.query_positions.data_ptr())
    for batch, page in [(4, 7), (1, 11), (3, 19)]:
        inputs = target_inputs(spec, batch)
        spec._prepare_query(inputs, batch, 4)
        target = {"target.0": SimpleNamespace(block_table=torch.full((batch, 16), page, dtype=torch.int32))}
        current = spec._query_attention_state(desc, batch, target)["draft.0"]
        assert pointers == (
            current.block_table.data_ptr(),
            current.indices.data_ptr(),
            current.query_positions.data_ptr(),
        )
        # Inspect the original capture views, not the newly returned objects.
        assert captured.block_table[:batch].eq(page).all()
        assert captured.block_table[batch:].eq(0).all()
        assert captured.indices[batch:].eq(-1).all()
        for request in range(batch):
            anchor = int(spec.anchors[request])
            expected_history = [value if 0 <= value < anchor - 2 else -1 for value in [1, 2, -1, 500]]
            for query in range(6):
                expected = expected_history + [max(anchor - 2, -1), max(anchor - 1, -1)]
                expected += [anchor + key if key <= query else -1 for key in range(6)]
                assert captured.indices[request, query].tolist() == expected


def test_capture_prepares_metadata_outside_model_graph(monkeypatch):
    spec = make_speculator()
    state = {"switched": False, "in_graph": False}
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 18, 3)
    calls = []

    @contextmanager
    def switch():
        state["switched"] = True
        yield
        state["switched"] = False

    prepare = spec._prepare_query
    metadata = spec._query_attention_state

    def checked_prepare(*args):
        assert not state["in_graph"]
        calls.append("prepare")
        return prepare(*args)

    def checked_metadata(*args):
        assert not state["in_graph"]
        calls.append("metadata")
        return metadata(*args)

    def model(count, attention, mode):
        assert state["in_graph"] and state["switched"]
        assert count == 18 and mode == CUDAGraphMode.FULL
        assert attention["draft.0"].indices.eq(-1).all()
        assert spec.input_buffers.is_padding[:count].all()
        calls.append("model")

    def capture(factory, **kwargs):
        assert state["switched"]
        forward = factory(desc, False)
        state["in_graph"] = True
        forward(CUDAGraphMode.FULL)
        state["in_graph"] = False

    monkeypatch.setattr(dfly_speculator, "communicator_switch", switch)
    spec._prepare_query = checked_prepare
    spec._query_attention_state = checked_metadata
    spec._run_query_model = model
    spec.cudagraph_manager = SimpleNamespace(needs_capture=lambda: True, capture=capture)
    spec.capture()
    assert calls == ["prepare", "metadata", "model"]


@pytest.mark.parametrize("fullgraph", [False, True])
def test_propose_prepares_current_inputs_before_execution(monkeypatch, fullgraph):
    spec = make_speculator(4)
    inputs = target_inputs(spec, 2)
    target = {"target.0": SimpleNamespace(block_table=torch.ones(2, 16, dtype=torch.int32))}
    batch = SimpleNamespace(
        num_reqs=2, query_start_loc=inputs.query_start_loc, positions=inputs.positions, idx_mapping=inputs.idx_mapping
    )
    seen = []

    def check(count, metadata, mode):
        assert count == (24 if fullgraph else 12)
        assert spec.source_rows[:2].tolist() == [5, 10]
        assert spec.input_buffers.input_ids[:12:6].tolist() == [1002, 1000]
        assert metadata["draft.0"].block_table[:2].eq(1).all()
        seen.append(mode)

    spec._run_query_model = check
    if fullgraph:
        desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 24, 4)
        # Actual FULL replay reads the persistent views established at capture.
        metadata = spec._query_attention_state(desc, 0, None)
        spec.cudagraph_manager = SimpleNamespace(
            dispatch=lambda *args: desc,
            run_fullgraph=lambda desc: check(desc.num_tokens, metadata, desc.cg_mode),
        )
    spec.propose(
        batch,
        target,
        None,
        inputs.last_hidden_states,
        None,
        inputs.num_sampled,
        inputs.num_rejected,
        inputs.last_sampled,
        None,
        None,
        None,
    )
    assert seen == [CUDAGraphMode.FULL if fullgraph else CUDAGraphMode.NONE]


def test_query_model_installs_and_clears_context_on_failure(monkeypatch):
    spec = make_speculator()
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, 18, 3)
    spec._prepare_query(None, 0, 3)
    metadata = spec._query_attention_state(desc, 0, None)

    @contextmanager
    def context(attention, config, **kwargs):
        assert config is spec._draft_vllm_config
        assert kwargs["is_padding"].all()
        yield

    def fail(*args):
        assert spec.model.layers[0].self_attn.context is metadata["draft.0"]
        raise RuntimeError("query failed")

    monkeypatch.setattr(dfly_speculator, "set_forward_context", context)
    spec._draft_tokens = fail
    with pytest.raises(RuntimeError, match="query failed"):
        spec._run_query_model(18, metadata, CUDAGraphMode.NONE)
    assert spec.model.layers[0].self_attn.context is None


def test_propose_rejects_query_block_beyond_capacity():
    spec = make_speculator()
    with pytest.raises(ValueError, match="exceeds"):
        spec.propose(SimpleNamespace(num_reqs=4), None, None, None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch,num_speculative_tokens", [(0, None), (2, 0)])
def test_empty_or_disabled_drafting_does_not_read_inputs(batch, num_speculative_tokens):
    spec = make_speculator()
    spec.draft_tokens.fill_(42)
    result = spec.propose(
        SimpleNamespace(num_reqs=batch),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        num_speculative_tokens=num_speculative_tokens,
    )
    assert result.shape == (batch, 5)
    assert result.eq(-1).all()


def test_memory_profile_keeps_mrope_shape_before_attention_binding():
    spec = make_speculator()
    spec.vllm_config.model_config.uses_mrope = True
    spec.vllm_config.model_config.mrope_num_dims = 3
    assert spec.mrope_positions is None
    observed = []
    spec.target.embed_input_ids = lambda ids: torch.zeros(ids.numel(), spec.hidden_size)
    spec.model.forward = lambda embeddings, positions: observed.append(positions.shape)
    result = spec.propose(
        SimpleNamespace(num_reqs=2),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        dummy_run=True,
        is_profile=True,
    )
    assert observed == [torch.Size([3, 12])]
    assert result.eq(-1).all()
