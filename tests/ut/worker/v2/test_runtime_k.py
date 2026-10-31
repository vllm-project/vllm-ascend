# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Runtime-K regressions using the imported upstream and Ascend classes.

Device operations are mocked by tests/ut/conftest.py. These are CPU contract
tests, not NPU graph-recording or kernel correctness tests.
"""

from contextlib import nullcontext
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as upstream
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import SpeculatorBatchDescriptor

from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import speculator as ascend
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph import AutoRegressiveAclGraphManager
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_legacy_pool_only_replays_maximum_k(k):
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.vllm_config = SimpleNamespace(num_speculative_tokens=4)
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=8)
    manager.graphs = {desc: object()}
    selected = manager.specialize_spec_tokens(desc, k)
    assert selected.cg_mode == (CUDAGraphMode.FULL if k == 4 else CUDAGraphMode.NONE)


def test_gqa_can_switch_k_and_preserves_complete_descriptor():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.vllm_config = SimpleNamespace(num_speculative_tokens=4)
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
        max_query_len=1,
        num_active_loras=2,
        num_ubatches=2,
    )
    descriptor_fields = {field.name: getattr(desc, field.name) for field in fields(desc)}
    short = SpeculatorBatchDescriptor(**descriptor_fields, num_speculative_tokens=2)
    maximum = replace(short, num_speculative_tokens=4)
    manager.graphs = {desc: object(), short: object(), maximum: object()}
    assert manager.specialize_spec_tokens(desc, 2) == short
    assert manager.specialize_spec_tokens(short, 4) == maximum
    missing = manager.specialize_spec_tokens(maximum, 3)
    assert missing == replace(maximum, cg_mode=CUDAGraphMode.NONE, num_speculative_tokens=3)


def test_ascend_does_not_select_upstream_fused_attention_loop():
    spec: Any = AscendTargetDependentARSpeculator.__new__(AscendTargetDependentARSpeculator)
    spec._configure_decode_metadata_update()
    assert spec.supports_decode_metadata_update is False


def test_prefill_fia_keeps_each_layers_parameters():
    spec: Any = AscendTargetDependentARSpeculator.__new__(AscendTargetDependentARSpeculator)
    spec.use_dcp = False
    metadata = {
        name: SimpleNamespace(block_tables=object(), actual_seq_lengths_q=[i], seq_lens_list=[i + 1])
        for i, name in enumerate(["draft.0", "draft.1"], start=1)
    }
    result = spec.build_fia_params(2, metadata, True, num_speculative_steps=2)
    assert len(result) == 2
    for params, (name, layer) in zip(result, metadata.items()):
        assert params == {
            "layer_name": name,
            "actual_seq_lengths": layer.actual_seq_lengths_q,
            "actual_seq_lengths_kv": layer.seq_lens_list,
            "block_table": layer.block_tables,
        }


def test_missing_metadata_fails_before_any_replay():
    spec: Any = AscendTargetDependentARSpeculator.__new__(AscendTargetDependentARSpeculator)
    spec.model_state = SimpleNamespace(attn_metadata=None)
    with pytest.raises(RuntimeError, match="metadata is required"):
        spec.build_draft_attn_metadatas(2, 2, False, num_speculative_steps=2)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_real_parent_decode_loop_tracks_k_transitions(mode):
    spec: Any = AscendTargetDependentARSpeculator.__new__(AscendTargetDependentARSpeculator)
    spec.num_speculative_steps = 4
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(2), query_start_loc=torch.arange(3))
    spec.idx_mapping = torch.arange(2)
    spec.current_draft_step = torch.tensor(0)
    spec._generate_draft = Mock()
    desc = BatchExecutionDescriptor(cg_mode=mode, num_tokens=2, num_reqs=2)
    for k in [4, 2, 0, 1, 4]:
        spec._generate_draft.reset_mock()
        spec._multi_step_decode(2, True, desc, None, torch.tensor([8, 8]), k)
        assert spec._generate_draft.call_count == max(0, k - 1)
        assert all(call.kwargs["num_speculative_steps"] == k for call in spec._generate_draft.call_args_list)
    assert spec.num_speculative_steps == 4


def test_real_propose_keeps_prefill_state_across_zero_and_one_k(monkeypatch):
    # Execute the actual upstream propose and decode orchestration; replace only
    # model/attention/device work with CPU stand-ins. In particular, K=0 must
    # still refresh prefill state before clearing the draft buffer.
    spec: Any = AscendTargetDependentARSpeculator.__new__(AscendTargetDependentARSpeculator)
    spec.num_speculative_steps = 4
    spec.max_num_reqs = 1
    spec.max_model_len = 32
    spec.dp_size, spec.dp_rank = 1, 0
    spec.pcp_manager = None
    spec.replicated_pcp = False
    spec.supports_decode_metadata_update = False
    spec.prefill_cudagraph_manager = spec.decode_cudagraph_manager = None
    spec.idx_mapping = torch.arange(1)
    spec.last_token_indices = torch.zeros(1, dtype=torch.int64)
    spec.sample_src_positions = torch.zeros(1, dtype=torch.int64)
    spec.current_draft_step = torch.tensor(0)
    spec.hidden_states = torch.zeros(1, 2)
    spec.draft_tokens = torch.full((1, 4), -1)
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(1), query_start_loc=torch.arange(2))
    spec.block_tables = SimpleNamespace(compute_slot_mappings=Mock(return_value={}))
    spec.kv_cache_config = object()
    spec._copy_request_inputs = spec._prepare_eplb_forward = Mock()
    spec._build_uniform_attn_metadata = Mock(return_value={})
    spec._configure_decode_metadata_update()
    state: dict[str, Any] = {"prefills": 0, "decode_steps": []}

    def prefill(*args, skip_sample=False, **kwargs):
        state["prefills"] += 1
        if not skip_sample:
            spec.draft_tokens[:, 0] = 10 * state["prefills"]

    def generate(*args, num_speculative_steps, **kwargs):
        step = int(spec.current_draft_step)
        state["decode_steps"].append(step)
        spec.draft_tokens[:, step] = 10 * state["prefills"] + step

    spec._prefill = prefill
    spec._generate_draft = generate
    monkeypatch.setattr(ascend, "disable_target_pcp_for_replicated_draft", lambda _: nullcontext())
    monkeypatch.setattr(ascend, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(ascend, "torch_gather_wrapper", nullcontext)
    monkeypatch.setattr(upstream, "prepare_prefill_inputs", Mock())
    monkeypatch.setattr(upstream, "prepare_decode_inputs", Mock())
    monkeypatch.setattr(upstream, "build_slot_mappings_by_layer", Mock(return_value={}))

    def dispatch(manager, num_reqs, num_tokens, *args, **kwargs):
        return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs), None

    monkeypatch.setattr(upstream, "dispatch_cg_and_sync_dp", dispatch)
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_tokens_after_padding=1,
        num_scheduled_tokens=torch.ones(1, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8]),
        seq_lens=torch.tensor([8]),
        idx_mapping=torch.arange(1),
        decode_graph_eligible=False,
    )
    for iteration, k in enumerate([4, 2, 0, 1, 4], start=1):
        state["decode_steps"].clear()
        result = spec.propose(batch, {}, {}, torch.zeros(1, 2), None, *[torch.zeros(1)] * 6, num_speculative_tokens=k)
        assert state["prefills"] == iteration
        assert state["decode_steps"] == list(range(1, k))
        if k == 0:
            assert result.tolist() == [[-1, -1, -1, -1]]
        else:
            assert result[0, :k].tolist() == [10 * iteration + step for step in range(k)]
    assert spec.num_speculative_steps == 4
