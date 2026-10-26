# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as graph_module
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as module
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


def _speculator() -> Any:
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.num_speculative_steps = 7
    spec.max_num_tokens = 32
    spec._context_slot_mappings = torch.arange(2 * 32, dtype=torch.int32).view(2, 32)
    spec.hidden_states = torch.randn(32, 12)
    spec.context_positions = torch.arange(32)
    return spec


@pytest.mark.parametrize("tokens", [0, 7, 16])
def test_context_graph_masks_only_rows_outside_current_batch(tokens):
    spec = _speculator()
    original = spec._context_slot_mappings.clone()
    precompute = MagicMock()
    spec._compat_precompute_context_kv = precompute
    spec._prepare_context_kv_for_graph(SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=2), tokens)
    assert torch.equal(spec._context_slot_mappings[:, :tokens], original[:, :tokens])
    assert torch.all(spec._context_slot_mappings[:, tokens:16] == module.dflash_speculator.PAD_SLOT_ID)
    assert torch.equal(spec._context_slot_mappings[:, 16:], original[:, 16:])
    precompute.assert_not_called()


@pytest.mark.parametrize("dummy", [False, True])
def test_context_graph_prefill_overflow_and_dummy_cache_writes(dummy):
    spec = _speculator()
    precompute = MagicMock()
    spec._compat_precompute_context_kv = precompute
    spec._prepare_context_kv_for_graph(SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=2), 24, dummy)
    if dummy:
        precompute.assert_not_called()
        assert torch.all(spec._context_slot_mappings[:, :16] == module.dflash_speculator.PAD_SLOT_ID)
    else:
        precompute.assert_called_once_with(16, 24)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
@pytest.mark.parametrize("dummy", [False, True])
def test_context_eager_path_precomputes_entire_current_context(mode, dummy):
    spec = _speculator()
    precompute = MagicMock()
    spec._compat_precompute_context_kv = precompute
    spec._prepare_context_kv_for_graph(SimpleNamespace(cg_mode=mode, num_reqs=2), 24, dummy)
    precompute.assert_called_once_with(0, 24, dummy)


@pytest.mark.parametrize("layer_groups", [None, [1, 0, 1]])
@pytest.mark.parametrize("dummy", [False, True])
def test_context_slice_preserves_each_layers_current_physical_slot_mapping(layer_groups, dummy):
    spec = _speculator()
    spec._layer_group_idx = layer_groups
    spec.model = SimpleNamespace(precompute_and_store_context_kv=MagicMock())
    for delta in (0, 100):
        spec._context_slot_mappings.add_(delta)
        spec._compat_precompute_context_kv(3, 9, dummy)
        states, positions, slots = spec.model.precompute_and_store_context_kv.call_args.args
        torch.testing.assert_close(states, spec.hidden_states[3:9])
        torch.testing.assert_close(positions, spec.context_positions[3:9])
        if dummy:
            assert slots is None
        elif layer_groups is None:
            torch.testing.assert_close(slots, spec._context_slot_mappings[0, 3:9])
        else:
            assert len(slots) == len(layer_groups)
            for observed, group in zip(slots, layer_groups):
                torch.testing.assert_close(observed, spec._context_slot_mappings[group, 3:9])


@pytest.mark.parametrize("requests,expected", [(1, 8), (2, 16), (4, 32), (8, 32)])
def test_context_graph_row_count_matches_upstream_capacity_bound(requests, expected):
    spec = _speculator()
    assert spec._compat_num_graph_context_tokens(requests) == expected


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_legacy_proposal_uses_block_table_cp_fields_and_prepares_prefix_before_query(monkeypatch, mode):
    spec = _speculator()
    spec.num_query_per_req = 7
    spec.max_num_reqs = 4
    spec.max_model_len = 4096
    spec.num_speculative_steps = 7
    spec.dp_size = 1
    spec.dp_rank = 0
    spec.pcp_manager = None
    spec._layer_group_idx = [0, 1]
    spec._group_causal = False
    spec.draft_kv_cache_group_id = 1
    spec.draft_kv_cache_group_ids = [1, 2]
    spec.kv_cache_config = object()
    spec.parallel_drafting_token_id = 99
    spec.sample_from_anchor = False
    spec.input_buffers = object()
    for name in ("sample_indices", "sample_pos", "sample_idx_mapping", "temperature", "seeds", "draft_tokens"):
        setattr(spec, name, torch.zeros(4, dtype=torch.int64))
    spec.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros(3, 32),
        input_block_tables=torch.zeros(3, 4, 2),
        kernel_block_sizes=[128] * 3,
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    calls = []
    prepare = MagicMock(side_effect=lambda *args: calls.append("inputs"))
    monkeypatch.setattr(module.dflash_speculator, "prepare_dflash_inputs", prepare)
    desc = SimpleNamespace(cg_mode=mode, num_reqs=2, num_tokens=14)
    monkeypatch.setattr(module.dflash_speculator, "dispatch_cg_and_sync_dp", lambda *args, **kwargs: (desc, None))
    monkeypatch.setattr(module.dflash_speculator, "build_slot_mappings_by_layer", lambda *args: {})
    spec._prepare_context_kv_for_graph = lambda *args: calls.append("context")

    def build_metadata(**kwargs):
        calls.append("metadata")
        return {}

    spec._build_uniform_attn_metadata = build_metadata
    spec._prepare_eplb_forward = lambda *args: None
    spec._generate_draft = lambda *args, **kwargs: calls.append("query")
    spec.query_cudagraph_manager = SimpleNamespace(run_fullgraph=lambda *args: calls.append("query"))
    batch = SimpleNamespace(num_reqs=2, num_tokens=8, seq_lens_cpu_upper_bound=torch.tensor([20, 30]))
    args = [torch.zeros(4, dtype=torch.int64)] * 6
    result = spec._propose_with_context_graph(batch, {}, {}, torch.randn(8, 12), None, *args)
    assert result.data_ptr() == spec.draft_tokens.data_ptr()
    assert calls == ["inputs", "inputs", "context", "metadata", "query"]
    assert prepare.call_count == 2
    assert all(call.args[19] == spec.block_tables.cp_size for call in prepare.call_args_list)
    assert spec.draft_max_seq_len == 37


def test_old_graph_capture_runs_context_prefix_before_query_forward(monkeypatch):
    calls = []
    manager = graph_module.DFlashAclGraphManager.__new__(graph_module.DFlashAclGraphManager)
    manager.speculator = object()
    manager.vllm_config = object()

    def old_capture(
        self,
        forward_fn,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
        max_model_len,
        causal,
        progress_bar_desc="Capture",
        **kwargs,
    ):
        assert self is manager
        assert progress_bar_desc == "Context and query"
        hook = kwargs.get("precompute_context_kv") if vllm_version_is("0.31.0") else kwargs.get("prepare_forward")
        if hook is not None:
            hook(2)
        forward_fn(2, 16, {}, {}, None, CUDAGraphMode.FULL)

    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "capture", old_capture)
    monkeypatch.setattr(graph_module, "communicator_switch", lambda: nullcontext())
    monkeypatch.setattr(graph_module, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(graph_module, "model_capture_wrapper", lambda *args: nullcontext())
    prefix_kwargs = (
        {"precompute_context_kv": lambda n: calls.append(("context", n))}
        if vllm_version_is("0.31.0")
        else {"prepare_forward": lambda n: calls.append(("context", n))}
    )
    manager.capture(
        lambda *args: calls.append(("query", args[0])),
        input_buffers=MagicMock(),
        block_tables=MagicMock(),
        attn_groups=[],
        kv_cache_config=MagicMock(),
        max_model_len=4096,
        causal=False,
        progress_bar_desc="Context and query",
        **prefix_kwargs,
    )
    assert calls == [("context", 2), ("query", 2)]


@pytest.mark.parametrize(
    "capture_descs,expected",
    [
        ({CUDAGraphMode.FULL: [SimpleNamespace(num_reqs=2, num_tokens=14)]}, True),
        ({CUDAGraphMode.FULL: [SimpleNamespace(num_reqs=2)], CUDAGraphMode.PIECEWISE: []}, True),
        (
            {CUDAGraphMode.FULL: [SimpleNamespace(num_reqs=2)], CUDAGraphMode.PIECEWISE: [SimpleNamespace(num_reqs=2)]},
            False,
        ),
        ({CUDAGraphMode.FULL: []}, False),
        ({}, False),
    ],
)
def test_compat_capture_uses_actual_graph_mode_dictionary(monkeypatch, capture_descs, expected):
    spec = _speculator()
    spec._reuse_draft_layout = True
    spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))
    spec.dp_size = 1
    spec.query_cudagraph_manager = SimpleNamespace(_capture_descs=capture_descs, capture=MagicMock())
    spec.sample_indices = torch.ones(4)
    spec.sample_pos = torch.ones(4)
    spec.sample_idx_mapping = torch.ones(4)
    spec.input_buffers = object()
    spec.block_tables = object()
    spec.attn_groups = object()
    spec.kv_cache_config = object()
    spec.max_model_len = 4096
    spec._group_causal = False
    original_capture = MagicMock()
    monkeypatch.delattr(module.DFlashSpeculator, "_num_graph_context_tokens", raising=False)
    monkeypatch.setattr(module.DFlashSpeculator, "capture", original_capture)
    spec.capture()
    assert spec._compat_context_graph is expected
    if expected:
        spec.query_cudagraph_manager.capture.assert_called_once()
        original_capture.assert_not_called()
        assert torch.all(spec.sample_indices == 0)
        assert torch.all(spec.sample_pos == 0)
        assert torch.all(spec.sample_idx_mapping == -1)
        assert torch.all(spec._context_slot_mappings == module.dflash_speculator.PAD_SLOT_ID)
    else:
        original_capture.assert_called_once()
        spec.query_cudagraph_manager.capture.assert_not_called()


@pytest.mark.parametrize(
    "capturing,mode,expected",
    [
        (True, CUDAGraphMode.FULL, True),
        (False, CUDAGraphMode.FULL, False),
    ],
)
def test_dflash_context_prefix_wrapper_sets_capture_context_and_forwards(monkeypatch, capturing, mode, expected):
    from vllm_ascend.worker.v2 import aclgraph_utils as wrapper_module

    original = SimpleNamespace(precompute_and_store_context_kv=MagicMock(return_value="stored"))
    wrapper = wrapper_module.ModelWithContext(original, is_draft_model=True)
    monkeypatch.setattr(wrapper_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    config = object()
    context = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(wrapper_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(wrapper_module, "set_forward_context", context)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(wrapper_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False, capturing=False))
    states, positions, slots = SimpleNamespace(shape=(16, 4096)), object(), [object(), object()]
    assert wrapper.precompute_and_store_context_kv(states, positions, slots) == "stored"
    original.precompute_and_store_context_kv.assert_called_once_with(states, positions, slots)
    context.assert_called_once_with(None, config, num_tokens=16, cudagraph_runtime_mode=CUDAGraphMode.FULL)
    assert wrapper_module._EXTRA_CTX.is_draft_model is True
    assert wrapper_module._EXTRA_CTX.capturing is expected


def test_native_capture_forwards_prefix_once_and_restores_model_on_error(monkeypatch):
    from vllm_ascend.worker.v2 import aclgraph_utils as wrapper_module

    original = SimpleNamespace(precompute_and_store_context_kv=MagicMock())
    spec = SimpleNamespace(model=original)
    manager = graph_module.DFlashAclGraphManager.__new__(graph_module.DFlashAclGraphManager)
    manager.speculator = spec
    manager.vllm_config = object()
    calls = []

    def capture(self, forward_fn, *args, precompute_context_kv=None, prepare_forward=None, **kwargs):
        assert spec.model.get_original_model() is original
        hook = precompute_context_kv if vllm_version_is("0.31.0") else prepare_forward
        hook(2)
        forward_fn(2)
        raise ValueError("capture failed")

    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "capture", capture)
    monkeypatch.setattr(graph_module, "communicator_switch", lambda: nullcontext())
    monkeypatch.setattr(graph_module, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(graph_module, "model_capture_wrapper", wrapper_module.model_capture_wrapper)
    prefix_kwargs = (
        {"precompute_context_kv": lambda n: calls.append(("context", n))}
        if vllm_version_is("0.31.0")
        else {"prepare_forward": lambda n: calls.append(("context", n))}
    )
    with pytest.raises(ValueError, match="capture failed"):
        manager.capture(
            lambda n: calls.append(("query", n)),
            input_buffers=MagicMock(),
            block_tables=MagicMock(),
            attn_groups=[],
            kv_cache_config=MagicMock(),
            max_model_len=4096,
            **prefix_kwargs,
        )
    assert calls == [("context", 2), ("query", 2)]
    assert spec.model is original


@pytest.mark.parametrize(
    "tp,dp,reuse,supported", [(2, 1, True, True), (1, 2, True, True), (1, 1, False, True), (1, 1, True, False)]
)
def test_context_backport_keeps_unsupported_paths_on_upstream(monkeypatch, tp, dp, reuse, supported):
    spec = _speculator()
    spec._reuse_draft_layout = reuse
    spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp))
    spec.dp_size = dp
    spec._supports_context_graph_backport = lambda: supported
    spec.query_cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: [SimpleNamespace(num_reqs=2)]}, capture=MagicMock()
    )
    parent = MagicMock()
    monkeypatch.delattr(module.DFlashSpeculator, "_num_graph_context_tokens", raising=False)
    monkeypatch.setattr(module.DFlashSpeculator, "capture", parent)
    spec.capture()
    assert spec._compat_context_graph is False
    parent.assert_called_once()
    spec.query_cudagraph_manager.capture.assert_not_called()


def test_context_backport_does_not_replace_native_upstream_capture(monkeypatch):
    spec = _speculator()
    spec.query_cudagraph_manager = None
    parent = MagicMock()
    monkeypatch.setattr(module.DFlashSpeculator, "_num_graph_context_tokens", MagicMock(), raising=False)
    monkeypatch.setattr(module.DFlashSpeculator, "capture", parent)
    spec.capture()
    assert spec._compat_context_graph is False
    parent.assert_called_once()
