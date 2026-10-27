# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for stable input and output storage across graph capture."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as core_cg
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

from vllm_ascend.models.glm5next.model import Glm5NextSPPPBuffer, Glm5NextSPPPSpec
from vllm_ascend.worker.v2.aclgraph_utils import Glm5NextSPPPGraphManager, ModelAclGraphManager
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState, Glm5NextSPPPModelState


def _state(spec, tensors=None):
    state = Glm5NextSPPPModelState.__new__(Glm5NextSPPPModelState)
    state._sp_pp_capture_layout = None
    state.bind_sp_pp_inputs(spec, tensors)
    return state


def _manager(spec, *, last_stage=False):
    manager = Glm5NextSPPPGraphManager.__new__(Glm5NextSPPPGraphManager)
    manager.model_runner = SimpleNamespace(_glm5next_sp_pp_spec=spec)
    manager.device = torch.device("cpu")
    manager.is_last_pp_rank = last_stage
    manager._sp_pp_output_buffer = None
    manager.intermediate_tensors = None
    manager.capture_sizes = [1, 5, 13]
    return manager


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize(
    "capture_options",
    [
        ((), {}, "Capturing CUDA graphs"),
        ((), {"progress_bar_desc": "SP+PP capture"}, "SP+PP capture"),
        ((False, False, None, "SP+PP capture", None), {}, "SP+PP capture"),
    ],
)
def test_capture_hooks_precede_slice_store_and_are_always_released(monkeypatch, fail, capture_options):
    spec = Glm5NextSPPPSpec(4, 0, 8, 3)
    active = Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu")).active
    state = _state(spec, active)
    manager = _manager(spec)
    original_addresses = {key: value.data_ptr() for key, value in active.tensors.items()}
    monkeypatch.setattr(AscendMambaHybridModelState, "prepare_dummy_inputs", lambda *args: {})
    seen = []
    optional_args, optional_kwargs, expected_desc = capture_options

    def parent_capture(
        self,
        model,
        model_state,
        input_buffers,
        inputs,
        block_tables,
        attn_groups,
        kv_cache_config,
        has_lora=False,
        use_aux_hidden_state_outputs=False,
        lora_capture_hook=None,
        progress_bar_desc="Capturing CUDA graphs",
        pcp_manager=None,
    ):
        assert (has_lora, use_aux_hidden_state_outputs, lora_capture_hook, progress_bar_desc, pcp_manager) == (
            False,
            False,
            None,
            expected_desc,
            None,
        )
        assert inputs is not active
        assert inputs.tensors is not active.tensors
        output_addresses = {key: value.data_ptr() for key, value in self.intermediate_tensors.tensors.items()}
        for total, rows in ((13, 4), (1, 1), (5, 2), (3, 1)):
            # The parent calls the dummy hook before slicing inputs and storing output.
            model_state.prepare_dummy_inputs(1, total)
            sliced = inputs[:total]
            assert all(value.shape[0] == rows for value in sliced.tensors.values())
            assert {key: value.data_ptr() for key, value in sliced.tensors.items()} == original_addresses
            assert {
                key: value.data_ptr() for key, value in self.intermediate_tensors.tensors.items()
            } == output_addresses
            assert all(value.shape[0] == rows for value in self.intermediate_tensors.tensors.values())
            assert all(output_addresses[key] != original_addresses[key] for key in original_addresses)
            for value in self.intermediate_tensors.tensors.values():
                value[:total] = torch.full_like(value, total)
            seen.append(rows)
        if fail:
            raise RuntimeError("capture injection")

    monkeypatch.setattr(ModelAclGraphManager, "capture", parent_capture)
    if fail:
        with pytest.raises(RuntimeError, match="capture injection"):
            manager.capture(None, state, None, active, None, [], None, *optional_args, **optional_kwargs)
    else:
        manager.capture(None, state, None, active, None, [], None, *optional_args, **optional_kwargs)
    assert seen == [4, 1, 2, 1]
    assert state._sp_pp_capture_layout is None
    assert manager._sp_pp_output_buffer is not None
    assert manager._sp_pp_output_buffer.capacity == 4
    assert active["hidden_states"].shape[0] == 4
    reference = weakref.ref(manager)
    del manager
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("first_stage,last_stage", [(True, False), (False, True)])
def test_capture_omits_only_stage_specific_buffers(monkeypatch, first_stage, last_stage):
    spec = Glm5NextSPPPSpec(4, 0, 8, 3)
    active = None if first_stage else Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu")).active
    state = _state(spec, active)
    manager = _manager(spec, last_stage=last_stage)
    monkeypatch.setattr(AscendMambaHybridModelState, "prepare_dummy_inputs", lambda *args: {})
    calls = []

    def parent_capture(self, model, model_state, input_buffers, inputs, *args, **kwargs):
        model_state.prepare_dummy_inputs(1, 5)
        if first_stage:
            assert inputs is None
        else:
            assert inputs is not active
            assert all(value.shape[0] == 2 for value in inputs.tensors.values())
        if last_stage:
            assert self.intermediate_tensors is None
        else:
            assert all(value.shape[0] == 2 for value in self.intermediate_tensors.tensors.values())
        calls.append(True)

    monkeypatch.setattr(ModelAclGraphManager, "capture", parent_capture)
    manager.capture(None, state, None, active, None, [], None)
    assert calls == [True]
    assert state._sp_pp_capture_layout is None
    if active is not None:
        assert active["hidden_states"].shape[0] == 4


def test_full_replay_rebinds_small_and_large_output_views(monkeypatch):
    spec = Glm5NextSPPPSpec(4, 0, 8, 3)
    manager = _manager(spec)
    manager._sp_pp_output_buffer = Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu"))
    manager.intermediate_tensors = manager._sp_pp_output_buffer.active
    addresses = {key: value.data_ptr() for key, value in manager.intermediate_tensors.tensors.items()}
    monkeypatch.setattr(
        ModelAclGraphManager, "run_fullgraph", lambda self, desc: self.intermediate_tensors[: desc.num_tokens]
    )
    for total, rows in ((1, 1), (16, 4), (3, 1), (9, 3)):
        output = manager.run_fullgraph(SimpleNamespace(num_tokens=total))
        assert all(value.shape[0] == rows for value in output.tensors.values())
        assert {key: value.data_ptr() for key, value in output.tensors.items()} == addresses


def test_release_and_recapture_drop_only_manager_owned_output(monkeypatch):
    spec = Glm5NextSPPPSpec(4, 0, 8, 3)
    active = Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu")).active
    state = _state(spec, active)
    manager = _manager(spec)
    manager._sp_pp_output_buffer = Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu"))
    manager.intermediate_tensors = manager._sp_pp_output_buffer.active
    output_reference = weakref.ref(manager._sp_pp_output_buffer)
    input_addresses = {key: value.data_ptr() for key, value in active.tensors.items()}
    monkeypatch.setattr(ModelAclGraphManager, "release_graphs", lambda self: None)
    manager.release_graphs()
    assert manager._sp_pp_output_buffer is None
    assert manager.intermediate_tensors is None
    gc.collect()
    assert output_reference() is None
    assert state.sp_pp_input_buffer is not None
    assert state.sp_pp_input_buffer.capacity == 4
    assert {key: value.data_ptr() for key, value in active.tensors.items()} == input_addresses
    monkeypatch.setattr(ModelAclGraphManager, "capture", lambda *args, **kwargs: None)
    manager.capture(None, state, None, active, None, [], None)
    assert manager._sp_pp_output_buffer is not None
    assert manager._sp_pp_output_buffer.capacity == 4
    assert state._sp_pp_capture_layout is None


@pytest.mark.parametrize(
    "graph_mode,decode,expected_mode,expected_tokens",
    [
        (CUDAGraphMode.FULL_DECODE_ONLY, True, CUDAGraphMode.FULL, (16, 16, 16)),
        (CUDAGraphMode.NONE, True, CUDAGraphMode.NONE, (9, 16, 9)),
        (CUDAGraphMode.FULL_DECODE_ONLY, False, CUDAGraphMode.NONE, (9, 16, 9)),
    ],
)
def test_supported_pp_dispatch_selects_same_token_bucket_on_each_stage(
    monkeypatch, graph_mode, decode, expected_mode, expected_tokens
):
    # Build real Core candidates while replacing only device and rank lookups.
    monkeypatch.setattr(core_cg, "current_platform", SimpleNamespace(get_global_graph_pool=lambda: None))
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 8, 16],
            max_cudagraph_capture_size=16,
        ),
        speculative_config=None,
    )
    per_stage = []
    for pp_rank in range(2):
        pp_group = SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == 1)
        with monkeypatch.context() as stage_patch:
            stage_patch.setattr(core_cg, "get_pp_group", lambda pp_group=pp_group: pp_group)
            manager = core_cg.CudaGraphManager(config, torch.device("cpu"), graph_mode, decode_query_len=1)
            # Supply the serving-time host flag without capturing a device graph.
            manager._graphs_captured = True
            selected = []
            for scheduled_tokens in (9, 16, 9):
                descriptor, dp_state = dispatch_cg_and_sync_dp(
                    manager,
                    num_reqs=scheduled_tokens if decode else 1,
                    num_tokens=scheduled_tokens,
                    uniform_token_count=1 if decode else None,
                    dp_size=1,
                    dp_rank=0,
                    max_query_len=1 if decode else scheduled_tokens,
                )
                assert dp_state is None
                assert descriptor.cg_mode == expected_mode
                selected.append(descriptor)
            assert tuple(desc.num_tokens for desc in selected) == expected_tokens
            per_stage.append(selected)
    assert per_stage[0] == per_stage[1]
