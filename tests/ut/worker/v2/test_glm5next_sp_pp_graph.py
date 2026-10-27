# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise existing parent slicing/store behavior through the private hooks."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.models.glm5next.model import Glm5NextSPPPBuffer, Glm5NextSPPPRuntime, Glm5NextSPPPSpec
from vllm_ascend.worker.v2.aclgraph_utils import (
    Glm5NextSPPPGraphManager,
    ModelAclGraphManager,
    collect_sorted_captured_token_sizes,
)
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState, Glm5NextSPPPModelState


def _state(runtime):
    state = Glm5NextSPPPModelState.__new__(Glm5NextSPPPModelState)
    state.sp_pp_runtime = runtime
    state._sp_pp_capture_layout = None
    return state


def _manager(runtime, *, last_stage=False):
    manager = Glm5NextSPPPGraphManager.__new__(Glm5NextSPPPGraphManager)
    manager.model_runner = SimpleNamespace(_glm5next_sp_pp_runtime=runtime)
    manager.device = torch.device("cpu")
    manager.is_last_pp_rank = last_stage
    manager._sp_pp_output_buffer = None
    manager.intermediate_tensors = None
    manager._capture_descs = {
        CUDAGraphMode.FULL: [SimpleNamespace(num_tokens=5), SimpleNamespace(num_tokens=1)],
        # PIECEWISE's NONE warmup also stores an intermediate output.
        CUDAGraphMode.PIECEWISE: [SimpleNamespace(num_tokens=13)],
    }
    # Match the Ascend parent constructor's public capture-size inventory.
    manager.capture_sizes = collect_sorted_captured_token_sizes(manager._capture_descs)
    return manager


def test_prepare_inputs_activates_before_parent_for_real_dummy_and_profile(monkeypatch):
    runtime = Glm5NextSPPPRuntime(Glm5NextSPPPSpec(4, 0, 8, 3), torch.device("cpu"))
    active = Glm5NextSPPPBuffer.allocate(runtime.spec, 16, torch.device("cpu")).active
    runtime.bind_input(active)
    state = _state(runtime)
    calls = []

    def parent_prepare(self, input_batch, req_states):
        calls.append(active["hidden_states"].shape[0])
        return {"existing_hybrid_inputs": req_states}

    monkeypatch.setattr(AscendMambaHybridModelState, "prepare_inputs", parent_prepare)
    for total in (1, 13, 3):
        assert state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=total), "kept") == {
            "existing_hybrid_inputs": "kept"
        }
    assert calls == [1, 4, 1]
    runtime.begin_step((1, 5, 4, 0))
    with pytest.raises(RuntimeError, match="physical token mismatch"):
        state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=6), "kept")
    assert calls == [1, 4, 1]
    runtime.end_step()


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
    runtime = Glm5NextSPPPRuntime(Glm5NextSPPPSpec(4, 0, 8, 3), torch.device("cpu"))
    active = Glm5NextSPPPBuffer.allocate(runtime.spec, 16, torch.device("cpu")).active
    runtime.bind_input(active)
    state = _state(runtime)
    manager = _manager(runtime)
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
        for total in (13, 1, 5, 3):
            # Mirrors the unmodified parent: dummy hook, input [:T], output store.
            model_state.prepare_dummy_inputs(1, total)
            sliced = inputs[:total]
            rows = runtime.spec.local_rows(total)
            assert all(value.shape[0] == rows for value in sliced.tensors.values())
            assert {key: value.data_ptr() for key, value in sliced.tensors.items()} == original_addresses
            assert all(self.intermediate_tensors[key].data_ptr() != sliced[key].data_ptr() for key in sliced.tensors)
            for key, value in self.intermediate_tensors.tensors.items():
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
    assert manager._sp_pp_output_buffer.capacity == 4
    # Capture callback did not resize the runner's independent input container.
    assert active["hidden_states"].shape[0] == 4
    reference = weakref.ref(manager)
    del manager
    gc.collect()
    assert reference() is None


def test_full_replay_rebinds_small_and_large_output_views(monkeypatch):
    runtime = Glm5NextSPPPRuntime(Glm5NextSPPPSpec(4, 0, 8, 3), torch.device("cpu"))
    manager = _manager(runtime)
    manager._sp_pp_output_buffer = Glm5NextSPPPBuffer.allocate(runtime.spec, 16, torch.device("cpu"))
    manager.intermediate_tensors = manager._sp_pp_output_buffer.active
    addresses = {key: value.data_ptr() for key, value in manager.intermediate_tensors.tensors.items()}
    monkeypatch.setattr(
        ModelAclGraphManager, "run_fullgraph", lambda self, desc: self.intermediate_tensors[: desc.num_tokens]
    )
    for total in (1, 16, 3, 9):
        output = manager.run_fullgraph(SimpleNamespace(num_tokens=total))
        assert all(value.shape[0] == runtime.spec.local_rows(total) for value in output.tensors.values())
        assert {key: value.data_ptr() for key, value in output.tensors.items()} == addresses


def test_release_and_recapture_drop_only_manager_owned_output(monkeypatch):
    runtime = Glm5NextSPPPRuntime(Glm5NextSPPPSpec(4, 0, 8, 3), torch.device("cpu"))
    runtime.bind_input(Glm5NextSPPPBuffer.allocate(runtime.spec, 16, torch.device("cpu")).active)
    manager = _manager(runtime)
    manager._sp_pp_output_buffer = Glm5NextSPPPBuffer.allocate(runtime.spec, 16, torch.device("cpu"))
    manager.intermediate_tensors = manager._sp_pp_output_buffer.active
    monkeypatch.setattr(ModelAclGraphManager, "release_graphs", lambda self: None)
    manager.release_graphs()
    assert manager._sp_pp_output_buffer is None
    assert manager.intermediate_tensors is None
    assert runtime.input_buffer.capacity == 4
    state = _state(runtime)
    monkeypatch.setattr(ModelAclGraphManager, "capture", lambda *args, **kwargs: None)
    manager.capture(None, state, None, runtime.input_buffer.active, None, [], None)
    assert manager._sp_pp_output_buffer is not None
    assert manager._sp_pp_output_buffer.capacity == 4
    assert state._sp_pp_capture_layout is None
