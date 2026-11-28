# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager

from vllm_ascend.ascend_config import is_draft_config_context
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.standalone_ar import speculator as module
from vllm_ascend.worker.v2.spec_decode.standalone_ar.aclgraph import StandaloneARAclGraphManager


@pytest.mark.parametrize("load_fails", [False, True])
def test_standalone_draft_compilation_does_not_inherit_target_layer_shapes(monkeypatch, load_fails):
    target_layer = SimpleNamespace(num_heads=32, num_kv_heads=8, head_size=128)
    draft_layer = SimpleNamespace(num_heads=16, num_kv_heads=8, head_size=128)
    target_context = {"target": target_layer}
    compilation = SimpleNamespace(static_forward_context=target_context, cudagraph_capture_sizes=[4, 8, 16])
    target_config = SimpleNamespace(compilation_config=compilation)
    speculator = object.__new__(module.AscendStandaloneARSpeculator)
    speculator.vllm_config = target_config
    target_model, draft_model, layer_names = object(), object(), {"target"}

    def load_parent(self, model, names):
        assert model is target_model and names is layer_names
        assert is_draft_config_context()
        assert self.vllm_config is not target_config
        assert self.vllm_config.compilation_config is not compilation
        context = self.vllm_config.compilation_config.static_forward_context
        assert context == {}
        context["draft"] = draft_layer
        self._draft_vllm_config = self.vllm_config
        if load_fails:
            raise RuntimeError("draft load failed")
        return draft_model

    monkeypatch.setattr(module.StandaloneARSpeculator, "load_draft_model", load_parent)
    if load_fails:
        with pytest.raises(RuntimeError, match="draft load failed"):
            speculator.load_draft_model(target_model, layer_names)
        assert target_context == {"target": target_layer}
    else:
        assert speculator.load_draft_model(target_model, layer_names) is draft_model
        assert target_context == {"target": target_layer, "draft": draft_layer}
        draft_context = speculator.draft_vllm_config.compilation_config.static_forward_context
        assert draft_context == {"draft": draft_layer}
        layer = next(iter(draft_context.values()))
        assert (layer.num_heads + 2 * layer.num_kv_heads) * layer.head_size == 4096
    assert speculator.vllm_config is target_config
    assert compilation.static_forward_context is target_context
    assert compilation.cudagraph_capture_sizes == [4, 8, 16]
    assert not is_draft_config_context()


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("target_bounds", [[7, 31], [63, 64]])
def test_correction_token_is_counted_once_in_standalone_attention(step, target_bounds):
    speculator = object.__new__(module.AscendStandaloneARSpeculator)
    speculator.expanded_positions = torch.arange(8)
    speculator.input_buffers = SimpleNamespace(positions=torch.arange(2))
    speculator.draft_is_prefilling = torch.tensor([True, False])
    speculator.vllm_config = SimpleNamespace(parallel_config=object(), model_config=SimpleNamespace(max_model_len=64))
    target = torch.tensor(target_bounds, dtype=torch.int32)
    # #60989's main propose includes the correction before builder dispatch.
    incoming = target + 1
    original = incoming.clone()
    qsl = np.array([0, 3, 8], dtype=np.int32)
    desc, slots, metadata = object(), torch.arange(8), {"draft": object()}

    with (
        patch.object(module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(module, "build_attn_metadata_factory", return_value=nullcontext()) as factory,
        patch.object(module.StandaloneARSpeculator, "_build_attn_metadata", return_value=metadata) as parent,
    ):
        actual = speculator._build_attn_metadata(2, desc, qsl, incoming, step, slot_mappings=slots)

    expected = incoming
    assert actual is metadata
    torch.testing.assert_close(parent.call_args.args[3], expected)
    torch.testing.assert_close(incoming, original)
    torch.testing.assert_close(target, torch.tensor(target_bounds, dtype=torch.int32))
    assert parent.call_args.args[0] == 2
    assert parent.call_args.args[1] is desc
    assert parent.call_args.args[2] is qsl
    assert parent.call_args.args[4] == step
    assert parent.call_args.args[7] is slots
    expected_positions = speculator.expanded_positions if step == 0 else speculator.input_buffers.positions
    assert factory.call_args.args[0] is expected_positions
    assert factory.call_args.args[1] == 8
    assert speculator.build_draft_attn_metadatas(2, 8, step == 0) == [metadata]
    assert speculator.build_param_source(2, metadata, step == 0).context is metadata


def test_standalone_graph_capture_has_ascend_input_buffer_fields(monkeypatch):
    def init_parent(self, config, device):
        self.max_num_reqs = 2
        self.max_num_tokens = 16
        self.input_buffers = InputBuffers(2, 16, device)

    monkeypatch.setattr(module.StandaloneARSpeculator, "__init__", init_parent)
    monkeypatch.setattr("vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda positions: None)
    speculator = module.AscendStandaloneARSpeculator(object(), torch.device("cpu"))

    batch = AscendInputBatch.make_dummy(2, 4, speculator.input_buffers)
    assert isinstance(speculator.input_buffers, AscendInputBuffers)
    assert speculator.input_buffers.dummy_num_tokens is None
    assert batch.num_tokens == 4
    np.testing.assert_array_equal(batch.seq_lens_np, [2, 2])
    assert speculator.input_buffers.query_start_loc.numel() == 4


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_standalone_graph_managers_are_bound_to_speculator(enforce_eager):
    speculator = object.__new__(module.AscendStandaloneARSpeculator)
    speculator.vllm_config = MagicMock()
    speculator.vllm_config.get_decode_cudagraph_query_lens.return_value = {5: 4}
    speculator.vllm_config.get_draft_decode_cudagraph_request_sizes.return_value = [1, 2, 4]
    speculator.speculative_config = SimpleNamespace(enforce_eager=enforce_eager)
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 3
    speculator.max_num_reqs = 4
    speculator.enable_adaptive_verification = False
    speculator.update_stream = object()
    prefill, decode = MagicMock(), MagicMock()
    with patch.object(module, "StandaloneARAclGraphManager", side_effect=[prefill, decode]) as factory:
        speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

    expected_mode = CUDAGraphMode.NONE if enforce_eager else CUDAGraphMode.FULL_DECODE_ONLY
    assert factory.call_args_list[0].args[2] == expected_mode
    assert factory.call_args_list[1].args[2] == expected_mode
    assert factory.call_args_list[0].args[3] == {5: 4}
    assert factory.call_args_list[1].kwargs["decode_query_lens"] == {1: 4}
    assert factory.call_args_list[1].kwargs["decode_request_sizes"] == [1, 2, 4]
    for manager in (prefill, decode):
        assert manager.speculator is speculator
        assert manager.update_stream is speculator.update_stream


@pytest.mark.parametrize("prefill", [False, True])
def test_standalone_capture_preserves_upstream_slots_and_single_step_forward(prefill):
    manager = object.__new__(StandaloneARAclGraphManager)
    manager.speculator = object()
    manager.is_draft_model_prefill = prefill
    forward_fn, slots = object(), torch.arange(8)
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.standalone_ar.aclgraph.communicator_switch", return_value=nullcontext()
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.standalone_ar.aclgraph.model_capture_wrapper", return_value=nullcontext()
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.standalone_ar.aclgraph.build_attn_metadata_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(SpeculatorCudaGraphManager, "capture") as native_capture,
    ):
        manager.capture(forward_fn, slot_mappings_buffer=slots, specialize_spec_tokens=True)

    native_capture.assert_called_once_with(manager, forward_fn, slot_mappings_buffer=slots, specialize_spec_tokens=True)
