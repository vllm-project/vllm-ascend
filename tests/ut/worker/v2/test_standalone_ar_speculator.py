# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import CrossAttentionSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as upstream_module
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import StandaloneARSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode.standalone_ar import speculator as module
from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import AscendStandaloneARSpeculator


@pytest.mark.parametrize("step,expected", [(0, [11, 21, 24]), (2, [12, 19, 24]), (3, [13, 20, 24])])
def test_attention_lengths_rewind_rejected_tokens(monkeypatch, step, expected):
    """Correction prefill retains the query span; decode removes rejected KV."""
    draft = object.__new__(AscendStandaloneARSpeculator)
    draft.max_model_len = 24
    draft._target_seq_lens_cpu = torch.tensor([10, 20, 23], dtype=torch.int32)
    draft._num_rejected_cpu = torch.tensor([0, 3, 1], dtype=torch.int32)
    draft.expanded_positions = torch.arange(12)
    draft.input_buffers = SimpleNamespace(positions=torch.arange(3) + 10)
    draft._draft_attn_config = SimpleNamespace(parallel_config=object())
    captured = {}

    @contextmanager
    def factory(positions, pad, is_prefilling, **kwargs):
        captured.update(kwargs, positions=positions, is_prefilling=is_prefilling)
        yield

    def build(self, *args):
        # Parent adds the step to the supplied CPU upper bound.
        captured["upper_bound"] = (args[3] + args[4]).clamp(max=self.max_model_len)
        captured["slot_mappings"] = args[-1]
        return {"draft.layer": captured}

    monkeypatch.setattr(module, "build_attn_metadata_factory", factory)
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(module, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(StandaloneARSpeculator, "_build_attn_metadata", build)
    slots = torch.arange(12, dtype=torch.int32).reshape(1, -1)
    batch_desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=12, num_reqs=3)
    result = draft._build_attn_metadata(
        3, batch_desc, np.array([0, 4, 8, 12]), torch.tensor([99, 99, 99]), step, slot_mappings=slots
    )
    assert result["draft.layer"] is captured
    assert captured["seq_lens_cpu"].tolist() == expected
    assert captured["upper_bound"].tolist() == expected
    assert captured["positions"] is (draft.expanded_positions if step == 0 else draft.input_buffers.positions)
    assert captured["attn_state"] == (
        AscendAttentionState.ChunkedPrefill if step == 0 else AscendAttentionState.DecodeOnly
    )
    assert captured["is_prefilling"].tolist() == [step == 0] * 3
    assert captured["slot_mappings"] is slots


def test_prefill_uses_exact_device_lengths(monkeypatch):
    draft = object.__new__(AscendStandaloneARSpeculator)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([10, 20, 99], dtype=torch.int32), seq_lens_cpu_upper_bound=torch.tensor([15, 25])
    )
    parent = MagicMock()
    monkeypatch.setattr(StandaloneARSpeculator, "_prefill", parent)
    rejected = torch.tensor([0, 3, 77])
    draft._prefill(batch, rejected, torch.tensor([1, 2]), 2, False, None)
    assert draft._target_seq_lens_cpu.tolist() == [10, 20]
    assert draft._num_rejected_cpu.tolist() == [0, 3]
    parent.assert_called_once()


def test_dummy_prefill_skips_host_copy(monkeypatch):
    draft = object.__new__(AscendStandaloneARSpeculator)
    monkeypatch.setattr(StandaloneARSpeculator, "_prefill", MagicMock())
    draft._prefill(object(), torch.empty(0), torch.empty(0), 2, True, None)
    assert not hasattr(draft, "_target_seq_lens_cpu")


@pytest.mark.parametrize(
    "parallel_field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_rejects_unadapted_parallel_topology(monkeypatch, parallel_field):
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    parallel = dict(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1)
    parallel[parallel_field] = 2
    with pytest.raises(NotImplementedError, match="PP=PCP=DCP=1"):
        AscendStandaloneARSpeculator(SimpleNamespace(parallel_config=SimpleNamespace(**parallel)), torch.device("cpu"))


@pytest.mark.parametrize(
    "compatibility,is_moe,draft_tp,message",
    [
        (True, False, 2, "310P"),
        (False, True, 2, "dense draft model"),
        (False, False, 1, "same draft and target TP"),
    ],
)
def test_rejects_unadapted_draft_configuration(monkeypatch, compatibility, is_moe, draft_tp, message):
    monkeypatch.setattr(
        module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: compatibility)
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=2,
        ),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                is_moe=is_moe, is_encoder_decoder=False, supports_multimodal_inputs=False
            ),
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=draft_tp),
        ),
    )
    with pytest.raises(NotImplementedError, match=message):
        AscendStandaloneARSpeculator(config, torch.device("cpu"))


@pytest.mark.parametrize("spec", [object(), object.__new__(CrossAttentionSpec)])
def test_rejects_draft_non_decoder_cache(spec):
    draft = object.__new__(AscendStandaloneARSpeculator)
    draft.draft_attn_layer_names = {"draft.layer"}
    kv_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["draft.layer"], kv_cache_spec=spec)])
    with pytest.raises(NotImplementedError, match="attention-only"):
        draft.set_attn(None, kv_config, None, None, [])


@pytest.mark.parametrize("field", ["is_encoder_decoder", "supports_multimodal_inputs"])
def test_rejects_draft_with_missing_encoder_inputs(monkeypatch, field):
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    draft_config = dict(is_moe=False, is_encoder_decoder=False, supports_multimodal_inputs=False)
    draft_config[field] = True
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(**draft_config)),
    )
    with pytest.raises(NotImplementedError, match="text-only decoder"):
        AscendStandaloneARSpeculator(config, torch.device("cpu"))


def test_real_metadata_builder_receives_expanded_query_layout(monkeypatch):
    """Exercise the inherited vLLM builder and Ascend wrappers together."""
    draft = object.__new__(AscendStandaloneARSpeculator)
    draft.max_model_len = draft.draft_max_seq_len = 32
    draft.dcp_size = 1
    draft._target_seq_lens_cpu = torch.tensor([10, 20], dtype=torch.int32)
    draft._num_rejected_cpu = torch.tensor([0, 2], dtype=torch.int32)
    draft.expanded_positions = torch.arange(7, dtype=torch.int64)
    draft.input_buffers = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32),
        seq_lens=torch.tensor([11, 21], dtype=torch.int32),
        positions=torch.zeros(7, dtype=torch.int64),
    )
    draft.block_tables = SimpleNamespace(input_block_tables=[torch.zeros(2, 2)], slot_mappings=torch.zeros(1, 5))
    draft.attn_groups, draft.kv_cache_config = [], object()
    draft.draft_is_prefilling = torch.zeros(2, dtype=torch.bool)
    draft._draft_attn_config = SimpleNamespace(parallel_config=object())
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return {"draft.layer": kwargs}

    monkeypatch.setattr(attn_utils, "build_attn_metadata", build)
    monkeypatch.setattr(module, "set_current_vllm_config", lambda _: nullcontext())
    slots = torch.arange(7, dtype=torch.int32).reshape(1, -1)
    draft._build_attn_metadata(
        2,
        BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=7, num_reqs=2),
        np.array([0, 3, 7], dtype=np.int32),
        torch.tensor([99, 99]),
        0,
        slot_mappings=slots,
    )
    assert captured["num_tokens"] == 7
    assert captured["query_start_loc_cpu"].tolist() == [0, 3, 7]
    assert captured["max_query_len"] == 4
    assert captured["seq_lens_cpu_upper_bound"].tolist() == [11, 21]
    assert captured["seq_lens_np"].tolist() == [11, 21]
    assert captured["positions"].data_ptr() == draft.expanded_positions.data_ptr()
    assert captured["slot_mappings"] is slots


def test_expanded_slots_match_ascend_cache_dtype(monkeypatch):
    draft = object.__new__(AscendStandaloneARSpeculator)
    draft.draft_attn_layer_names = {"draft.layer"}
    draft._draft_attn_config = object()
    draft.max_num_tokens, draft.max_num_reqs = 8, 4
    draft.device = torch.device("cpu")
    monkeypatch.setattr(StandaloneARSpeculator, "set_attn", MagicMock())
    monkeypatch.setattr(module, "set_current_vllm_config", lambda _: nullcontext())
    draft.set_attn(
        None,
        SimpleNamespace(kv_cache_groups=[]),
        SimpleNamespace(num_kv_cache_groups=2, slot_mappings=torch.zeros(2, 8, dtype=torch.int32)),
        None,
        [],
    )
    assert draft.expanded_slot_mappings.dtype == torch.int32
    assert draft.expanded_slot_mappings.shape == (2, 12)


def test_forward_uses_draft_config_and_unwraps_tuple(monkeypatch):
    draft = object.__new__(AscendStandaloneARSpeculator)
    draft._draft_attn_config = object()
    draft.vllm_config = object()
    hidden = torch.zeros(3, 4)
    draft.model = MagicMock(return_value=(hidden, torch.ones(3, 4)))
    draft._prepare_eplb_forward = MagicMock()
    forward_context = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(module, "set_forward_context", forward_context)
    monkeypatch.setattr(module, "set_current_vllm_config", lambda _: nullcontext())
    ids, positions = torch.arange(3), torch.arange(3)
    assert draft._run_model(ids, positions, {"draft.layer": object()}, {}, None) is hidden
    assert forward_context.call_args.args[1] is draft.attn_vllm_config
    assert forward_context.call_args.kwargs["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    draft.model.assert_called_once_with(input_ids=ids, positions=positions)


def test_decode_loop_overwrites_rejected_kv_positions(monkeypatch):
    """Run the inherited loop and check positions/lengths seen by each forward."""
    draft = object.__new__(AscendStandaloneARSpeculator)
    draft.max_model_len = 32
    draft.num_speculative_steps = 3
    draft.expanded_positions = torch.tensor([10, 17])
    draft.last_token_indices = torch.tensor([0, 1])
    draft.input_buffers = SimpleNamespace(
        positions=torch.zeros(2, dtype=torch.int64),
        seq_lens=torch.zeros(2, dtype=torch.int32),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        input_ids=torch.zeros(2, dtype=torch.int32),
    )
    draft.arange_gpu = torch.arange(3, dtype=torch.int32)
    draft.draft_tokens = torch.tensor([[31, 0, 0], [32, 0, 0]])
    draft.current_draft_step = torch.tensor(0)
    draft.idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    draft.temperature = torch.zeros(2)
    draft.seeds = torch.zeros(2, dtype=torch.int64)
    draft.draft_logits = None
    draft.sample_draft = MagicMock(side_effect=[torch.tensor([41, 42]), torch.tensor([51, 52])])
    forwards = []

    def run(input_ids, positions, *args):
        forwards.append((input_ids.clone(), positions.clone(), draft.input_buffers.seq_lens.clone()))
        return torch.zeros(2, 4)

    draft._run_model = run
    batch = SimpleNamespace(
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        idx_mapping=draft.idx_mapping,
        seq_lens_cpu_upper_bound=torch.tensor([10, 20], dtype=torch.int32),
    )
    draft.block_tables = SimpleNamespace(compute_slot_mappings=MagicMock(return_value=torch.zeros(1, 2)))
    draft.kv_cache_config = object()
    draft._build_uniform_attn_metadata = MagicMock(return_value={})
    monkeypatch.setattr(upstream_module, "build_slot_mappings_by_layer", lambda *args: {})
    draft._multi_step_decode(batch, torch.tensor([0, 3]), 2, False, None)
    assert [(ids.tolist(), positions.tolist(), lengths.tolist()) for ids, positions, lengths in forwards] == [
        ([31, 32], [11, 18], [12, 19]),
        ([41, 42], [12, 19], [13, 20]),
    ]
    assert draft.draft_tokens.tolist() == [[31, 41, 51], [32, 42, 52]]
    assert [call.kwargs["step"] for call in draft._build_uniform_attn_metadata.call_args_list] == [2, 3]
