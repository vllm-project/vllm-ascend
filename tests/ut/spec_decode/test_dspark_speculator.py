#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for Ascend DSpark model loading and target contracts."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import vllm.forward_context as forward_context
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.models.dspark_aux import (
    DSparkAuxHiddenContract,
    DSparkAuxHiddenFormat,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
    _copy_config_with_draft_capture_sizes,
    _derive_draft_capture_sizes,
)

_HIDDEN = 8
_FC_IN = 5 * _HIDDEN  # concatenated aux hidden states
# Patch where load_draft_model looks it up (the speculator module binding).
_ROT_MATRIX = "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_matrix"


def _spec(vllm_config: SimpleNamespace) -> AscendDSparkSpeculator:
    """Bypass the heavy ``__init__``; ``load_draft_model`` only reads
    ``self.vllm_config`` and the patched parent call."""
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    spec.aux_hidden_contract = None
    return spec


def _fake_draft() -> SimpleNamespace:
    fc = torch.nn.Linear(_FC_IN, _HIDDEN, bias=False)
    with torch.no_grad():
        fc.weight.copy_(torch.randn_like(fc.weight))
    return SimpleNamespace(model=SimpleNamespace(fc=fc))


def _quarot_config() -> SimpleNamespace:
    quarot = {"rotation_map": {"global_rotation": "x.safetensors"}}
    return SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"optional": {"quarot": quarot}}),
        model_config=SimpleNamespace(model="/fake"),
    )


def _bf16_config() -> SimpleNamespace:
    return SimpleNamespace(quant_config=None, model_config=SimpleNamespace())


def _no_call(*args, **kwargs):
    raise AssertionError("get_rotation_matrix must not be called without a rotation path")


class TestLoadDraftModel:
    """Cover QuaRot post-load handling and Aux Hidden negotiation."""

    @pytest.fixture
    def captured(self, monkeypatch):
        """Stub the heavy parent ``load_draft_model`` to return a fake draft and
        snapshot its fc weight before the override mutates it in place."""
        out: dict = {}

        def _load(self, target_model, target_attn_layer_names):
            draft = _fake_draft()
            out["before"] = draft.model.fc.weight.data.clone()
            out["draft"] = draft
            return draft

        monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
        return out

    def test_rotates_fc_for_quarot_target(self, captured, monkeypatch):
        # R = 2*I -> W @ R == 2*W, an expectation independent of process_weight.
        monkeypatch.setattr(_ROT_MATRIX, lambda path: torch.eye(_HIDDEN) * 2.0)
        draft = _spec(_quarot_config()).load_draft_model(MagicMock(), set())
        before = captured["before"]
        assert draft is captured["draft"]
        assert torch.allclose(draft.model.fc.weight.data, 2.0 * before, atol=1e-6)
        assert not torch.allclose(draft.model.fc.weight.data, before)

    def test_noop_for_bf16_target(self, captured, monkeypatch):
        monkeypatch.setattr(_ROT_MATRIX, _no_call)
        draft = _spec(_bf16_config()).load_draft_model(MagicMock(), set())
        assert torch.equal(draft.model.fc.weight.data, captured["before"])

    def test_negotiates_draft_declared_aux_hidden_contract(self, monkeypatch):
        contract = DSparkAuxHiddenContract(
            format=DSparkAuxHiddenFormat.RAW_PREFIX_SUM,
            layer_ids=(1,),
            capture_point="post_layer_raw_prefix_sum",
            target_hidden_size=_HIDDEN,
            dtype=torch.bfloat16,
        )
        draft = _fake_draft()
        draft.get_required_dspark_aux_hidden_state_contract = lambda: contract
        monkeypatch.setattr(
            DSparkSpeculator,
            "load_draft_model",
            lambda *_args: draft,
        )
        target = MagicMock()

        loaded = _spec(_bf16_config()).load_draft_model(target, set())

        assert loaded is draft
        target.configure_dspark_aux_hidden_state_contract.assert_called_once_with(contract)

    def test_rejects_target_without_required_aux_capability(self, monkeypatch):
        contract = DSparkAuxHiddenContract(
            format=DSparkAuxHiddenFormat.RAW_PREFIX_SUM,
            layer_ids=(1,),
            capture_point="post_layer_raw_prefix_sum",
            target_hidden_size=_HIDDEN,
            dtype=torch.bfloat16,
        )
        draft = _fake_draft()
        draft.get_required_dspark_aux_hidden_state_contract = lambda: contract
        monkeypatch.setattr(
            DSparkSpeculator,
            "load_draft_model",
            lambda *_args: draft,
        )

        with pytest.raises(ValueError, match="does not expose"):
            _spec(_bf16_config()).load_draft_model(SimpleNamespace(), set())


@pytest.mark.parametrize(
    ("capture_sizes", "target_query_len", "draft_query_len", "expected"),
    [
        ([16, 32], 8, 7, [14, 28]),
        ([12, 24], 6, 5, [10, 20]),
        ([1, 2, 4], 8, 7, []),
    ],
)
def test_derive_draft_capture_sizes_preserves_target_request_buckets(
    capture_sizes,
    target_query_len,
    draft_query_len,
    expected,
):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=max(capture_sizes),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    assert (
        _derive_draft_capture_sizes(
            config,
            target_query_len,
            draft_query_len,
        )
        == expected
    )


def test_copy_config_with_draft_capture_sizes_preserves_runtime_state():
    static_forward_context = {"draft.mla": object()}
    compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[16, 32],
        max_cudagraph_capture_size=32,
        static_forward_context=static_forward_context,
    )
    config = SimpleNamespace(compilation_config=compilation_config)

    draft_config = _copy_config_with_draft_capture_sizes(config, [14, 28])

    assert draft_config is not config
    assert draft_config.compilation_config is not compilation_config
    assert draft_config.compilation_config.cudagraph_capture_sizes == [14, 28]
    assert draft_config.compilation_config.max_cudagraph_capture_size == 28
    assert draft_config.compilation_config.static_forward_context is static_forward_context
    assert compilation_config.cudagraph_capture_sizes == [16, 32]
    assert compilation_config.max_cudagraph_capture_size == 32


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_graph_manager_respects_draft_enforce_eager(monkeypatch, enforce_eager):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.speculative_config = SimpleNamespace(enforce_eager=enforce_eager)
    spec.attn_cg_support = SimpleNamespace(min_cg_support=AttentionCGSupport.UNIFORM_BATCH)
    spec.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    spec.num_speculative_steps = 5
    spec.num_query_per_req = 5
    spec.device = torch.device("cpu")
    spec.update_stream = object()
    spec.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6], max_cudagraph_capture_size=6),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    manager = MagicMock()
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.dspark.speculator.DFlashAclGraphManager", manager)
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    expected = CUDAGraphMode.NONE if enforce_eager else CUDAGraphMode.FULL_DECODE_ONLY
    assert manager.call_args.args[2] == expected
    assert manager.call_args.kwargs["speculator"] is spec
    assert manager.call_args.args[0].compilation_config.cudagraph_capture_sizes == [5]
    assert spec.query_cudagraph_manager.update_stream is spec.update_stream


def test_update_draft_attn_metadata_updates_mla_decode_schema():
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.num_query_per_req = 7
    mla_metadata = SimpleNamespace(
        decode=SimpleNamespace(actual_seq_lengths_q=[7, 14]),
    )

    updated = spec._update_draft_attn_metadata(
        {"draft.mla": mla_metadata},
        num_reqs_padded=4,
    )

    assert updated["draft.mla"] is mla_metadata
    assert mla_metadata.decode.actual_seq_lengths_q == [7, 14, 21, 28]
    assert not hasattr(mla_metadata, "actual_seq_lengths_q")


def test_update_draft_attn_metadata_keeps_gqa_schema_compatible():
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.num_query_per_req = 5
    gqa_metadata = SimpleNamespace(actual_seq_lengths_q=[5])

    spec._update_draft_attn_metadata(
        {"draft.gqa": gqa_metadata},
        num_reqs_padded=3,
    )

    assert gqa_metadata.actual_seq_lengths_q == [5, 10, 15]


def test_build_draft_is_prefilling_zeros_padded_requests():
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.input_batch = SimpleNamespace(
        num_reqs=2,
        is_prefilling_np=np.array([True, False, True, True]),
    )

    is_prefilling = spec._build_draft_is_prefilling(4)

    torch.testing.assert_close(
        is_prefilling,
        torch.tensor([True, False, False, False]),
    )


@pytest.mark.parametrize("legacy_api", [False, True])
@pytest.mark.parametrize("dummy_run,skip_attn", [(True, True), (True, False), (False, False)])
def test_propose_does_not_reuse_target_dp_counts_for_memory_profile(monkeypatch, legacy_api, dummy_run, skip_attn):
    module = "vllm_ascend.worker.v2.spec_decode.dspark.speculator"
    counts = torch.tensor([16, 32], dtype=torch.int32)
    target_sync = counts if legacy_api else SimpleNamespace(num_tokens_across_dp=counts)
    spec = _spec(_bf16_config())
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(16, dtype=torch.int64))
    spec.max_num_tokens = 16
    spec.max_num_reqs = 2
    spec._build_draft_is_prefilling = MagicMock(return_value=torch.zeros(2, dtype=torch.bool))
    monkeypatch.setattr(f"{module}.vllm_version_is", lambda _: legacy_api)
    monkeypatch.setattr(f"{module}.build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(f"{module}.build_draft_attn_metadata_factory", lambda *args: nullcontext())
    parent_propose = MagicMock(return_value=torch.zeros(2, 5, dtype=torch.int64))
    monkeypatch.setattr(DSparkSpeculator, "propose", parent_propose)

    spec.propose(
        SimpleNamespace(num_reqs=2, num_tokens=16),
        {},
        {},
        torch.zeros(16, _HIDDEN),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        num_tokens_across_dp=target_sync if legacy_api else None,
        dp_sync=None if legacy_api else target_sync,
        dummy_run=dummy_run,
        skip_attn_for_dummy_run=skip_attn,
        is_profile=dummy_run and skip_attn,
    )

    forwarded_sync = parent_propose.call_args.args[11]
    if dummy_run and skip_attn:
        # None makes the draft forward synchronize its own query count (2 * 5),
        # rather than asserting that the target count (16) equals 10.
        assert forwarded_sync is None
    else:
        assert forwarded_sync is target_sync
    torch.testing.assert_close(counts, torch.tensor([16, 32], dtype=torch.int32))


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_memory_profile_recomputes_draft_dp_counts_in_forward_context(monkeypatch, dp_rank):
    """Exercise the real upstream profile branch and forward-context DP check."""
    module = "vllm_ascend.worker.v2.spec_decode.dspark.speculator"
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=dp_rank,
            is_moe_model=True,
        )
    )
    spec = _spec(config)
    spec.max_num_tokens = 16
    spec.max_num_reqs = 2
    spec.max_model_len = 128
    spec.num_query_per_req = 5
    spec.hidden_states = torch.zeros(16, _HIDDEN)
    spec.context_positions = torch.zeros(16, dtype=torch.int64)
    spec.input_buffers = SimpleNamespace(positions=spec.context_positions)
    spec.draft_tokens = torch.zeros(2, 5, dtype=torch.int64)
    spec.model = SimpleNamespace(precompute_and_store_context_kv=MagicMock())
    spec._prepare_eplb_forward = MagicMock()
    spec._build_draft_is_prefilling = MagicMock(return_value=torch.zeros(2, dtype=torch.bool))
    monkeypatch.setattr(f"{module}.vllm_version_is", lambda _: False)
    monkeypatch.setattr(f"{module}.build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(f"{module}.build_draft_attn_metadata_factory", lambda *args: nullcontext())
    # Stub only distributed transport / device context setup, not the upstream
    # propose path or the DPMetadata.make assertion that failed in production.
    coordinate = MagicMock(return_value=(False, torch.tensor([10, 10]), None))
    monkeypatch.setattr(forward_context, "coordinate_batch_across_dp", coordinate)
    monkeypatch.setattr(forward_context.current_platform, "set_additional_forward_context", lambda **kwargs: {})
    monkeypatch.setattr(forward_context, "create_forward_context", lambda *args, **kwargs: SimpleNamespace())
    observed = []

    def generate(num_reqs, num_tokens, **kwargs):
        with forward_context.set_forward_context(
            None,
            config,
            num_tokens=num_tokens,
            num_tokens_across_dp=kwargs["num_tokens_across_dp"],
        ):
            observed.append((num_reqs, num_tokens))

    spec._generate_draft = generate
    target_counts = torch.tensor([16, 16])
    spec.propose(
        SimpleNamespace(num_reqs=2, num_tokens=16, seq_lens_cpu_upper_bound=np.array([8, 8])),
        {},
        {},
        torch.zeros(16, _HIDDEN),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        dp_sync=SimpleNamespace(num_tokens_across_dp=target_counts),
        dummy_run=True,
        skip_attn_for_dummy_run=True,
        is_profile=True,
    )

    assert observed == [(2, 10)]
    coordinate.assert_called_once_with(
        num_tokens_unpadded=10,
        parallel_config=config.parallel_config,
        allow_microbatching=False,
    )
    torch.testing.assert_close(target_counts, torch.tensor([16, 16]))
