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
"""Unit tests for the dspark speculative-decoding proposer."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask

# 0 = single-DP (no padding); >0 = multi-DP where num_input_tokens >
# num_query_total, the out-of-bounds regime.
MULTI_DP_PADDING_SIZES = [0, 8, 32]
_NUM_SPECULATIVE_TOKENS = 3
_MAX_BATCH_SIZE = 2
_MAX_NUM_TOKENS = 8
_HIDDEN_SIZE = 16


@pytest.mark.parametrize(
    ("dcp_size", "pcp_enabled", "expected_submit"),
    [(1, False, True), (2, False, False), (1, True, False)],
)
def test_build_draft_metadata_submits_only_non_cp_device_tasks(
    dcp_size: int,
    pcp_enabled: bool,
    expected_submit: bool,
):
    tasks = [DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None, group_id) for group_id in (7, 9)]

    class DraftMetadataProvider:
        def __init__(self, task):
            self.enabled = False
            self.task = task

        def enable_device_metadata(self):
            self.enabled = True

        def take_device_metadata_tasks(self):
            return (self.task,)

        def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
            return SimpleNamespace()

    builders = [DraftMetadataProvider(task) for task in tasks]
    groups = [
        SimpleNamespace(
            kv_cache_group_id=group_id,
            layer_names=[f"draft.attn.{group_id}"],
            get_metadata_builder=lambda builder=builder: builder,
        )
        for group_id, builder in enumerate(builders)
    ]
    executor = MagicMock()
    proposer = SimpleNamespace(
        draft_attn_groups=groups,
        use_compress=False,
        method="dspark",
        runner=SimpleNamespace(device_metadata_executor=executor),
        dcp_size=dcp_size,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=2 if pcp_enabled else 1)
        ),
        sliding_window=None,
        _per_group_block_table_buffers={group_id: torch.ones((1, 1), dtype=torch.int32) for group_id in range(2)},
        _per_group_query_slot_mapping_buffers={group_id: torch.zeros(1, dtype=torch.int32) for group_id in range(2)},
        _prepare_dspark_group_metadata=lambda metadata, *_args, **_kwargs: metadata,
    )
    common_attn_metadata = SimpleNamespace(
        num_reqs=1,
        block_table_tensor=torch.ones((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
    )

    metadata, first = AscendSpecDecodeBaseProposer.build_draft_attn_metadata(
        proposer,
        common_attn_metadata,
        num_input_tokens=1,
        num_actual_tokens=1,
    )

    assert metadata[0]["draft.attn.0"] is first
    assert all(not builder.enabled for builder in builders)
    if expected_submit:
        executor.submit.assert_called_once_with(tasks)
    else:
        executor.submit.assert_not_called()


@pytest.mark.parametrize("has_task", [True, False])
def test_dspark_device_metadata_executor_forward_lifecycle(has_task: bool):
    events = []
    executor = SimpleNamespace(submission_in_flight=False)

    def release():
        events.append("release")

    executor.release = release
    runner = SimpleNamespace(
        device_metadata_executor=executor,
        dcp_manager=None,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        _sync_metadata_across_dp=lambda num_tokens, **kwargs: (num_tokens, torch.tensor(1), None),
        dynamic_eplb=False,
        eplb_heat_collection_status=False,
    )
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.runner = runner
    proposer.method = "dspark"
    proposer.model = SimpleNamespace(combine_hidden_states=lambda hidden_states: hidden_states)
    proposer.hidden_size = 4
    proposer.use_cuda_graph = False
    proposer.dcp_size = 1
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_mla=True))
    proposer.draft_window_size = None
    proposer.supports_mm_inputs = False
    proposer.slot_mapping_group = [torch.zeros(2, dtype=torch.int32)]
    proposer.seq_lens_group = [torch.zeros(2, dtype=torch.int32)]
    proposer.query_start_loc_group = [torch.zeros(3, dtype=torch.int32)]
    proposer._pad_draft_buffers = MagicMock()
    proposer.uses_mrope = False
    proposer.positions = torch.arange(2, dtype=torch.int32)
    proposer.parallel_drafting = True
    proposer.token_indices_to_sample = torch.zeros(2, dtype=torch.int32)
    proposer.enable_enpu = False
    proposer._update_full_graph_params_if_needed = MagicMock()
    proposer.set_inputs_first_pass = MagicMock()
    proposer.build_draft_attn_metadata = MagicMock()
    proposer._context_slot_mapping_buffers = None
    proposer.build_model_inputs_first_pass = MagicMock()

    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        batch_size=lambda: 1,
        num_reqs=1,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        block_table_tensor=torch.ones((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
    )
    proposer.set_inputs_first_pass.return_value = (
        1,
        torch.tensor([0], dtype=torch.int32),
        common_attn_metadata,
        None,
    )

    def build_metadata(*args, **kwargs):
        events.append("submit" if has_task else "build")
        executor.submission_in_flight = has_task
        metadata = {"draft.attn": SimpleNamespace(num_prefills=0)}
        return [metadata], metadata["draft.attn"]

    proposer.build_draft_attn_metadata.side_effect = build_metadata

    def run_draft(**kwargs):
        events.append("run")
        return torch.ones((1, 3), dtype=torch.int64)

    proposer._runnable = run_draft
    forward_context = SimpleNamespace(moe_layer_index=-1, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    context_executors = []

    @contextmanager
    def forward_context_manager(*args, **kwargs):
        context_executors.append(kwargs["device_metadata_executor"])
        events.append("context")
        yield

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer._HIDDEN_STATE_DRAFTER_TYPES", (object,)),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.set_ascend_forward_context",
            forward_context_manager,
        ),
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_forward_context", return_value=forward_context),
    ):
        result = AscendSpecDecodeBaseProposer._propose(
            proposer,
            3,
            target_token_ids=torch.ones(1, dtype=torch.int64),
            target_positions=torch.zeros(1, dtype=torch.int32),
            target_hidden_states=torch.ones((1, 4)),
            next_token_ids=torch.ones(1, dtype=torch.int64),
            token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
            common_attn_metadata=common_attn_metadata,
            target_model_batch_desc=SimpleNamespace(uniform=True),
            sampling_metadata=MagicMock(),
        )

    assert torch.equal(result, torch.ones((1, 3), dtype=torch.int64))
    assert context_executors == [executor if has_task else None]
    assert events == (["submit", "context", "run", "release"] if has_task else ["build", "context", "run"])


@pytest.fixture(autouse=True)
def _stub_device_properties(monkeypatch):
    """CPU CI has no NPU: ``init_device_properties_triton`` is skipped when
    ``HAS_TRITON`` is false, leaving ``_NUM_VECTORCORE`` unset, so
    ``get_vectorcore_num`` asserts. ``set_inputs_first_pass`` sizes the kernel
    grid via ``_compute_num_programs`` -> ``get_vectorcore_num``; stub the
    device-property globals so the grid computation runs on CPU. The kernel
    itself is mocked per-test, and the small inputs here yield a ``(1,)`` grid
    either way (matching ``test_kernel_called_with_has_num_rejected``)."""
    monkeypatch.setattr("vllm_ascend.ops.triton.triton_utils._NUM_AICORE", 8)
    monkeypatch.setattr("vllm_ascend.ops.triton.triton_utils._NUM_VECTORCORE", 8)
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.dspark_proposer.get_ascend_config",
        lambda: SimpleNamespace(
            dynamic_spec_config=SimpleNamespace(method=None, method_params={})
        ),
    )


class _DSparkProposerTestBase:
    """Shared helpers for ``AscendDSparkProposer`` tests."""

    @staticmethod
    def _make_vllm_config(hf_config: SimpleNamespace, draft_sample_method: str) -> SimpleNamespace:
        """Build the minimal config consumed by the DSpark initializer."""
        draft_model_config = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: _HIDDEN_SIZE)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_sample_method=draft_sample_method,
                draft_model_config=draft_model_config,
            )
        )

    @classmethod
    def _make_proposer(
        cls,
        *,
        max_num_tokens: int,
        num_reqs: int,
        block_size: int,
        hf_config: SimpleNamespace | None = None,
        draft_attn_causal: bool | None = None,
        draft_sample_method: str = "greedy",
    ):
        device = torch.device("cpu")
        vllm_config = cls._make_vllm_config(hf_config or SimpleNamespace(), draft_sample_method)

        def mock_parent_init(
            proposer: AscendDSparkProposer,
            vllm_config: SimpleNamespace,
            device: torch.device,
            runner: object | None = None,
        ) -> None:
            del runner
            proposer.draft_model_config = vllm_config.speculative_config.draft_model_config
            proposer.num_speculative_tokens = block_size
            proposer.max_batch_size = num_reqs
            proposer.max_num_tokens = max_num_tokens
            proposer.dtype = torch.float32
            proposer.device = device
            proposer.hidden_size = _HIDDEN_SIZE
            proposer.hidden_states = torch.empty(0)
            proposer._dflash_hidden_states = torch.empty(0)
            proposer.model = (
                SimpleNamespace(get_draft_attn_causal=lambda: [draft_attn_causal])
                if draft_attn_causal is not None
                else SimpleNamespace()
            )

        dynamic_spec_config = SimpleNamespace(method="", method_params={})
        with (
            patch.object(AscendDSparkProposer.__base__, "__init__", mock_parent_init),
            patch(
                "vllm_ascend.spec_decode.dspark_proposer.get_ascend_config",
                return_value=SimpleNamespace(
                    dynamic_spec_config=dynamic_spec_config,
                ),
            ),
        ):
            proposer = AscendDSparkProposer(vllm_config, device)
        num_query_total = num_reqs * proposer.num_query_per_req
        proposer.positions = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer.positions[:num_query_total] = torch.arange(num_query_total, dtype=torch.int32)
        proposer.parallel_drafting_token_id = 0
        proposer.kv_cache_gid = 0
        proposer._dflash_num_context = 0

        proposer.input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._context_positions_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._slot_mapping_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._dspark_seed_buffer = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._dflash_hidden_states = torch.zeros((max_num_tokens, 8), dtype=torch.float32, device=device)
        proposer.arange_dflash = torch.arange(max_num_tokens + 1, dtype=torch.int32, device=device)
        proposer.token_arange_np = np.arange(max_num_tokens + 1, dtype=np.int32)

        gid = 0
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=gid,
                kv_cache_spec=SimpleNamespace(block_size=block_size),
                layer_names=["L0"],
            )
        ]
        proposer._layer_group_idx = [gid]
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)
        proposer._per_group_block_tables = {gid: block_table}
        proposer._per_group_block_table_buffers = {gid: block_table}
        slot = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._per_group_slot_mappings = {gid: slot}
        proposer._per_group_kernel_block_sizes = {gid: block_size}
        proposer._per_group_query_slot_mapping_buffers = {gid: slot.clone()}
        proposer._per_group_context_slot_mapping_buffers = {gid: slot.clone()}
        return proposer

    @staticmethod
    def _invoke_set_inputs_first_pass(
        proposer,
        *,
        num_reqs,
        block_size,
        seq_len=128,
        host_seq_len=None,
        async_metadata=False,
        context=None,
        num_rejected=None,
        with_optional_attrs=False,
    ):
        """Drive ``set_inputs_first_pass`` with a configurable cad.

        ``context`` sets ``query_start_loc_cpu[num_reqs]`` so the proposer
        copies ``context`` rows of target hidden states (0 by default).
        Returns ``(num_query_total, token_indices, cad, extra,
        next_token_ids, target_hidden_states)``.
        """
        next_token_ids = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        target_hidden_states = torch.arange(num_reqs * 8, dtype=torch.float32).reshape(num_reqs, 8)
        query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        if context is not None:
            query_start_loc_cpu[num_reqs] = context
        if host_seq_len is None:
            host_seq_len = seq_len
        seq_lens_cpu = torch.full((num_reqs,), host_seq_len, dtype=torch.int32)
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=torch.full((num_reqs,), seq_len, dtype=torch.int32),
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu=None if async_metadata else seq_lens_cpu,
            max_seq_len=seq_len,
        )
        if with_optional_attrs:
            cad.actual_seq_lengths_q = [0] * num_reqs
            cad.decode_token_per_req = 0
        num_query_total, token_indices, cad, extra = proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=next_token_ids,
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected,
        )
        return num_query_total, token_indices, cad, extra, next_token_ids, target_hidden_states


class TestDSparkPositionsFullUnderMultiDp(_DSparkProposerTestBase):
    """Guard: under multi-DP the dspark draft proposer must hand DSA attention a
    full-length positions buffer so ``positions[:num_input_tokens]`` never reads
    out of bounds (the slice is DP-padded and may exceed the local query size)."""

    @staticmethod
    def _call_set_inputs_first_pass(proposer, *, num_reqs, block_size):
        # query_start_loc_cpu[num_reqs] is 0 so _dflash_num_context becomes 0.
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=torch.zeros(num_reqs + 1, dtype=torch.int32),
            seq_lens=torch.full((num_reqs,), 128, dtype=torch.int32),
            _seq_lens_cpu=torch.full((num_reqs,), 128, dtype=torch.int32),
            seq_lens_cpu=torch.full((num_reqs,), 128, dtype=torch.int32),
            max_seq_len=128,
        )
        proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=torch.zeros((num_reqs, 8), dtype=torch.float32),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )
        return cad

    @pytest.mark.parametrize("dp_padding", MULTI_DP_PADDING_SIZES)
    def test_positions_not_pre_sliced(self, monkeypatch, dp_padding):
        """``cad.positions`` must be the full buffer, not ``[:num_query_total]``."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)

        # DSA attention slices positions[:num_input_tokens] (DP-padded); a
        # pre-slice to num_query_total reads out of bounds under multi-DP.
        assert cad.positions.shape[0] == max_num_tokens
        assert cad.positions[:num_input_tokens].shape[0] == num_input_tokens

    @pytest.mark.parametrize("dp_padding", [8, 32])
    def test_positions_full_and_padded_for_dsa(self, monkeypatch, dp_padding):
        """After set_inputs_first_pass + _pad_draft_buffers, positions[:num_input]
        is full-length and zero-padded in the DP region."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_query_total:num_input_tokens] = -999
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)
        proposer._pad_draft_buffers(num_query_total, num_input_tokens)

        dsa_slice = cad.positions[:num_input_tokens]
        assert dsa_slice.shape[0] == num_input_tokens
        assert torch.all(dsa_slice[num_query_total:] == 0)


class TestPadDraftBuffersBeforeBuild(_DSparkProposerTestBase):
    """Guard: ``_pad_draft_buffers`` must zero the DP-padding region of positions
    and run before ``build_draft_attn_metadata``, so the attention backend reads
    valid (zero) padding instead of stale values."""

    def test_zeros_dp_padding_region(self):
        """``_pad_draft_buffers`` zeros positions / input_ids / slot_mapping in
        the DP-padding region."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_actual:num_input] = -999
        proposer.input_ids[num_actual:num_input] = -999
        proposer._slot_mapping_buffer[num_actual:num_input] = -999
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            buf[num_actual:num_input] = -999

        proposer._pad_draft_buffers(num_actual, num_input)

        assert torch.all(proposer.positions[num_actual:num_input] == 0)
        assert torch.all(proposer.input_ids[num_actual:num_input] == proposer.parallel_drafting_token_id)
        assert torch.all(proposer._slot_mapping_buffer[num_actual:num_input] == -1)
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            assert torch.all(buf[num_actual:num_input] == -1)
        assert torch.all(proposer.positions[:num_actual] != -999)

    def test_noop_without_dp_padding(self):
        """Single-DP (num_input <= num_actual) leaves buffers untouched."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        snapshot = proposer.positions.clone()
        proposer._pad_draft_buffers(num_actual, num_actual)
        assert torch.equal(proposer.positions, snapshot)


class TestDSparkInitialization(_DSparkProposerTestBase):
    """Tests for DSpark initialization configuration."""

    @pytest.mark.parametrize(
        ("hf_config", "expected_sample_from_anchor", "expected_num_query_per_req", "draft_sample_method"),
        [
            pytest.param(SimpleNamespace(), True, _NUM_SPECULATIVE_TOKENS, "greedy"),
            pytest.param(
                SimpleNamespace(sample_from_anchor=False), False, 1 + _NUM_SPECULATIVE_TOKENS, "probabilistic"
            ),
        ],
    )
    def test_configures_anchor_sampling(
        self,
        hf_config: SimpleNamespace,
        expected_sample_from_anchor: bool,
        expected_num_query_per_req: int,
        draft_sample_method: str,
    ) -> None:
        """Verify the bonus-anchor flag selects the expected query layout."""
        proposer = self._make_proposer(
            max_num_tokens=_MAX_NUM_TOKENS,
            num_reqs=_MAX_BATCH_SIZE,
            block_size=_NUM_SPECULATIVE_TOKENS,
            hf_config=hf_config,
            draft_sample_method=draft_sample_method,
        )
        expected_max_query_tokens = _MAX_BATCH_SIZE * (1 + _NUM_SPECULATIVE_TOKENS)
        assert proposer.sample_from_anchor is expected_sample_from_anchor
        assert proposer.num_query_per_req == expected_num_query_per_req
        assert proposer.max_query_tokens == expected_max_query_tokens
        assert proposer._dspark_draft_buffer.shape == (_MAX_BATCH_SIZE, 1 + _NUM_SPECULATIVE_TOKENS)


# fmt: off
class TestSetPerGroupAttnMetadata(_DSparkProposerTestBase):
    """``set_per_group_attn_metadata`` stores the runner-provided per-group
    block table / slot mapping into the read-only dicts the proposer consults
    during ``set_inputs_first_pass``."""

    def test_stores_block_table_and_slot_mapping(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        # a gid not pre-populated by _make_proposer (which only seeds gid=0)
        gid = 7
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32)
        slot_mapping = torch.full((max_num_tokens,), 42, dtype=torch.int32)

        proposer.set_per_group_attn_metadata(gid, block_table, slot_mapping)

        assert proposer._per_group_block_tables[gid] is block_table
        assert proposer._per_group_slot_mappings[gid] is slot_mapping

    def test_overwrites_existing_gid(self):
        num_reqs, block_size, max_num_tokens = 2, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        gid = 0  # already populated by _make_proposer
        old_block_table = proposer._per_group_block_tables[gid]
        new_block_table = torch.ones((num_reqs, 16), dtype=torch.int32)
        new_slot_mapping = torch.ones(max_num_tokens, dtype=torch.int32)

        proposer.set_per_group_attn_metadata(gid, new_block_table, new_slot_mapping)

        assert proposer._per_group_block_tables[gid] is new_block_table
        assert proposer._per_group_slot_mappings[gid] is new_slot_mapping
        assert proposer._per_group_block_tables[gid] is not old_block_table


class TestDSparkInitValidation:
    """Validate DSpark-specific buffers and DFlash graph overrides."""

    @staticmethod
    def _make_vllm_config(
        *,
        num_speculative_tokens,
        max_batch_size,
        max_num_tokens,
        draft_sample_method,
        hidden_size=8,
    ):
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            draft_sample_method=draft_sample_method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(),
                get_hidden_size=lambda: hidden_size
            ),
        )
        return SimpleNamespace(speculative_config=speculative_config)

    @staticmethod
    def _stub_dflash_init(
        monkeypatch,
        *,
        num_speculative_tokens,
        max_batch_size,
        max_num_tokens,
        dtype,
        device,
    ):
        """Replace the heavy DFlash/Eagle base init with a stub that only sets
        the attributes DSpark's ``__init__`` subsequently reads."""

        def _stub(self, vllm_config, device, runner=None):
            self.num_speculative_tokens = num_speculative_tokens
            self.max_batch_size = max_batch_size
            self.max_num_tokens = max_num_tokens
            self.dtype = dtype
            self.device = device
            self.draft_model_config = vllm_config.speculative_config.draft_model_config
            # present so the ``del`` in DSpark.__init__ succeeds
            self.hidden_size = 0
            self.hidden_states = None
            self._dflash_hidden_states = None
            self.use_cuda_graph = True

        monkeypatch.setattr(AscendDflashProposer, "__init__", _stub)

    def test_greedy_allocates_dspark_buffers(self, monkeypatch):
        device = torch.device("cpu")
        num_spec, max_batch, max_num_tokens, hidden = 5, 16, 256, 8
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=num_spec,
            max_batch_size=max_batch,
            max_num_tokens=max_num_tokens,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=num_spec,
            max_batch_size=max_batch,
            max_num_tokens=max_num_tokens,
            draft_sample_method="greedy",
            hidden_size=hidden,
        )
        proposer = AscendDSparkProposer(vllm_config, device)

        blk = 1 + num_spec
        max_query_tokens = max_batch * (1 + num_spec)
        # DSpark-specific draft / seed buffers.
        assert proposer._dspark_draft_buffer.shape == (max_batch, blk)
        assert proposer._dspark_draft_buffer.dtype == torch.int64
        assert proposer._dspark_seed_buffer.shape == (max_batch,)
        assert proposer._dspark_seed_buffer.dtype == torch.int64
        # hidden_size / hidden states come from the draft model config.
        assert proposer.hidden_size == hidden
        assert proposer.hidden_states.shape == (max_num_tokens, hidden)
        assert proposer._dflash_hidden_states.shape == (max_num_tokens, hidden)
        # Static DSpark preserves the base proposer's graph-mode decision.
        assert proposer.use_cuda_graph is True
        # anchor-first: N query tokens per request, no bonus token (unlike
        # DFlash's 1+N).
        assert proposer.max_query_tokens == max_query_tokens
        assert proposer.positions.shape == (max_query_tokens,)
        assert proposer.positions.dtype == torch.int32
        assert proposer._slot_mapping_buffer.shape == (max_query_tokens,)
        # per-group bookkeeping dicts start empty / None.
        assert proposer._per_group_block_tables == {}
        assert proposer._per_group_slot_mappings == {}
        assert proposer._context_slot_mapping_buffers is None

    def test_dynamic_verify_length_disables_graph(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_ascend_config",
            lambda: SimpleNamespace(
                dynamic_spec_config=SimpleNamespace(method="dspark", method_params={})
            ),
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.DynamicSpecScheduler",
            MagicMock(),
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="greedy",
        )

        proposer = AscendDSparkProposer(vllm_config, device)

        assert proposer.use_cuda_graph is False


class TestDSparkGraphDummyRun(_DSparkProposerTestBase):
    def test_query_runnable_does_not_inject_hidden_states(self):
        source = inspect.getsource(AscendSpecDecodeBaseProposer._run_merged_draft)
        assert 'elif self.method != "dspark":' in source

    def test_builds_group_metadata_without_writing_context_kv(self, monkeypatch):
        proposer = self._make_proposer(
            max_num_tokens=64,
            num_reqs=2,
            block_size=3,
            draft_attn_causal=False,
        )
        metadata = SimpleNamespace(
            attn_mask=object(),
            attn_state=None,
            actual_seq_lengths_q=[3, 6],
            block_tables=torch.zeros((2, 4), dtype=torch.int32),
            slot_mapping=torch.zeros(6, dtype=torch.int32),
            causal=False,
        )
        builder = MagicMock()
        builder.build_for_graph_capture.return_value = metadata
        proposer.draft_attn_groups[0].get_metadata_builder = lambda: builder
        proposer.runner = SimpleNamespace(
            _sync_metadata_across_dp=lambda n, **kwargs: (n, None, None),
            optimistic_seq_lens_cpu=torch.zeros(2, dtype=torch.int32),
            seq_lens=torch.zeros(2, dtype=torch.int32),
        )
        proposer.vllm_config = SimpleNamespace()
        proposer.use_cuda_graph = True
        proposer.token_indices_to_sample = torch.zeros(6, dtype=torch.int32)
        proposer._get_positions = lambda n: proposer.positions[:n]
        proposer._runnable = MagicMock()
        proposer.model.precompute_and_store_context_kv = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context",
            lambda *args, **kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_forward_context",
            lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer._EXTRA_CTX",
            SimpleNamespace(capturing=True),
        )

        proposer.dummy_run(
            num_tokens=6,
            num_reqs=2,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
        )

        builder.build_for_graph_capture.assert_called_once()
        proposer.model.precompute_and_store_context_kv.assert_not_called()
        call_metadata = proposer._runnable.call_args.kwargs[
            "multi_steps_attn_metadata"
        ]
        assert call_metadata == [{"L0": metadata}]
        assert metadata.attn_mask is None
        assert metadata.attn_state == AscendAttentionState.ChunkedPrefill

    def test_pads_capture_metadata_to_target_graph_bucket(self, monkeypatch):
        proposer = self._make_proposer(
            max_num_tokens=64,
            num_reqs=2,
            block_size=3,
            draft_attn_causal=False,
        )
        metadata = SimpleNamespace(
            attn_mask=None,
            attn_state=None,
            actual_seq_lengths_q=[3, 6, 8],
            block_tables=torch.zeros((3, 4), dtype=torch.int32),
            slot_mapping=torch.zeros(8, dtype=torch.int32),
            causal=False,
        )
        builder = MagicMock()
        builder.build_for_graph_capture.return_value = metadata
        proposer.draft_attn_groups[0].get_metadata_builder = lambda: builder
        proposer.runner = SimpleNamespace(
            _sync_metadata_across_dp=lambda n, **kwargs: (n, None, None),
            optimistic_seq_lens_cpu=torch.zeros(2, dtype=torch.int32),
            seq_lens=torch.zeros(2, dtype=torch.int32),
        )
        proposer.vllm_config = SimpleNamespace()
        proposer.use_cuda_graph = True
        proposer.token_indices_to_sample = torch.zeros(6, dtype=torch.int32)
        proposer._get_positions = lambda n: proposer.positions[:n]
        proposer._runnable = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context",
            lambda *args, **kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_forward_context",
            lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer._EXTRA_CTX",
            SimpleNamespace(capturing=True),
        )

        proposer.dummy_run(
            num_tokens=8,
            num_reqs=2,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=SimpleNamespace(num_tokens=8),
        )

        common_metadata = builder.build_for_graph_capture.call_args.args[0]
        assert common_metadata.num_actual_tokens == 6
        assert common_metadata.num_input_tokens == 8
        assert common_metadata.num_reqs == 3
        assert common_metadata.query_start_loc_cpu.tolist() == [0, 3, 6, 8]
        assert proposer._runnable.call_args.kwargs["num_input_tokens"] == 8


class TestDSparkGraphRuntimePadding(_DSparkProposerTestBase):
    @staticmethod
    def _query_start_loc(values: list[int], capacity: int = 16):
        array = np.zeros(capacity, dtype=np.int32)
        array[: len(values)] = values
        return SimpleNamespace(np=array, copy_to_gpu=MagicMock())

    def test_single_request_pads_eight_query_tokens_to_nine_token_graph(self):
        proposer = self._make_proposer(
            max_num_tokens=64,
            num_reqs=4,
            block_size=8,
        )
        query_start_loc = self._query_start_loc([0, 8])

        metadata_reqs = proposer.pad_query_start_loc_for_graph(
            query_start_loc,
            num_input_tokens=9,
            real_num_reqs=1,
            graph_num_reqs=1,
        )

        assert metadata_reqs == 2
        assert query_start_loc.np[:3].tolist() == [0, 8, 9]
        query_start_loc.copy_to_gpu.assert_called_once_with()

    def test_three_requests_match_four_request_graph_capture_layout(self):
        proposer = self._make_proposer(
            max_num_tokens=64,
            num_reqs=4,
            block_size=8,
        )
        query_start_loc = self._query_start_loc([0, 8, 16, 24])

        metadata_reqs = proposer.pad_query_start_loc_for_graph(
            query_start_loc,
            num_input_tokens=36,
            real_num_reqs=3,
            graph_num_reqs=4,
        )

        assert metadata_reqs == 5
        assert query_start_loc.np[:6].tolist() == [0, 8, 16, 24, 32, 36]
        query_start_loc.copy_to_gpu.assert_called_once_with()

    @pytest.mark.parametrize("num_speculative_tokens", [1, 3, 5, 7, 8, 15])
    @pytest.mark.parametrize(
        ("real_num_reqs", "graph_num_reqs"),
        [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8)],
    )
    def test_padding_contract_is_parameterized_by_draft_width_and_batch(
        self,
        num_speculative_tokens,
        real_num_reqs,
        graph_num_reqs,
    ):
        """The N-query draft layout must match an (N+1)-wide target bucket.

        This deliberately exercises widths other than the GLM-5.2 checkpoint's
        current N=8, including the Ascend decode-round upper boundary N=15.
        """
        graph_num_tokens = graph_num_reqs * (num_speculative_tokens + 1)
        proposer = self._make_proposer(
            max_num_tokens=graph_num_tokens + 16,
            num_reqs=graph_num_reqs,
            block_size=num_speculative_tokens,
        )
        real_boundaries = [
            req_idx * num_speculative_tokens
            for req_idx in range(real_num_reqs + 1)
        ]
        query_start_loc = self._query_start_loc(
            real_boundaries,
            capacity=graph_num_reqs + 3,
        )

        metadata_reqs = proposer.pad_query_start_loc_for_graph(
            query_start_loc,
            num_input_tokens=graph_num_tokens,
            real_num_reqs=real_num_reqs,
            graph_num_reqs=graph_num_reqs,
        )

        expected_boundaries = [
            req_idx * num_speculative_tokens
            for req_idx in range(graph_num_reqs + 1)
        ]
        expected_boundaries.append(graph_num_tokens)
        assert metadata_reqs == graph_num_reqs + 1
        assert query_start_loc.np[: metadata_reqs + 1].tolist() == (
            expected_boundaries
        )
        query_start_loc.copy_to_gpu.assert_called_once_with()

    @pytest.mark.parametrize("num_speculative_tokens", [3, 5, 8, 15])
    def test_non_anchor_layout_uses_exact_target_graph_width(
        self,
        num_speculative_tokens,
    ):
        """sample_from_anchor=False already has N+1 real query rows."""
        graph_num_reqs = 4
        real_num_reqs = 3
        query_width = num_speculative_tokens + 1
        graph_num_tokens = graph_num_reqs * query_width
        proposer = self._make_proposer(
            max_num_tokens=graph_num_tokens + 16,
            num_reqs=graph_num_reqs,
            block_size=num_speculative_tokens,
            hf_config=SimpleNamespace(sample_from_anchor=False),
        )
        query_start_loc = self._query_start_loc(
            [req_idx * query_width for req_idx in range(real_num_reqs + 1)],
            capacity=graph_num_reqs + 2,
        )

        metadata_reqs = proposer.pad_query_start_loc_for_graph(
            query_start_loc,
            num_input_tokens=graph_num_tokens,
            real_num_reqs=real_num_reqs,
            graph_num_reqs=graph_num_reqs,
        )

        assert metadata_reqs == graph_num_reqs
        assert query_start_loc.np[: metadata_reqs + 1].tolist() == [
            req_idx * query_width for req_idx in range(graph_num_reqs + 1)
        ]
        query_start_loc.copy_to_gpu.assert_called_once_with()


class TestSetInputsFirstPassOutputs(_DSparkProposerTestBase):
    """``set_inputs_first_pass`` returns the anchor-first query budget and
    rewrites the common attention metadata into the DSpark cross-attention
    shape (N query tokens per request, non-causal, chunked-prefill state)."""

    @pytest.fixture(autouse=True)
    def _mock_kernel(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel",
            MagicMock(),
        )

    def test_return_value_and_token_indices(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        num_query_total, token_indices, _cad, extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )[:4]
        assert num_query_total == num_reqs * block_size
        assert token_indices.shape == (num_reqs * block_size,)
        assert token_indices.dtype == torch.int32
        # 4th return slot is unused (no per-group attn metadata tuple here).
        assert extra is None

    def test_seed_buffer_copied_from_next_tokens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        self._invoke_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)
        expected = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        assert torch.equal(proposer._dspark_seed_buffer[:num_reqs], expected)
        assert torch.all(proposer._dspark_seed_buffer[num_reqs:] == 0)

    def test_context_hidden_states_copied(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        self._invoke_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size, context=num_reqs)
        assert proposer._dflash_num_context == num_reqs
        expected = torch.arange(num_reqs * 8, dtype=torch.float32).reshape(num_reqs, 8)
        assert torch.equal(proposer._dflash_hidden_states[:num_reqs], expected)

    def test_query_slot_kernel_uses_logical_block_size(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel",
            kernel,
        )
        num_reqs, num_speculative_tokens, max_num_tokens = 1, 7, 32
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=num_speculative_tokens,
        )
        proposer.draft_attn_groups[0].kv_cache_spec.block_size = 384
        proposer._per_group_kernel_block_sizes[0] = 128

        self._invoke_set_inputs_first_pass(
            proposer,
            num_reqs=num_reqs,
            block_size=num_speculative_tokens,
            seq_len=720,
        )

        kwargs = kernel[1,].call_args.kwargs
        assert proposer.draft_attn_groups[0].kv_cache_spec.block_size == 384
        assert kwargs["block_size"] == 128

    def test_cad_rewritten_to_cross_attention_shape(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        num_query_total, _, cad, _ = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, with_optional_attrs=True
        )[:4]
        # token budgets reflect anchor-first (N per request, no bonus).
        assert cad.num_actual_tokens == num_query_total
        assert cad.num_input_tokens == num_query_total
        assert cad.max_query_len == block_size
        assert cad.max_seq_len == 128 + block_size
        # attention is non-causal cross-attention over the draft query block.
        assert cad.causal is False
        assert cad.attn_mask is None
        assert cad.attn_state == AscendAttentionState.ChunkedPrefill
        # positions is the full buffer (DSA slices it), not a pre-slice.
        assert cad.positions is proposer.positions
        # slot mapping is a slice of the primary group's query buffer (shares
        # storage from offset 0); a fresh slice is not identity-equal, so check
        # the underlying storage and length instead.
        assert cad.slot_mapping.data_ptr() == proposer._per_group_query_slot_mapping_buffers[0].data_ptr()
        assert cad.slot_mapping.shape[0] == num_query_total
        # optional attrs the proposer rewrites when present.
        assert cad.actual_seq_lengths_q == [block_size] * num_reqs
        assert cad.decode_token_per_req == block_size

    def test_cad_uses_model_reported_causality(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            draft_attn_causal=True,
        )
        _, _, cad, _ = self._invoke_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)[:4]

        assert cad.causal is True

    def test_cad_query_start_loc_and_seq_lens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)[
            :4
        ]
        expected_qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * block_size
        assert torch.equal(cad.query_start_loc, expected_qsl)
        assert torch.equal(cad.query_start_loc_cpu, expected_qsl)
        # seq_lens grow by block_size when no tokens were rejected.
        expected = torch.full((num_reqs,), 128 + block_size, dtype=torch.int32)
        assert torch.equal(cad.seq_lens, expected)
        assert torch.equal(cad._seq_lens_cpu, expected)
        assert torch.equal(cad.seq_lens_cpu, expected)


class TestSetInputsFirstPassRejectedTokens(_DSparkProposerTestBase):
    """The ``has_num_rejected`` branch must shrink ``seq_lens`` by the rejected
    token count before adding the draft block size, and flag the kernel."""

    def test_seq_lens_subtracts_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer,
            num_reqs=num_reqs,
            block_size=block_size,
            host_seq_len=126,
            async_metadata=True,
            num_rejected=rejected,
        )[:4]
        # effective = seq_lens(128) - rejected(2) = 126; then + block_size(5) = 131.
        assert torch.equal(cad.seq_lens, torch.full((num_reqs,), 128 - 2 + block_size, dtype=torch.int32))
        expected_host = torch.full((num_reqs,), 126 + block_size, dtype=torch.int32)
        assert torch.equal(cad._seq_lens_cpu, expected_host)
        assert cad.seq_lens_cpu is None

    def test_kernel_called_with_has_num_rejected(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel",
            kernel,
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        self._invoke_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected)
        # The proposer calls the kernel as ``kernel[1,](...)`` (Triton-style
        # grid indexing), so the call lands on the indexed sub-mock.
        sub = kernel[1,]
        assert sub.called
        kwargs = sub.call_args.kwargs
        assert kwargs["HAS_NUM_REJECTED"] is True
        assert kwargs["num_rejected_tokens_ptr"] is rejected
        assert kwargs["SAMPLE_FROM_ANCHOR"] is True


class TestInitializeAttnBackend(_DSparkProposerTestBase):
    """Initialization preserves each group's logical kernel block size."""

    @staticmethod
    def _make_proposer_for_init():
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.vllm_config = SimpleNamespace()
        proposer.device = torch.device("cpu")
        proposer.runner = SimpleNamespace(device_metadata_executor=None)
        proposer.dcp_size = 1
        return proposer

    @pytest.mark.parametrize(
        ("dcp_size", "pcp_enabled", "has_executor", "expected_tokens"),
        [
            (1, False, True, 8),
            (2, False, True, None),
            (1, True, True, None),
            (1, False, False, None),
        ],
    )
    def test_initializes_device_metadata_only_for_eligible_dsa_draft(
        self,
        monkeypatch,
        dcp_size,
        pcp_enabled,
        has_executor,
        expected_tokens,
    ):
        class DraftBuilder:
            def __init__(self):
                self.max_num_tokens = None

            def enable_dspark_device_metadata(self, max_num_tokens):
                self.max_num_tokens = max_num_tokens

        backend = MagicMock()
        backend.full_cls_name.return_value = "fake.backend"
        layer = MagicMock()
        layer.get_attn_backend.return_value = backend
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *args, **kwargs: {"L0": layer},
        )
        proposer = self._make_proposer_for_init()
        proposer.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: {"L0"})
        proposer.max_query_tokens = 8
        proposer.max_num_tokens = 16
        proposer.dcp_size = dcp_size
        proposer.runner.device_metadata_executor = object() if has_executor else None
        kv_cache_spec = MagicMock(block_size=128)
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["L0"], kv_cache_spec=kv_cache_spec)]
        )
        builder = DraftBuilder()

        with (
            patch.object(AttentionGroup, "create_metadata_builders"),
            patch.object(AttentionGroup, "get_metadata_builder", return_value=builder),
            patch("vllm_ascend.spec_decode.dspark_proposer.AscendDSAMetadataBuilder", DraftBuilder),
            patch("vllm_ascend.spec_decode.dspark_proposer.enable_pcp", return_value=pcp_enabled),
        ):
            proposer.initialize_attn_backend(kv_cache_config)

        assert builder.max_num_tokens == expected_tokens

    def test_initialization_tracks_logical_block_size_per_gid(self, monkeypatch):
        manager_specs = [MagicMock(), MagicMock()]
        for spec in manager_specs:
            spec.block_size = 384

        backend = MagicMock()
        backend.full_cls_name.return_value = "fake.backend"
        layers = {}
        for gid in range(2):
            layer = MagicMock()
            layer.get_attn_backend.return_value = backend
            layers[f"L{gid}"] = layer
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: layers,
        )

        proposer = self._make_proposer_for_init()
        proposer.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: {"L0", "L1"})
        proposer.max_query_tokens = 8
        proposer.max_num_tokens = 16
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[f"L{gid}"],
                    kv_cache_spec=manager_specs[gid],
                )
                for gid in range(2)
            ],
        )

        with patch.object(AttentionGroup, "create_metadata_builders") as create_builders:
            proposer.initialize_attn_backend(
                kv_cache_config,
                kernel_block_sizes=[128, 64],
            )

        assert [spec.block_size for spec in manager_specs] == [384, 384]
        assert proposer._per_group_kernel_block_sizes == {0: 128, 1: 64}
        assert [group.kv_cache_group_id for group in proposer.draft_attn_groups] == [0, 1]
        assert proposer.kernel_block_size == 128
        assert [call.kwargs["kernel_block_size"] for call in create_builders.call_args_list] == [128, 64]

    @pytest.mark.parametrize("draft_uses_mla", [False, True], ids=["gqa", "mla"])
    def test_mixed_target_and_dspark_group_creates_one_draft_attention_group(self, monkeypatch, draft_uses_mla: bool):
        page_size = 488448
        target_layer = "language_model.model.layers.3.self_attn.attn"
        draft_layers = [f"model.layers.{layer_idx}.self_attn.attn" for layer_idx in range(93, 98)]
        target_spec = MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            page_size_padded=page_size,
        )
        if draft_uses_mla:
            draft_spec = MLAAttentionSpec(
                block_size=384,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
                page_size_padded=page_size,
                non_causal_multi_token_decode=True,
            )
        else:
            draft_spec = FullAttentionSpec(
                block_size=384,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.bfloat16,
                page_size_padded=page_size,
            )
        mixed_spec = UniformTypeKVCacheSpecs.from_specs(
            {
                target_layer: target_spec,
                **{layer_name: draft_spec for layer_name in draft_layers},
            }
        )
        assert mixed_spec is not None

        backend = MagicMock()
        backend.full_cls_name.return_value = "fake.gqa.backend"
        layers = {}
        for layer_name in draft_layers:
            layer = MagicMock()
            layer.get_attn_backend.return_value = backend
            layers[layer_name] = layer
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *args, **kwargs: layers,
        )

        proposer = self._make_proposer_for_init()
        proposer.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: set(draft_layers))
        proposer.max_query_tokens = 16
        proposer.max_num_tokens = 32
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[target_layer, *draft_layers],
                    kv_cache_spec=mixed_spec,
                )
            ]
        )

        with patch.object(AttentionGroup, "create_metadata_builders"):
            proposer.initialize_attn_backend(
                kv_cache_config,
                kernel_block_sizes=[128],
            )

        assert len(proposer.draft_attn_groups) == 1
        assert set(proposer.draft_attn_groups[0].layer_names) == set(draft_layers)
        assert proposer.draft_attn_groups[0].kv_cache_group_id == 0
        assert proposer._layer_group_idx == [0] * 5
