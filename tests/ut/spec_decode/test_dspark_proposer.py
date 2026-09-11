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
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.spec_decode.dspark_proposer as dspark_proposer_module
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
    ("dcp_size", "pcp_size", "expected_submit"),
    [(1, 1, True), (2, 1, False), (1, 2, False)],
)
def test_build_draft_metadata_submits_only_non_cp_device_tasks(
    dcp_size: int,
    pcp_size: int,
    expected_submit: bool,
):
    tasks = [DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None, group_id) for group_id in (7, 9)]

    class DraftMetadataProvider:
        def __init__(self, task):
            self.task = task

        def enable_device_metadata(self):
            pass

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
            parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        ),
        _per_group_block_table_buffers={group_id: torch.ones((1, 1), dtype=torch.int32) for group_id in range(2)},
        _per_group_query_slot_mapping_buffers={group_id: torch.zeros(1, dtype=torch.int32) for group_id in range(2)},
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
    if expected_submit:
        executor.submit.assert_called_once_with(tasks)
    else:
        executor.submit.assert_not_called()


@pytest.mark.parametrize(
    ("has_task", "failure"),
    [(True, None), (False, None), (True, "run"), (True, "context")],
)
def test_dspark_device_metadata_executor_forward_lifecycle(has_task: bool, failure: str | None):
    events = []
    executor = SimpleNamespace(submission_in_flight=False)

    def release():
        events.append("release")
        executor.submission_in_flight = False

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
    proposer.get_model = lambda: proposer.model
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
    proposer._prepare_parallel_draft_seq_lens_cpu = MagicMock()
    proposer.uses_mrope = False
    proposer.positions = torch.arange(2, dtype=torch.int32)
    proposer.parallel_drafting = True
    proposer.token_indices_to_sample = torch.zeros(2, dtype=torch.int32)
    proposer.enable_enpu = False
    proposer._update_full_graph_params_if_needed = MagicMock()
    proposer.set_inputs_first_pass = MagicMock()
    proposer.build_draft_attn_metadata = MagicMock()

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
        if failure == "run":
            raise RuntimeError("draft run failed")
        return torch.ones((1, 1), dtype=torch.int64)

    proposer._runnable = run_draft
    forward_context = SimpleNamespace(moe_layer_index=-1, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    context_executors = []

    @contextmanager
    def forward_context_manager(*args, **kwargs):
        context_executors.append(kwargs["device_metadata_executor"])
        events.append("context")
        if failure == "context":
            raise RuntimeError("draft context failed")
        yield

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.DSparkDeepseekV4ForCausalLM", object),
        patch("vllm_ascend.spec_decode.llm_base_proposer.set_ascend_forward_context", forward_context_manager),
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_forward_context", return_value=forward_context),
    ):
        call = lambda: AscendSpecDecodeBaseProposer._propose(
            proposer,
            target_token_ids=torch.ones(1, dtype=torch.int64),
            target_positions=torch.zeros(1, dtype=torch.int32),
            target_hidden_states=torch.ones((1, 4)),
            next_token_ids=torch.ones(1, dtype=torch.int64),
            token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
            common_attn_metadata=common_attn_metadata,
            target_model_batch_desc=SimpleNamespace(uniform=True),
            sampling_metadata=MagicMock(),
        )
        if failure is None:
            result = call()
            assert torch.equal(result, torch.ones((1, 1), dtype=torch.int64))
        else:
            with pytest.raises(RuntimeError, match=f"draft {failure} failed"):
                call()

    assert context_executors == [executor if has_task else None]
    expected_events = ["submit" if has_task else "build", "context"]
    if failure != "context":
        expected_events.append("run")
    if has_task and failure is None:
        expected_events.append("release")
    assert events == expected_events
    if failure is not None:
        assert executor.submission_in_flight


class _DSparkProposerTestBase:
    """Shared helpers for ``AscendDSparkProposer`` tests."""

    @staticmethod
    def _make_vllm_config(hf_config: SimpleNamespace) -> SimpleNamespace:
        """Build the minimal config consumed by the DSpark initializer."""
        draft_model_config = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: _HIDDEN_SIZE)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_sample_method="greedy",
                draft_model_config=draft_model_config,
                num_speculative_tokens=_NUM_SPECULATIVE_TOKENS,
            ),
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
    ):
        device = torch.device("cpu")
        vllm_config = cls._make_vllm_config(hf_config or SimpleNamespace())

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

        with patch.object(AscendDSparkProposer.__base__, "__init__", mock_parent_init):
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

    # fmt: off
    @staticmethod
    def _invoke_set_inputs_first_pass(
        proposer,
        *,
        num_reqs,
        block_size,
        seq_len=128,
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
        target_hidden_states = torch.arange(
            num_reqs * 8, dtype=torch.float32
        ).reshape(num_reqs, 8)
        query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        if context is not None:
            query_start_loc_cpu[num_reqs] = context
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=torch.full((num_reqs,), seq_len, dtype=torch.int32),
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


# fmt: on


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
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
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
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
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

    def test_must_precede_build(self):
        """build_draft_attn_metadata reads positions but does not zero it, so
        _pad_draft_buffers must run first."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        def capture_build():
            captured = {}

            def fake_build(common_attn_metadata, num_input_tokens, num_actual_tokens):
                captured["region"] = common_attn_metadata.positions[num_actual:num_input].clone()
                return None, common_attn_metadata

            return captured, fake_build

        ok = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        ok.positions[num_actual:num_input] = -999
        cap_ok, build_ok = capture_build()
        ok.build_draft_attn_metadata = build_ok
        ok._pad_draft_buffers(num_actual, num_input)
        ok.build_draft_attn_metadata(SimpleNamespace(positions=ok.positions), num_input, num_actual)
        assert torch.all(cap_ok["region"] == 0)

        bug = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        bug.positions[num_actual:num_input] = -999
        cap_bug, build_bug = capture_build()
        bug.build_draft_attn_metadata = build_bug
        bug.build_draft_attn_metadata(SimpleNamespace(positions=bug.positions), num_input, num_actual)
        bug._pad_draft_buffers(num_actual, num_input)
        assert torch.all(cap_bug["region"] == -999)

    def test_called_before_build_in_propose(self):
        """In ``_propose`` the ``_pad_draft_buffers`` call must precede
        ``build_draft_attn_metadata``."""
        src = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
        pad_idx = src.find("self._pad_draft_buffers(")
        build_idx = src.find("self.build_draft_attn_metadata(")
        # Only assert when both calls live directly in _propose; a refactor that
        # extracts them elsewhere leaves this guard inert rather than brittle.
        if pad_idx != -1 and build_idx != -1:
            assert pad_idx < build_idx, (
                "_pad_draft_buffers must be called before build_draft_attn_metadata "
                "in _propose, otherwise the attention backend reads un-zeroed "
                "positions in the DP-padding region."
            )


class TestDSparkInitialization(_DSparkProposerTestBase):
    """Tests for DSpark initialization configuration."""

    @pytest.mark.parametrize(
        ("additional_config", "draft_sample_method", "message"),
        [
            pytest.param(
                {"enable_reduce_sample": True},
                "greedy",
                "does not support enable_reduce_sample",
                id="reduce-sample-bypasses-markov-head",
            ),
            pytest.param(
                {},
                "probabilistic",
                "probabilistic draft sampling is not supported",
                id="probabilistic-draft-sampling",
            ),
        ],
    )
    def test_rejects_invalid_config_before_parent_initialization(
        self,
        additional_config: dict,
        draft_sample_method: str,
        message: str,
    ) -> None:
        vllm_config = self._make_vllm_config(SimpleNamespace())
        vllm_config.additional_config = additional_config
        vllm_config.speculative_config.draft_sample_method = draft_sample_method
        parent_init = MagicMock()

        with (
            pytest.raises(ValueError, match=message),
            patch.object(
                AscendDSparkProposer.__base__,
                "__init__",
                parent_init,
            ),
        ):
            AscendDSparkProposer(vllm_config, torch.device("cpu"))

        parent_init.assert_not_called()

    def test_rejects_k3_checkpoint_block_size_mismatch_before_parent_initialization(
        self,
        monkeypatch,
    ) -> None:
        class FakeK3DSparkConfig(SimpleNamespace):
            pass

        monkeypatch.setattr(
            dspark_proposer_module,
            "K3DSparkConfig",
            FakeK3DSparkConfig,
        )
        vllm_config = self._make_vllm_config(
            FakeK3DSparkConfig(
                block_size=5,
            )
        )
        vllm_config.speculative_config.num_speculative_tokens = 7
        parent_init = MagicMock()

        with (
            pytest.raises(ValueError, match="checkpoint block_size \\(5\\); got 7"),
            patch.object(
                AscendDSparkProposer.__base__,
                "__init__",
                parent_init,
            ),
        ):
            AscendDSparkProposer(vllm_config, torch.device("cpu"))

        parent_init.assert_not_called()

    @pytest.mark.parametrize(
        ("hf_config", "expected_sample_from_anchor", "expected_num_query_per_req"),
        [
            pytest.param(SimpleNamespace(), True, _NUM_SPECULATIVE_TOKENS),
            pytest.param(SimpleNamespace(sample_from_anchor=False), False, 1 + _NUM_SPECULATIVE_TOKENS),
        ],
    )
    def test_configures_anchor_sampling(
        self,
        hf_config: SimpleNamespace,
        expected_sample_from_anchor: bool,
        expected_num_query_per_req: int,
    ) -> None:
        """Verify the anchor-sampling setting selects the expected query layout."""
        proposer = self._make_proposer(
            max_num_tokens=_MAX_NUM_TOKENS,
            num_reqs=_MAX_BATCH_SIZE,
            block_size=_NUM_SPECULATIVE_TOKENS,
            hf_config=hf_config,
        )
        expected_max_query_tokens = _MAX_BATCH_SIZE * expected_num_query_per_req
        assert proposer.sample_from_anchor is expected_sample_from_anchor
        assert proposer.num_query_per_req == expected_num_query_per_req
        assert proposer.max_query_tokens == expected_max_query_tokens


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
    """``AscendDSparkProposer.__init__`` rejects probabilistic draft sampling
    (unsupported on the v1 model runner) and, for the greedy path, allocates
    the DSpark-specific draft/seed buffers and overrides the DFlash
    query-token / cudagraph defaults."""

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

        monkeypatch.setattr(AscendDflashProposer, "__init__", _stub)

    def test_probabilistic_rejected(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="probabilistic",
        )
        with pytest.raises(ValueError, match="probabilistic"):
            AscendDSparkProposer(vllm_config, device)

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
        max_query_tokens = max_batch * num_spec
        # DSpark-specific draft / seed buffers.
        assert proposer._dspark_draft_buffer.shape == (max_batch, blk)
        assert proposer._dspark_draft_buffer.dtype == torch.int64
        assert proposer._dspark_seed_buffer.shape == (max_batch,)
        assert proposer._dspark_seed_buffer.dtype == torch.int64
        # hidden_size / hidden states come from the draft model config.
        assert proposer.hidden_size == hidden
        assert proposer.hidden_states.shape == (max_num_tokens, hidden)
        assert proposer._dflash_hidden_states.shape == (max_num_tokens, hidden)
        # DSpark runs eager only (Ascend cudagraph unsupported on this path).
        assert proposer.use_cuda_graph is False
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


class TestSetInputsFirstPassOutputs(_DSparkProposerTestBase):
    """``set_inputs_first_pass`` returns the anchor-first query budget and
    rewrites the common attention metadata into the DSpark query-block shape."""

    @pytest.fixture(autouse=True)
    def _mock_kernel(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )

    def test_return_value_and_token_indices(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        num_query_total, token_indices, _cad, extra = (
            self._invoke_set_inputs_first_pass(
                proposer, num_reqs=num_reqs, block_size=block_size
            )[:4]
        )
        assert num_query_total == num_reqs * block_size
        assert token_indices.shape == (num_reqs * block_size,)
        assert token_indices.dtype == torch.int32
        # 4th return slot is unused (no per-group attn metadata tuple here).
        assert extra is None

    def test_seed_buffer_copied_from_next_tokens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )
        expected = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        assert torch.equal(proposer._dspark_seed_buffer[:num_reqs], expected)
        assert torch.all(proposer._dspark_seed_buffer[num_reqs:] == 0)

    def test_context_hidden_states_copied(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, context=num_reqs
        )
        assert proposer._dflash_num_context == num_reqs
        expected = torch.arange(num_reqs * 8, dtype=torch.float32).reshape(num_reqs, 8)
        assert torch.equal(proposer._dflash_hidden_states[:num_reqs], expected)

    def test_query_slot_kernel_uses_logical_block_size(self):
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

        kernel = dspark_proposer_module.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
        kwargs = kernel[1,].call_args.kwargs
        assert proposer.draft_attn_groups[0].kv_cache_spec.block_size == 384
        assert kwargs["block_size"] == 128

    def test_cad_rewritten_to_cross_attention_shape(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
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
        assert (
            cad.slot_mapping.data_ptr()
            == proposer._per_group_query_slot_mapping_buffers[0].data_ptr()
        )
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
        _, _, cad, _ = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )[:4]

        assert cad.causal is True

    def test_cad_query_start_loc_and_seq_lens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )[:4]
        expected_qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * block_size
        assert torch.equal(cad.query_start_loc, expected_qsl)
        assert torch.equal(cad.query_start_loc_cpu, expected_qsl)
        # seq_lens grow by block_size when no tokens were rejected.
        assert torch.equal(cad.seq_lens, torch.full((num_reqs,), 128 + block_size, dtype=torch.int32))


class TestSetInputsFirstPassRejectedTokens(_DSparkProposerTestBase):
    """The ``has_num_rejected`` branch must shrink ``seq_lens`` by the rejected
    token count before adding the draft block size, and flag the kernel."""

    def test_seq_lens_subtracts_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected
        )[:4]
        # effective = seq_lens(128) - rejected(2) = 126; then + block_size(5) = 131.
        assert torch.equal(
            cad.seq_lens, torch.full((num_reqs,), 128 - 2 + block_size, dtype=torch.int32)
        )

    def test_kernel_called_with_has_num_rejected(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            kernel,
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected
        )
        # The proposer calls the kernel as ``kernel[1,](...)`` (Triton-style
        # grid indexing), so the call lands on the indexed sub-mock.
        sub = kernel[1,]
        assert sub.called
        kwargs = sub.call_args.kwargs
        assert kwargs["HAS_NUM_REJECTED"] is True
        assert kwargs["num_rejected_tokens_ptr"] is rejected
        assert kwargs["SAMPLE_FROM_ANCHOR"] is True


class TestInitializeAttnBackendErrors(_DSparkProposerTestBase):
    """``initialize_attn_backend`` raises clearly when the draft model does not
    expose the DSpark layer-name API, or when no draft attention groups can be
    built from the kv-cache groups."""

    @staticmethod
    def _make_proposer_for_init():
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        )
        # The real proposer constructor always sets this field.  These
        # lightweight initializer tests bypass __init__, so preserve that
        # production invariant with a generic non-K3 draft config.
        proposer.draft_model_config = SimpleNamespace(hf_config=object())
        proposer.device = torch.device("cpu")
        proposer.runner = SimpleNamespace(device_metadata_executor=None)
        proposer.dcp_size = 1
        return proposer

    @pytest.mark.parametrize(
        ("dcp_size", "pcp_size", "has_executor", "expected_tokens"),
        [
            (1, 1, True, 8),
            (2, 1, True, None),
            (1, 2, True, None),
            (1, 1, False, None),
        ],
    )
    def test_initializes_device_metadata_only_for_eligible_dsa_draft(
        self,
        monkeypatch,
        dcp_size,
        pcp_size,
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
        proposer.vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
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
        ):
            proposer.initialize_attn_backend(kv_cache_config)

        assert builder.max_num_tokens == expected_tokens

    def test_model_without_draft_layer_names_raises(self, monkeypatch):
        # get_layers_from_vllm_config is called first; stub it so the model
        # check is what actually fails.
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: {},
        )
        proposer = self._make_proposer_for_init()
        # model lacks get_draft_kv_cache_layer_names entirely.
        proposer.model = SimpleNamespace()

        kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        with pytest.raises(RuntimeError, match="get_draft_kv_cache_layer_names"):
            proposer.initialize_attn_backend(kv_cache_config)

    def test_no_draft_attn_groups_raises(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: {},
        )
        proposer = self._make_proposer_for_init()
        # draft layer names exist, but no kv-cache group names overlap them.
        proposer.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: {"L0"})

        non_overlapping_group = SimpleNamespace(layer_names=["OTHER_LAYER"])
        kv_cache_config = SimpleNamespace(kv_cache_groups=[non_overlapping_group])
        with pytest.raises(RuntimeError, match="registered draft attention groups"):
            proposer.initialize_attn_backend(kv_cache_config)

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
        proposer.model = SimpleNamespace(
            get_draft_kv_cache_layer_names=lambda: {"L0", "L1"}
        )
        assert not isinstance(
            proposer.draft_model_config.hf_config,
            dspark_proposer_module.K3DSparkConfig,
        )
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
        assert [g.kv_cache_group_id for g in proposer.draft_attn_groups] == [0, 1]
        assert set(proposer._per_group_query_slot_mapping_buffers) == {0, 1}
        assert set(proposer._per_group_context_slot_mapping_buffers) == {0, 1}
        assert proposer.kernel_block_size == 128
        assert [
            call.kwargs["kernel_block_size"]
            for call in create_builders.call_args_list
        ] == [128, 64]

    def test_k3_initialization_enables_rope_on_mla_builders(self, monkeypatch):
        class FakeK3Config:
            pass

        class FakeMLABuilder:
            def __init__(self):
                self.use_mla_rope = False

        fake_builder = FakeMLABuilder()

        def create_metadata_builders(group, *args, **kwargs):
            del args, kwargs
            group.metadata_builders = [fake_builder]

        backend = MagicMock()
        backend.full_cls_name.return_value = "fake.backend"
        layer = MagicMock()
        layer.get_attn_backend.return_value = backend
        monkeypatch.setattr(
            dspark_proposer_module,
            "get_layers_from_vllm_config",
            lambda *args, **kwargs: {"L0": layer},
        )
        monkeypatch.setattr(dspark_proposer_module, "K3DSparkConfig", FakeK3Config)
        monkeypatch.setattr(
            dspark_proposer_module,
            "AscendMLAMetadataBuilder",
            FakeMLABuilder,
        )
        monkeypatch.setattr(
            AttentionGroup,
            "create_metadata_builders",
            create_metadata_builders,
        )

        proposer = self._make_proposer_for_init()
        proposer.draft_model_config = SimpleNamespace(hf_config=FakeK3Config())
        proposer.model = SimpleNamespace(
            get_draft_kv_cache_layer_names=lambda: {"L0"}
        )
        proposer.max_query_tokens = 8
        proposer.max_num_tokens = 16
        manager_spec = MagicMock()
        manager_spec.block_size = 128
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["L0"],
                    kv_cache_spec=manager_spec,
                )
            ]
        )

        proposer.initialize_attn_backend(kv_cache_config)

        assert fake_builder.use_mla_rope is dspark_proposer_module.K3_DSPARK_USE_MLA_ROPE

    def test_kernel_block_size_falls_back_to_cache_spec(self):
        proposer = self._make_proposer_for_init()

        assert (
            proposer._resolve_kernel_block_size(
                0,
                SimpleNamespace(block_size=384),
                None,
            )
            == 384
        )
# fmt: on
