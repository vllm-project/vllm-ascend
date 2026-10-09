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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.spec_decode.llm_base_proposer as lbp
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _norm(obj):
    if isinstance(obj, SimpleNamespace):
        return ("ns", tuple(sorted((k, _norm(v)) for k, v in vars(obj).items())))
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return ("obj", type(obj).__name__)


def _make_proposer(
    *,
    num_speculative_tokens=5,
    method="mtp",
    use_cuda_graph=False,
    dcp_size=1,
    parallel_drafting=False,
):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.method = method
    proposer.use_cuda_graph = use_cuda_graph
    proposer.dcp_size = dcp_size
    proposer.parallel_drafting = parallel_drafting
    proposer._mtp_metadata_keepalive = None
    proposer.attn_layer_names = ["model.layers.61.self_attn.attn"]
    proposer.draft_attn_groups = [SimpleNamespace(name="g0")]
    return proposer


def _make_window_events(proposer, pre_record=True):
    events = [torch.npu.ExternalEvent() for _ in range(proposer.num_speculative_tokens - 1)]
    if pre_record:
        main = torch.npu.current_stream()
        for event in events:
            event.record(main)
    return events


def _patch_env(monkeypatch, overlap_flag):
    monkeypatch.setattr(lbp, "get_ascend_config", lambda: SimpleNamespace(multistream_mtp_metadata_overlap=overlap_flag))
    calls = []
    arg_signatures = []

    def fake_update(self, draft_index, *args, **kwargs):
        calls.append(draft_index)
        arg_signatures.append(
            (
                draft_index,
                tuple(_norm(a) for a in args),
                tuple(sorted((k, _norm(v)) for k, v in kwargs.items())),
            )
        )
        return SimpleNamespace(chain=draft_index), SimpleNamespace(step=draft_index)

    monkeypatch.setattr(AscendSpecDecodeBaseProposer, "attn_update_stack_num_spec_norm", fake_update)
    # The bare test proposer has no vllm_config / backend wiring; report no
    # cache-only attention groups so the prebuild takes the regular path.
    monkeypatch.setattr(AscendSpecDecodeBaseProposer, "_is_cache_only_draft_attn_group", lambda self, group: False)
    return calls, arg_signatures


def _run_prebuild(proposer, window_events=None):
    multi_steps = [{"model.layers.61.self_attn.attn": SimpleNamespace(step=0)}]
    events = proposer._prebuild_draft_attn_metadata(
        multi_steps,
        common_attn_metadata=SimpleNamespace(base=0),
        batch_size=2,
        num_input_tokens=2,
        used_update_positions=SimpleNamespace(pos=0),
        aclgraph_runtime_mode=None,
        draft_cp_kwargs={},
        dcp_mtp_inputs=None,
        window_events=window_events,
    )
    return multi_steps, events


class TestMtpMetadataStreamSingleton:
    def test_returns_same_object(self):
        first = lbp.mtp_metadata_stream()
        assert lbp.mtp_metadata_stream() is first


class TestOverlapGateHelper:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, True),
            ({"method": "eagle"}, False),
            ({"use_cuda_graph": True}, False),
            ({"dcp_size": 2}, False),
        ],
    )
    def test_gate_matrix(self, monkeypatch, kwargs, expected):
        _patch_env(monkeypatch, overlap_flag=True)
        proposer = _make_proposer(**kwargs)
        assert proposer._mtp_metadata_overlap_enabled() == expected

    def test_gate_off_when_flag_false(self, monkeypatch):
        _patch_env(monkeypatch, overlap_flag=False)
        proposer = _make_proposer()
        assert proposer._mtp_metadata_overlap_enabled() is False


class TestPrebuildGating:
    @pytest.mark.parametrize(
        "kwargs,expect_overlap",
        [
            ({}, True),
            ({"method": "eagle"}, False),  # gating fails -> serial prebuild still runs
            ({"use_cuda_graph": True}, False),  # gating fails -> serial prebuild still runs
            ({"dcp_size": 2}, False),  # should_update_next_steps=False -> prebuild skipped
            ({"parallel_drafting": True}, False),  # should_update_next_steps=False -> skipped
        ],
    )
    def test_gating_matrix(self, monkeypatch, kwargs, expect_overlap):
        calls, _ = _patch_env(monkeypatch, overlap_flag=True)
        proposer = _make_proposer(**kwargs)
        windows = _make_window_events(proposer)
        multi_steps, events = _run_prebuild(proposer, window_events=windows)
        expect_prebuild = not (kwargs.get("parallel_drafting") or kwargs.get("dcp_size", 1) > 1)
        if expect_overlap:
            assert events is not None and len(events) == proposer.num_speculative_tokens - 1
        else:
            assert events is None
        if expect_prebuild:
            assert calls == [1, 2, 3, 4]
            assert len(multi_steps) == proposer.num_speculative_tokens
        else:
            assert calls == []
            assert len(multi_steps) == 1

    def test_flag_off_runs_serial_loop_ignoring_windows(self, monkeypatch):
        calls, _ = _patch_env(monkeypatch, overlap_flag=False)
        proposer = _make_proposer()
        multi_steps, events = _run_prebuild(proposer, window_events=_make_window_events(proposer))
        assert events is None
        assert proposer._mtp_metadata_keepalive is None
        assert calls == [1, 2, 3, 4]
        assert len(multi_steps) == proposer.num_speculative_tokens

    def test_overlap_requires_window_events(self, monkeypatch):
        _patch_env(monkeypatch, overlap_flag=True)
        proposer = _make_proposer()
        with pytest.raises(AssertionError):
            _run_prebuild(proposer, window_events=None)


class TestPrebuildOverlapEquivalence:
    def test_overlap_matches_serial_metadata_and_sets_keepalive(self, monkeypatch):
        calls, arg_signatures = _patch_env(monkeypatch, overlap_flag=True)
        # Serial reference: same fake update, switch only the config flag off.
        monkeypatch.setattr(lbp, "get_ascend_config", lambda: SimpleNamespace(multistream_mtp_metadata_overlap=False))
        serial_proposer = _make_proposer()
        serial_steps, serial_events = _run_prebuild(serial_proposer, window_events=None)
        assert serial_events is None
        assert serial_proposer._mtp_metadata_keepalive is None
        serial_signatures = list(arg_signatures)
        assert [sig[1][0] for sig in serial_signatures] == [
            ("ns", (("base", 0),)),
            ("ns", (("chain", 1),)),
            ("ns", (("chain", 2),)),
            ("ns", (("chain", 3),)),
        ]

        # Overlap run: flag back on, gated by pre-recorded lm_head windows.
        monkeypatch.setattr(lbp, "get_ascend_config", lambda: SimpleNamespace(multistream_mtp_metadata_overlap=True))
        overlap_proposer = _make_proposer()
        windows = _make_window_events(overlap_proposer)
        overlap_steps, events = _run_prebuild(overlap_proposer, window_events=windows)

        assert len(events) == overlap_proposer.num_speculative_tokens - 1
        # The overlap branch passed the side-stream prebuild exactly the same
        # per-call arguments (index, chain progression, scalars, kwargs) as
        # the serial branch.
        assert arg_signatures[len(serial_signatures) :] == serial_signatures
        serial_steps_seq = [entry["model.layers.61.self_attn.attn"].step for entry in serial_steps]
        overlap_steps_seq = [entry["model.layers.61.self_attn.attn"].step for entry in overlap_steps]
        assert overlap_steps_seq == serial_steps_seq == [0, 1, 2, 3, 4]
        assert overlap_proposer._mtp_metadata_keepalive is not None
        assert len(overlap_proposer._mtp_metadata_keepalive) == 3
        assert overlap_proposer._mtp_metadata_keepalive[0] is overlap_steps
