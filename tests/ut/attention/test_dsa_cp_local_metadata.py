#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for the DSA CP local token metadata build path.

Covers the lazy AscendC op resolution helper and the three-level fallback
dispatch (AscendC aclnn op -> triton -> torch) in
``AscendDSACPMetadataBuilder._build_local_token_metadata``.

The AscendC kernel itself is not exercised here (UTs run on CPU with mocked
torch_npu); its numerical equivalence is verified by e2e tests on NPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

# Golden case from the _build_local_token_metadata docstring:
# TP=3, num_input_tokens=45, 9 requests with seq lens [1..9], rank 1
# (local_start=15, local_end=30).
NUM_REQS = 9
NUM_INPUT_TOKENS = 45
QSL_RANK1 = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
SEQ_LENS_RANK1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
EXPECTED_LOCAL_START = 15
EXPECTED_LOCAL_END = 30
EXPECTED_TOKENS_PER_RANK = 15
EXPECTED_LOCAL_QSL = [0, 0, 0, 0, 0, 0, 6, 13, 15]
EXPECTED_LOCAL_SEQ_LENS = [0, 0, 0, 0, 0, 6, 7, 2, 0]


def _make_builder() -> AscendDSACPMetadataBuilder:
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder._zero_i32 = torch.tensor([0], dtype=torch.int32)
    return builder


def _tp_group(tp_size: int = 3, rank: int = 1):
    return SimpleNamespace(world_size=tp_size, rank_in_group=rank)


def _fake_npu_tensor():
    """Minimal stand-in with .device.type != "cpu".

    Inside the AscendC/triton dispatch branches the query_start_loc tensor is
    only handed to the (mocked) device op, never used in torch computations,
    so a proxy is sufficient to steer the dispatch on CPU.
    """
    return SimpleNamespace(device=SimpleNamespace(type="npu"))


class TestLazyOpResolution:
    def test_resolves_and_caches(self):
        fake_op = MagicMock()
        mock_ns = MagicMock(npu_dsa_local_metadata=fake_op)
        with (
            patch.object(dsa_cp, "_dsa_local_metadata_op_resolved", False),
            patch.object(dsa_cp, "_dsa_local_metadata_op", None),
            patch("torch.ops._C_ascend", mock_ns, create=True),
        ):
            assert dsa_cp._get_dsa_local_metadata_op() is fake_op
            # Second call must return the cached op without re-resolving.
            assert dsa_cp._get_dsa_local_metadata_op() is fake_op

    def test_marks_side_effect(self):
        fake_op = MagicMock()
        mock_ns = MagicMock(npu_dsa_local_metadata=fake_op)
        with (
            patch.object(dsa_cp, "_dsa_local_metadata_op_resolved", False),
            patch.object(dsa_cp, "_dsa_local_metadata_op", None),
            patch("torch.ops._C_ascend", mock_ns, create=True),
            patch("torch.fx.node.has_side_effect") as has_side_effect,
        ):
            dsa_cp._get_dsa_local_metadata_op()
        has_side_effect.assert_called_once_with(fake_op)

    def test_returns_none_when_op_missing(self):
        # spec_set=[] makes every attribute access raise AttributeError,
        # simulating a build without the custom op compiled in.
        with (
            patch.object(dsa_cp, "_dsa_local_metadata_op_resolved", False),
            patch.object(dsa_cp, "_dsa_local_metadata_op", None),
            patch("torch.ops._C_ascend", MagicMock(spec_set=[]), create=True),
        ):
            assert dsa_cp._get_dsa_local_metadata_op() is None


class TestBuildLocalTokenMetadataDispatch:
    def test_uses_ascendc_op_when_available(self):
        builder = _make_builder()
        op = MagicMock()
        qsl_proxy = _fake_npu_tensor()
        seq_lens = torch.tensor(SEQ_LENS_RANK1, dtype=torch.int32)
        local_qsl = torch.zeros(NUM_REQS + 1, dtype=torch.int32)
        local_sl = torch.zeros(NUM_REQS, dtype=torch.int32)

        with (
            patch.object(dsa_cp, "get_tp_group", return_value=_tp_group()),
            patch.object(dsa_cp, "_get_dsa_local_metadata_op", return_value=op) as get_op,
        ):
            builder._build_local_token_metadata(
                num_reqs=NUM_REQS,
                num_input_tokens=NUM_INPUT_TOKENS,
                query_start_loc=qsl_proxy,
                seq_lens=seq_lens,
                local_query_start_loc=local_qsl,
                local_seq_lens=local_sl,
            )

        get_op.assert_called_once()
        op.assert_called_once()
        args = op.call_args.args
        assert args[0] is qsl_proxy
        assert args[1] is seq_lens
        assert args[2] is local_qsl
        assert args[3] is local_sl
        # start_pos_out is None -> zero sentinel passed instead.
        assert args[4] is builder._zero_i32
        assert args[5] == EXPECTED_LOCAL_START
        assert args[6] == EXPECTED_LOCAL_END
        assert args[7] == NUM_REQS
        # compute_start_pos must be False when start_pos_out is None.
        assert args[8] is False

    def test_ascendc_op_receives_start_pos_buffer(self):
        builder = _make_builder()
        op = MagicMock()
        start_pos_out = torch.zeros(NUM_REQS, dtype=torch.int32)

        with (
            patch.object(dsa_cp, "get_tp_group", return_value=_tp_group()),
            patch.object(dsa_cp, "_get_dsa_local_metadata_op", return_value=op),
        ):
            builder._build_local_token_metadata(
                num_reqs=NUM_REQS,
                num_input_tokens=NUM_INPUT_TOKENS,
                query_start_loc=_fake_npu_tensor(),
                seq_lens=torch.tensor(SEQ_LENS_RANK1, dtype=torch.int32),
                local_query_start_loc=torch.zeros(NUM_REQS + 1, dtype=torch.int32),
                local_seq_lens=torch.zeros(NUM_REQS, dtype=torch.int32),
                start_pos_out=start_pos_out,
            )

        args = op.call_args.args
        assert args[4] is start_pos_out
        assert args[8] is True

    def test_falls_back_to_triton_when_op_missing(self):
        builder = _make_builder()
        launch = MagicMock()
        triton_kernel = MagicMock()
        triton_kernel.__getitem__.return_value = launch

        with (
            patch.object(dsa_cp, "get_tp_group", return_value=_tp_group()),
            patch.object(dsa_cp, "_get_dsa_local_metadata_op", return_value=None),
            patch.object(dsa_cp, "HAS_TRITON", True),
            patch.object(dsa_cp, "build_local_metadata_triton", triton_kernel),
            patch.object(dsa_cp, "triton", MagicMock(next_power_of_2=MagicMock(return_value=16))),
        ):
            builder._build_local_token_metadata(
                num_reqs=NUM_REQS,
                num_input_tokens=NUM_INPUT_TOKENS,
                query_start_loc=_fake_npu_tensor(),
                seq_lens=torch.tensor(SEQ_LENS_RANK1, dtype=torch.int32),
                local_query_start_loc=torch.zeros(NUM_REQS + 1, dtype=torch.int32),
                local_seq_lens=torch.zeros(NUM_REQS, dtype=torch.int32),
            )

        launch.assert_called_once()
        args, kwargs = launch.call_args
        assert args[7] == NUM_REQS
        assert kwargs["BLOCK_NUM_REQS"] == 16
        assert kwargs["COMPUTE_START_POS"] is False


class TestBuildLocalTokenMetadataTorchFallback:
    """Numerical validation of the torch fallback against the docstring golden
    case; the AscendC kernel must produce identical results (verified on NPU
    by e2e)."""

    def test_golden_case_rank1(self):
        builder = _make_builder()
        max_num_seqs = 12
        local_qsl = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        local_sl = torch.zeros(max_num_seqs, dtype=torch.int32)

        with patch.object(dsa_cp, "get_tp_group", return_value=_tp_group()):
            local_start, local_end, tokens_per_rank, num_tokens_pad, out_qsl, out_sl = (
                builder._build_local_token_metadata(
                    num_reqs=NUM_REQS,
                    num_input_tokens=NUM_INPUT_TOKENS,
                    query_start_loc=torch.tensor(QSL_RANK1, dtype=torch.int32),
                    seq_lens=torch.tensor(SEQ_LENS_RANK1, dtype=torch.int32),
                    local_query_start_loc=local_qsl,
                    local_seq_lens=local_sl,
                )
            )

        assert local_start == EXPECTED_LOCAL_START
        assert local_end == EXPECTED_LOCAL_END
        assert tokens_per_rank == EXPECTED_TOKENS_PER_RANK
        assert num_tokens_pad == NUM_INPUT_TOKENS
        assert out_qsl.tolist() == EXPECTED_LOCAL_QSL
        assert out_sl.tolist() == EXPECTED_LOCAL_SEQ_LENS
        # The tail beyond num_reqs must stay zero: the AscendC kernel only
        # writes [0, num_reqs] and relies on the pre-zeroed buffers.
        assert local_qsl[NUM_REQS + 1 :].tolist() == [0] * (max_num_seqs - NUM_REQS)
        assert local_sl[NUM_REQS:].tolist() == [0] * (max_num_seqs - NUM_REQS)

    def test_start_pos_and_boundary_crossing(self):
        # 1 rank (tp=1): local_start=0, local_end=num_input_tokens.
        # Request 1 crosses nothing; start_pos = seq_len - q_len.
        builder = _make_builder()
        num_reqs = 2
        local_qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
        local_sl = torch.zeros(num_reqs, dtype=torch.int32)
        start_pos_out = torch.zeros(num_reqs, dtype=torch.int32)

        with patch.object(dsa_cp, "get_tp_group", return_value=_tp_group(tp_size=1, rank=0)):
            builder._build_local_token_metadata(
                num_reqs=num_reqs,
                num_input_tokens=5,
                query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
                seq_lens=torch.tensor([10, 7], dtype=torch.int32),
                local_query_start_loc=local_qsl,
                local_seq_lens=local_sl,
                start_pos_out=start_pos_out,
            )

        assert local_qsl.tolist() == [0, 2, 5]
        assert local_sl.tolist() == [10, 7]
        assert start_pos_out.tolist() == [8, 4]

    def test_requests_beyond_local_end_are_masked(self):
        # tp=4, rank 3: local slice [45, 60); only the tail of the batch hits.
        builder = _make_builder()
        num_reqs = 4
        local_qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
        local_sl = torch.zeros(num_reqs, dtype=torch.int32)

        with patch.object(dsa_cp, "get_tp_group", return_value=_tp_group(tp_size=4, rank=3)):
            _, _, _, _, out_qsl, out_sl = builder._build_local_token_metadata(
                num_reqs=num_reqs,
                num_input_tokens=60,
                query_start_loc=torch.tensor([0, 10, 20, 40, 55], dtype=torch.int32),
                seq_lens=torch.tensor([10, 10, 20, 15], dtype=torch.int32),
                local_query_start_loc=local_qsl,
                local_seq_lens=local_sl,
            )

        # tp=4, rank 3: local slice [45, 60); requests 0-2 end before 45 and
        # are fully masked out, request 3 straddles the boundary
        # (qs=40 clamps to 45, qe=55 stays -> lql=10).
        assert out_qsl.tolist() == [0, 0, 0, 0, 10]
        # offset = qe - lqe = 55 - 55 = 0 -> local_seq_len = 15.
        assert out_sl.tolist() == [0, 0, 0, 15]
