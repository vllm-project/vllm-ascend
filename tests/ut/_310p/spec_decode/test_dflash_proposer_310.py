#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    _compute_slots_for_block_size_310,
    _copy_and_expand_inputs_ascendc,
    _validate_num_spec_tokens_310,
)


def test_compute_slots_supports_mixed_physical_block_sizes():
    positions = torch.tensor([63, 64, 127, 128], dtype=torch.int32)
    request_ids = torch.zeros(4, dtype=torch.long)
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)

    slots_64 = _compute_slots_for_block_size_310(positions, request_ids, block_table, 64)
    slots_128 = _compute_slots_for_block_size_310(positions, request_ids, block_table, 128)

    assert slots_64.tolist() == [703, 704, 767, 768]
    assert slots_128.tolist() == [1343, 1344, 1407, 1408]


@pytest.mark.parametrize("num_speculative_tokens", [6, 8, 15])
def test_num_spec_validation_supports_k_above_five(num_speculative_tokens):
    _validate_num_spec_tokens_310(num_speculative_tokens)


def test_num_spec_validation_rejects_query_beyond_kernel_capacity():
    with pytest.raises(ValueError, match="supports at most 15"):
        _validate_num_spec_tokens_310(16)


class TestCopyAndExpandInputsAscendC(TestBase):
    def _make_self(self, num_query_total, num_context):
        return SimpleNamespace(
            device=torch.device("cpu"),
            vllm_config=SimpleNamespace(speculative_config=None),
            runner=SimpleNamespace(max_num_reqs=16),
            parallel_drafting_token_id=999,
            kernel_block_size=128,
            num_speculative_tokens=3,
            input_ids=torch.zeros(num_query_total, dtype=torch.int32),
            positions=torch.zeros(num_query_total, dtype=torch.int32),
            _slot_mapping_buffer=torch.zeros(num_query_total, dtype=torch.int32),
            _context_positions_buffer=torch.zeros(num_context, dtype=torch.int32),
            _context_slot_mapping_buffer=torch.zeros(num_context, dtype=torch.int32),
        )

    def _run(self, fake_self, target_positions, num_context, batch_size, num_query_per_req, captured):
        num_query_total = batch_size * num_query_per_req

        cad = SimpleNamespace(
            slot_mapping=torch.zeros(num_context, dtype=torch.int32),
            query_start_loc=torch.tensor([0, num_context], dtype=torch.int32),
            seq_lens=torch.tensor([num_context], dtype=torch.int32),
            block_table_tensor=torch.zeros(batch_size, 8, dtype=torch.int32),
        )

        def fake_op(next_token_ids, tpos, *args, **kwargs):
            captured["tpos"] = tpos
            captured.setdefault("num_rejected", []).append(args[4])
            n = tpos.shape[0]
            return (
                torch.zeros(num_query_total, dtype=torch.int32),
                torch.zeros(num_query_total, dtype=torch.int32),
                torch.zeros(num_query_total, dtype=torch.int32),
                torch.arange(n, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
                torch.zeros(batch_size * 3, dtype=torch.int32),
            )

        mock_ascend = MagicMock()
        mock_ascend.npu_copy_and_expand_dflash_inputs.side_effect = fake_op
        mock_ops = MagicMock()
        mock_ops._C_ascend = mock_ascend

        with patch.object(torch, "ops", mock_ops):
            _copy_and_expand_inputs_ascendc(
                fake_self,
                next_token_ids=torch.tensor([5], dtype=torch.int32),
                target_positions=target_positions,
                cad=cad,
                num_rejected_tokens_gpu=None,
                num_query_per_req=num_query_per_req,
                batch_size=batch_size,
                num_context=num_context,
                sample_from_anchor=False,
            )

    def test_mrope_positions_reduced_to_row0(self):
        # MRoPE models feed positions as [3, num_context]; the op must receive a
        # flat [num_context] vector (row 0) so the context outputs are sized by the
        # token count, not the mrope dim (which would size them as 3).
        num_context = 17
        target_positions = torch.stack(
            [
                torch.arange(num_context, dtype=torch.int32),
                torch.arange(num_context, dtype=torch.int32) + 100,
                torch.arange(num_context, dtype=torch.int32) + 200,
            ]
        )
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        captured = {}

        self._run(fake_self, target_positions, num_context, batch_size=1, num_query_per_req=4, captured=captured)

        self.assertEqual(captured["tpos"].dim(), 1)
        self.assertEqual(captured["tpos"].shape[0], num_context)
        torch.testing.assert_close(captured["tpos"], torch.arange(num_context, dtype=torch.int32))

    def test_1d_positions_passthrough(self):
        # Regular RoPE already provides a 1D [num_context] positions vector.
        num_context = 12
        target_positions = torch.arange(num_context, dtype=torch.int32)
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        captured = {}

        self._run(fake_self, target_positions, num_context, batch_size=1, num_query_per_req=4, captured=captured)

        self.assertEqual(captured["tpos"].dim(), 1)
        self.assertEqual(captured["tpos"].shape[0], num_context)

    def test_exact_piecewise_zero_rejection_buffer_is_persistent_and_bounded(self):
        num_context = 12
        target_positions = torch.arange(num_context, dtype=torch.int32)
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        captured = {}

        with patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310.is_310p_dflash_piecewise",
            return_value=True,
        ):
            self._run(
                fake_self,
                target_positions,
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured=captured,
            )
            self._run(
                fake_self,
                target_positions,
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured=captured,
            )

        first, second = captured["num_rejected"]
        self.assertEqual(fake_self._zero_num_rejected_buffer_310.shape, (16,))
        self.assertEqual(first.shape, (1,))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, torch.zeros(1, dtype=torch.int32))
