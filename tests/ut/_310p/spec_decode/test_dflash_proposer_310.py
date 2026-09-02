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

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.config import CUDAGraphMode

import vllm_ascend._310p.spec_decode.dflash_proposer_310 as dflash_proposer_310
from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    AscendDflashProposer310,
    _compute_slots_for_block_size_310,
    _copy_and_expand_inputs_ascendc,
    _validate_num_spec_tokens_310,
    wrap_dummy_run_with_draft_flag,
)


class _RejectInt64Add(TorchDispatchMode):
    """Model the 310P dynamic-int64 Add alignment restriction."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.add.Tensor and any(
            isinstance(arg, torch.Tensor) and arg.dtype == torch.int64 for arg in args
        ):
            raise AssertionError("310P slot mapping must not launch int64 Add")
        return func(*args, **kwargs)


def test_dflash_private_index_fill_avoids_dynamic_int64_add():
    tensor = torch.arange(6).reshape(2, 3)
    indices = torch.tensor([0, -1], dtype=torch.int64)

    with _RejectInt64Add():
        output = dflash_proposer_310._index_fill_without_add_310p_dflash(
            tensor,
            1,
            indices,
            99,
        )

    torch.testing.assert_close(
        output,
        torch.tensor([[99, 1, 99], [99, 4, 99]]),
    )


def test_dflash_prepare_next_tokens_does_not_use_shared_index_fill():
    class BackupTokens:
        def __init__(self):
            self.np = np.zeros(4, dtype=np.int64)
            self.gpu = torch.zeros(4, dtype=torch.int64)

        def copy_to_gpu(self, num_reqs):
            self.gpu[:num_reqs].copy_(torch.from_numpy(self.np[:num_reqs]))

    fake_self = SimpleNamespace(backup_next_token_ids=BackupTokens())
    requests = {
        "request-0": SimpleNamespace(get_token_id=MagicMock(return_value=31)),
        "request-1": SimpleNamespace(get_token_id=MagicMock(return_value=42)),
    }
    gpu_input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens_no_spec=torch.tensor([3, 3]),
        req_ids=["request-0", "request-1"],
        vocab_size=100,
    )

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.DeviceOperator.index_fill",
        side_effect=AssertionError("shared index_fill must stay out of 310P DFlash"),
    ):
        next_token_ids, valid_counts = AscendDflashProposer310.prepare_next_token_ids_padded(
            fake_self,
            sampled_token_ids=torch.tensor(
                [[11, 12, -1], [21, 22, -1]],
                dtype=torch.int64,
            ),
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=torch.tensor([-1], dtype=torch.int64),
            num_discarded_requests=1,
        )

    assert next_token_ids.tolist() == [12, 42]
    assert valid_counts.tolist() == [2, 0]


def test_dummy_capture_prepares_dual_rope_before_graph_capture():
    prepare_rope = MagicMock(return_value="prepared")
    finish_rope = MagicMock()
    positions = torch.arange(160, dtype=torch.int32)
    fake_self = SimpleNamespace(
        vllm_config=None,
        _get_positions=MagicMock(return_value=positions),
        _prepare_full_decode_draft_rope=prepare_rope,
        _finish_full_decode_draft_rope=finish_rope,
    )

    def original(
        self,
        num_tokens,
        *,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        is_profile=False,
    ):
        del self, num_tokens, aclgraph_runtime_mode, is_profile
        return "captured"

    wrapped = wrap_dummy_run_with_draft_flag(original)
    assert (
        wrapped(
            fake_self,
            160,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
        )
        == "captured"
    )
    prepare_rope.assert_called_once_with(
        query_positions=positions,
        query_actual_tokens=160,
        descriptor_tokens=160,
        runtime_mode=CUDAGraphMode.FULL,
    )
    finish_rope.assert_called_once_with("prepared")


def test_hybrid_dummy_capture_clamps_rope_to_draft_query_capacity():
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )
    prepare_rope = MagicMock(return_value="prepared")
    finish_rope = MagicMock()
    fake_self = SimpleNamespace(
        vllm_config=config,
        max_query_tokens=32,
        _get_positions=MagicMock(
            side_effect=lambda num_tokens: torch.arange(
                num_tokens,
                dtype=torch.int32,
            ),
        ),
        _prepare_full_decode_draft_rope=prepare_rope,
        _finish_full_decode_draft_rope=finish_rope,
    )

    def original(
        self,
        num_tokens,
        *,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        is_profile=False,
    ):
        del self, num_tokens, aclgraph_runtime_mode, is_profile
        return "captured"

    wrapped = wrap_dummy_run_with_draft_flag(original)
    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert (
            wrapped(
                fake_self,
                160,
                aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            )
            == "captured"
        )

    fake_self._get_positions.assert_called_once_with(32)
    prepare_rope.assert_called_once()
    rope_call = prepare_rope.call_args.kwargs
    torch.testing.assert_close(
        rope_call["query_positions"],
        torch.arange(32, dtype=torch.int32),
    )
    assert rope_call["query_actual_tokens"] == 32
    assert rope_call["descriptor_tokens"] == 32
    assert rope_call["runtime_mode"] == CUDAGraphMode.PIECEWISE
    finish_rope.assert_called_once_with("prepared")


def test_hybrid_capability_reserves_draft_rope_capacity_before_capture():
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )
    fake_self = SimpleNamespace(
        vllm_config=config,
        runner=SimpleNamespace(max_num_tokens=400),
        _get_positions=MagicMock(return_value=torch.arange(64, dtype=torch.int32)),
    )

    wrapped = wrap_dummy_run_with_draft_flag(lambda self, *args, **kwargs: "captured")
    with (
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310.configure_draft_rope_capacity_310",
        ) as reserve,
    ):
        assert wrapped(fake_self, 64) == "captured"

    reserve.assert_called_once_with(400)


def test_compute_slots_supports_mixed_physical_block_sizes():
    positions = torch.tensor([63, 64, 127, 128], dtype=torch.int32)
    request_ids = torch.zeros(4, dtype=torch.long)
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)

    slots_64 = _compute_slots_for_block_size_310(positions, request_ids, block_table, 64)
    slots_128 = _compute_slots_for_block_size_310(positions, request_ids, block_table, 128)

    assert slots_64.tolist() == [703, 704, 767, 768]
    assert slots_128.tolist() == [1343, 1344, 1407, 1408]


def test_compute_slots_avoids_dynamic_int64_add_on_310p():
    positions = torch.tensor(
        list(range(75, 91)) + list(range(86, 102)),
        dtype=torch.int32,
    )
    request_ids = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)
    block_table = torch.tensor(
        [[1020], [350]],
        dtype=torch.int32,
    )

    with _RejectInt64Add():
        slots = _compute_slots_for_block_size_310(
            positions,
            request_ids,
            block_table,
            128,
            use_int32_math=True,
        )

    assert slots.dtype == torch.int32
    assert slots.tolist() == list(range(130635, 130651)) + list(range(44886, 44902))


def test_hybrid_uses_int32_draft_address_math_without_changing_other_modes():
    config = SimpleNamespace()
    with (
        patch.object(
            dflash_proposer_310,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch.object(
            dflash_proposer_310,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
    ):
        assert dflash_proposer_310._uses_int32_draft_address_math_310(config)

    with (
        patch.object(
            dflash_proposer_310,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch.object(
            dflash_proposer_310,
            "is_310p_dflash_full_and_piecewise",
            return_value=False,
        ),
    ):
        assert not dflash_proposer_310._uses_int32_draft_address_math_310(config)


def test_dflash_seq_lens_update_avoids_dynamic_int64_add_on_310p():
    batch_size = 2
    num_query_per_req = 16
    num_context = 32
    fake_self = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        num_speculative_tokens=15,
        _dflash_hidden_states=torch.zeros((num_context, 4)),
        _slot_mapping_buffer=torch.zeros(
            batch_size * num_query_per_req,
            dtype=torch.int32,
        ),
        arange_dflash=torch.arange(64, dtype=torch.int32),
        token_arange_np=np.arange(64, dtype=np.int32),
    )
    cad = SimpleNamespace(
        num_reqs=batch_size,
        seq_lens=torch.tensor([75, 70], dtype=torch.int64),
        max_seq_len=75,
    )

    with (
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310._copy_and_expand_inputs_ascendc",
            return_value=torch.zeros(30, dtype=torch.int32),
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310.is_310p_dflash_full_decode_only",
            return_value=True,
        ),
        _RejectInt64Add(),
    ):
        AscendDflashProposer310.set_inputs_first_pass(
            fake_self,
            target_token_ids=torch.zeros(num_context, dtype=torch.int32),
            next_token_ids=torch.zeros(batch_size, dtype=torch.int32),
            target_positions=torch.arange(num_context, dtype=torch.int32),
            target_hidden_states=torch.zeros((num_context, 4)),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=torch.tensor([0, 11], dtype=torch.int32),
        )

    assert cad.seq_lens.dtype == torch.int32
    assert cad.seq_lens.tolist() == [91, 75]


def test_dflash_seq_lens_preserve_legacy_dtype_outside_fdo():
    batch_size = 2
    num_query_per_req = 16
    num_context = 32
    fake_self = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        num_speculative_tokens=15,
        _dflash_hidden_states=torch.zeros((num_context, 4)),
        _slot_mapping_buffer=torch.zeros(
            batch_size * num_query_per_req,
            dtype=torch.int32,
        ),
        arange_dflash=torch.arange(64, dtype=torch.int32),
        token_arange_np=np.arange(64, dtype=np.int32),
    )
    cad = SimpleNamespace(
        num_reqs=batch_size,
        seq_lens=torch.tensor([75, 70], dtype=torch.int64),
        max_seq_len=75,
    )

    with (
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310._copy_and_expand_inputs_ascendc",
            return_value=torch.zeros(30, dtype=torch.int32),
        ),
        patch(
            "vllm_ascend._310p.spec_decode.dflash_proposer_310.is_310p_dflash_full_decode_only",
            return_value=False,
        ),
    ):
        AscendDflashProposer310.set_inputs_first_pass(
            fake_self,
            target_token_ids=torch.zeros(num_context, dtype=torch.int32),
            next_token_ids=torch.zeros(batch_size, dtype=torch.int32),
            target_positions=torch.arange(num_context, dtype=torch.int32),
            target_hidden_states=torch.zeros((num_context, 4)),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=torch.tensor([0, 11], dtype=torch.int32),
        )

    assert cad.seq_lens.dtype == torch.int64
    assert cad.seq_lens.tolist() == [91, 75]


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

    def test_hybrid_zero_rejection_buffer_depends_on_effective_piecewise_mode(self):
        num_context = 12
        target_positions = torch.arange(num_context, dtype=torch.int32)
        fake_self = self._make_self(num_query_total=4, num_context=num_context)
        fake_self.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash"),
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            ),
            additional_config={
                "ascend_compilation_config": {
                    "dflash_full_and_piecewise_capture_config": {
                        "piecewise_capture_size": 64,
                        "full_capture_size": 160,
                    },
                },
            },
        )

        with (
            patch(
                "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.spec_decode.dflash_proposer_310.get_forward_context",
                return_value=SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                ),
            ),
        ):
            captured = {}
            self._run(
                fake_self,
                target_positions,
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured=captured,
            )
            piecewise_buffer = fake_self._zero_num_rejected_buffer_310

        del fake_self._zero_num_rejected_buffer_310
        with (
            patch(
                "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.spec_decode.dflash_proposer_310.get_forward_context",
                return_value=SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                ),
            ),
        ):
            self._run(
                fake_self,
                target_positions,
                num_context,
                batch_size=1,
                num_query_per_req=4,
                captured={},
            )

        self.assertEqual(piecewise_buffer.shape, (16,))
        self.assertFalse(hasattr(fake_self, "_zero_num_rejected_buffer_310"))
