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

from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_ascend._310p.sample.rejection_sampler as rejection_sampler_310_module
import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from tests.ut.base import TestBase
from vllm_ascend._310p.sample.rejection_sampler import (
    AscendRejectionSampler310,
    _force_pytorch_rejection_path,
    _rejection_greedy_sample_310,
    _rejection_greedy_sample_pytorch_310,
)


class _RejectUnalignedGreedyCopyOps(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (
            torch.ops.aten.index_put_.default,
            torch.ops.aten._index_put_impl_.default,
        ):
            raise AssertionError("Advanced index writes can lower to an unaligned Add on 310P")
        if func is torch.ops.aten.copy_.default:
            destination = args[0]
            byte_offset = destination.storage_offset() * destination.element_size()
            if byte_offset % 64:
                raise AssertionError("Sub-row copy destination is not 64-byte aligned on 310P")
        if func is torch.ops.aten.add.Tensor:
            lhs = args[0]
            if isinstance(lhs, torch.Tensor) and lhs.numel() <= 4:
                raise AssertionError("Add with a small leading tensor is unsafe on 310P")
        return func(*args, **kwargs)


class TestForcePytorchRejectionPath(TestBase):
    def test_sampler_always_binds_fused_greedy_entry(self):
        sampler = AscendRejectionSampler310.__new__(AscendRejectionSampler310)
        expected = object()

        with (
            patch.object(
                rejection_sampler_310_module,
                "_force_pytorch_rejection_path",
                return_value=nullcontext(),
            ) as force_path,
            patch.object(
                rejection_sampler_310_module.AscendRejectionSampler,
                "forward",
                return_value=expected,
            ),
        ):
            actual = sampler.forward(None, None, None, None)

        self.assertIs(actual, expected)
        force_path.assert_called_once_with(
            sampler.sample_recovered_tokens,
            greedy_fn=_rejection_greedy_sample_310,
        )

    def test_disables_triton_and_binds_recovered_then_restores(self):
        orig_triton = rejection_sampler_module.HAS_TRITON
        orig_recovered = rejection_sampler_module.sample_recovered_tokens
        orig_greedy = rejection_sampler_module.rejection_greedy_sample_pytorch

        def sentinel(*args, **kwargs):
            return None

        def sentinel_greedy(*args, **kwargs):
            return None

        with _force_pytorch_rejection_path(sentinel, greedy_fn=sentinel_greedy):
            # 310P has no working Triton; the base sampler must take PyTorch paths.
            self.assertFalse(rejection_sampler_module.HAS_TRITON)
            self.assertIs(rejection_sampler_module.sample_recovered_tokens, sentinel)
            self.assertIs(
                rejection_sampler_module.rejection_greedy_sample_pytorch,
                sentinel_greedy,
            )

        self.assertEqual(rejection_sampler_module.HAS_TRITON, orig_triton)
        self.assertIs(rejection_sampler_module.sample_recovered_tokens, orig_recovered)
        self.assertIs(
            rejection_sampler_module.rejection_greedy_sample_pytorch,
            orig_greedy,
        )

    def test_restores_on_exception(self):
        orig_triton = rejection_sampler_module.HAS_TRITON
        orig_recovered = rejection_sampler_module.sample_recovered_tokens

        with self.assertRaises(RuntimeError), _force_pytorch_rejection_path(lambda *a, **k: None):
            raise RuntimeError("boom")

        self.assertEqual(rejection_sampler_module.HAS_TRITON, orig_triton)
        self.assertIs(rejection_sampler_module.sample_recovered_tokens, orig_recovered)

    def test_alignment_safe_greedy_copy_avoids_small_tensor_scalar_add(self):
        output_token_ids = torch.full((2, 4), -1, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([3, 6], dtype=torch.int32)
        draft_token_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
        target_argmax = torch.tensor([1, 9, 3, 4, 5, 6], dtype=torch.int32)
        bonus_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)

        with _RejectUnalignedGreedyCopyOps():
            _rejection_greedy_sample_pytorch_310(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                [3, 3],
                3,
            )

        torch.testing.assert_close(
            output_token_ids,
            torch.tensor(
                [
                    [1, 9, -1, -1],
                    [4, 5, 6, 8],
                ],
                dtype=torch.int32,
            ),
        )

    def test_all_greedy_routes_to_fused_op(self):
        output_token_ids = torch.full((2, 4), -1, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([3, 6], dtype=torch.int32)
        draft_token_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
        target_argmax = draft_token_ids.to(torch.int64)
        bonus_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
        calls = []

        def fused_op(*args):
            calls.append(args)
            args[4].fill_(42)

        with patch.object(
            rejection_sampler_310_module,
            "_get_rejection_sample_greedy_310_op",
            return_value=fused_op,
        ):
            _rejection_greedy_sample_310(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                [3, 3],
                3,
            )

        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(output_token_ids, torch.full_like(output_token_ids, 42))

    def test_mixed_greedy_keeps_pytorch_fallback(self):
        output_token_ids = torch.full((2, 4), -1, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([3, 6], dtype=torch.int32)
        draft_token_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
        target_argmax = torch.tensor([1, 9, 3, 4, 5, 6], dtype=torch.int64)
        bonus_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
        is_greedy = torch.tensor([True, False])

        with patch.object(
            rejection_sampler_310_module,
            "_get_rejection_sample_greedy_310_op",
            side_effect=AssertionError("mixed batches must not query the fused op"),
        ):
            _rejection_greedy_sample_310(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                [3, 3],
                3,
                is_greedy,
            )

        torch.testing.assert_close(
            output_token_ids,
            torch.tensor(
                [
                    [1, 9, -1, -1],
                    [-1, -1, -1, -1],
                ],
                dtype=torch.int32,
            ),
        )
