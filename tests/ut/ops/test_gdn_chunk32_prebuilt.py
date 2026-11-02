# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""CPU-only regression tests for prebuilt A5 chunk32 descriptors."""

from types import SimpleNamespace
from unittest import TestCase, main

import torch
import vllm  # noqa: F401 - Initialize vLLM before Ascend adaptors.

from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata


class NoIterationTuple(tuple):
    def __iter__(self):
        raise AssertionError("Per-layer adaptor must not rebuild chunk descriptors")


class TestChunk32Prebuilt(TestCase):
    def build(self, cu):
        return _build_non_spec_chunked_prefill_metadata(
            SimpleNamespace(),
            torch.tensor(cu, dtype=torch.int32),
            torch.device("cpu"),
            build_device_metadata=False,
        )

    def test_two_sequences(self):
        m = self.build([0, 64, 128])
        self.assertEqual(m.chunk_indices_chunk32_host, (0, 0, 0, 1, 1, 0, 1, 1))
        self.assertEqual(m.chunk_indices_chunk64_host, (0, 0, 1, 0))
        self.assertIsNone(m.cu_seqlens_kern)

    def test_compacted_sequence_ids_and_tails(self):
        m = self.build([0, 0, 33, 33, 98, 98])
        self.assertEqual(m.cu_seqlens_kern, (0, 33, 98))
        self.assertEqual(m.chunk_indices_chunk32_host, (0, 0, 0, 1, 1, 0, 1, 1, 1, 2))
        self.assertEqual(m.keep_meta.tolist(), [False, True, False, True, False])

    def test_all_zero_segments(self):
        for cu in ([0, 0], [0, 0, 0]):
            with self.subTest(cu=cu):
                self.assertEqual(self.build(cu).chunk_indices_chunk32_host, ())

    def test_chunk_boundary_and_rebuild_between_batches(self):
        a = self.build([0, 31, 63, 96])
        self.assertEqual(a.chunk_indices_chunk32_host, (0, 0, 1, 0, 2, 0, 2, 1))
        b = self.build([0, 32])
        self.assertEqual(b.chunk_indices_chunk32_host, (0, 0))
        self.assertEqual(a.chunk_indices_chunk32_host, (0, 0, 1, 0, 2, 0, 2, 1))

    def test_layers_reuse_exact_prebuilt_object(self):
        # Initialize the ops package first to avoid the device adaptor import cycle.
        from vllm_ascend.device.device_op import A5DeviceAdaptor

        for cu in ([0, 64, 128], [0, 0, 33, 33, 98, 98]):
            with self.subTest(cu=cu):
                m = self.build(cu)
                indices = m.chunk_indices_chunk32_host
                effective = m.cu_seqlens_host if m.keep_meta is None else m.cu_seqlens_kern
                if m.keep_meta is None:
                    m.cu_seqlens_host = NoIterationTuple(effective)
                else:
                    m.cu_seqlens_kern = NoIterationTuple(effective)
                tokens = cu[-1]
                state = torch.zeros(len(cu) - 1, 2, 4, 4)
                seen = []

                def fused(q, k, v, g, beta, indices=indices, effective=effective, seen=seen, tokens=tokens, **kwargs):
                    self.assertIs(kwargs["chunk_indices"], indices)
                    self.assertEqual(kwargs["chunk_size"], 32)
                    self.assertFalse(kwargs["use_exp2"])
                    self.assertFalse(kwargs["disable_recompute"])
                    self.assertTrue(kwargs["use_qk_l2norm_in_kernel"])
                    self.assertTrue(kwargs["use_beta_sigmoid_in_kernel"])
                    self.assertEqual(kwargs["cu_seqlens"], effective)
                    seen.append(kwargs["chunk_indices"])
                    return torch.zeros(1, tokens, 2, 4), kwargs["initial_state"].clone()

                for _ in range(3):
                    _, final = A5DeviceAdaptor.fla_gdn_prefill(
                        torch.zeros(1, tokens, 1, 4),
                        torch.zeros(1, tokens, 1, 4),
                        torch.zeros(1, tokens, 2, 4),
                        torch.zeros(1, tokens, 2),
                        torch.zeros(1, tokens, 2),
                        state,
                        0.5,
                        m,
                        fused,
                    )
                    self.assertTrue(torch.equal(final, state))
                self.assertEqual(len(seen), 3)

    def test_device_metadata_branch_preserves_chunk64(self):
        builder = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=SimpleNamespace(linear_num_value_heads=32)),
                parallel_config=SimpleNamespace(tensor_parallel_size=1),
            )
        )
        m = _build_non_spec_chunked_prefill_metadata(
            builder,
            torch.tensor([0, 64, 128], dtype=torch.int32),
            torch.device("cpu"),
            build_device_metadata=True,
        )
        self.assertEqual(m.chunk_indices_chunk32_host, (0, 0, 0, 1, 1, 0, 1, 1))
        self.assertEqual(m.chunk_indices_chunk64.tolist(), [[0, 0], [1, 0]])


if __name__ == "__main__":
    main(verbosity=2)
