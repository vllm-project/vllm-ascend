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

"""CPU-only unit tests for the A2/A3 FLA NPU GDN prefill adaptor.

Tests the BaseDeviceAdaptor.fla_gdn_prefill argument shim around the FLA
kernel, verifying the optimized parameter contract:

* q/k/v are handed over head-first (BNSD) without copying when the transpose
  is already contiguous, and as sequence-major (BSND) otherwise;
* q/k normalization is delegated to the kernel (use_qk_l2norm_in_kernel=True);
* beta sigmoid is delegated to the kernel (use_beta_sigmoid_in_kernel=True);
* the recurrent state keeps its [N, Hv, Dk, Dv] layout (state_v_first=True);
* use_exp2=False, disable_recompute=False for the A2/A3 optimized path.
"""

from types import SimpleNamespace
from unittest import TestCase, main

import torch
import vllm  # noqa: F401 - Initialize vLLM before Ascend adaptors.
import vllm_ascend.ops  # noqa: F401 - Break the device_op import cycle.

# Typical A2/A3 GDN prefill shape.
BATCH = 1
TOKENS = 64
HK = 16
HV = 32
HEAD_DIM = 128
STATE_SLOTS = 2
CHUNK64_INDICES = (0, 0, 1, 0)


def make_meta(cu_seqlens=(0, 32, 64), keep_meta=None, cu_seqlens_kern=None):
    """Build the subset of GDNChunkedPrefillMetadata the adaptor reads."""
    return SimpleNamespace(
        cu_seqlens_host=cu_seqlens,
        cu_seqlens_kern=cu_seqlens_kern,
        chunk_indices_chunk64_host=CHUNK64_INDICES,
        keep_meta=keep_meta,
    )


def make_inputs(tokens=TOKENS, dtype=torch.bfloat16):
    """Build contract-shaped q/k/v/g/beta tensors, all sequence-major."""
    q = torch.randn(BATCH, tokens, HK, HEAD_DIM, dtype=dtype)
    k = torch.randn(BATCH, tokens, HK, HEAD_DIM, dtype=dtype)
    v = torch.randn(BATCH, tokens, HV, HEAD_DIM, dtype=dtype)
    g = torch.randn(BATCH, tokens, HV, HEAD_DIM, dtype=dtype)
    beta = torch.randn(BATCH, tokens, HV, dtype=dtype)
    return q, k, v, g, beta


def make_state(slots=STATE_SLOTS, dtype=torch.bfloat16, fill=5.0):
    """Recurrent state in the caller's [N, Hv, Dk, Dv] layout."""
    return torch.full((slots, HV, HEAD_DIM, HEAD_DIM), fill, dtype=dtype)


class TestFlaGdnPrefillAdaptor(TestCase):
    """Verify BaseDeviceAdaptor.fla_gdn_prefill forwards the kernel contract."""

    # ------------------------------------------------------------------ #
    # 1. Kernel contract: all parameters match the A2/A3 optimized path.  #
    # ------------------------------------------------------------------ #
    def test_forwards_the_kernel_contract_for_sequence_major_inputs(self):
        """BSND layout, all kwargs matching the optimized A2/A3 contract."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        q, k, v, g, beta = make_inputs()
        state = make_state()
        calls = []

        def fused_fwd(q_in, k_in, v_in, g_in, beta_in, **kwargs):
            calls.append({"q": q_in, "k": k_in, "v": v_in, "g": g_in, "beta": beta_in, **kwargs})
            return torch.zeros_like(q_in), torch.zeros_like(state)

        BaseDeviceAdaptor.fla_gdn_prefill(
            q, k, v, g, beta, state, 0.125, make_meta(), fused_fwd
        )
        call = calls[0]

        self.assertEqual(call["chunk_size"], 64)
        self.assertFalse(call["use_exp2"])
        self.assertTrue(call["use_qk_l2norm_in_kernel"])
        self.assertTrue(call["use_beta_sigmoid_in_kernel"])
        self.assertFalse(call["disable_recompute"])
        self.assertTrue(call["state_v_first"])
        self.assertFalse(call["allow_neg_eigval"])
        self.assertTrue(call["output_final_state"])
        self.assertEqual(call["layout"], "BSND")
        self.assertEqual(call["scale"], 0.125)

    # ------------------------------------------------------------------ #
    # 2. Layout selection: BNSD when head-first transpose is contiguous. #
    # ------------------------------------------------------------------ #
    def test_head_first_contiguous_uses_bnsd_without_copying(self):
        """When q/k/v transpose(1,2) is contiguous, use BNSD and pass the view."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        tokens = 32
        # Create contiguous BSND tensors whose transpose(1,2) is also contiguous.
        q = torch.randn(BATCH, HK, tokens, HEAD_DIM, dtype=torch.bfloat16).transpose(1, 2)
        k = torch.randn(BATCH, HK, tokens, HEAD_DIM, dtype=torch.bfloat16).transpose(1, 2)
        v = torch.randn(BATCH, HV, tokens, HEAD_DIM, dtype=torch.bfloat16).transpose(1, 2)

        calls = []

        def fused_fwd(q_in, k_in, v_in, g_in, beta_in, **kwargs):
            calls.append(kwargs)
            for actual in (q_in, k_in, v_in):
                self.assertTrue(actual.is_contiguous())
            return torch.zeros_like(q_in), torch.zeros_like(v_in)

        gate = torch.randn(BATCH, tokens, HV, HEAD_DIM, dtype=torch.bfloat16)
        raw_beta = torch.randn(BATCH, tokens, HV, dtype=torch.bfloat16)

        BaseDeviceAdaptor.fla_gdn_prefill(
            q, k, v, gate, raw_beta, make_state(), 0.125, make_meta(), fused_fwd
        )
        self.assertEqual(calls[0]["layout"], "BNSD")

    def test_non_contiguous_transpose_uses_bsnd(self):
        """When the head-first transpose is NOT contiguous, fall back to BSND."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        tokens = 64
        # A slice breaks contiguity after transpose(1,2).
        q = torch.randn(BATCH, tokens * 2, HK, HEAD_DIM, dtype=torch.bfloat16)[:, :tokens]
        k = torch.randn(BATCH, tokens * 2, HK, HEAD_DIM, dtype=torch.bfloat16)[:, :tokens]
        v = torch.randn(BATCH, tokens * 2, HV, HEAD_DIM, dtype=torch.bfloat16)[:, :tokens]

        calls = []

        def fused_fwd(q_in, k_in, v_in, g_in, beta_in, **kwargs):
            calls.append(kwargs)
            for actual in (q_in, k_in, v_in):
                self.assertTrue(actual.is_contiguous())
            return torch.zeros_like(q_in), torch.zeros_like(v_in)

        gate = torch.randn(BATCH, tokens, HV, HEAD_DIM, dtype=torch.bfloat16)
        raw_beta = torch.randn(BATCH, tokens, HV, dtype=torch.bfloat16)

        BaseDeviceAdaptor.fla_gdn_prefill(
            q, k, v, gate, raw_beta, make_state(), 0.125, make_meta(), fused_fwd
        )
        self.assertEqual(calls[0]["layout"], "BSND")

    # ------------------------------------------------------------------ #
    # 3. State passthrough: state_v_first=True, no transpose, no copy.   #
    # ------------------------------------------------------------------ #
    def test_recurrent_state_is_passed_through_untransposed(self):
        """initial_state must reach the kernel as-is (state_v_first=True)."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        state = make_state(fill=7.0)
        original_ptr = state.data_ptr()

        calls = []

        def fused_fwd(q_in, k_in, v_in, g_in, beta_in, **kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["initial_state"].data_ptr(), original_ptr)
            return torch.zeros_like(q_in), kwargs["initial_state"].clone()

        q, k, v, g, beta = make_inputs()
        output, final = BaseDeviceAdaptor.fla_gdn_prefill(
            q, k, v, g, beta, state, 0.125, make_meta(), fused_fwd
        )
        self.assertTrue(torch.equal(final, state))

    # ------------------------------------------------------------------ #
    # 4. keep_meta: scatter kept segments, preserve empty ones.           #
    # ------------------------------------------------------------------ #
    def test_keep_meta_scatters_kept_segments_and_preserves_empty_ones(self):
        """When keep_meta filters empty segments, only kept rows reach kernel."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        keep = torch.tensor([False, True, False, True, False])
        meta = make_meta(
            cu_seqlens=(0, 0, 33, 33, 98, 98),
            keep_meta=keep,
            cu_seqlens_kern=(0, 33, 98),
        )
        state = make_state(slots=5, fill=3.0)
        original_state = state.clone()

        tokens = 98
        q = torch.randn(BATCH, tokens, HK, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(BATCH, tokens, HK, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.randn(BATCH, tokens, HV, HEAD_DIM, dtype=torch.bfloat16)
        gate = torch.randn(BATCH, tokens, HV, HEAD_DIM, dtype=torch.bfloat16)
        raw_beta = torch.randn(BATCH, tokens, HV, dtype=torch.bfloat16)

        kernel_final = torch.full_like(state[[1, 3]], 99.0)

        def fused_fwd(q_in, k_in, v_in, g_in, beta_in, **kwargs):
            self.assertEqual(kwargs["cu_seqlens"], (0, 33, 98))
            self.assertEqual(kwargs["initial_state"].shape[0], 2)
            torch.testing.assert_close(kwargs["initial_state"], original_state[[1, 3]])
            return torch.zeros_like(q_in), kernel_final

        output, final = BaseDeviceAdaptor.fla_gdn_prefill(
            q, k, v, gate, raw_beta, state, 0.125, meta, fused_fwd
        )

        expected = original_state.clone()
        expected[[1, 3]] = 99.0
        torch.testing.assert_close(final, expected)
        torch.testing.assert_close(final[~keep], state[~keep])

    # ------------------------------------------------------------------ #
    # 5. Dtype casting: g → float32, beta → v.dtype.                     #
    # ------------------------------------------------------------------ #
    def test_g_and_beta_are_cast_to_the_kernel_dtypes(self):
        """beta/g may arrive as bf16 or fp32; kernel always sees fp32 g and bf16 beta."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        for g_dtype in (torch.bfloat16, torch.float32):
            for beta_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(g_dtype=g_dtype, beta_dtype=beta_dtype):
                    q, k, v, _, _ = make_inputs()
                    g = torch.randn(BATCH, TOKENS, HV, HEAD_DIM, dtype=g_dtype)
                    beta = torch.randn(BATCH, TOKENS, HV, dtype=beta_dtype)

                    calls = []

                    def fused_fwd(q_in, k_in, v_in, g_in, beta_in, **kwargs):
                        calls.append({"g": g_in, "beta": beta_in})
                        return torch.zeros_like(q_in), torch.zeros_like(v_in)

                    BaseDeviceAdaptor.fla_gdn_prefill(
                        q, k, v, g, beta, make_state(), 0.125, make_meta(), fused_fwd
                    )

                    self.assertEqual(calls[0]["g"].dtype, torch.float32)
                    self.assertEqual(calls[0]["beta"].dtype, v.dtype)
                    self.assertTrue(calls[0]["g"].is_contiguous())
                    self.assertTrue(calls[0]["beta"].is_contiguous())

    # ------------------------------------------------------------------ #
    # 6. Layer reuse: the same prebuilt metadata is reused across layers. #
    # ------------------------------------------------------------------ #
    def test_layers_reuse_exact_prebuilt_object(self):
        """Verify the adaptor does not rebuild chunk descriptors per layer."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        meta = make_meta()
        indices = meta.chunk_indices_chunk64_host
        effective = meta.cu_seqlens_host

        tokens = TOKENS
        state = make_state()
        seen = []

        def fused(q, k, v, g, beta, indices=indices, effective=effective, seen=seen, tokens=tokens, **kwargs):
            self.assertIs(kwargs["chunk_indices"], indices)
            self.assertEqual(kwargs["cu_seqlens"], effective)
            seen.append(kwargs["chunk_indices"])
            return torch.zeros(1, tokens, 2, 4), kwargs["initial_state"].clone()

        for _ in range(3):
            _, final = BaseDeviceAdaptor.fla_gdn_prefill(
                torch.zeros(1, tokens, 1, 4),
                torch.zeros(1, tokens, 1, 4),
                torch.zeros(1, tokens, 2, 4),
                torch.zeros(1, tokens, 2),
                torch.zeros(1, tokens, 2),
                state,
                0.5,
                meta,
                fused,
            )
            self.assertTrue(torch.equal(final, state))
        self.assertEqual(len(seen), 3)

    # ------------------------------------------------------------------ #
    # 7. Full integration: compacted state scatter + layout + dtype.     #
    # ------------------------------------------------------------------ #
    def test_compacted_state_scatter_and_layout(self):
        """End-to-end: keep_meta scatter, BSND layout, beta passthrough."""
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        keep = torch.tensor([False, True, False, True, False])
        meta = make_meta(
            cu_seqlens=(0, 0, 33, 33, 98, 98),
            keep_meta=keep,
            cu_seqlens_kern=(0, 33, 98),
        )
        state = torch.arange(5 * 2 * 4 * 4).reshape(5, 2, 4, 4).float()
        original_state = state.clone()

        tokens = 98
        qkv = torch.arange(tokens * 2 * 4).reshape(1, tokens, 2, 4).to(torch.bfloat16)
        gate = torch.zeros(1, tokens, 2)
        raw_beta = torch.linspace(-2, 2, tokens * 2).reshape(1, tokens, 2).to(torch.bfloat16)

        def fused(
            q,
            k,
            v,
            g,
            beta,
            qkv=qkv,
            raw_beta=raw_beta,
            original_state=original_state,
            **kwargs,
        ):
            self.assertEqual(kwargs["layout"], "BSND")
            for actual in (q, k, v):
                self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(beta, raw_beta)
            torch.testing.assert_close(kwargs["initial_state"], original_state[[1, 3]])
            self.assertEqual(kwargs["cu_seqlens"], (0, 33, 98))
            return qkv.clone(), kwargs["initial_state"] + 1000

        output, final = BaseDeviceAdaptor.fla_gdn_prefill(
            qkv, qkv, qkv, gate, raw_beta, state, 0.5, meta, fused
        )
        expected_state = original_state.clone()
        expected_state[[1, 3]] += 1000
        torch.testing.assert_close(output, qkv)
        torch.testing.assert_close(final, expected_state)
        torch.testing.assert_close(state, original_state)


if __name__ == "__main__":
    main(verbosity=2)