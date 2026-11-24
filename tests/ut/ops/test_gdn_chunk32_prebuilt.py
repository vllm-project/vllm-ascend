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

from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import pytest
import torch
import vllm  # noqa: F401 - Initialize vLLM before Ascend adaptors.

from vllm_ascend.attention.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile


class NoIterationTuple(tuple):
    def __iter__(self):
        raise AssertionError("Per-layer adaptor must not rebuild chunk descriptors")


@pytest.mark.parametrize(
    "device_type,chunk32_enabled,chunk_size,use_exp2,kernel_l2norm,disable_recompute,state_v_first",
    [
        (AscendDeviceType.A2, False, 64, False, True, False, True),
        (AscendDeviceType.A3, False, 64, False, True, False, True),
        (AscendDeviceType.A5, True, 32, False, True, False, True),
        (AscendDeviceType.A5, False, 64, True, True, True, True),
    ],
)
@pytest.mark.parametrize("head_first", [False, True])
def test_fla_kernel_contract_isolated_by_device_and_chunk_size(
    device_type, chunk32_enabled, chunk_size, use_exp2, kernel_l2norm, disable_recompute, state_v_first, head_first
):
    """Preserve each device's kernel contract, with chunk32 enabled only on A5."""
    from vllm_ascend.device.device_op import get_device_adaptor

    profile = get_hardware_profile(device_type)
    if not chunk32_enabled:
        profile = replace(profile, capabilities=profile.capabilities - {HardwareCapability.FLA_GDN_PREFILL_CHUNK32})
    # Empty segments exercise compacted descriptors and state restoration.
    cu = torch.tensor([0, 0, 33, 33, 98, 98], dtype=torch.int32)
    q = torch.arange(98 * 2 * 4).reshape(1, 98, 2, 4).to(torch.bfloat16)
    v = torch.arange(98 * 2 * 3).reshape(1, 98, 2, 3).to(torch.bfloat16)
    if head_first:
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    # Non-square state dimensions expose accidental state-layout changes.
    state = torch.arange(5 * 2 * 3 * 4).reshape(5, 2, 3, 4).float()
    original_state = state.clone()
    gate = torch.zeros(1, 98, 2)
    raw_beta = torch.linspace(-2, 2, 196).reshape(1, 98, 2).to(torch.bfloat16)
    beta = raw_beta if chunk32_enabled else raw_beta.float().sigmoid().to(v.dtype)
    expected_output = v.clone()
    fused = Mock(side_effect=lambda *args, **kw: (expected_output, kw["initial_state"] + 1000))
    with (
        patch("vllm_ascend.attention.gdn_attn_builder.get_current_hardware_profile", return_value=profile),
        patch("vllm_ascend.device.device_op.get_current_hardware_profile", return_value=profile),
        patch(
            "vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd",
            side_effect=lambda x: torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype),
        ) as l2norm,
    ):
        metadata = _build_non_spec_chunked_prefill_metadata(
            SimpleNamespace(), cu, torch.device("cpu"), build_device_metadata=False
        )
        output, final_state = get_device_adaptor().fla_gdn_prefill(q, q, v, gate, beta, state, 0.5, metadata, fused)

    fused.assert_called_once()
    args, kwargs = fused.call_args
    assert kwargs["chunk_size"] == chunk_size
    assert kwargs["use_exp2"] is use_exp2
    assert kwargs["disable_recompute"] is disable_recompute
    assert kwargs.get("use_beta_sigmoid_in_kernel", False) is chunk32_enabled
    assert kwargs["use_qk_l2norm_in_kernel"] is kernel_l2norm
    assert kwargs["state_v_first"] is state_v_first
    assert kwargs["layout"] == ("BNSD" if head_first else "BSND")
    assert kwargs["cu_seqlens"] == (0, 33, 98)
    assert metadata.chunk_indices_chunk64_host == (0, 0, 1, 0, 1, 1)
    if chunk32_enabled:
        assert kwargs["chunk_indices"] is metadata.chunk_indices_chunk32_host
        assert kwargs["chunk_indices"] == (0, 0, 0, 1, 1, 0, 1, 1, 1, 2)
    else:
        assert metadata.chunk_indices_chunk32_host is None
        assert kwargs["chunk_indices"] is metadata.chunk_indices_chunk64_host
    torch.testing.assert_close(args[4], beta)
    expected_kernel_state = original_state[[1, 3]]
    if not state_v_first:
        expected_kernel_state = expected_kernel_state.transpose(-1, -2).contiguous()
    torch.testing.assert_close(kwargs["initial_state"], expected_kernel_state)
    assert l2norm.call_count == (0 if kernel_l2norm else 2)
    expected_state = original_state.clone()
    expected_state[[1, 3]] += 1000
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)
    torch.testing.assert_close(state, original_state)


class TestChunk32Prebuilt(TestCase):
    def setUp(self):
        # Metadata and adaptor look up the profile in their own modules.
        # Patch both so chunk32 tests do not depend on the host device family.
        for module in ("vllm_ascend.attention.gdn_attn_builder", "vllm_ascend.device.device_op"):
            profile_patch = patch(
                f"{module}.get_current_hardware_profile",
                return_value=get_hardware_profile(AscendDeviceType.A5),
            )
            profile_patch.start()
            self.addCleanup(profile_patch.stop)

    def test_non_a5_preserves_chunk64_without_building_chunk32(self):
        from vllm_ascend.attention import gdn_attn_builder

        original = gdn_attn_builder.prepare_chunk_indices

        def reject_chunk32(cu, chunk_size):
            self.assertNotEqual(chunk_size, 32)
            return original(cu, chunk_size)

        for device_type in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType._310P):
            with (
                self.subTest(device_type=device_type),
                patch(
                    "vllm_ascend.attention.gdn_attn_builder.get_current_hardware_profile",
                    return_value=get_hardware_profile(device_type),
                ),
                patch("vllm_ascend.attention.gdn_attn_builder.prepare_chunk_indices", side_effect=reject_chunk32),
            ):
                metadata = self.build([0, 64, 128])
                self.assertIsNone(metadata.chunk_indices_chunk32_host)
                self.assertEqual(metadata.chunk_indices_chunk64_host, (0, 0, 1, 0))

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
                seen: list[object] = []

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

    def test_missing_chunk32_metadata_rejects_before_kernel(self):
        from vllm_ascend.device.device_op import A5DeviceAdaptor

        metadata = self.build([0, 32])
        metadata.chunk_indices_chunk32_host = None
        fused = Mock()
        qkv = torch.zeros(1, 32, 1, 4)
        gate = torch.zeros(1, 32, 1)
        with self.assertRaisesRegex(ValueError, "requires prebuilt chunk32 metadata"):
            A5DeviceAdaptor.fla_gdn_prefill(qkv, qkv, qkv, gate, gate, torch.zeros(1, 1, 4, 4), 0.5, metadata, fused)
        fused.assert_not_called()

    def test_compacted_state_scatter_and_layout(self):
        from vllm_ascend.device.device_op import A5DeviceAdaptor

        for head_first in (False, True):
            with self.subTest(head_first=head_first):
                metadata = self.build([0, 0, 33, 33, 98, 98])
                state = torch.arange(5 * 2 * 4 * 4).reshape(5, 2, 4, 4).float()
                original_state = state.clone()
                qkv = torch.arange(98 * 2 * 4).reshape(1, 98, 2, 4).to(torch.bfloat16)
                if head_first:
                    qkv = qkv.transpose(1, 2).contiguous().transpose(1, 2)
                gate = torch.zeros(1, 98, 2)
                raw_beta = torch.linspace(-2, 2, 196).reshape(1, 98, 2).to(torch.bfloat16)

                def fused(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    head_first=head_first,
                    qkv=qkv,
                    raw_beta=raw_beta,
                    original_state=original_state,
                    **kwargs,
                ):
                    self.assertEqual(kwargs["layout"], "BNSD" if head_first else "BSND")
                    expected_qkv = qkv.transpose(1, 2) if head_first else qkv
                    for actual in (q, k, v):
                        torch.testing.assert_close(actual, expected_qkv)
                        self.assertTrue(actual.is_contiguous())
                    torch.testing.assert_close(beta, raw_beta)
                    torch.testing.assert_close(kwargs["initial_state"], original_state[[1, 3]])
                    self.assertEqual(kwargs["cu_seqlens"], (0, 33, 98))
                    return qkv.clone(), kwargs["initial_state"] + 1000

                output, final = A5DeviceAdaptor.fla_gdn_prefill(
                    qkv, qkv, qkv, gate, raw_beta, state, 0.5, metadata, fused
                )
                expected_state = original_state.clone()
                expected_state[[1, 3]] += 1000
                torch.testing.assert_close(output, qkv)
                torch.testing.assert_close(final, expected_state)
                torch.testing.assert_close(state, original_state)


if __name__ == "__main__":
    main(verbosity=2)
