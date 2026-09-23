# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import kda


@pytest.mark.parametrize(
    "dtype,head_dim,cu_seqlens,fused_norm",
    [
        (torch.bfloat16, 128, (0, 2, 3), True),
        (torch.float16, 128, (0, 2, 3), False),
        (torch.bfloat16, 64, (0, 2, 3), False),
        (torch.bfloat16, 128, (0, 0, 3), False),
    ],
)
@pytest.mark.parametrize("tensor_metadata", [False, True])
def test_chunk_normalization_dispatch_preserves_prefill_contract(
    monkeypatch, dtype, head_dim, cu_seqlens, fused_norm, tensor_metadata
):
    # Projection slices may be strided; the operator needs contiguous BSND.
    q, k, v = (torch.randn(1, 3, 1, head_dim * 2, dtype=dtype)[..., :head_dim] for _ in range(3))
    raw_gate = torch.randn(1, 3, 1, head_dim, dtype=torch.float32)
    beta = torch.full((1, 3, 1), 0.25)
    state = torch.randn(2, 1, head_dim, head_dim)
    chunks = (0, 0, 1, 0) if cu_seqlens[1] else (1, 0)
    descriptor = torch.tensor(cu_seqlens) if tensor_metadata else cu_seqlens
    chunk_descriptor = torch.tensor(chunks).reshape(-1, 2) if tensor_metadata else chunks
    expected_output, expected_state = torch.empty_like(v), torch.empty_like(state)
    normalized = []

    def normalize(value):
        assert value.is_contiguous()
        result = value * 2
        normalized.append(result)
        return result

    def chunk(q_arg, k_arg, v_arg, gate_arg, beta_arg, scale, chunk_size, **kwargs):
        assert q_arg.is_contiguous() and k_arg.is_contiguous() and v_arg.is_contiguous()
        torch.testing.assert_close(q_arg, q if fused_norm else normalized[0])
        torch.testing.assert_close(k_arg, k if fused_norm else normalized[1])
        torch.testing.assert_close(v_arg, v)
        assert gate_arg is raw_gate and beta_arg is beta
        assert kwargs["initial_state"] is state
        assert kwargs["cu_seqlens"] == cu_seqlens
        assert kwargs["chunk_indices"] == chunks
        assert kwargs["layout"] == "BSND" and chunk_size == 64
        assert scale == head_dim**-0.5
        assert kwargs["use_qk_l2norm_in_kernel"] is fused_norm
        assert not kwargs["use_beta_sigmoid_in_kernel"]
        assert kwargs["use_gate_in_kernel"] and kwargs["state_v_first"] and kwargs["output_final_state"]
        assert not kwargs["disable_recompute"] and not kwargs["return_intermediate_states"]
        return expected_output, expected_state, *([None] * 10)

    monkeypatch.setattr(kda, "l2norm_fwd", normalize)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_kda_fwd", chunk, raising=False)
    output, final_state = kda.run_chunk_kda(
        q,
        k,
        v,
        raw_gate,
        beta,
        state,
        descriptor,
        chunk_descriptor,
        torch.zeros(1),
        torch.zeros(head_dim),
        lower_bound=-4.0,
    )
    assert output is expected_output and final_state is expected_state
    assert len(normalized) == (0 if fused_norm else 2)


@pytest.mark.parametrize("kdim,vdim", [(64, 128), (128, 64), (128, 256), (32, 32)])
def test_chunk_rejects_unsupported_head_dimensions_before_launch(monkeypatch, kdim, vdim):
    q = torch.empty(1, 3, 1, kdim, dtype=torch.bfloat16)
    v = torch.empty(1, 3, 1, vdim, dtype=torch.bfloat16)
    launch = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_kda_fwd", launch, raising=False)
    with pytest.raises(ValueError, match="K=V=64 or K=V=128"):
        kda.run_chunk_kda(
            q,
            q,
            v,
            q,
            torch.empty(1, 3, 1),
            torch.empty(1, 1, vdim, kdim),
            (0, 3),
            (0, 0),
            torch.zeros(1),
            torch.zeros(kdim),
            lower_bound=None,
        )
    launch.assert_not_called()


@pytest.mark.parametrize("device_descriptor", ["cu_seqlens", "chunk_indices"])
def test_chunk_requires_host_metadata_without_reading_device_values(monkeypatch, device_descriptor):
    q = torch.empty(1, 3, 1, 128, dtype=torch.bfloat16)
    metadata = {"cu_seqlens": (0, 3), "chunk_indices": (0, 0)}
    # Meta tensors have no readable values, exercising the check before tolist().
    metadata[device_descriptor] = torch.empty(2, device="meta", dtype=torch.int64)
    launch = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_kda_fwd", launch, raising=False)
    with pytest.raises(ValueError, match=f"{device_descriptor} must be prepared on CPU"):
        kda.run_chunk_kda(
            q,
            q,
            q,
            q,
            torch.empty(1, 3, 1),
            torch.empty(1, 1, 128, 128),
            metadata["cu_seqlens"],
            metadata["chunk_indices"],
            torch.zeros(1),
            torch.zeros(128),
            lower_bound=None,
        )
    launch.assert_not_called()
