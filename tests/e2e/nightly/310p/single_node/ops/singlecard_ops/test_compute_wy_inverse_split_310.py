# SPDX-License-Identifier: Apache-2.0
"""Gate-tripped WY regression: independent CPU solve and repeated native calls."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64
HEAD_DIM = 128
COSINE_THRESHOLD = 0.999999
REPEAT_COUNT = 5
KEY_HEADS = 2
VALUE_HEADS = 4
ROW_SUM_THRESHOLD = 0.75


def _minimum_chunk_cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    assert actual.shape == expected.shape, "Shape mismatch must not be hidden by reshaping"
    assert torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item()
    actual_rows = actual.double().reshape(-1, CHUNK_SIZE * HEAD_DIM)
    expected_rows = expected.double().reshape_as(actual_rows)
    actual_norm = actual_rows.norm(dim=-1)
    expected_norm = expected_rows.norm(dim=-1)
    both_zero = (actual_norm == 0) & (expected_norm == 0)
    similarity = torch.nn.functional.cosine_similarity(actual_rows, expected_rows, dim=-1, eps=1e-30)
    similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
    return similarity.min().item()


def _make_inputs_cpu(batch: int, tokens: int, decay: float) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(42)
    q = (torch.randn(batch, tokens, KEY_HEADS, HEAD_DIM, generator=generator) * 0.05).half()
    direction = torch.nn.functional.normalize(torch.randn(batch, 1, KEY_HEADS, HEAD_DIM, generator=generator), dim=-1)
    k = direction.expand(batch, tokens, KEY_HEADS, HEAD_DIM).contiguous().half()
    v = (torch.randn(batch, tokens, VALUE_HEADS, HEAD_DIM, generator=generator) * 0.1).half()
    g = torch.full((batch, tokens, VALUE_HEADS), -decay, dtype=torch.float32)
    beta = torch.full((batch, tokens, VALUE_HEADS), 0.8, dtype=torch.float16)
    return q, k, v, g, beta


def _reference_cpu(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta = inputs
    batch, tokens = k.shape[:2]

    # Independent FP64 CPU oracle on the actual FP16 inputs. No NPU fallback
    # and no dependency on the native inverse/split implementation.
    key = k.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2).transpose(1, 2).double()
    key = key.reshape(batch, VALUE_HEADS, -1, CHUNK_SIZE, HEAD_DIM)
    value = v.transpose(1, 2).double().reshape_as(key)
    beta_rows = beta.transpose(1, 2).double().reshape(batch, VALUE_HEADS, -1, CHUNK_SIZE)
    cumulative_g = g.transpose(1, 2).double().reshape_as(beta_rows).cumsum(dim=-1)
    lower_decay = (cumulative_g.unsqueeze(-1) - cumulative_g.unsqueeze(-2)).tril(diagonal=-1).exp()
    key_beta = key * beta_rows.unsqueeze(-1)
    a = -(key_beta @ key.transpose(-1, -2) * lower_decay).tril(diagonal=-1)
    assert (a.abs().sum(dim=-1).amax(dim=-1) >= ROW_SUM_THRESHOLD).all(), "Inputs must exercise the large-row-sum gate"
    system = torch.eye(CHUNK_SIZE, dtype=torch.float64) - a
    w_ref = torch.linalg.solve_triangular(system, key_beta * cumulative_g.exp().unsqueeze(-1), upper=False)
    u_ref = torch.linalg.solve_triangular(system, value * beta_rows.unsqueeze(-1), upper=False)
    w_ref = w_ref.reshape(batch, VALUE_HEADS, tokens, HEAD_DIM).half()
    u_ref = u_ref.reshape_as(w_ref).half()
    expected_g = cumulative_g.reshape(batch, VALUE_HEADS, tokens).float()
    return q.transpose(1, 2), k.transpose(1, 2), w_ref, u_ref, expected_g


@pytest.mark.parametrize("batch,tokens", [(1, 64), (1, 832), (1, 1280), (4, 128), (4, 2048)])
@pytest.mark.parametrize("decay", [0.001, 0.1, 0.3])
def test_inverse_split_large_row_sum_repeated(batch: int, tokens: int, decay: float):
    enable_custom_op()
    cpu_inputs = _make_inputs_cpu(batch, tokens, decay)
    reference = _reference_cpu(cpu_inputs)
    inputs = tuple(tensor.npu() for tensor in cpu_inputs)
    first: tuple[torch.Tensor, ...] | None = None
    for _ in range(REPEAT_COUNT):
        output = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(*inputs, CHUNK_SIZE)
        torch.npu.synchronize()
        current = tuple(tensor.cpu() for tensor in output)
        assert len(current) == 5
        assert all(torch.isfinite(tensor).all().item() for tensor in current)
        torch.testing.assert_close(current[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(current[1], reference[1], rtol=0, atol=0)
        torch.testing.assert_close(current[4], reference[4], rtol=1e-5, atol=1e-5)
        assert _minimum_chunk_cosine(current[2], reference[2]) >= COSINE_THRESHOLD
        assert _minimum_chunk_cosine(current[3], reference[3]) >= COSINE_THRESHOLD
        if first is not None:
            assert all(torch.equal(left, right) for left, right in zip(current, first))
        else:
            first = current
