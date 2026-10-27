# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.models.glm5next.model import (
    Glm5NextSPPPError,
    _normalize_sp_pp_state_fake,
    _normalize_sp_pp_state_impl,
)


@pytest.mark.parametrize("num_tokens", [1, 3, 4, 5, 17])
@pytest.mark.parametrize("tp_rank", range(4))
@pytest.mark.parametrize("tail_shape", [(8,), (4, 8), (4, 1), (4, 4)])
def test_normalization_matches_full_gather_trim_and_reshard(num_tokens, tp_rank, tail_shape):
    tp_size = 4
    local_tokens = (num_tokens + tp_size - 1) // tp_size
    torch.manual_seed(0)
    all_states = torch.randn(tp_size * local_tokens, *tail_shape)
    # Invalid SP rows may contain outputs from biased mHC kernels.
    all_states[num_tokens:] = float("nan")
    if all_states.shape[0] > num_tokens:
        all_states[-1] = float("inf")
    local = all_states[tp_rank * local_tokens : (tp_rank + 1) * local_tokens].clone()
    original = local.clone()
    gathered_trimmed = all_states[:num_tokens]
    full_padded = torch.cat((gathered_trimmed, torch.zeros(tp_size * local_tokens - num_tokens, *tail_shape)), dim=0)
    expected = full_padded[tp_rank * local_tokens : (tp_rank + 1) * local_tokens]

    actual = _normalize_sp_pp_state_impl(local, num_tokens, tp_rank, tp_size)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(local, original, equal_nan=True)
    assert actual.data_ptr() != local.data_ptr()


def test_normalization_preserves_original_runner_padding():
    # Physical T includes runner padding. Only the additional SP row is zero.
    local = torch.tensor([[7.0], [float("nan")]])
    result = _normalize_sp_pp_state_impl(local, 5, 2, 4)
    torch.testing.assert_close(result, torch.tensor([[7.0], [0.0]]))


def test_normalization_rejects_an_incorrect_local_shape():
    with pytest.raises(Glm5NextSPPPError, match="local physical token count"):
        _normalize_sp_pp_state_impl(torch.ones(3, 8), 5, 0, 4)


def test_normalization_fake_preserves_metadata_without_alias():
    with FakeTensorMode():
        source = torch.empty(2, 4, 8, dtype=torch.bfloat16)
        result = _normalize_sp_pp_state_fake(source, 5, 2, 4)
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert result.device == source.device
        assert result is not source


def test_registered_normalization_has_fake_dispatch_and_symbolic_token_schema():
    schema = str(torch.ops.vllm.glm5next_sp_pp_normalize.default._schema)
    assert "SymInt num_tokens" in schema
    with FakeTensorMode():
        source = torch.empty(2, 4, 8, dtype=torch.bfloat16)
        result = torch.ops.vllm.glm5next_sp_pp_normalize(source, 5, 2, 4)
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert result.device == source.device
        assert result is not source
