# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.glm5next.kda import Glm5NextGDNHostMetadataBuilder, Glm5NextPCPMetadataMixin
from vllm_ascend.models.glm5next.recoverssm_metadata import (
    AscendKDARecoverSSMMetadataBuilder,
    Glm5NextRecoverSSMMetadataBuilder,
)
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


def _make_permutation_builder(builder_cls=Glm5NextPCPMetadataMixin):
    builder = builder_cls.__new__(builder_cls)
    builder._pcp_size = 2
    builder._is_pcp_decode_sharded = True
    builder._pcp_gather_indices = torch.empty(16, dtype=torch.int64)
    return builder


def _make_permutation_context(local_tokens, global_tokens):
    return AscendPCPAttentionContext(
        global_batch=SimpleNamespace(is_dummy=True, num_tokens_after_padding=global_tokens),
        global_block_tables=(),
        global_slot_mappings=torch.empty(0),
        hidden_restore_idx=torch.arange(global_tokens),
        local_num_tokens=local_tokens,
    )


@pytest.mark.parametrize(
    "local_tokens,global_tokens,expected",
    [
        (2, 4, [0, 1, 2, 3]),
        (2, 3, [0, 1, 2, 0]),
        (3, 3, [0, 1, 2, 0, 1, 2]),
    ],
)
def test_glm_metadata_refreshes_capture_indices_without_mutating_pcp(local_tokens, global_tokens, expected):
    builder = _make_permutation_builder()
    capture_context = _make_permutation_context(local_tokens, global_tokens)
    captured = builder._attach_pcp_permutation(SimpleNamespace(pcp_context=capture_context))
    indices = captured.pcp_context.padded_gather_idx
    assert indices.tolist() == expected
    assert capture_context.padded_gather_idx is None
    # The graph keeps this address while subsequent batches change ownership.
    source = torch.arange(2 * local_tokens).flip(0) % global_tokens
    live_context = replace(
        capture_context,
        global_batch=SimpleNamespace(is_dummy=False),
        padded_gather_idx=source,
    )
    live = builder._attach_pcp_permutation(SimpleNamespace(pcp_context=live_context))
    assert live.pcp_context.padded_gather_idx.data_ptr() == indices.data_ptr()
    assert indices.data_ptr() != source.data_ptr()
    torch.testing.assert_close(indices, source)
    assert live_context.padded_gather_idx is source


@pytest.mark.parametrize("invalid", ["missing-layout", "short-layout", "capacity", "empty-local", "empty-global"])
def test_glm_metadata_rejects_invalid_permutations(invalid):
    builder = _make_permutation_builder()
    context = _make_permutation_context(2, 4)
    if invalid == "missing-layout":
        context.global_batch.is_dummy = False
    elif invalid == "short-layout":
        context = replace(context, padded_gather_idx=torch.arange(3))
    elif invalid == "capacity":
        builder._pcp_gather_indices = torch.empty(3, dtype=torch.int64)
    elif invalid == "empty-local":
        context = replace(context, local_num_tokens=0)
    else:
        context.global_batch.num_tokens_after_padding = 0
    with pytest.raises(ValueError, match="GLM KDA PCP"):
        builder._attach_pcp_permutation(SimpleNamespace(pcp_context=context))


@pytest.mark.parametrize(
    "builder_cls,base_cls",
    [
        pytest.param(Glm5NextGDNHostMetadataBuilder, AscendGDNHostMetadataBuilder, id="gdn"),
        pytest.param(Glm5NextRecoverSSMMetadataBuilder, AscendKDARecoverSSMMetadataBuilder, id="recoverssm"),
    ],
)
def test_glm_build_attaches_model_owned_permutation(builder_cls, base_cls):
    builder = _make_permutation_builder(builder_cls)
    context = _make_permutation_context(2, 4)
    common = SimpleNamespace()
    base_metadata = SimpleNamespace(pcp_context=context)
    with patch.object(base_cls, "build", return_value=base_metadata) as build:
        metadata = builder.build(0, common)
    build.assert_called_once_with(0, common)
    assert metadata is base_metadata
    assert metadata.pcp_context.padded_gather_idx.data_ptr() == builder._pcp_gather_indices.data_ptr()
    assert metadata.pcp_context.padded_gather_idx.tolist() == [0, 1, 2, 3]
    assert context.padded_gather_idx is None


def test_replicated_decode_skips_permutation_until_prefill():
    builder = _make_permutation_builder()
    builder._is_pcp_decode_sharded = False
    builder._pcp_gather_indices.fill_(-1)
    source = torch.tensor([0, 1, 0, 1])
    context = replace(
        _make_permutation_context(2, 2),
        global_batch=SimpleNamespace(is_dummy=False, is_prefilling_np=torch.tensor([False]), has_prefill=False),
        padded_gather_idx=source,
    )
    metadata = SimpleNamespace(pcp_context=context)
    assert builder._attach_pcp_permutation(metadata) is metadata
    assert metadata.pcp_context is context
    assert metadata.pcp_context.padded_gather_idx is source
    assert torch.all(builder._pcp_gather_indices == -1)

    # A later prefill must still refresh the persistent graph-owned storage.
    context.global_batch.is_prefilling_np[0] = True
    context.global_batch.has_prefill = True
    metadata = builder._attach_pcp_permutation(metadata)
    assert metadata.pcp_context is not context
    assert metadata.pcp_context.padded_gather_idx.data_ptr() == builder._pcp_gather_indices.data_ptr()
    torch.testing.assert_close(metadata.pcp_context.padded_gather_idx, source)
    assert context.padded_gather_idx is source
