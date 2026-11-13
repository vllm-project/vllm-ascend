# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention import gdn_recoverssm_metadata as metadata_module
from vllm_ascend.attention.gdn_attn_builder import (
    AscendGDNAttentionMetadataBuilder,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)
from vllm_ascend.attention.gdn_recoverssm_metadata import (
    AscendGDNRecoverSSMAttentionBackend,
    AscendGDNRecoverSSMCommitMetadata,
    AscendGDNRecoverSSMMetadata,
    AscendGDNRecoverSSMMetadataBuilder,
)
from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder


def test_gdn_builder_uses_gdn_base_without_kda_inheritance():
    assert AscendGDNRecoverSSMMetadataBuilder.__bases__ == (AscendGDNAttentionMetadataBuilder,)
    assert not issubclass(AscendGDNRecoverSSMMetadataBuilder, KimiK3KDAMetadataBuilder)
    assert AscendKDARecoverSSMMetadataBuilder.__bases__ == (KimiK3KDAMetadataBuilder,)
    assert (
        AscendGDNRecoverSSMMetadataBuilder._normalize_graph_padding
        is AscendKDARecoverSSMMetadataBuilder._normalize_graph_padding
    )


def test_gdn_recoverssm_backend_identity():
    assert AscendGDNRecoverSSMMetadataBuilder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
    assert AscendGDNRecoverSSMAttentionBackend.get_name() == "GDN_RECOVERSSM"
    assert AscendGDNRecoverSSMAttentionBackend.get_builder_cls() is AscendGDNRecoverSSMMetadataBuilder


def test_build_forces_active_decodes_onto_recoverssm_path(monkeypatch):
    builder = AscendGDNRecoverSSMMetadataBuilder.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder.use_spec_decode = True
    builder._capturing = False
    builder._ones = torch.ones(3, dtype=torch.int32)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
    builder.kv_cache_spec = SimpleNamespace(block_size=16)
    builder._get_context = Mock(return_value=object())

    starts = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=3,
        is_prefilling=torch.tensor([False, False, True]),
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        block_table_tensor=torch.tensor([[4], [5], [6]], dtype=torch.int32),
        compute_num_computed_tokens=Mock(),
        pcp_context=object(),
    )
    state_indices = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    conv = GDNSpecCausalConv1dMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        cache_indices=state_indices,
        num_accepted_tokens=torch.tensor([3, 2], dtype=torch.int32),
    )
    base_metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=2,
        num_actual_tokens=2,
        spec_query_start_loc=conv.query_start_loc,
        spec_state_indices_tensor=state_indices,
        num_accepted_tokens=conv.num_accepted_tokens,
    )
    base_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=conv,
        actual_seq_lengths=None,
    )
    parent_build = Mock(return_value=base_metadata)
    monkeypatch.setattr(AscendGDNAttentionMetadataBuilder, "build", parent_build)

    result = builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor([3, 2, 1], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([-1, 0, -1], dtype=torch.int32),
    )

    assert isinstance(result, AscendGDNRecoverSSMMetadata)
    assert parent_build.call_args.kwargs["num_decode_draft_tokens_cpu"].tolist() == [1, 1, -1]
    assert result.spec_state_indices_tensor.tolist() == [[10], [20]]
    assert result.num_accepted_tokens.tolist() == [1, 1]
    assert result.spec_decode_metadata.spec_causal_conv1d.cache_indices.tolist() == [10, 20]
    assert result.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens.tolist() == [1, 1]
    assert result.recoverssm_commit.request_indices is None
    assert result.pcp_context is common.pcp_context


def test_commit_uses_gdn_context_and_returns_align_postprocess_metadata():
    context = Mock()
    block_table = torch.tensor([[7], [8]], dtype=torch.int32)
    num_computed_tokens = torch.tensor([16, 32], dtype=torch.int32)
    request_indices = torch.tensor([1, 0], dtype=torch.int32)
    commit = AscendGDNRecoverSSMCommitMetadata(
        state_indices=torch.tensor([[3], [4]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        request_indices=request_indices,
        block_table=block_table,
        num_computed_tokens=num_computed_tokens,
        block_size=64,
    )
    metadata = AscendGDNRecoverSSMMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=3,
        num_actual_tokens=3,
        recoverssm_commit=commit,
        recoverssm_context=context,
    )
    accepted = torch.tensor([2, 1], dtype=torch.int32)

    postprocess = metadata.commit_recoverssm_state(accepted)

    context.commit.assert_called_once()
    args = context.commit.call_args.args
    kwargs = context.commit.call_args.kwargs
    assert args[0] is accepted
    torch.testing.assert_close(args[1], torch.tensor([3, 4], dtype=torch.int32))
    torch.testing.assert_close(args[2], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert kwargs["request_indices"] is request_indices
    assert kwargs["block_table"] is block_table
    assert kwargs["num_computed_tokens"] is num_computed_tokens
    assert kwargs["mamba_block_size"] == 64
    assert postprocess.num_spec_decodes == 2
    assert postprocess.request_indices is request_indices
    assert postprocess.block_table is block_table
    assert postprocess.num_computed_tokens is num_computed_tokens
    assert postprocess.block_size == 64


def test_context_factory_uses_gdn_pages_without_global_kda_patch(monkeypatch):
    builder = AscendGDNRecoverSSMMetadataBuilder.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder._context = None
    builder.layer_names = ["gdn.1", "gdn.2"]
    builder.num_spec = 4
    layers = [object(), object()]
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=dict(zip(builder.layer_names, layers))),
        scheduler_config=SimpleNamespace(max_num_seqs=12),
    )
    create = Mock(return_value=object())
    monkeypatch.setattr(metadata_module.GDNRecoverSSMCommitContext, "create", create)
    assert builder._get_context() is create.return_value
    assert builder._get_context() is create.return_value
    create.assert_called_once_with(layers, spec_query_len=5, max_num_reqs=12)
