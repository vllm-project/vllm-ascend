# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_ascend.ops import gdn_recoverssm_metadata as metadata_module
from vllm_ascend.ops.gdn_recoverssm_metadata import AscendGDNRecoverSSMMetadataBuilder


def test_prefill_mixed_with_decode_has_both_conv_descriptors(monkeypatch):
    builder = AscendGDNRecoverSSMMetadataBuilder.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder.use_full_cuda_graph = False
    starts = torch.tensor([0, 1, 4], dtype=torch.int32)
    m = SimpleNamespace(query_start_loc_cpu=starts, query_start_loc=starts)
    metadata = SimpleNamespace(
        num_spec_decodes=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        non_spec_query_start_loc=starts,
        non_spec_state_indices_tensor=torch.tensor([2, 5], dtype=torch.int32),
        has_initial_state=torch.tensor([True, False]),
    )
    chunk_builder = Mock(return_value=object())
    monkeypatch.setattr(metadata_module, "_build_non_spec_chunked_prefill_metadata", chunk_builder)
    builder._build_conv_metadata(m, metadata, None)
    assert metadata.non_spec_prefill_metadata.chunk is chunk_builder.return_value
    assert metadata.non_spec_decode_metadata.causal_conv1d.query_start_loc.tolist() == [0, 1]
    assert metadata.non_spec_decode_metadata.causal_conv1d.cache_indices.tolist() == [2]
    assert metadata.prefill_state_indices.tolist() == [5]
    assert metadata.prefill_has_initial_state.tolist() == [False]
    assert metadata.prefill_query_start_loc.tolist() == [0, 3]
    # Omitting build_device_metadata uses its True default for the Triton fallback.
    assert chunk_builder.call_args.args[1].tolist() == [0, 3]
    assert chunk_builder.call_args.kwargs == {}


def test_context_factory_uses_gdn_pages_without_global_kda_patch(monkeypatch):
    builder = AscendGDNRecoverSSMMetadataBuilder.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder.recoverssm_context = None
    builder.layer_names = ["gdn.1", "gdn.2"]
    builder.num_spec = 4
    layers = [object(), object()]
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=dict(zip(builder.layer_names, layers))),
        scheduler_config=SimpleNamespace(max_num_seqs=12),
    )
    create = Mock(return_value=object())
    monkeypatch.setattr(metadata_module.GDNRecoverSSMCommitContext, "create", create)
    assert builder._get_recoverssm_context() is create.return_value
    assert builder._get_recoverssm_context() is create.return_value
    create.assert_called_once_with(layers, spec_query_len=5, max_num_reqs=12)
