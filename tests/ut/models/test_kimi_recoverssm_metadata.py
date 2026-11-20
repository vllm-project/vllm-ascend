# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.ops.test_gdn_attn_builder import BatchSpec, _make_builder, create_common_attn_metadata
from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder
from vllm_ascend.ops import kda_pcp


def test_context_is_reused_only_while_all_layer_buffers_remain_bound(monkeypatch):
    layers = [object(), object()]
    builder = AscendKimiKDARecoverSSMMetadataBuilder.__new__(AscendKimiKDARecoverSSMMetadataBuilder)
    builder.layer_names = ["layer0", "layer1"]
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=dict(zip(builder.layer_names, layers))),
        num_speculative_tokens=7,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    builder.recoverssm_context = None
    old_context = SimpleNamespace(is_bound_to=MagicMock(return_value=True))
    new_context = SimpleNamespace(is_bound_to=MagicMock(return_value=True))
    create = MagicMock(side_effect=[old_context, new_context])
    monkeypatch.setattr("vllm_ascend.models.kimi_k3.recoverssm_metadata.KDARecoverSSMCommitContext.create", create)

    assert builder._get_recoverssm_context() is old_context
    create.assert_called_once_with(layers, spec_query_len=8, max_num_reqs=8)
    assert builder._get_recoverssm_context() is old_context
    old_context.is_bound_to.assert_called_once_with(layers)
    old_context.is_bound_to.return_value = False
    assert builder._get_recoverssm_context() is new_context
    assert create.call_count == 2


@pytest.mark.parametrize("pcp_size", [1, 2])
def test_recoverssm_pcp_builder_allocates_buffers_without_mutating_runner_config(pcp_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=4),
    )

    def init_base(self, spec, layers, vllm_config, device):
        self.decode_cudagraph_max_bs = vllm_config.compilation_config.max_cudagraph_capture_size

    with (
        patch.object(AscendKDARecoverSSMMetadataBuilder, "__init__", init_base),
        patch(
            "vllm_ascend.ops.kda_pcp.get_pcp_group",
            return_value=SimpleNamespace(rank_in_group=1),
        ),
    ):
        builder = AscendKimiKDARecoverSSMMetadataBuilder(None, [], config, torch.device("cpu"))

    assert config.compilation_config.max_cudagraph_capture_size == 4
    assert builder.decode_cudagraph_max_bs == 4 * pcp_size
    assert builder.pcp_rank == (1 if pcp_size > 1 else 0)
    assert builder._pcp_restore_idx.numel() == 16 * pcp_size
    assert builder._pcp_local_idx.numel() == 16


def test_recoverssm_pcp_build_attaches_permutations_and_padding_masks():
    builder = object.__new__(AscendKimiKDARecoverSSMMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 0
    builder._pcp_restore_idx = torch.zeros(8, dtype=torch.long)
    builder._pcp_restore_mask = torch.zeros(8, dtype=torch.bool)
    builder._pcp_local_idx = torch.zeros(4, dtype=torch.long)
    builder._pcp_local_mask = torch.zeros(4, dtype=torch.bool)
    context = SimpleNamespace(
        local_num_tokens=2,
        hidden_restore_idx=torch.tensor([0, 2, 1]),
        local_token_indices=torch.tensor([0, 2]),
    )
    metadata = SimpleNamespace(pcp_context=context)

    with patch.object(AscendKDARecoverSSMMetadataBuilder, "build", return_value=metadata) as build_base:
        result = builder.build("batch", fast_build=True)

    build_base.assert_called_once_with("batch", fast_build=True)
    assert result.pcp_restore_idx.tolist() == [0, 2, 1, 0]
    assert result.pcp_restore_mask.tolist() == [True, True, True, False]
    assert result.pcp_local_idx.tolist() == [0, 2]
    assert result.pcp_local_mask.tolist() == [True, True]


@pytest.mark.parametrize("for_capture", [False, True])
def test_recoverssm_real_builder_propagates_pcp_context(for_capture):
    device = torch.device("cpu")
    group = SimpleNamespace(rank_in_group=1)

    def make_recoverssm(spec, layers, config, device):
        config.parallel_config.prefill_context_parallel_size = 2
        config.cache_config.use_kda_recoverssm = True
        config.compilation_config.max_cudagraph_capture_size = 2
        config.compilation_config.static_forward_context = {name: object() for name in layers}
        builder = AscendKimiKDARecoverSSMMetadataBuilder(spec, layers, config, device)
        assert config.compilation_config.max_cudagraph_capture_size == 2
        return builder

    with patch.object(kda_pcp, "get_pcp_group", return_value=group):
        builder = _make_builder(
            device=device,
            num_heads=32,
            num_speculative_tokens=1,
            builder_cls=make_recoverssm,
            cudagraph_mode=CUDAGraphMode.FULL if for_capture else CUDAGraphMode.NONE,
        )
    # Avoid allocating model-layer state; the metadata builders themselves are real.
    recoverssm_context = SimpleNamespace(is_bound_to=lambda layers: True)
    builder.recoverssm_context = recoverssm_context
    common = create_common_attn_metadata(BatchSpec(seq_lens=[8] * 4, query_lens=[1] * 4), 16, device)
    context = SimpleNamespace(
        local_num_tokens=2,
        hidden_restore_idx=torch.tensor([0, 2, 1, 3]),
        local_token_indices=None if for_capture else torch.tensor([1, 3]),
        global_batch=SimpleNamespace(num_tokens=4),
    )
    common.pcp_context = context

    # Four global tokens exceed the rank-local capture limit of two, but fit
    # the builder's scaled limit. Capture dummies have no local_token_indices.
    assert builder.decode_cudagraph_max_bs == 4
    if for_capture:
        metadata = builder.build_for_cudagraph_capture(common)
    else:
        metadata = builder.build(0, common, torch.ones(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32))

    assert metadata.pcp_context is context
    assert metadata.recoverssm_context is recoverssm_context
    assert metadata.recoverssm_commit is not None
    assert metadata.pcp_restore_idx.tolist() == [0, 2, 1, 3]
    assert metadata.pcp_restore_mask.tolist() == [True, True, True, True]
    assert metadata.pcp_local_idx.tolist() == ([2, 3] if for_capture else [1, 3])
    assert metadata.pcp_local_mask.tolist() == [True, True]

    if for_capture:
        assert builder.use_full_cuda_graph
        assert metadata.spec_query_start_loc.data_ptr() == builder.spec_query_start_loc.data_ptr()
        addresses = [metadata.pcp_restore_idx.data_ptr(), metadata.pcp_local_idx.data_ptr()]
        context.hidden_restore_idx = torch.tensor([3, 1, 2])
        context.local_token_indices = torch.tensor([0])
        replay = builder.build(0, common, torch.ones(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32))
        assert [replay.pcp_restore_idx.data_ptr(), replay.pcp_local_idx.data_ptr()] == addresses
        assert replay.pcp_restore_idx.tolist() == [3, 1, 2, 0]
        assert replay.pcp_restore_mask.tolist() == [True, True, True, False]
        assert replay.pcp_local_idx.tolist() == [0, 0]
        assert replay.pcp_local_mask.tolist() == [True, False]
