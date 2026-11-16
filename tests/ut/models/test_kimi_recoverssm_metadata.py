# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder


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
            "vllm_ascend.models.kimi_k3.recoverssm_metadata.get_pcp_group",
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
