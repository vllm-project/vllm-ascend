# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

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
