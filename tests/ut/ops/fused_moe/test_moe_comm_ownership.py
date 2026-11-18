# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as comm


@pytest.mark.parametrize("ep_size,deepgemm", [(1, False), (8, False), (8, True)])
def test_model_snapshots_and_layer_lookup_survive_draft_registration(monkeypatch, ep_size, deepgemm):
    monkeypatch.setattr(comm, "_MoECommMethods", {})
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(deepgemm_mega_moe=deepgemm))
    for name in ("AllGatherCommImpl", "AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl", "DeepGemmCommImpl"):
        monkeypatch.setattr(comm, name, lambda config: SimpleNamespace(moe_config=config))
    target = SimpleNamespace(ep_size=ep_size, num_experts=384, num_local_experts=384 // ep_size)
    draft = SimpleNamespace(ep_size=ep_size, num_experts=128, num_local_experts=128 // ep_size)
    comm.setup_moe_comm_method(target)
    target_snapshot = {kind: comm.get_moe_comm_method(kind) for kind in MoECommType}
    comm.setup_moe_comm_method(draft)
    draft_snapshot = {kind: comm.get_moe_comm_method(kind) for kind in MoECommType}
    if deepgemm:
        kinds = [MoECommType.DEEPGEMM]
        assert target_snapshot[MoECommType.ALLTOALL] is None
    else:
        kinds = [kind for kind in MoECommType if kind != MoECommType.DEEPGEMM]
        if ep_size == 1:
            kinds = [MoECommType.ALLGATHER]
    for kind in kinds:
        assert target_snapshot[kind] is not None
        assert draft_snapshot[kind] is not None
        assert target_snapshot[kind] is comm.get_moe_comm_method(kind, target)
        assert draft_snapshot[kind] is comm.get_moe_comm_method(kind, draft)
        assert target_snapshot[kind] is not draft_snapshot[kind]


def test_deepgemm_uses_common_prepare_execute_finalize_contract(monkeypatch):
    prepare_finalize = Mock()
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithAll2All", lambda config: prepare_finalize)
    backend = comm.DeepGemmCommImpl(SimpleNamespace())
    assert backend.token_dispatcher is None
    tokens, logits, weights, ids, result = (object() for _ in range(5))
    layer = SimpleNamespace(_deepgemm_v41=SimpleNamespace(apply=Mock(return_value=result)))
    backend.prepare(tokens, logits)
    prepare_finalize.prepare.assert_called_once()
    inputs = SimpleNamespace(layer=layer, hidden_states=tokens, topk_weights=weights, topk_ids=ids)
    assert backend.fused_experts(inputs) is result
    layer._deepgemm_v41.apply.assert_called_once_with(layer, tokens, weights, ids)
    backend.finalize(result, reduce_results=False)
    prepare_finalize.finalize.assert_called_once_with(result, False, None)
    backend.pad_and_split_input_ids(ids)
    prepare_finalize.pad_and_split_input_ids.assert_called_once_with(ids)
