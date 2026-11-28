# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as comm


@pytest.mark.parametrize("ep_size,deepgemm", [(1, False), (8, False), (8, True)])
def test_model_snapshots_and_layer_lookup_survive_draft_registration(monkeypatch, ep_size, deepgemm):
    monkeypatch.setattr(comm, "_MoECommMethods", {})
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(deepgemm_mega_moe=deepgemm))
    for name in (
        "AllGatherCommImpl",
        "AlltoAllCommImpl",
        "MC2CommImpl",
        "FusedMC2CommImpl",
        "DeepGemmFusedMC2CommImpl",
    ):
        monkeypatch.setattr(comm, name, lambda config, name=name: SimpleNamespace(moe_config=config, backend=name))
    monkeypatch.setattr(comm, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(comm, "deepgemm_device_supported", lambda _: True)
    target = SimpleNamespace(ep_group=object(), ep_size=ep_size, num_experts=384, num_local_experts=384 // ep_size)
    draft_experts = target.num_experts if deepgemm else 128
    draft = SimpleNamespace(ep_size=ep_size, num_experts=draft_experts, num_local_experts=draft_experts // ep_size)
    comm.setup_moe_comm_method(target, comm.QuantType.W4A8MXFP)
    target_snapshot = {kind: comm.get_moe_comm_method(kind) for kind in MoECommType}
    comm.setup_moe_comm_method(draft)
    draft_snapshot = {kind: comm.get_moe_comm_method(kind) for kind in MoECommType}
    if ep_size > 1:
        assert target_snapshot[MoECommType.FUSED_MC2].backend == (
            "DeepGemmFusedMC2CommImpl" if deepgemm else "FusedMC2CommImpl"
        )
        assert draft_snapshot[MoECommType.FUSED_MC2].backend == "FusedMC2CommImpl"
    kinds = list(MoECommType) if ep_size > 1 else [MoECommType.ALLGATHER]
    for kind in kinds:
        assert target_snapshot[kind] is not None
        assert draft_snapshot[kind] is not None
        assert target_snapshot[kind] is comm.get_moe_comm_method(kind, target)
        assert draft_snapshot[kind] is comm.get_moe_comm_method(kind, draft)
        assert target_snapshot[kind] is not draft_snapshot[kind]


def test_deepgemm_uses_common_prepare_execute_finalize_contract(monkeypatch):
    prepare_finalize = Mock()
    make_prepare = Mock(return_value=prepare_finalize)
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithAll2All", make_prepare)
    monkeypatch.setattr(comm, "TokenDispatcherWithMC2", Mock)
    backend = comm.DeepGemmFusedMC2CommImpl(SimpleNamespace())
    make_prepare.assert_called_once_with(backend.moe_config)
    tokens, logits, weights, ids, result = (object() for _ in range(5))
    layer = SimpleNamespace(_deepgemm_mega_moe=SimpleNamespace(apply=Mock(return_value=result)))
    backend.prepare(tokens, logits)
    prepare_finalize.prepare.assert_called_once_with(
        hidden_states=tokens, router_logits=logits, replace_allreduce=True, quant_type=comm.QuantType.NONE
    )
    inputs = SimpleNamespace(layer=layer, hidden_states=tokens, topk_weights=weights, topk_ids=ids)
    assert backend.fused_experts(inputs) is result
    layer._deepgemm_mega_moe.apply.assert_called_once_with(layer, tokens, weights, ids)
    backend.finalize(result, reduce_results=False)
    prepare_finalize.finalize.assert_called_once_with(result, False, None)
    backend.pad_and_split_input_ids(ids)
    prepare_finalize.pad_and_split_input_ids.assert_called_once_with(ids)


def test_local_prepare_preserves_an_empty_rank(monkeypatch):
    from vllm_ascend.ops.fused_moe import prepare_finalize as pf

    monkeypatch.setattr(pf, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pf, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pf, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    monkeypatch.setattr(comm, "TokenDispatcherWithMC2", Mock)
    local = comm.DeepGemmFusedMC2CommImpl(SimpleNamespace())
    tokens, logits, ids = torch.empty(0, 128), torch.empty(0, 16), torch.empty(0, dtype=torch.long)
    prepared = local.prepare(tokens, logits)
    assert prepared.hidden_states is tokens and prepared.router_logits is logits
    assert local.pad_and_split_input_ids(ids) is ids
    assert local.finalize(tokens, False) is tokens
