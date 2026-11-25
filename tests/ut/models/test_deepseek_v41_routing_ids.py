# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.ops.fused_moe import prepare_finalize as prepare_module
from vllm_ascend.ops.fused_moe.router import fused_topk_router as router_module


def _router(*, hash_routing=True, vision=False):
    token_ids = torch.arange(1024)
    table = torch.stack((token_ids % 4, (token_ids + 1) % 4), dim=-1).to(torch.int32)
    return router_module.AscendFusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        scoring_func="sqrtsoftplus",
        tid2eid=table if hash_routing else None,
        bias_vl=torch.zeros(4) if vision else None,
    )


@pytest.fixture(autouse=True)
def _mock_moe_config(monkeypatch):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A2))


def _config(*, dp=2, pcp=2, tp=2, ep=False, microbatch=False, max_tokens=8, capture_tokens=9):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            enable_expert_parallel=ep,
            world_size_across_dp=dp * pcp * tp,
            pipeline_parallel_size=1,
            use_ubatching=microbatch,
        ),
        model_config=SimpleNamespace(get_num_experts=lambda: 256),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=capture_tokens),
    )


def _model(routers, *, replay_start=None, start=0):
    model = object.__new__(model_module.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.start_layer = start
    model.end_layer = len(routers)
    model.decoder_replay_start = len(routers) if replay_start is None else replay_start
    model.layers = [SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace(router=router))) for router in routers]
    return model


@pytest.mark.parametrize("capture_tokens,expected_local_capacity", [(None, 8), (9, 12)])
def test_cache_setup_supports_dp_pcp_and_graph_capacity(capture_tokens, expected_local_capacity):
    routers = [_router(hash_routing=False), _router(), _router(), _router(hash_routing=False, vision=True)]
    model = _model(routers)
    dp, pcp = 2, 2
    model._init_routing_input_ids_cache(_config(dp=dp, pcp=pcp, tp=4, max_tokens=5, capture_tokens=capture_tokens))

    assert routers[0].routing_input_ids_buffer is None
    buffer = routers[1].routing_input_ids_buffer
    assert buffer.dtype == torch.int64
    assert buffer.shape == (expected_local_capacity * dp * pcp,)
    assert [router.is_routing_input_ids_producer for router in routers] == [False, True, False, False]
    assert routers[2].routing_input_ids_buffer is buffer
    assert routers[3].routing_input_ids_buffer is buffer
    assert list(model.buffers()) == [buffer]
    assert not model.state_dict()


@pytest.mark.parametrize(
    "device,ep_size,max_tokens,expected_buffer",
    [
        (AscendDeviceType.A3, 16, 12, False),
        (AscendDeviceType.A3, 1, 12, True),
        (AscendDeviceType.A2, 16, 8, False),
        (AscendDeviceType.A2, 16, 12, True),
    ],
    ids=["a3-ep", "a3-single-rank-ep", "a2-mc2-only", "a2-mc2-and-allgather"],
)
def test_cache_setup_requires_reachable_allgather(monkeypatch, device, ep_size, max_tokens, expected_buffer):
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: get_hardware_profile(device))
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 8)
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: False)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=0))

    router = _router()
    model = _model([router])
    model._init_routing_input_ids_cache(
        _config(
            dp=ep_size // 4 if ep_size > 1 else 1,
            pcp=1,
            tp=4 if ep_size > 1 else 1,
            ep=True,
            max_tokens=max_tokens,
            capture_tokens=None,
        )
    )

    assert router.is_routing_input_ids_producer == expected_buffer
    assert (router.routing_input_ids_buffer is not None) == expected_buffer
    assert len(list(model.buffers())) == int(expected_buffer)


def test_cache_setup_skips_microbatches_and_layers_without_id_routing():
    routers = [_router(), _router()]
    model = _model(routers)
    model._init_routing_input_ids_cache(_config(microbatch=True))
    assert all(router.routing_input_ids_buffer is None for router in routers)
    assert not list(model.buffers())

    routers = [_router(hash_routing=False) for _ in range(2)]
    model = _model(routers)
    model._init_routing_input_ids_cache(_config())
    assert all(router.routing_input_ids_buffer is None for router in routers)
    assert not list(model.buffers())


def test_cache_setup_separates_pp_ownership_replay_and_model_instances():
    model_routers = []
    for _ in range(2):
        routers = [_router(), _router(), _router(), _router(hash_routing=False), _router(), _router()]
        model = _model(routers, replay_start=3, start=1)
        # A PP placeholder outside the owned slice must never be inspected.
        model.layers[0] = torch.nn.Identity()
        model._init_routing_input_ids_cache(_config())
        model_routers.append(routers)
        assert routers[0].routing_input_ids_buffer is None
        assert routers[1].is_routing_input_ids_producer
        assert routers[2].routing_input_ids_buffer is routers[1].routing_input_ids_buffer
        assert routers[3].routing_input_ids_buffer is None
        assert routers[4].is_routing_input_ids_producer
        assert routers[5].routing_input_ids_buffer is routers[4].routing_input_ids_buffer
        assert routers[4].routing_input_ids_buffer is not routers[1].routing_input_ids_buffer
    assert model_routers[0][1].routing_input_ids_buffer is not model_routers[1][1].routing_input_ids_buffer


def _fake_hash(**kwargs):
    ids = kwargs["input_ids"]
    selected = kwargs["tid2eid"][torch.where(ids == -1, 0, ids)].to(torch.int64)
    scores = torch.nn.functional.softplus(kwargs["x"]).sqrt()
    return scores.gather(1, selected), selected.to(torch.int32), None


def _peer_gather(offset):
    def gather(tensor, dim=0):
        # Simulate a peer with different token IDs while retaining padding
        # zeros and the -1 sentinel on both ranks.
        peer = torch.where(tensor > 0, tensor + offset, tensor)
        return torch.cat((tensor, peer), dim=dim)

    return gather


@pytest.mark.parametrize("dp,pcp,dtype", [(2, 1, torch.int32), (1, 2, torch.int64)])
def test_hash_routing_reuses_dp_pcp_layout_and_refreshes_each_forward(monkeypatch, dp, pcp, dtype):
    """Use the real prepare path; only peer communication and the NPU hash op are mocked."""
    monkeypatch.setattr(prepare_module, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    context = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    monkeypatch.setattr(router_module, "_EXTRA_CTX", context)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", context)
    monkeypatch.setattr(router_module.torch.ops._C_ascend, "moe_gating_top_k_hash", _fake_hash, raising=False)
    dp_group = SimpleNamespace(all_gather=_peer_gather(100))
    pcp_group = SimpleNamespace(all_gather=_peer_gather(200))
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: pcp_group)
    config = SimpleNamespace(is_sequence_parallel=False, dp_size=dp, pcp_size=pcp, dp_group=dp_group)
    routers = [_router() for _ in range(4)]
    uncached = _router()
    buffer = torch.full((64,), -99, dtype=torch.int64)
    for index, router in enumerate(routers):
        router.routing_input_ids_buffer = buffer
        router.is_routing_input_ids_producer = index == 0
    input_storage = torch.empty(4, dtype=dtype)

    # The first two forwards reuse both shape and input address; later ones
    # shrink and grow to detect stale values and active-row slicing errors.
    retained_view = buffer.view(dtype)[:0]
    for forward_index, num_tokens in enumerate((3, 3, 0, 1, 4)):
        input_storage.copy_(torch.arange(4, dtype=dtype) + 10 + 10 * forward_index)
        input_storage[0] = -1
        input_ids = input_storage[:num_tokens]
        original_ids = input_ids.clone()
        local_hidden = input_ids.float().unsqueeze(1).repeat(1, 4)
        local_logits = torch.arange(num_tokens * 4, dtype=torch.float32).view(num_tokens, 4) / 10
        context.max_tokens_across_dp = num_tokens + 1
        dp_rows = (num_tokens + 1) * dp if dp > 1 else num_tokens
        context.max_tokens_across_pcp = dp_rows + 1
        for layer_index, router in enumerate(routers):
            prepare = prepare_module.PrepareAndFinalizeWithAllGather(config)
            gather = MagicMock(wraps=prepare.all_gather_input_ids)
            monkeypatch.setattr(prepare, "all_gather_input_ids", gather)
            context.moe_comm_method = SimpleNamespace(prepare_finalize=prepare)
            prepared = prepare.prepare(local_hidden, local_logits)
            expected_ids = prepared.hidden_states[:, 0].to(dtype)
            actual = router._compute_routing(
                prepared.hidden_states, prepared.router_logits, torch.int32, input_ids=input_ids
            )
            assert gather.call_count == int(layer_index == 0)
            if layer_index == len(routers) - 1:
                if forward_index == 0:
                    retained_view = router._prepare_routing_input_ids(input_ids, expected_ids.numel())
                    assert retained_view.data_ptr() == buffer.data_ptr()
                    gather.assert_not_called()
                elif forward_index == 1:
                    # A consumer holding the old view sees the producer's
                    # second update without recomputing its routing IDs.
                    torch.testing.assert_close(retained_view, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(buffer.view(dtype)[: expected_ids.numel()], expected_ids, rtol=0, atol=0)
            expected = uncached._compute_routing(
                prepared.hidden_states, prepared.router_logits, torch.int32, input_ids=input_ids
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(input_ids, original_ids, rtol=0, atol=0)


def test_sp_path_gathers_once_after_layer_prepare(monkeypatch):
    monkeypatch.setattr(prepare_module, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    context = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    monkeypatch.setattr(router_module, "_EXTRA_CTX", context)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", context)
    seen_id_gathers = []
    layer_context_ready = False

    def gather(value):
        assert layer_context_ready
        if value.ndim == 1:
            seen_id_gathers.append(value.clone())
        # Four equal SP shards. Preserve a nontrivial peer order.
        return torch.cat([value, value.flip(0), value, value.flip(0)], dim=0)

    monkeypatch.setattr(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", gather)
    config = SimpleNamespace(is_sequence_parallel=True, dp_size=2, pcp_size=2)
    producer, consumer = _router(), _router()
    buffer = torch.full((16,), -99, dtype=torch.int64)
    for router in (producer, consumer):
        router.routing_input_ids_buffer = buffer
    producer.is_routing_input_ids_producer = True
    input_ids = torch.tensor([-1, 19], dtype=torch.int32)
    for router in (producer, consumer):
        layer_context_ready = True
        prepare = prepare_module.PrepareAndFinalizeWithAllGather(config)
        context.moe_comm_method = SimpleNamespace(prepare_finalize=prepare)
        prepared = prepare.prepare(input_ids.float().unsqueeze(1).repeat(1, 4), torch.ones(2, 4))
        actual = router._prepare_routing_input_ids(input_ids, prepared.router_logits.shape[0])
        torch.testing.assert_close(actual, prepared.hidden_states[:, 0].to(input_ids.dtype))
        layer_context_ready = False
    assert len(seen_id_gathers) == 1
    assert seen_id_gathers[0].dtype == input_ids.dtype


def test_mc2_does_not_consume_or_update_cached_ids(monkeypatch):
    producer, consumer = _router(), _router()
    buffer = torch.full((8,), 73, dtype=torch.int64)
    for router in (producer, consumer):
        router.routing_input_ids_buffer = buffer
    producer.is_routing_input_ids_producer = True
    gather = MagicMock(side_effect=lambda ids: torch.cat((ids, ids.flip(0))))
    split = MagicMock(side_effect=lambda ids: ids[:2])
    context = SimpleNamespace(
        moe_comm_type=MoECommType.MC2,
        moe_comm_method=SimpleNamespace(
            pad_and_split_input_ids=split,
            prepare_finalize=SimpleNamespace(all_gather_input_ids=gather),
        ),
    )
    monkeypatch.setattr(router_module, "_EXTRA_CTX", context)
    input_ids = torch.tensor([-1, 12, 25, 37], dtype=torch.int32)
    for router in (producer, consumer):
        actual = router._prepare_routing_input_ids(input_ids, 2)
        torch.testing.assert_close(actual, torch.tensor([-1, 12], dtype=input_ids.dtype))
    assert split.call_count == 2
    gather.assert_not_called()
    assert torch.equal(buffer, torch.full_like(buffer, 73))

    # A later all-gather forward must overwrite the old shared storage even
    # though no Python model-level cache reset ran between these forwards.
    context.moe_comm_type = MoECommType.ALLGATHER
    expected = torch.tensor([-1, 12, 25, 37, 37, 25, 12, -1], dtype=input_ids.dtype)
    for router in (producer, consumer):
        actual = router._prepare_routing_input_ids(input_ids, 8)
        torch.testing.assert_close(actual, expected)
    gather.assert_called_once()


def test_replay_prefix_and_tail_refresh_their_own_rows(monkeypatch):
    routers = [_router() for _ in range(4)]
    model = _model(routers, replay_start=2)
    model._init_routing_input_ids_cache(_config(tp=1))
    gather = MagicMock(side_effect=lambda ids: torch.cat((ids, ids + 100)))
    context = SimpleNamespace(
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(all_gather_input_ids=gather)),
    )
    monkeypatch.setattr(router_module, "_EXTRA_CTX", context)
    for offset in (0, 10):
        prefix_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32) + offset
        prefix = routers[0]._prepare_routing_input_ids(prefix_ids, 8)
        torch.testing.assert_close(routers[1]._prepare_routing_input_ids(prefix_ids, 8), prefix)
        prefix_snapshot = prefix.clone()
        replay_ids = prefix_ids[-2:]
        tail = routers[2]._prepare_routing_input_ids(replay_ids, 4)
        torch.testing.assert_close(routers[3]._prepare_routing_input_ids(replay_ids, 4), tail)
        torch.testing.assert_close(tail, torch.cat((replay_ids, replay_ids + 100)))
        torch.testing.assert_close(prefix, prefix_snapshot)
    assert gather.call_count == 4
