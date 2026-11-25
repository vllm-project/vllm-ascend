# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm_ascend.ops.fused_moe import mega_moe_context as context
from vllm_ascend.ops.fused_moe import moe_comm_method as comms
from vllm_ascend.quantization.quant_type import QuantType


def _get_mega_moe_ccl_buffer_size(moe_expert_num, num_topk, server_num=0):
    return moe_expert_num * num_topk + server_num


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setattr(context, "_agree", lambda _, value: [value, value])
    monkeypatch.setattr(comms, "_MoECommMethods", {})
    scope = SimpleNamespace(device_group=object())
    calls = []
    allocated: set[object] = set()

    def factory(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        max_recv_token_num,
        dispatch_quant_mode,
        dispatch_quant_out_dtype,
    ):
        kwargs = locals().copy()
        # Model a device with only enough HBM for one communication window.
        assert not allocated
        allocated.add(group)
        calls.append(group)
        return SimpleNamespace(
            **kwargs,
            ep_world_size=8,
            context=torch.tensor([len(calls)]),
            topo_type=0,
            rank_num_per_server=8,
            rank_id=0,
            group_name="mc2",
            _ctx_manager=object(),
            destroy=Mock(side_effect=lambda: allocated.remove(group)),
            ccl_buffer_size=4096,
        )

    def layout(experts, topk, quant=QuantType.W4A8MXFP, hidden=64, intermediate=512, tokens=128):
        cfg = SimpleNamespace(num_experts=experts, num_local_experts=experts // 8, mc2_group=scope)

        def params(mode, dtype):
            return dict(
                num_experts=experts,
                num_topk=topk,
                num_max_tokens_per_rank=tokens,
                hidden=hidden,
                intermediate_hidden=intermediate,
                max_recv_token_num=0,
                dispatch_quant_mode=mode,
                dispatch_quant_out_dtype=dtype,
            )

        comm = SimpleNamespace(
            get_symm_buffer_for_mega_moe=factory,
            mega_moe_buffer_params=params,
            moe_config=cfg,
            mega_moe_symm_buffer=None,
        )
        return {context.ExpertLayout(quant, experts, cfg.num_local_experts, topk, hidden, intermediate): comm}

    return layout, scope, calls


def test_smaller_draft_one_factory_and_stable_context_after_restore(pair, monkeypatch):
    layout, scope, calls = pair
    main, draft = layout(384, 6), layout(128, 3)
    assert context.try_share_mega_moe_buffer(main, draft, scope, True)
    target, proposer = next(iter(main.values())), next(iter(draft.values()))
    owner, view = target.mega_moe_symm_buffer, proposer.mega_moe_symm_buffer
    captured = owner.context
    assert calls == [scope.device_group]
    assert (owner.num_topk, view.num_topk, view.num_experts) == (6, 3, 128)
    assert view.context is captured and view._ctx_manager is owner._ctx_manager
    barrier = Mock()
    monkeypatch.setattr(context.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(context.torch.npu, "current_device", lambda: 0)
    monkeypatch.setattr(comms, "get_mc2_group", lambda: scope)
    # A different MoE resource may cancel teardown after this one prepares.
    for index, group in enumerate([scope.device_group, object(), object()]):
        for method in (proposer, target):
            assert comms.FusedMC2CommImpl.prepare_hccl_teardown(method)
        scope.device_group = group
        for method in (proposer, target):
            assert comms.FusedMC2CommImpl.refresh_hccl_runtime_state(method)
        assert barrier.call_count == index and len(calls) == index + 1
        assert owner.context is view.context is captured
        assert captured.item() == index + 1 and view.group is owner.group is group
        if index == 0:
            owner.destroy.assert_not_called()


def test_restore_retries_barrier_with_existing_allocation(pair, monkeypatch):
    layout, scope, calls = pair
    main, draft = layout(384, 6), layout(128, 3)
    assert context.try_share_mega_moe_buffer(main, draft, scope, True)
    shared_buffer = next(iter(main.values()))._shared_mega_moe_buffer
    old_group, new_group = scope.device_group, object()
    scope.device_group = new_group
    monkeypatch.setattr(context.torch.distributed, "barrier", Mock(side_effect=[RuntimeError("barrier failed"), None]))
    monkeypatch.setattr(context.torch.npu, "current_device", lambda: 0)
    with pytest.raises(RuntimeError, match="barrier failed"):
        shared_buffer.restore(scope)
    assert shared_buffer.owner.group is old_group
    shared_buffer.restore(scope)
    assert calls == [old_group, new_group]
    assert shared_buffer.owner.group is shared_buffer.views[0].group is new_group


@pytest.mark.parametrize(
    "step,successful_steps,allocations,copied",
    [("release", 0, 1, False), ("validate", 1, 2, False), ("copy context", 2, 2, True)],
)
def test_peer_restore_failure_stops_before_next_collective(
    pair, monkeypatch, step, successful_steps, allocations, copied
):
    layout, scope, calls = pair
    main = layout(384, 6)
    assert context.try_share_mega_moe_buffer(main, layout(128, 3), scope, True)
    shared_buffer = next(iter(main.values()))._shared_mega_moe_buffer
    scope.device_group = object()
    peer_results = iter([None] * successful_steps + ["injected failure"])

    def agree(scope: object, local: object) -> list[object]:
        if isinstance(local, tuple):
            return [local, local]
        return [local, next(peer_results)]

    monkeypatch.setattr(context, "_agree", agree)
    barrier = Mock()
    monkeypatch.setattr(context.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(context.torch.npu, "current_device", lambda: 0)
    with pytest.raises(RuntimeError, match=f"{step} on rank 1: injected failure"):
        shared_buffer.restore(scope)
    barrier.assert_not_called()
    assert len(calls) == allocations
    assert shared_buffer.owner.context.item() == (2 if copied else 1)
    assert shared_buffer.owner.group is not scope.device_group


def test_restore_factory_failure_after_tag_creation_requires_new_group(pair, monkeypatch):
    layout, scope, calls = pair
    main = layout(384, 6)
    assert context.try_share_mega_moe_buffer(main, layout(128, 3), scope, True)
    shared_buffer = next(iter(main.values()))._shared_mega_moe_buffer
    factory = shared_buffer.factory
    tagged_groups = []
    failed_group = scope.device_group = object()

    def allocate(group, **params):
        tagged_groups.append(group)
        if group is failed_group:
            raise RuntimeError("device registration failed after creating the HCCL tag")
        return factory(group, **params)

    monkeypatch.setattr(shared_buffer, "factory", allocate)
    monkeypatch.setattr(context.torch.distributed, "barrier", Mock())
    monkeypatch.setattr(context.torch.npu, "current_device", lambda: 0)
    with pytest.raises(RuntimeError, match="device registration failed"):
        shared_buffer.restore(scope)
    with pytest.raises(RuntimeError, match="recreate the MC2 group"):
        shared_buffer.restore(scope)
    assert tagged_groups == [failed_group]

    scope.device_group = object()
    shared_buffer.restore(scope)
    assert tagged_groups == [failed_group, scope.device_group]
    assert len(calls) == 2
    assert shared_buffer.owner.group is shared_buffer.views[0].group is scope.device_group


def test_restore_partial_allocation_requires_new_group(pair, monkeypatch):
    layout, scope, calls = pair
    main = layout(384, 6)
    assert context.try_share_mega_moe_buffer(main, layout(128, 3), scope, True)
    shared_buffer = next(iter(main.values()))._shared_mega_moe_buffer
    scope.device_group = object()
    monkeypatch.setattr(context, "_agree", lambda _, state: [state, (False, True, False)])
    with pytest.raises(RuntimeError, match="recreate the MC2 group"):
        shared_buffer.restore(scope)
    shared_buffer.owner.destroy.assert_not_called()
    assert len(calls) == 1


def test_published_rank_participates_in_peer_restore_retry(pair, monkeypatch):
    layout, scope, calls = pair
    main = layout(384, 6)
    assert context.try_share_mega_moe_buffer(main, layout(128, 3), scope, True)
    target = next(iter(main.values()))
    target._mega_moe_hccl_state_stale = False
    monkeypatch.setattr(comms, "get_mc2_group", lambda: scope)
    monkeypatch.setattr(
        context, "_agree", lambda _, value: [value, (False, True, False) if isinstance(value, tuple) else value]
    )
    barrier = Mock()
    monkeypatch.setattr(context.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(context.torch.npu, "current_device", lambda: 0)
    assert comms.FusedMC2CommImpl.refresh_hccl_runtime_state(target)
    barrier.assert_called_once()
    assert len(calls) == 1


def test_window_without_release_api_keeps_target_lifecycle(pair):
    layout, scope, calls = pair
    main = layout(384, 6)
    target = next(iter(main.values()))
    mode, dtype, _ = comms.moe_utils._get_cann_mega_moe_quant_settings(QuantType.W4A8MXFP)
    params = target.mega_moe_buffer_params(mode, dtype)
    owner = target.get_symm_buffer_for_mega_moe(scope.device_group, **params)
    del owner.destroy
    target.mega_moe_symm_buffer = owner
    with pytest.raises(RuntimeError, match="target already owns the MegaMoE buffer"):
        context.try_share_mega_moe_buffer(main, layout(128, 3), scope, True)
    assert calls == [scope.device_group]
    assert target.mega_moe_symm_buffer is owner
    assert not hasattr(target, "_shared_mega_moe_buffer")


@pytest.mark.parametrize(
    "changes",
    [
        dict(quant=QuantType.W8A8MXFP),
        dict(quant=QuantType.NONE),
        dict(experts=512),
        dict(topk=7),
        dict(hidden=128),
        dict(intermediate=1024),
        dict(tokens=256),
    ],
)
def test_incompatible_draft_never_initializes_a_second_buffer(pair, changes):
    layout, scope, calls = pair
    draft = layout(**(dict(experts=128, topk=3) | changes))
    assert not context.try_share_mega_moe_buffer(layout(384, 6), draft, scope, True)
    assert not calls
    assert next(iter(draft.values())).mega_moe_symm_buffer is None


@pytest.mark.parametrize("draft_topk", [3, 6])
def test_equal_expert_counts_preserve_native_registry(pair, draft_topk):
    layout, scope, calls = pair
    main, draft = layout(384, 6), layout(384, draft_topk)
    target, proposer = next(iter(main.values())), next(iter(draft.values()))
    key = comms._moe_config_key(comms.MoECommType.FUSED_MC2, target.moe_config)
    comms._MoECommMethods[key] = proposer
    assert not context.try_share_mega_moe_buffer(main, draft, scope, True)
    assert comms._MoECommMethods[key] is proposer
    assert proposer.mega_moe_symm_buffer is target.mega_moe_symm_buffer is None
    assert not calls


def test_initialized_draft_keeps_its_own_buffer(pair):
    layout, scope, calls = pair
    main, draft = layout(384, 6), layout(128, 3)
    proposer = next(iter(draft.values()))
    existing = proposer.mega_moe_symm_buffer = object()
    assert not context.try_share_mega_moe_buffer(main, draft, scope, True)
    assert proposer.mega_moe_symm_buffer is existing
    assert next(iter(main.values())).mega_moe_symm_buffer is None
    assert not calls


def test_overlapping_execution_disables_sharing(pair):
    layout, scope, calls = pair
    assert not context.try_share_mega_moe_buffer(layout(384, 6), layout(128, 3), scope, False)
    assert not calls


@pytest.mark.parametrize("peer_only", [False, True])
def test_size_query_exceeds_window(pair, monkeypatch, peer_only):
    layout, scope, calls = pair
    if peer_only:

        def agree(_, value):
            return [value, (True, False) if value == (True, True) else value]

        monkeypatch.setattr(context, "_agree", agree)
    else:
        monkeypatch.setattr(
            __import__(__name__, fromlist=["_"]), "_get_mega_moe_ccl_buffer_size", lambda moe_expert_num: 4097
        )
    main, draft = layout(384, 6), layout(128, 3)
    with pytest.raises(RuntimeError, match="Disable CANN MegaMoE"):
        context.try_share_mega_moe_buffer(main, draft, scope, True)
    assert not hasattr(next(iter(main.values())), "_shared_mega_moe_buffer")
    assert next(iter(draft.values())).mega_moe_symm_buffer is None
    assert calls == [scope.device_group]


def test_peer_disagreement_avoids_collective_allocation(pair, monkeypatch):
    layout, scope, calls = pair
    monkeypatch.setattr(context, "_agree", lambda _, local: [local, None])
    assert not context.try_share_mega_moe_buffer(layout(384, 6), layout(128, 3), scope, True)
    assert not calls


@pytest.mark.parametrize("target_quant", [QuantType.W4A8, QuantType.W4A8MXFP])
def test_w4a8_target_and_unquantized_draft_keep_native_communication(pair, target_quant):
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    layout, scope, calls = pair

    def loaded_model(experts, topk, quant):
        ((shape, method),) = layout(experts, topk, quant).items()
        layer = object.__new__(AscendRoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = cfg = method.moe_config
        cfg.experts_per_token = topk
        cfg.hidden_dim, cfg.intermediate_size_per_partition = shape.hidden, shape.intermediate
        layer.quant_method = SimpleNamespace(quant_method=SimpleNamespace(quant_type=quant))
        for kind in (comms.MoECommType.FUSED_MC2, comms.MoECommType.ALLTOALL, comms.MoECommType.ALLGATHER):
            key = comms._moe_config_key(kind, cfg)
            comms._MoECommMethods[key] = method if kind is comms.MoECommType.FUSED_MC2 else object()
        return layer, method

    target_layer, target = loaded_model(384, 6, target_quant)
    draft_layer, proposer = loaded_model(128, 3, QuantType.NONE)
    target_buffer = target.mega_moe_symm_buffer = object()
    before = comms._MoECommMethods.copy()

    assert not context.try_share_mega_moe_buffer(
        context.fused_layouts(target_layer), context.fused_layouts(draft_layer), scope, True
    )

    assert not calls
    assert target.mega_moe_symm_buffer is target_buffer
    assert proposer.mega_moe_symm_buffer is None
    assert comms._MoECommMethods.keys() == before.keys()
    assert all(comms._MoECommMethods[key] is method for key, method in before.items())


@pytest.mark.parametrize(
    "target_type,arch,device,mega,transport,kv_only,use_ubatching,expected",
    [
        ("deepseek_v41", "DeepseekV41DSparkModel", "A5", True, "FUSED_MC2", False, False, True),
        ("deepseek_v41", "DeepseekV41DSparkModel", "A3", True, "FUSED_MC2", False, False, False),
        ("deepseek_v41", "DeepseekV41DSparkModel", "A5", False, "FUSED_MC2", False, False, False),
        ("deepseek_v41", "DeepseekV41DSparkModel", "A5", True, "FUSED_MC2", True, False, False),
        ("deepseek_v41", "DeepseekV41DSparkModel", "A5", True, "MC2", False, False, False),
        ("deepseek_v4", "DeepseekV41DSparkModel", "A5", True, "FUSED_MC2", False, False, False),
        ("qwen3_moe", "DeepseekV41DSparkModel", "A5", True, "FUSED_MC2", False, False, False),
        ("deepseek_v4", "DSparkDraftModel", "A5", True, "FUSED_MC2", False, False, False),
        ("qwen3", "Qwen3DSparkModel", "A5", True, "FUSED_MC2", False, False, False),
        ("deepseek_v41", "DeepseekV41DSparkModel", "A5", True, "FUSED_MC2", False, True, True),
    ],
)
def test_only_channel_v41_dspark_attempts_sharing_after_native_load(
    monkeypatch, target_type, arch, device, mega, transport, kv_only, use_ubatching, expected
):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    from vllm_ascend.device import hardware_profile
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.distributed import parallel_state
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

    spec = object.__new__(AscendDSparkSpeculator)
    spec.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
    spec.context_kv_only = kv_only
    spec.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=target_type)))
    spec._lmhead_tp_wrap_draft_logits = lambda model: None
    spec.vllm_config.parallel_config = SimpleNamespace(use_ubatching=use_ubatching)
    target, draft, scope = object(), SimpleNamespace(), object()
    bind = Mock()
    layouts = Mock(side_effect=lambda model: model)
    get_group = Mock(return_value=scope)
    agree = Mock()
    entrypoint = Mock(wraps=context.try_share_dspark_mega_moe_buffer)
    monkeypatch.setattr(context, "try_share_dspark_mega_moe_buffer", entrypoint)
    monkeypatch.setattr(context, "try_share_mega_moe_buffer", bind)
    monkeypatch.setattr(context, "fused_layouts", layouts)
    monkeypatch.setattr(context, "_agree", agree)
    monkeypatch.setattr(parallel_state, "get_mc2_group", get_group)
    profile = hardware_profile.get_hardware_profile(AscendDeviceType[device])
    monkeypatch.setattr(context, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(context, "use_cann_megamoe", lambda _: mega)
    monkeypatch.setattr(context, "select_moe_comm_method", lambda n, cfg: comms.MoECommType[transport])

    def load(self, model, names):
        entrypoint.assert_not_called()
        layouts.assert_not_called()
        get_group.assert_not_called()
        bind.assert_not_called()
        assert model is target
        return draft

    def post_process(config):
        assert config is spec.vllm_config
        entrypoint.assert_called_once_with(
            target, draft, spec.vllm_config, spec.draft_model_config, context_kv_only=kv_only
        )
        if expected:
            bind.assert_called_once_with(target, draft, scope, not use_ubatching)
        else:
            bind.assert_not_called()

    draft.post_process = Mock(side_effect=post_process)
    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    assert spec.load_draft_model(target, set()) is draft
    draft.post_process.assert_called_once_with(spec.vllm_config)
    entrypoint.assert_called_once_with(
        target, draft, spec.vllm_config, spec.draft_model_config, context_kv_only=kv_only
    )
    if expected:
        bind.assert_called_once_with(target, draft, scope, not use_ubatching)
        assert layouts.call_args_list == [call(target), call(draft)]
        get_group.assert_called_once_with()
    else:
        bind.assert_not_called()
        layouts.assert_not_called()
        get_group.assert_not_called()
        agree.assert_not_called()
