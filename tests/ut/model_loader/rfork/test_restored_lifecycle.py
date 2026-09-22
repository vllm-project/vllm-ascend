# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def rfork_helpers(monkeypatch):
    """Load the real RFork helpers with only their external imports stubbed."""

    def install(name, **attributes):
        module = _stub(monkeypatch, name, **attributes)
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, child_name, module, raising=False)
        return module

    def package(name):
        return install(name, __path__=[])

    package("vllm")
    package("vllm.config")
    install("vllm.config", ModelConfig=object, VllmConfig=object)
    install("vllm.config.load", LoadConfig=object)
    install("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0)
    install("vllm.distributed.parallel_state", get_ep_group=lambda: None, get_pp_group=lambda: None)
    install("vllm.logger", logger=logging.getLogger("rfork-restored-lifecycle-test"))

    package("vllm.model_executor")
    package("vllm.model_executor.model_loader")
    install("vllm.model_executor.model_loader", register_model_loader=lambda name: lambda cls: cls)

    class BaseModelLoader:
        def __init__(self, load_config):
            self.load_config = load_config

    install("vllm.model_executor.model_loader.base_loader", BaseModelLoader=BaseModelLoader)
    install(
        "vllm.model_executor.model_loader.utils",
        initialize_model=lambda **kwargs: None,
        process_weights_after_loading=lambda *args, **kwargs: None,
    )
    package("vllm.utils")
    install("vllm.utils.torch_utils", set_default_torch_dtype=lambda dtype: nullcontext())
    package("vllm.model_executor.layers")
    rope_dict = {"baseline": object()}
    install("vllm.model_executor.layers.rotary_embedding", _ROPE_DICT=rope_dict)

    package("vllm_ascend")
    package("vllm_ascend.device")
    install("vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=0))
    install("vllm_ascend.device.hardware_profile", get_current_hardware_profile=lambda: SimpleNamespace())
    package("vllm_ascend.model_loader")
    package("vllm_ascend.model_loader.rfork")

    safety = _load_module(monkeypatch, "rfork_restored_lifecycle_safety", "safety.py")
    install(
        "vllm_ascend.model_loader.rfork.safety",
        mutable_weights_bypass_reason=safety.mutable_weights_bypass_reason,
        ensure_no_registered_rfork_weights=safety.ensure_no_registered_rfork_weights,
    )
    install("vllm_ascend.model_loader.rfork.config", RForkConfig=object)
    install(
        "vllm_ascend.model_loader.rfork.identity",
        _resolve_sharded_dp_rank=lambda *args, **kwargs: None,
        build_compatibility_fingerprint=lambda *args, **kwargs: "fingerprint",
    )
    install("vllm_ascend.model_loader.rfork.session", RForkSession=object)
    install(
        "vllm_ascend.model_loader.rfork.types",
        RForkFallbackCleanupResult=object,
        RForkIdentity=object,
        RForkLifecycleState=object,
        RForkSeedServiceStartResult=object,
    )

    package("vllm_ascend.eplb")
    package("vllm_ascend.eplb.adaptor")

    class VllmEplbAdaptor:
        _registered_moe_layers = []

    adaptor_module = install("vllm_ascend.eplb.adaptor.vllm_adaptor", VllmEplbAdaptor=VllmEplbAdaptor)
    package("vllm_ascend.ops")
    package("vllm_ascend.ops.fused_moe")

    class AscendMoERunner(Module):
        def __init__(self, quant_method=None):
            super().__init__()
            self._quant_method = quant_method

    install("vllm_ascend.ops.fused_moe.fused_moe", AscendMoERunner=AscendMoERunner)

    class AscendRoutedExperts:
        moe_counter = -1

    routed_experts_module = install("vllm_ascend.ops.fused_moe.routed_experts", AscendRoutedExperts=AscendRoutedExperts)
    rotary_module = install(
        "vllm_ascend.ops.rotary_embedding",
        _cos_sin_cache=None,
        _cos_cache=None,
        _sin_cache=None,
    )

    class DynamoHookRegistry(dict):
        pass

    dynamo_hooks = DynamoHookRegistry()
    install("torch._dynamo.convert_frame", _bytecode_hooks=dynamo_hooks)

    loader = _load_module(monkeypatch, "rfork_restored_lifecycle_loader", "rfork_loader.py")
    return SimpleNamespace(
        loader=loader,
        safety=safety,
        AscendMoERunner=AscendMoERunner,
        adaptor_module=adaptor_module,
        routed_experts_module=routed_experts_module,
        rotary_module=rotary_module,
        dynamo_hooks=dynamo_hooks,
        rope_dict=rope_dict,
    )


@pytest.mark.parametrize(
    ("model_sleep", "vllm_sleep", "weight_transfer", "expected"),
    [
        (True, False, False, "sleep mode"),
        (False, True, False, "sleep mode"),
        (False, False, True, "online weight transfer (weight_transfer_config)"),
        (False, False, False, None),
        (True, False, True, "sleep mode"),
    ],
)
def test_mutable_weights_bypass_reason_reports_sleep_and_weight_transfer(
    rfork_helpers, model_sleep, vllm_sleep, weight_transfer, expected
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=vllm_sleep),
        weight_transfer_config=object() if weight_transfer else None,
    )
    model_config = SimpleNamespace(enable_sleep_mode=model_sleep)

    assert rfork_helpers.safety.mutable_weights_bypass_reason(vllm_config, model_config) == expected


def _vllm_config_with_sessions(**session_blocks):
    """Build a vllm_config whose RFork sessions report the given registered blocks."""
    load_config = SimpleNamespace()
    for attr, blocks in session_blocks.items():
        backend = SimpleNamespace(snapshot_registered_weight_blocks=lambda blocks=blocks: list(blocks))
        setattr(load_config, attr, SimpleNamespace(transfer_backend=backend))
    return SimpleNamespace(load_config=load_config)


@pytest.mark.parametrize(
    ("session_blocks", "expected_holder"),
    [
        ({"rfork_session": [(0x1000, 256)]}, "main"),
        ({"rfork_draft_session": [(0x2000, 512)]}, "draft"),
        ({"rfork_session": [(0x1000, 256)], "rfork_draft_session": [(0x2000, 512)]}, "main, draft"),
    ],
)
def test_ensure_no_registered_rfork_weights_rejects_while_storage_is_registered(
    rfork_helpers, session_blocks, expected_holder
):
    vllm_config = _vllm_config_with_sessions(**session_blocks)

    with pytest.raises(RuntimeError, match=r"reload_weights is not supported") as excinfo:
        rfork_helpers.safety.ensure_no_registered_rfork_weights(vllm_config, "reload_weights")
    assert f"({expected_holder} model)" in str(excinfo.value)


@pytest.mark.parametrize(
    "vllm_config",
    [
        SimpleNamespace(),
        SimpleNamespace(load_config=None),
        # No RFork session at all: a non-RFork load format leaves the attributes unset.
        SimpleNamespace(load_config=SimpleNamespace()),
        # A session exists but registration was released or never completed.
        _vllm_config_with_sessions(rfork_session=[]),
        _vllm_config_with_sessions(rfork_session=[], rfork_draft_session=[]),
    ],
)
def test_ensure_no_registered_rfork_weights_allows_unregistered_sessions(rfork_helpers, vllm_config):
    rfork_helpers.safety.ensure_no_registered_rfork_weights(vllm_config, "reload_weights")


def test_ensure_no_registered_rfork_weights_ignores_sessions_without_a_backend(rfork_helpers):
    """A session mid-construction exposes no usable snapshot; treat it as holding nothing."""
    load_config = SimpleNamespace(
        rfork_session=SimpleNamespace(transfer_backend=None),
        rfork_draft_session=SimpleNamespace(transfer_backend=SimpleNamespace()),
    )

    rfork_helpers.safety.ensure_no_registered_rfork_weights(SimpleNamespace(load_config=load_config), "reload_weights")


def _add_dynamo_hook(hooks, owner):
    handle = RemovableHandle(hooks)
    hooks[handle.id] = owner.bytecode_hook
    owner._bytecode_hook_handle = handle
    return handle.id


def test_fallback_reset_restores_ascend_globals_and_only_removes_new_dynamo_hooks(rfork_helpers):
    class HookOwner(Module):
        def bytecode_hook(self, *args, **kwargs):
            pass

    adaptor = rfork_helpers.adaptor_module.VllmEplbAdaptor
    baseline_layer = object()
    baseline_registry = [baseline_layer]
    adaptor._registered_moe_layers = baseline_registry
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 17

    baseline_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = baseline_caches

    baseline_owner = HookOwner()
    baseline_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, baseline_owner)
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace())
    snapshot = rfork_helpers.loader._snapshot_process_global_model_state(vllm_config)

    baseline_registry.append(object())
    adaptor._registered_moe_layers = [object()]
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 999
    dirty_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = dirty_caches

    discarded_owner = HookOwner()
    discarded_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, discarded_owner)
    assert discarded_hook_id not in snapshot.dynamo_bytecode_hook_ids

    rfork_helpers.loader._reset_process_global_model_state(vllm_config, snapshot=snapshot)

    assert adaptor._registered_moe_layers is baseline_registry
    assert adaptor._registered_moe_layers == [baseline_layer]
    assert rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter == 17
    assert (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) == baseline_caches
    assert baseline_hook_id in rfork_helpers.dynamo_hooks
    assert rfork_helpers.dynamo_hooks[baseline_hook_id].__self__ is baseline_owner
    assert discarded_hook_id not in rfork_helpers.dynamo_hooks
