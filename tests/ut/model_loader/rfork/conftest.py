# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from torch.nn import Module

from .rfork_test_support import RFORK_ROOT, _load_module, _stub


@pytest.fixture
def local_weights(monkeypatch):
    return _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.local_weights", "local_weights.py")


@pytest.fixture
def runtime(monkeypatch, local_weights):
    """Load the real lease/session code with external runtime dependencies stubbed."""
    prefix = "vllm_ascend.model_loader.rfork"

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name):
        full_name = f"{prefix}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, RFORK_ROOT / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        return module

    stub("vllm.logger", logger=logging.getLogger("rfork-release-test"))
    stub("vllm.utils.network_utils", get_ip=lambda: "127.0.0.1", join_host_port=lambda host, port: f"{host}:{port}")
    stub(f"{prefix}.identity", build_seed_key=lambda **kwargs: "model-key")
    stub(
        f"{prefix}.transfer_backend",
        RForkTransferBackend=lambda **kwargs: Mock(registered_structural_digest="registered-digest"),
    )

    class StartupError(RuntimeError):
        def __init__(self, message, *, handle=None):
            super().__init__(message)
            self.handle = handle

    stub(
        f"{prefix}.seed_server",
        RForkSeedServerHandle=Mock,
        RForkSeedServerStartupError=StartupError,
        start_rfork_server=Mock(),
    )
    types = load("types")
    config = load("config")
    load("manifest")
    load("tensor_layout")
    load("seed_client")
    client = load("planner_client")
    session = load("session")
    monkeypatch.setattr(session.atexit, "register", lambda callback: None)
    cfg = config.RForkConfig(
        "model",
        "strategy",
        "http://planner",
        request_timeout_sec=0.1,
        lease_release_retry_interval_sec=0.001,
        seed_removal_supported=None,
    )
    identity = types.RForkIdentity(0, 0, compatibility_fingerprint="fingerprint")
    lease = types.SeedLease("127.0.0.1", 1234, "private-user-id", 0, "model-key", lease_ttl_sec=60.0)
    return SimpleNamespace(types=types, client=client, session=session, config=cfg, identity=identity, lease=lease)


@pytest.fixture
def tensor_runtime(monkeypatch, local_weights):
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-tensor-safety-test"))
    _stub(monkeypatch, "vllm.utils", __path__=[])
    _stub(
        monkeypatch,
        "vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        join_host_port=lambda host, port: f"{host}:{port}",
    )
    for module_name in (
        "vllm_ascend",
        "vllm_ascend.model_loader",
        "vllm_ascend.model_loader.rfork",
    ):
        _stub(monkeypatch, module_name, __path__=[])

    @dataclass(frozen=True)
    class _SeedTransferInfo:
        session_id: str
        weights: dict
        formats: dict | None = None

    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=_SeedTransferInfo)
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.manifest", "manifest.py")
    tensor_layout = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.tensor_layout", "tensor_layout.py")
    transfer_backend = _load_module(
        monkeypatch, "vllm_ascend.model_loader.rfork.transfer_backend", "transfer_backend.py"
    )
    return SimpleNamespace(
        tensor_layout=tensor_layout,
        transfer_backend=transfer_backend,
        RForkTransferBackend=transfer_backend.RForkTransferBackend,
        SeedTransferInfo=_SeedTransferInfo,
    )


@pytest.fixture
def rfork_helpers(monkeypatch, local_weights):
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
    install(
        "vllm.distributed.parallel_state",
        get_ep_group=lambda: None,
        get_pp_group=lambda: None,
        get_world_group=lambda: SimpleNamespace(world_size=1, cpu_group=None),
    )
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
    package("vllm_ascend.models")

    class AscendDeepSeekMTP(Module):
        pass

    mtp_module = install("vllm_ascend.models.deepseek_mtp", AscendDeepSeekMTP=AscendDeepSeekMTP)
    package("vllm_ascend.device")
    install("vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=0))
    install("vllm_ascend.device.hardware_profile", get_current_hardware_profile=lambda: SimpleNamespace())
    package("vllm_ascend.model_loader")
    package("vllm_ascend.model_loader.rfork")

    safety = _load_module(monkeypatch, "rfork_restored_lifecycle_safety", "safety.py")
    install("vllm_ascend.model_loader.rfork.safety", mutable_weights_bypass_reason=safety.mutable_weights_bypass_reason)
    rfork_config_module = _load_module(monkeypatch, "rfork_restored_lifecycle_config", "config.py")
    install(
        "vllm_ascend.model_loader.rfork.config",
        RForkConfig=rfork_config_module.RForkConfig,
        RForkConfigurationError=rfork_config_module.RForkConfigurationError,
        unknown_extra_config_keys=rfork_config_module.unknown_extra_config_keys,
    )
    install(
        "vllm_ascend.model_loader.rfork.identity",
        _resolve_sharded_dp_rank=lambda *args, **kwargs: None,
        build_compatibility_fingerprint=lambda *args, **kwargs: "fingerprint",
    )
    install("vllm_ascend.model_loader.rfork.session", RForkSession=object)
    types_module = _load_module(monkeypatch, "rfork_restored_lifecycle_types", "types.py")
    install(
        "vllm_ascend.model_loader.rfork.types",
        RForkFallbackCleanupResult=types_module.RForkFallbackCleanupResult,
        RForkIdentity=types_module.RForkIdentity,
        RForkLifecycleState=types_module.RForkLifecycleState,
        RForkSeedServiceStartResult=types_module.RForkSeedServiceStartResult,
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
        mtp_module=mtp_module,
    )
