# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import inspect
import sys
from functools import wraps
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.triton_utils import dispatcher

from vllm_ascend import triton_kernel_overrides as overrides
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.platform import NPUPlatform


@pytest.fixture
def suffix_options():
    return {
        "local_scan": {"MAX_DEPTH": 3, "BLOCK": 256},
        "finalize": {
            "HAS_CORPUS": True,
            "ADAPTIVE": True,
            "HASH_MUL": -7046029254386353131,
            "NUM_BUCKETS": 64,
            "SLOTS": 8,
            "MAX_DEPTH": 3,
            "DEPTH_PO2": 4,
            "K": 1,
            "K_PO2": 1,
            "NB_PO2": 8,
        },
        "ingest": {"HASH_MUL": -7046029254386353131, "NUM_BUCKETS": 64, "SLOTS": 8, "BLOCK": 1024},
        "observe": {"K": 3, "K_PO2": 4},
    }


@pytest.mark.parametrize("operation", ["local_scan", "finalize", "ingest", "observe"])
@pytest.mark.parametrize("keyword_launch", [False, True])
def test_dispatch_reaches_suffix_launchers(monkeypatch, suffix_options, operation, keyword_launch):
    """Check adapter selection, alias rebinding and the upstream call ABI."""
    target = "vllm.v1.worker.gpu.spec_decode.suffix.speculator"
    module = importlib.import_module(target)
    name = f"_suffix_{operation}_kernel"
    original = getattr(module, name)
    key = f"{target}.{name}"
    impl = overrides.TRITON_KERNEL_OVERRIDES[key]
    names = getattr(original, "arg_names", None)
    if names is None:
        names = tuple(inspect.signature(original).parameters)
    names = tuple(names)
    assert tuple(inspect.signature(impl).parameters)[:-1] == names
    assert original.__module__ == target

    calls = []

    @wraps(impl)
    def record_launch(*args, **kwargs):
        calls.append((args, kwargs))

    # Restore the real module and registry after dispatcher's identity sweep.
    monkeypatch.setattr(module, name, original)
    monkeypatch.setattr(dispatcher, "_registry", {})
    alias = ModuleType("_suffix_dispatch_alias")
    vars(alias)["kernel"] = original
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    dispatcher.register_kernels({key: record_launch})
    kernel = getattr(module, name)
    assert vars(alias)["kernel"] is kernel

    options = suffix_options[operation]
    args = {name: object() for name in names if name not in options}
    if keyword_launch:
        # Reverse insertion order: binding must follow upstream parameter order.
        kernel[(999,)](**dict(reversed(list((args | options).items()))))
    else:
        # Match SuffixSpeculator's positional tensors + keyword constants.
        kernel[(999,)](*args.values(), **options, num_warps=1)
    if keyword_launch:
        # Dispatcher fallback binds all-keyword calls in the upstream parameter order.
        assert calls == [(tuple((args | options)[name] for name in names), {})]
    else:
        assert calls == [(tuple(args.values()), options | {"num_warps": 1})]


def test_registration_is_idempotent_and_retries_failures(monkeypatch):
    monkeypatch.setattr(overrides, "_kernels_registered", False)
    register = Mock(side_effect=[RuntimeError("registration failed"), None])
    monkeypatch.setattr(dispatcher, "register_kernels", register)
    with pytest.raises(RuntimeError, match="registration failed"):
        overrides.register_triton_kernels()
    assert not overrides._kernels_registered
    overrides.register_triton_kernels()
    overrides.register_triton_kernels()
    assert register.call_count == 2
    register.assert_called_with(overrides.TRITON_KERNEL_OVERRIDES)


@pytest.mark.parametrize("compatibility,has_triton", [(False, True), (False, False), (True, True)])
def test_platform_routes_registration(monkeypatch, compatibility, has_triton):
    platform = importlib.import_module("vllm_ascend.platform")
    triton_utils = importlib.import_module("vllm.triton_utils")
    sampling = importlib.import_module("vllm_ascend._310p.worker.v2.sample.dispatch")
    family = DeviceAdaptorFamily.COMPATIBILITY if compatibility else object()
    monkeypatch.setattr(platform, "get_current_hardware_profile", lambda: SimpleNamespace(device_adaptor_family=family))
    monkeypatch.setattr(triton_utils, "HAS_TRITON", has_triton)
    register_suffix = Mock()
    register_sampling = Mock()
    monkeypatch.setattr(overrides, "register_triton_kernels", register_suffix)
    monkeypatch.setattr(sampling, "register_sampling_kernel_overrides", register_sampling)
    NPUPlatform.register_triton_kernel_overrides()
    assert register_suffix.call_count == int(not compatibility and has_triton)
    assert register_sampling.call_count == int(compatibility)
