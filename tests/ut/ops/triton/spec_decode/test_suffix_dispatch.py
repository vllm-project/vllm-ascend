# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.triton_utils import dispatcher

from tests.ut.ops.triton.spec_decode.conftest import KernelCall
from vllm_ascend import triton_kernel_overrides as overrides
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.platform import NPUPlatform


@pytest.mark.parametrize("operation", ["local_scan", "finalize", "ingest", "observe"])
@pytest.mark.parametrize("keyword_launch", [False, True])
def test_dispatch_reaches_suffix_launchers(monkeypatch, recording_kernel, suffix_options, operation, keyword_launch):
    """Exercise real dispatcher binding and the production launch adapters."""
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

    calls: list[KernelCall] = []
    impl_module = importlib.import_module(impl.__module__)
    monkeypatch.setattr(impl_module, name, recording_kernel(operation, calls))
    if operation == "observe":
        monkeypatch.setattr(impl_module, "_suffix_clear_observed_kernel", recording_kernel("clear", calls))
    # Restore the real module and registry after dispatcher's identity sweep.
    monkeypatch.setattr(module, name, original)
    monkeypatch.setattr(dispatcher, "_registry", {})
    alias = ModuleType("_suffix_dispatch_alias")
    vars(alias)["kernel"] = original
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    dispatcher.register_kernels({key: impl})
    kernel = getattr(module, name)
    assert vars(alias)["kernel"] is kernel

    tensor = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2056)
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 12)
    options = suffix_options[operation]
    args = {name: object() for name in names if name not in options}
    args.update(
        token_ids_ptr=tensor,
        token_ids_stride=2048,
        idx_mapping_ptr=SimpleNamespace(shape=(7,)),
        scratch_ptr=scratch,
        scratch_stride=8,
        num_blocks=8,
        Lp1=2049,
        corpus_len=4096,
        num_docs=2,
        stats_ptr=SimpleNamespace(shape=(18, 3, 2)),
    )
    args = {name: args[name] for name in names if name not in options}
    if keyword_launch:
        # Reverse insertion order: binding must follow upstream parameter order.
        kernel[(999,)](**dict(reversed(list((args | options).items()))))
    else:
        # Match SuffixSpeculator's positional tensors + keyword constants.
        kernel[(999,)](*args.values(), **options, num_warps=1)
    expected_grid = {"local_scan": (7, 8), "finalize": (7,), "ingest": (2, 2), "observe": (18,)}
    assert calls[0][1] == expected_grid[operation]
    expected_values = args | {"token_ids_stride": 2056, "scratch_stride": 12}
    expected_args = tuple(expected_values[name] for name in args)
    if operation == "observe":
        expected_args += (7,)
        assert [call[0] for call in calls] == ["observe", "clear"]
        assert calls[0][3]["REQS_PO2"] == 8
    assert calls[0][2] == expected_args
    if operation == "finalize":
        assert calls[0][3]["K_PO2"] == 2
        assert calls[0][3]["DEPTH_PO2"] == 4


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
