# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU-only A5 availability and real norm-forward dispatch tests.

The extension import, dispatcher registrations, and NPU backends are mocked.
Production availability helpers and Ascend forward_oot methods execute directly.
"""

import builtins
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
import torch_npu
import vllm.envs as vllm_envs

from vllm_ascend import utils
from vllm_ascend.ops import layernorm


@pytest.fixture(autouse=True)
def isolated_availability(monkeypatch):
    utils.enable_a5_add_rms_norm_bias.cache_clear()
    monkeypatch.setattr(utils, "is_950", lambda: True)
    monkeypatch.setattr(utils, "_CUSTOM_OP_ENABLED", None)
    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    yield
    utils.enable_a5_add_rms_norm_bias.cache_clear()


@pytest.fixture
def packaged_operator(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_CUSTOM_OP_BASE_DIR", str(tmp_path))
    config = (
        tmp_path
        / "_cann_ops_custom"
        / "vendors"
        / utils._CUSTOM_OP_VENDOR_DIR
        / "op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"AddRmsNormBias": {"opFile": {"value": "add_rms_norm_bias_apt"}}}))
    bootstrap = Mock()
    monkeypatch.setattr(utils, "bootstrap_custom_op_env", bootstrap)
    extension_import = Mock()
    real_import = builtins.__import__

    def import_extension_only(name, *args, **kwargs):
        if name == "vllm_ascend.vllm_ascend_C":
            extension_import()
            return sys.modules["vllm_ascend"]
        if name == "vllm_ascend.meta_registration":
            raise AssertionError("The A5 norm uses its existing C++ Meta registration")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_extension_only)
    namespace = SimpleNamespace(npu_add_rms_norm_bias=Mock())
    monkeypatch.setattr(torch.ops, "_C_ascend", namespace)
    dispatch = Mock(return_value=True)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", dispatch)
    generic = Mock(side_effect=AssertionError("The dedicated A5 gate must not call the generic gate"))
    monkeypatch.setattr(utils, "enable_custom_op", generic)
    return SimpleNamespace(
        config=config,
        bootstrap=bootstrap,
        extension_import=extension_import,
        namespace=namespace,
        dispatch=dispatch,
        generic=generic,
    )


@pytest.mark.parametrize("generic_cached", (None, False, True))
def test_a5_success_is_cached_without_changing_generic_gate(packaged_operator, monkeypatch, generic_cached):
    monkeypatch.setattr(utils, "_CUSTOM_OP_ENABLED", generic_cached)
    assert utils.enable_a5_add_rms_norm_bias() is True
    assert utils.enable_a5_add_rms_norm_bias() is True
    assert utils._CUSTOM_OP_ENABLED is generic_cached
    packaged_operator.bootstrap.assert_called_once_with()
    packaged_operator.extension_import.assert_called_once_with()
    assert packaged_operator.dispatch.call_count == 2
    packaged_operator.dispatch.assert_has_calls(
        [call("_C_ascend::npu_add_rms_norm_bias", "Meta"), call("_C_ascend::npu_add_rms_norm_bias", "PrivateUse1")],
        any_order=True,
    )
    packaged_operator.generic.assert_not_called()


@pytest.mark.parametrize(
    ("is_a5", "batch_invariant"),
    ((False, False), (False, True), (True, True)),
    ids=("legacy", "legacy-batch-invariant", "a5-batch-invariant"),
)
def test_hardware_and_batch_guards_do_not_load(packaged_operator, monkeypatch, is_a5, batch_invariant):
    monkeypatch.setattr(utils, "is_950", lambda: is_a5)
    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(utils, "_CUSTOM_OP_ENABLED", True)
    assert utils.enable_a5_add_rms_norm_bias() is False
    assert utils._CUSTOM_OP_ENABLED is True
    packaged_operator.bootstrap.assert_not_called()
    packaged_operator.extension_import.assert_not_called()
    packaged_operator.dispatch.assert_not_called()


@pytest.mark.parametrize(
    "config_text",
    (None, "{broken json", "null", "[]", '{"OtherOperator": {}}'),
    ids=("missing-file", "malformed-json", "null", "list", "missing-op"),
)
def test_missing_or_invalid_package_falls_back_without_loading(packaged_operator, config_text):
    if config_text is None:
        packaged_operator.config.unlink()
    else:
        packaged_operator.config.write_text(config_text)
    assert utils.enable_a5_add_rms_norm_bias() is False
    assert utils._CUSTOM_OP_ENABLED is None
    packaged_operator.bootstrap.assert_not_called()
    packaged_operator.extension_import.assert_not_called()
    packaged_operator.dispatch.assert_not_called()


@pytest.mark.parametrize("missing", ("schema", "Meta", "PrivateUse1", "dispatcher-error"))
def test_incomplete_registration_falls_back(packaged_operator, monkeypatch, missing):
    monkeypatch.setattr(utils, "_CUSTOM_OP_ENABLED", True)
    if missing == "schema":
        del packaged_operator.namespace.npu_add_rms_norm_bias
    elif missing == "dispatcher-error":
        packaged_operator.dispatch.side_effect = RuntimeError("operator does not exist")
    else:
        packaged_operator.dispatch.side_effect = lambda _name, key: key != missing
    assert utils.enable_a5_add_rms_norm_bias() is False
    assert utils._CUSTOM_OP_ENABLED is True
    packaged_operator.extension_import.assert_called_once_with()


def test_unavailable_result_is_cached_independently(packaged_operator, monkeypatch):
    monkeypatch.setattr(utils, "_CUSTOM_OP_ENABLED", False)
    packaged_operator.config.unlink()
    assert utils.enable_a5_add_rms_norm_bias() is False
    packaged_operator.config.write_text(json.dumps({"AddRmsNormBias": {}}))
    # Installation changes require a fresh process (or explicit cache invalidation).
    assert utils.enable_a5_add_rms_norm_bias() is False
    assert utils._CUSTOM_OP_ENABLED is False
    packaged_operator.extension_import.assert_not_called()


@pytest.mark.parametrize("error", ("extension not installed", "libother_dependency.so not found"))
def test_unrelated_import_failure_does_not_retry(packaged_operator, error):
    packaged_operator.extension_import.side_effect = ImportError(error)
    assert utils.enable_a5_add_rms_norm_bias() is False
    assert utils._CUSTOM_OP_ENABLED is None
    packaged_operator.bootstrap.assert_called_once_with()
    packaged_operator.extension_import.assert_called_once_with()
    packaged_operator.dispatch.assert_not_called()


def test_missing_vendor_library_retries_with_vendor_path(packaged_operator):
    packaged_operator.extension_import.side_effect = [ImportError("libcust_opapi.so not found"), None]
    assert utils.enable_a5_add_rms_norm_bias() is True
    assert utils._CUSTOM_OP_ENABLED is None
    assert packaged_operator.bootstrap.call_args_list == [call(), call(include_vendor_lib=True)]
    assert packaged_operator.extension_import.call_count == 2


def test_failed_vendor_library_retry_falls_back(packaged_operator, monkeypatch):
    monkeypatch.setattr(utils, "_CUSTOM_OP_ENABLED", True)
    packaged_operator.extension_import.side_effect = ImportError("libcust_opapi.so not found")
    assert utils.enable_a5_add_rms_norm_bias() is False
    assert utils._CUSTOM_OP_ENABLED is True
    assert packaged_operator.bootstrap.call_args_list == [call(), call(include_vendor_lib=True)]
    assert packaged_operator.extension_import.call_count == 2
    packaged_operator.dispatch.assert_not_called()


@pytest.fixture
def norm_backends(monkeypatch):
    base_y = torch.full((2, 4), 3.0)
    output_residual = torch.full((2, 4), 7.0)

    def fused(_x, _residual, _weight, beta, _epsilon):
        y = base_y.clone()
        if beta is not None:
            y.add_(beta)
        return y, None, output_residual

    custom = Mock(side_effect=fused)
    eager = Mock(side_effect=lambda *args: (base_y.clone(), None, output_residual))
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(npu_add_rms_norm_bias=custom))
    monkeypatch.setattr(torch_npu, "npu_add_rms_norm", eager, raising=False)
    return custom, eager, base_y, output_residual


def selected_gates(monkeypatch, is_a5, available):
    monkeypatch.setattr(layernorm, "is_950", lambda: is_a5)
    a5 = Mock(return_value=available)
    generic = Mock(return_value=available)
    selected, unused = (a5, generic) if is_a5 else (generic, a5)
    unused.side_effect = AssertionError("The other device's enable gate must not run")
    monkeypatch.setattr(layernorm, "enable_a5_add_rms_norm_bias", a5)
    monkeypatch.setattr(layernorm, "enable_custom_op", generic)
    return selected, unused


@pytest.mark.parametrize("is_a5", (False, True), ids=("legacy", "a5"))
@pytest.mark.parametrize("available", (False, True), ids=("fallback", "custom"))
@pytest.mark.parametrize("has_beta", (False, True), ids=("no-beta", "signed-beta"))
def test_rms_residual_selects_device_gate_and_adds_beta_once(norm_backends, monkeypatch, is_a5, available, has_beta):
    selected_gate, unused_gate = selected_gates(monkeypatch, is_a5, available)
    custom, eager, base_y, output_residual = norm_backends
    beta = torch.tensor((-0.5, 0.0, 0.25, 1.0)) if has_beta else None
    layer = SimpleNamespace(weight=torch.tensor((0.5, 1.0, 1.5, 2.0)), bias=beta, variance_epsilon=1e-6)
    x, residual = torch.zeros((2, 4)), torch.ones((2, 4))
    y, result_residual = layernorm.AscendRMSNorm.forward_oot(layer, x, residual)
    torch.testing.assert_close(y, base_y if beta is None else base_y + beta, rtol=0, atol=0)
    assert result_residual is output_residual
    selected_gate.assert_called_once_with()
    unused_gate.assert_not_called()
    chosen, unused = (custom, eager) if available else (eager, custom)
    chosen.assert_called_once()
    unused.assert_not_called()
    args = chosen.call_args.args
    assert args[0] is x and args[1] is residual and args[2] is layer.weight
    assert args[-1] == layer.variance_epsilon
    if available:
        assert args[3] is beta


@pytest.mark.parametrize("is_a5", (False, True), ids=("legacy", "a5"))
@pytest.mark.parametrize("available", (False, True), ids=("fallback", "custom"))
def test_gemma_residual_keeps_weight_plus_one_and_no_beta(norm_backends, monkeypatch, is_a5, available):
    selected_gate, unused_gate = selected_gates(monkeypatch, is_a5, available)
    custom, eager, base_y, output_residual = norm_backends
    layer = SimpleNamespace(weight=torch.tensor((-0.5, 0.0, 0.5, 1.0)), variance_epsilon=1e-6)
    x, residual = torch.zeros((2, 4)), torch.ones((2, 4))
    y, result_residual = layernorm.AscendGemmaRMSNorm.forward_oot(layer, x, residual)
    torch.testing.assert_close(y, base_y, rtol=0, atol=0)
    assert result_residual is output_residual
    selected_gate.assert_called_once_with()
    unused_gate.assert_not_called()
    chosen, unused = (custom, eager) if available else (eager, custom)
    chosen.assert_called_once()
    unused.assert_not_called()
    args = chosen.call_args.args
    assert args[0] is x and args[1] is residual
    torch.testing.assert_close(args[2], 1.0 + layer.weight, rtol=0, atol=0)
    assert args[-1] == layer.variance_epsilon
    if available:
        assert args[3] is None


@pytest.mark.parametrize("is_a5", (False, True), ids=("legacy", "a5"))
@pytest.mark.parametrize("bias_state", ("absent", "not-loaded", "loaded"))
def test_rms_without_residual_preserves_bias_loaded_semantics(monkeypatch, is_a5, bias_state):
    selected_gate, unused_gate = selected_gates(monkeypatch, is_a5, True)
    selected_gate.side_effect = AssertionError("No residual must not enable a fused add-norm operator")
    base_y = torch.full((2, 4), 3.0)
    rms = Mock(side_effect=lambda *args: (base_y.clone(), None))
    monkeypatch.setattr(torch_npu, "npu_rms_norm", rms, raising=False)
    beta = None if bias_state == "absent" else torch.tensor((-0.5, 0.0, 0.25, 1.0))
    layer = SimpleNamespace(weight=torch.ones(4), bias=beta, bias_loaded=bias_state == "loaded", variance_epsilon=1e-6)
    x = torch.zeros((2, 4))
    y = layernorm.AscendRMSNorm.forward_oot(layer, x)
    torch.testing.assert_close(y, base_y + beta if layer.bias_loaded else base_y, rtol=0, atol=0)
    rms.assert_called_once()
    args = rms.call_args.args
    assert args[0] is x and args[1] is layer.weight and args[2] == layer.variance_epsilon
    selected_gate.assert_not_called()
    unused_gate.assert_not_called()


@pytest.mark.parametrize("is_a5", (False, True), ids=("legacy", "a5"))
def test_gemma_without_residual_keeps_device_operator(monkeypatch, is_a5):
    selected_gate, unused_gate = selected_gates(monkeypatch, is_a5, True)
    selected_gate.side_effect = AssertionError("No residual must not enable a fused add-norm operator")
    output = torch.full((2, 4), 3.0)
    gemma = Mock(return_value=output)
    monkeypatch.setattr(layernorm.DeviceOperator, "npu_gemma_rms_norm", gemma)
    layer = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    x = torch.zeros((2, 4))
    assert layernorm.AscendGemmaRMSNorm.forward_oot(layer, x) is output
    gemma.assert_called_once()
    args = gemma.call_args.args
    assert args[0] is x and args[1] is layer.weight and args[2] == layer.variance_epsilon
    selected_gate.assert_not_called()
    unused_gate.assert_not_called()


@pytest.mark.parametrize("kind", ("rms", "gemma"))
def test_a5_forward_uses_real_cached_helper(packaged_operator, norm_backends, monkeypatch, kind):
    monkeypatch.setattr(layernorm, "is_950", lambda: True)
    monkeypatch.setattr(layernorm, "enable_a5_add_rms_norm_bias", utils.enable_a5_add_rms_norm_bias)
    monkeypatch.setattr(layernorm, "enable_custom_op", packaged_operator.generic)
    layer = SimpleNamespace(weight=torch.ones(4), bias=None, variance_epsilon=1e-6)
    cls = layernorm.AscendRMSNorm if kind == "rms" else layernorm.AscendGemmaRMSNorm
    custom, eager, _, _ = norm_backends
    packaged_operator.extension_import.assert_not_called()
    for _ in range(2):
        cls.forward_oot(layer, torch.zeros((2, 4)), torch.ones((2, 4)))
    assert custom.call_count == 2
    eager.assert_not_called()
    packaged_operator.extension_import.assert_called_once_with()
    packaged_operator.generic.assert_not_called()
    assert utils._CUSTOM_OP_ENABLED is None
