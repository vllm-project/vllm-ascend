# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.reload.inplace import reload_weights

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.gdn import _refresh_recurrent_gate_bias
from vllm_ascend.worker.worker import NPUWorker


@pytest.fixture(autouse=True)
def _use_a5_profile(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.worker.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A5),
    )


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType._310P])
@pytest.mark.parametrize("operation", ["reload_weights", "finish_weight_update"])
def test_non_a5_weight_update_does_not_import_gdn(monkeypatch, device_type, operation):
    monkeypatch.setattr(
        "vllm_ascend.worker.worker.get_current_hardware_profile", lambda: get_hardware_profile(device_type)
    )
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = MagicMock()
    worker.weight_transfer_engine = MagicMock()
    worker._weight_update_active = True
    original_import = builtins.__import__

    def reject_gdn_import(name, *args, **kwargs):
        if name == "vllm_ascend.ops.gdn" or name.startswith(("fla_npu", "triton")):
            raise AssertionError(f"Non-A5 weight update imported optional GDN dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gdn_import)
    getattr(worker, operation)()

    worker.model_runner.get_model.assert_not_called()
    if operation == "reload_weights":
        worker.model_runner.reload_weights.assert_called_once_with()
    else:
        worker.weight_transfer_engine.finish_weight_update.assert_called_once_with()
        worker.model_runner.reset_lora_state.assert_called_once_with()
        assert not worker._weight_update_active


def _make_model():
    model = nn.Module()
    model.linear_attn = nn.Module()
    layer = model.linear_attn
    layer.dt_bias = nn.Parameter(torch.tensor([0.3, -1.7], dtype=torch.bfloat16))
    layer.register_buffer("_recurrent_dt_bias_fp32", torch.empty(2), persistent=False)
    _refresh_recurrent_gate_bias(layer)
    return model


def _make_worker(model, use_v2=True):
    worker = NPUWorker.__new__(NPUWorker)
    worker.use_v2_model_runner = use_v2
    config = SimpleNamespace(model_config=SimpleNamespace(quantization=None), load_config=None, lora_config=None)

    def load(**kwargs):
        with torch.no_grad():
            reload_weights(config, model, **kwargs)

    worker.model_runner = SimpleNamespace(
        model=model, get_model=lambda: model, reload_weights=load, reset_lora_state=MagicMock()
    )
    return worker


@pytest.mark.parametrize("use_v2", [False, True])
def test_kernel_format_reload_refreshes_gdn_bias_in_place(use_v2):
    model = _make_model()
    worker = _make_worker(model, use_v2)
    layer = model.linear_attn
    cached = layer._recurrent_dt_bias_fp32
    address = cached.data_ptr()

    for values in ([1.5, 2.5], [-2.0, 0.75]):
        replacement = torch.tensor(values, dtype=torch.bfloat16)
        worker.reload_weights(
            weights_iterator=[("linear_attn.dt_bias", replacement)],
            weights_path=None,
            is_checkpoint_format=False,
        )
        torch.testing.assert_close(layer.dt_bias, replacement, rtol=0, atol=0)
        torch.testing.assert_close(cached, replacement.float(), rtol=0, atol=0)
        assert layer._recurrent_dt_bias_fp32 is cached
        assert cached.data_ptr() == address
        assert cached.dtype == torch.float32
        assert "linear_attn._recurrent_dt_bias_fp32" not in model.state_dict()


def test_reload_model_without_gdn_buffer():
    model = nn.Linear(2, 2, bias=False)
    worker = _make_worker(model)
    replacement = torch.full_like(model.weight, 2.0)
    worker.reload_weights(weights_iterator=[("weight", replacement)], weights_path=None, is_checkpoint_format=False)
    torch.testing.assert_close(model.weight, replacement)
    assert list(model.named_buffers()) == []


def test_failed_reload_does_not_refresh_gdn_bias():
    model = _make_model()
    worker = _make_worker(model)
    before = model.linear_attn._recurrent_dt_bias_fp32.clone()

    def failed_reload(**kwargs):
        with torch.no_grad():
            model.linear_attn.dt_bias.fill_(4)
        raise RuntimeError("reload failed")

    worker.model_runner.reload_weights = failed_reload
    with pytest.raises(RuntimeError, match="reload failed"):
        worker.reload_weights(is_checkpoint_format=False)
    torch.testing.assert_close(model.linear_attn._recurrent_dt_bias_fp32, before)


def test_weight_transfer_refreshes_gdn_bias_only_after_finish():
    model = _make_model()
    worker = _make_worker(model)
    worker._weight_update_active = True
    worker.weight_transfer_engine = MagicMock()
    layer = model.linear_attn
    cached = layer._recurrent_dt_bias_fp32
    before, address = cached.clone(), cached.data_ptr()

    def update_chunk(_):
        with torch.no_grad():
            layer.dt_bias.fill_(2)

    def finish():
        with torch.no_grad():
            layer.dt_bias.fill_(3)

    worker.weight_transfer_engine.update_weights.side_effect = update_chunk
    worker.weight_transfer_engine.finish_weight_update.side_effect = finish
    worker.update_weights({})
    torch.testing.assert_close(cached, before)
    worker.finish_weight_update()
    torch.testing.assert_close(cached, layer.dt_bias.float(), rtol=0, atol=0)
    assert cached.data_ptr() == address
    assert not worker._weight_update_active
    worker.model_runner.reset_lora_state.assert_called_once_with()


def test_failed_weight_transfer_finish_does_not_refresh_gdn_bias():
    model = _make_model()
    worker = _make_worker(model)
    worker._weight_update_active = True
    worker.weight_transfer_engine = MagicMock()
    before = model.linear_attn._recurrent_dt_bias_fp32.clone()

    def failed_finish():
        with torch.no_grad():
            model.linear_attn.dt_bias.fill_(4)
        raise RuntimeError("finish failed")

    worker.weight_transfer_engine.finish_weight_update.side_effect = failed_finish
    with pytest.raises(RuntimeError, match="finish failed"):
        worker.finish_weight_update()
    torch.testing.assert_close(model.linear_attn._recurrent_dt_bias_fp32, before)
    worker.model_runner.reset_lora_state.assert_not_called()
