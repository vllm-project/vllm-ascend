# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only Worker RPC probes; never imported by normal serving."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCWeightTransferEngine
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

SUFFIXES = ("tid2eid", "bias_vl", "e_score_correction_bias", "attn_sink")
PATTERNS = ["*.mlp.gate." + s for s in SUFFIXES[:3]] + ["*.self_attn.attn_sink"]


def is_tracked(name):
    return name.rsplit(".", 1)[-1] in SUFFIXES


def variant_tensor(tensor):
    if not tensor.dtype.is_floating_point:
        return (tensor + 1) % 8
    return (tensor.to(torch.bfloat16) + 0.015625).to(tensor.dtype)


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class RetentionProbe:
    # Worker injects this extension as a mixin; these members come from Worker.
    model_runner: NPUModelRunner | NPUModelRunnerV2
    model_config: ModelConfig
    weight_transfer_engine: NPUIPCWeightTransferEngine
    _sleep_saved_parameters: dict[str, torch.Tensor]
    _wsc_events: list[dict[str, object]]
    _wsc_hook_events: list[dict[str, object]]
    _wsc_phase: str

    def wsc_state(self):
        model = self.model_runner.model
        params = {
            name: {"sha256": digest(p), "ptr": p.data_ptr(), "shape": list(p.shape), "dtype": str(p.dtype)}
            for name, p in model.named_parameters()
            if is_tracked(name)
        }
        return {
            "parameters": params,
            "saved": {n: digest(p) for n, p in self._sleep_saved_parameters.items()},
            "configured": self.model_config.sleep_preserve_parameter_names,
            "old_snapshot_present": any(hasattr(m, "_runtime_tensor_backup") for m in model.modules()),
        }

    def wsc_install(self):
        self._wsc_events = []
        self._wsc_hook_events = []
        model = self.model_runner.model
        original_reinit = model.reinitialize_runtime_params

        def monitored_reinit(*args, **kwargs):
            self._wsc_hook_events.append({"stage": self._wsc_phase + ":hook-before", "state": self.wsc_state()})
            result = original_reinit(*args, **kwargs)
            self._wsc_hook_events.append({"stage": self._wsc_phase + ":hook-after", "state": self.wsc_state()})
            return result

        model.reinitialize_runtime_params = monitored_reinit
        engine = self.weight_transfer_engine
        for method in ("start_weight_update", "finish_weight_update"):
            original = getattr(engine, method)

            def monitored(*args, _original=original, _method=method, **kwargs):
                self._wsc_phase = _method
                self._wsc_events.append({"stage": _method + ":before", "state": self.wsc_state()})
                result = _original(*args, **kwargs)
                self._wsc_events.append({"stage": _method + ":after", "state": self.wsc_state()})
                return result

            setattr(engine, method, monitored)
        return self.wsc_state()

    def wsc_variant(self):
        with torch.no_grad():
            for name, param in self.model_runner.model.named_parameters():
                if is_tracked(name):
                    param.copy_(variant_tensor(param))
        return self.wsc_state()

    def wsc_poison(self):
        # The PR keeps the graph weight pool resident. Destroy just these test
        # values after the real CPU backup to prove wake restores the backup,
        # rather than passing merely because the resident values survived sleep.
        names = self._sleep_saved_parameters
        assert names
        with torch.no_grad():
            for name in names:
                p = self.model_runner.model.get_parameter(name)
                p.fill_(float("nan") if p.dtype.is_floating_point else -1)
        return self.wsc_state()

    def wsc_configure(self, mode):
        self.model_config.sleep_preserve_parameter_names = PATTERNS.copy() if mode == "retain" else []
        return self.wsc_state()

    def wsc_events(self):
        return self._wsc_events

    def wsc_hook_events(self):
        return self._wsc_hook_events
