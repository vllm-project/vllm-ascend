# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side evidence for native parameter retention boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCWeightTransferEngine
    from vllm_ascend.worker.v1.model_runner import NPUModelRunner
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

SUFFIXES = ("tid2eid", "bias_vl", "e_score_correction_bias", "attn_sink")
PATTERNS = ["*.mlp.gate." + s for s in SUFFIXES[:3]] + ["*.self_attn.attn_sink"]


def is_tracked(name):
    return name.rsplit(".", 1)[-1] in SUFFIXES


def variant_tensor(tensor, num_experts=8):
    if not tensor.dtype.is_floating_point:
        return (tensor + 1) % num_experts
    return (tensor.to(torch.bfloat16) + 0.015625).to(tensor.dtype)


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class RetentionProbe:
    model_runner: NPUModelRunner | NPUModelRunnerV2
    model_config: ModelConfig
    weight_transfer_engine: NPUIPCWeightTransferEngine
    _sleep_saved_parameters: dict[str, torch.Tensor]
    _restore_sleep_parameters: Callable[[torch.nn.Module], None]
    _wsc_events: list[dict[str, object]]
    _wsc_finalize_events: list[dict[str, object]]
    _wsc_poison_restore: bool
    _wsc_poisoned: dict[str, object]

    def wsc_saved_state(self):
        # Device storage is unmapped during sleep: read only the real CPU backup.
        return {"saved": {n: digest(p) for n, p in self._sleep_saved_parameters.items()}}

    def wsc_state(self):
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

        model = self.model_runner.model
        parameters = {}
        kernel_names: dict[int, str] = {}
        aliases = {}
        meta_names = []
        for name, parameter in model.named_parameters():
            if not is_tracked(name):
                continue
            if parameter.is_meta:
                # Native reload uses temporary meta placeholders. Inspect the
                # retained kernel tensor, which is what captured graphs use.
                module_name, local_name = name.rsplit(".", 1)
                info = get_layerwise_info(model.get_submodule(module_name))
                assert info.kernel_tensors is not None
                parameter = info.kernel_tensors[0][local_name]
                meta_names.append(name)
            # Separate meta placeholders may refer to one original parameter.
            # Deduplicate by the resolved object, retaining the alias evidence.
            if id(parameter) in kernel_names:
                aliases[name] = kernel_names[id(parameter)]
                continue
            kernel_names[id(parameter)] = name
            parameters[name] = parameter
        return {
            "reload_meta_parameters": meta_names,
            "reload_kernel_aliases": aliases,
            "parameters": {
                name: {"sha256": digest(p), "ptr": p.data_ptr(), "shape": list(p.shape), "dtype": str(p.dtype)}
                for name, p in parameters.items()
            },
            **self.wsc_saved_state(),
            "configured": self.model_config.sleep_preserve_parameter_names,
            "old_snapshot_present": any(hasattr(m, "_runtime_tensor_backup") for m in model.modules()),
        }

    def wsc_install(self):
        from vllm.model_executor.model_loader import reload

        self._wsc_events = []
        self._wsc_finalize_events = []
        self._wsc_poison_restore = False
        self._wsc_poisoned = {}
        original_restore = self._restore_sleep_parameters

        def monitored_restore(model):
            # Poison after allocator.wake_up maps storage, before Worker copies
            # its CPU snapshot. Never touch an unmapped allocation during sleep.
            if self._wsc_poison_restore:
                self._wsc_poison_restore = False
                with torch.no_grad():
                    for name in self._sleep_saved_parameters:
                        param = model.get_parameter(name)
                        param.fill_(float("nan") if param.dtype.is_floating_point else -1)
                self._wsc_poisoned = self.wsc_state()
            return original_restore(model)

        self._restore_sleep_parameters = monitored_restore
        original_finalize = reload.finalize_layerwise_reload

        def monitored_finalize(*args, **kwargs):
            self._wsc_finalize_events.append({"stage": "finalize:before", "state": self.wsc_state()})
            result = original_finalize(*args, **kwargs)
            self._wsc_finalize_events.append({"stage": "finalize:after", "state": self.wsc_state()})
            return result

        reload.finalize_layerwise_reload = monitored_finalize
        engine = self.weight_transfer_engine
        for method in ("start_weight_update", "finish_weight_update"):
            original = getattr(engine, method)

            def monitored(*args, _original=original, _method=method, **kwargs):
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
                    param.copy_(variant_tensor(param, self.model_config.hf_config.n_routed_experts))
        return self.wsc_state()

    def wsc_poison(self):
        assert self._sleep_saved_parameters
        self._wsc_poison_restore = True
        return self.wsc_saved_state()

    def wsc_poisoned_state(self):
        return self._wsc_poisoned

    def wsc_configure(self, mode):
        self.model_config.sleep_preserve_parameter_names = PATTERNS.copy() if mode == "retain" else []
        return self.wsc_state()

    def wsc_events(self):
        return self._wsc_events

    def wsc_finalize_events(self):
        return self._wsc_finalize_events
