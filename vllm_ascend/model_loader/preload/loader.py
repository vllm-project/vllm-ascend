# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc
import itertools
import sys
from contextlib import contextmanager

import torch
from torch.utils.hooks import RemovableHandle
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheState

from .config import validate_preload_config

_STATIC_ROPE_CACHES = ("_cos_sin_cache", "_cos_cache", "_sin_cache", "_cos_sin_flat")


def _clear_rope_caches() -> None:
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

    from vllm_ascend.ops import rotary_embedding as rope

    _ROPE_DICT.clear()
    for name in _STATIC_ROPE_CACHES:
        setattr(rope, name, None)


@contextmanager
def _model_construction_state(vllm_config: VllmConfig):
    """Roll back constructor registries when upstream falls back to disk."""
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

    from vllm_ascend.ops import rotary_embedding as rope

    rope_dict = dict(_ROPE_DICT)
    caches = {name: getattr(rope, name) for name in _STATIC_ROPE_CACHES}
    context = vllm_config.compilation_config.static_forward_context
    previous_context = dict(context)
    dynamo = sys.modules.get("torch._dynamo.convert_frame")
    previous_hooks = frozenset(getattr(dynamo, "_bytecode_hooks", {}))
    _clear_rope_caches()
    try:
        yield
    except BaseException:
        # A compiled model can register a bound callback during construction.
        # Detach only callbacks created by this failed attempt, otherwise the
        # discarded meta model stays alive across the normal disk fallback.
        dynamo = sys.modules.get("torch._dynamo.convert_frame")
        hooks = getattr(dynamo, "_bytecode_hooks", {})
        for hook_id, callback in list(hooks.items()):
            if hook_id in previous_hooks:
                continue
            owner = getattr(callback, "__self__", None)
            handle = getattr(owner, "_bytecode_hook_handle", None)
            if (
                isinstance(owner, torch.nn.Module)
                and isinstance(handle, RemovableHandle)
                and handle.id == hook_id
                and handle.hooks_dict_ref() is hooks
            ):
                handle.remove()
        context.clear()
        context.update(previous_context)
        _ROPE_DICT.clear()
        _ROPE_DICT.update(rope_dict)
        for name, value in caches.items():
            setattr(rope, name, value)
        raise


def _rebind_rope_caches(model: torch.nn.Module) -> None:
    from vllm_ascend.ops import rotary_embedding as rope

    for module in model.modules():
        if isinstance(
            module, (rope.AscendRotaryEmbedding, rope.AscendYaRNRotaryEmbedding, rope.AscendGemma4RotaryEmbedding)
        ):
            rope._cos_sin_cache = module.cos_sin_cache
            if not isinstance(module, rope.AscendRotaryEmbedding):
                # YaRN and Gemma4 publish only the main table at construction;
                # IPC replaces that buffer but cannot update its global alias.
                return
            cos = module._rope_derived_cos_cache
            sin = module._rope_derived_sin_cache
            rope._cos_cache, rope._sin_cache = cos, sin
            width = cos.shape[-1]
            if (
                cos.untyped_storage().data_ptr() == sin.untyped_storage().data_ptr()
                and cos.stride() == sin.stride() == (2 * width, 1)
                and sin.storage_offset() == cos.storage_offset() + width
            ):
                # Restore the daemon's interleaved view without allocating.
                flat = cos.as_strided((cos.shape[0] * 2, width), (width, 1))
            else:
                # Upstream copy mode clones the two views independently.
                # Own the derived fast-path cache as a model buffer as well.
                flat = torch.stack((cos, sin), dim=1).flatten(0, 1)
                module.register_buffer("_rope_preload_interleaved_cache", flat, persistent=False)
            rope._cos_sin_flat = flat
            return


class AscendIpcModelLoader(IpcModelLoader):
    """Keep upstream transport/lifecycle and specialize NPU model binding."""

    def get_external_weight_memory(self, vllm_config: VllmConfig) -> int:
        validate_preload_config(vllm_config, mode=self.mode)
        if self.mode == "copy":
            # Wait before the worker snapshot, but do not subtract these
            # weights from its independent budget. The worker must later
            # remove a confirmed producer release from the profiling delta.
            self.resident_cache_memory = self._daemon_memory()
            return 0
        return super().get_external_weight_memory(vllm_config)

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "") -> torch.nn.Module:
        validate_preload_config(vllm_config, mode=self.mode)
        self._released_cache_memory = 0
        model = super().load_model(vllm_config, model_config, prefix)
        if self._released_cache_memory:
            model._ascend_preload_released_cache_bytes = self._released_cache_memory
        return model

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        if self.mode == "zero_copy":
            raise ValueError("In-place reload would mutate daemon-owned zero_copy weights")
        super().load_weights(model, model_config)

    def _fallback_load(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str) -> torch.nn.Module:
        # The upstream exception handler has exited, so failed constructor
        # tracebacks and model cycles can be reclaimed before disk allocation.
        gc.collect()
        torch.accelerator.empty_cache()
        return super()._fallback_load(vllm_config, model_config, prefix)

    def _send_release(self) -> None:
        if self.mode == "copy":
            # Finish asynchronous reads from imported NPU allocations before
            # asking their owner to release them. This runs once at startup.
            torch.npu.synchronize()
            free_before = torch.npu.mem_get_info()[0]
        super()._send_release()
        if self.mode == "copy":
            self._released_cache_memory = max(0, torch.npu.mem_get_info()[0] - free_before)

    def _build_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str, state: WeightCacheState
    ) -> torch.nn.Module:
        with _model_construction_state(vllm_config):
            model = super()._build_model(vllm_config, model_config, prefix, state)
            _rebind_rope_caches(model)
            return model

    def _apply_entries(self, model: torch.nn.Module, state: WeightCacheState, device_index: int) -> None:
        expected = {
            name: (tensor.shape, tensor.dtype)
            for name, tensor in itertools.chain(
                model.named_parameters(remove_duplicate=False), model.named_buffers(remove_duplicate=False)
            )
        }
        provided = set(state.entries) | set(state.aliases)
        missing = set(expected) - provided
        if missing:
            raise ValueError(f"Incomplete Ascend weight cache: missing {sorted(missing)}")
        for alias, canonical in state.aliases.items():
            if canonical not in state.entries:
                raise ValueError(f"Invalid cached alias {alias}: missing {canonical}")
        super()._apply_entries(model, state, device_index)
        actual = dict(model.named_parameters(remove_duplicate=False))
        actual.update(dict(model.named_buffers(remove_duplicate=False)))
        for name, (shape, dtype) in expected.items():
            if actual[name].shape != shape or actual[name].dtype != dtype:
                raise ValueError(f"Cached tensor {name} has an incompatible shape or dtype")
        remaining = [
            name for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()) if tensor.is_meta
        ]
        if remaining:
            raise ValueError(f"Unresolved meta tensors in Ascend weight cache: {remaining}")
