#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Rebuild host-offloaded (Engram ``cpu_offload``) parameters on their host storage.

Where this patch hooks, why it is needed, how it works and when it can be removed are
documented with the patch registry entry, ``worker/patch_offload_reload.py`` in
``vllm_ascend/patch/__init__.py``.
"""

from importlib import import_module
from typing import Any

import torch
from vllm.logger import logger

_LAYERWISE_MODULE = "vllm.model_executor.model_loader.reload.layerwise"
_META_MODULE = "vllm.model_executor.model_loader.reload.meta"

# Producer side of the attribute-name contract; imported lazily so worker
# startup does not depend on vLLM shipping the DeepSeek-V4.1 package.
_EMBEDDING_MODULE = "vllm_ascend.models.deepseek_v41.engram.embedding"

# Import-time idempotence guard: `install()` must run once per process, before
# the first reload, and rebinding twice would wrap the wrappers.
_installed = False


def _host_offload_storage_attr() -> str:
    return import_module(_EMBEDDING_MODULE).HOST_OFFLOAD_STORAGE_ATTR


def _host_offload_storage(tensor: torch.Tensor) -> torch.Tensor | None:
    """Host range ``tensor`` was created on, or ``None`` for a device tensor."""
    storage = tensor.__dict__.get(_host_offload_storage_attr())
    return storage if isinstance(storage, torch.Tensor) else None


def _make_materialize_layer(meta_module: Any):
    original = meta_module.materialize_layer

    def materialize_layer(layer: torch.nn.Module, info: Any) -> None:
        """``materialize_layer`` that rebuilds offloaded parameters on host storage."""
        parameters, buffers = meta_module.get_layer_params_buffers(layer)
        kernel_parameters = info.kernel_tensors[0] if info.kernel_tensors is not None else {}
        offloaded: dict[str, torch.Tensor] = {}
        for name, tensor in {**parameters, **buffers}.items():
            host_storage = _host_offload_storage(tensor)
            if host_storage is None or not tensor.is_meta:
                continue
            if tuple(tensor.size()) != tuple(host_storage.size()) or tensor.dtype != host_storage.dtype:
                logger.warning_once(
                    "Host-offloaded parameter %s changed layout (%s %s -> %s %s); materializing it on the device.",
                    name,
                    tuple(tensor.size()),
                    tensor.dtype,
                    tuple(host_storage.size()),
                    host_storage.dtype,
                )
                continue
            # ``torch.frombuffer`` storage cannot be resized, so the parameter
            # keeps aliasing the range the lookup kernel already reads. Reusing
            # the captured kernel parameter keeps the object identity the
            # reload's weight loaders were bound to.
            param = kernel_parameters.get(name)
            if param is None or tuple(param.size()) != tuple(host_storage.size()):
                param = torch.nn.Parameter(host_storage, requires_grad=False)
            layer.register_parameter(name, param)
            offloaded[name] = host_storage

        if not offloaded:
            original(layer, info)
            return

        info.host_offload_params = offloaded
        logger.debug("Materialized host-offloaded parameters %s on their host storage", sorted(offloaded))

        # Every other meta tensor of the layer still needs its device materialization.
        if any(name not in offloaded and tensor.is_meta for name, tensor in parameters.items()):
            original(layer, info)

    return materialize_layer


def _make_copy_and_restore_kernel_tensors(layerwise_module: Any):
    original = layerwise_module._copy_and_restore_kernel_tensors

    def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: Any) -> None:
        """``_copy_and_restore_kernel_tensors`` without the offloaded device round trip."""
        offloaded = getattr(info, "host_offload_params", None)
        if not offloaded:
            original(layer, info)
            return

        parameters, buffers = info.kernel_tensors
        for name, host_storage in offloaded.items():
            param = getattr(layer, name)
            # The parameter was rebuilt on the host range, so the range already
            # holds the update: a cross-device copy would be the per-rank
            # materialization this patch exists to avoid.
            assert param.data_ptr() == host_storage.data_ptr(), (
                f"host-offloaded parameter {name!r} no longer aliases its host storage"
            )

        for name, param in parameters.items():
            if name in offloaded:
                continue
            param.data.copy_(getattr(layer, name))
        non_persistent = info.kernel_non_persistent_buffers
        loaded_tensor_names = {name for name, _ in info.loaded_weights}
        for name, buffer in buffers.items():
            if name not in layer._buffers:
                continue
            if name in non_persistent and name not in loaded_tensor_names:
                continue
            buffer.data.copy_(getattr(layer, name))

        layerwise_module._place_kernel_tensors(layer, info)
        del info.host_offload_params

    return _copy_and_restore_kernel_tensors


def install() -> None:
    """Rebind the layerwise reload hooks; no-op when vLLM does not expose them."""
    global _installed
    if _installed:
        return
    try:
        # Both modules are rebound below, which ``ModuleType`` does not describe.
        layerwise_module: Any = import_module(_LAYERWISE_MODULE)
        meta_module: Any = import_module(_META_MODULE)
    except ImportError as exc:
        logger.warning_once("Host-offload aware layerwise reload disabled: %s", exc)
        _installed = True
        return

    if not hasattr(meta_module, "materialize_layer") or not hasattr(meta_module, "get_layer_params_buffers"):
        logger.warning_once("Host-offload aware layerwise reload disabled: %s has no materialize_layer", _META_MODULE)
        _installed = True
        return
    if not hasattr(layerwise_module, "materialize_layer") or not hasattr(layerwise_module, "get_layer_params_buffers"):
        logger.warning_once(
            "Host-offload aware layerwise reload disabled: %s does not relay the reload helpers", _LAYERWISE_MODULE
        )
        _installed = True
        return
    if not hasattr(layerwise_module, "_copy_and_restore_kernel_tensors") or not hasattr(
        layerwise_module, "_place_kernel_tensors"
    ):
        logger.warning_once(
            "Host-offload aware layerwise reload disabled: %s has no kernel-tensor hooks", _LAYERWISE_MODULE
        )
        _installed = True
        return

    # ``layerwise`` imports the helper by name, so the call site reads
    # ``layerwise.materialize_layer``; rebinding only ``meta.materialize_layer``
    # would leave the reload on the upstream implementation. Both names are
    # rebound: the relay is what the reload calls, the definition also covers
    # modules that imported it from ``meta`` before this patch was applied.
    patched_materialize_layer = _make_materialize_layer(meta_module)
    meta_module.materialize_layer = patched_materialize_layer
    layerwise_module.materialize_layer = patched_materialize_layer
    layerwise_module._copy_and_restore_kernel_tensors = _make_copy_and_restore_kernel_tensors(layerwise_module)
    _installed = True


install()
