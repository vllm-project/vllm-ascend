# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bind the whole process tree to the NPUs selected by ``--device-ids``.

``vLLM`` resolves the model architecture (and probes multimodal support) while
building the engine config, i.e. in the API server process and before any
worker binds its NPU.  That import path reaches
``vllm.third_party.flash_linear_attention.ops.utils.get_all_max_shared_mem()``,
which loops over ``device_count()`` and queries every *visible* device.  With
``ASCEND_RT_VISIBLE_DEVICES=4,5 --device-ids 1`` the server therefore creates a
small ACL context on chip 4 as well, and ``npu-smi`` reports a ``vllm`` process
on both cards even though only chip 5 serves the model.

Fix: resolve ``--device-ids`` against ``ASCEND_RT_VISIBLE_DEVICES`` once, before
any NPU call, and rewrite the environment to exactly the selected physical
chips while renumbering the ids to ``0..N-1``.  Everything downstream --
``ParallelConfig.assigned_physical_gpu_ids``,
``NPUPlatform.logical_device_id_to_visible_device_id`` and
``torch.npu.set_device`` -- resolves the same physical devices, so no other
chip is ever touched.  Worker processes inherit the narrowed environment, so
their device ordinals stay consistent as well.
"""

from __future__ import annotations

import os

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import logger

_DEVICE_CONTROL_ENV_VAR = "ASCEND_RT_VISIBLE_DEVICES"


def _split_visible_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _narrow_visible_devices_to_device_ids(engine_args) -> None:
    """Rewrite the visible NPU set so it contains only ``--device-ids``."""
    device_ids = getattr(engine_args, "device_ids", None)
    if not device_ids:
        return

    # Ray selects devices through placement groups; --device-ids is ignored and
    # vLLM already warns about it.
    if getattr(engine_args, "distributed_executor_backend", None) == "ray":
        return

    # UUIDs need the CANN-side translation, which already resolves them to the
    # physical chip; rewriting the environment would break that mapping.
    if any(not isinstance(device_id, int) for device_id in device_ids):
        return

    current_env = os.environ.get(_DEVICE_CONTROL_ENV_VAR, "")
    visible = _split_visible_devices(current_env)
    if not visible:
        # Nothing restricts us today: the ids are already physical chip ids.
        physical = list(device_ids)
    else:
        # Integer ids are zero-based indices into the visible list.  Leave an
        # out-of-range value alone so vLLM reports it with its own message.
        if any(device_id < 0 or device_id >= len(visible) for device_id in device_ids):
            return
        physical = [visible[device_id] for device_id in device_ids]

    if len(set(physical)) != len(physical):
        # vLLM rejects duplicate ids later; keep the error path untouched.
        return

    new_env = ",".join(physical)
    if new_env == current_env:
        return

    os.environ[_DEVICE_CONTROL_ENV_VAR] = new_env
    engine_args.device_ids = list(range(len(physical)))
    logger.info(
        "Narrowed %s from %s to %s for device ids %s; no other NPU will be initialized.",
        _DEVICE_CONTROL_ENV_VAR,
        current_env or "<unset>",
        new_env,
        engine_args.device_ids,
    )


_original_create_engine_config = EngineArgs.create_engine_config


def _create_engine_config_patched(self, *args, **kwargs):
    # Must run before create_model_config(): resolving the model class imports
    # modules that enumerate and initialize every visible NPU.
    _narrow_visible_devices_to_device_ids(self)
    return _original_create_engine_config(self, *args, **kwargs)


EngineArgs.create_engine_config = _create_engine_config_patched
