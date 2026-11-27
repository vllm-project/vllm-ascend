# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Model-level validation for Triton-only Ascend implementations.

Some OOT model modules import Triton kernels while the module is imported.
Those models cannot be made safe by an operator-level dispatch check: on 310P
the import itself must be rejected before vLLM resolves the model class.
"""

from collections.abc import Iterable
from functools import wraps

from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)

# These model implementations currently import Triton-only operators during
# model-module import and do not provide a 310P native implementation.
TRITON_ONLY_ARCHITECTURES = frozenset(
    {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
        "Step5ForCausalLM",
        "Step5ForConditionalGeneration",
        "Step5MTP",
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "Glm5NextMTPModel",
        "DeepseekV41ForCausalLM",
        "DeepseekV41ForConditionalGeneration",
        "DeepseekV41DSparkModel",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    }
)


def _architectures(model_config) -> tuple[str, ...]:
    values: list[str] = []
    for config_attr in ("hf_config", "hf_text_config"):
        config = getattr(model_config, config_attr, None)
        architectures = getattr(config, "architectures", None) or ()
        if isinstance(architectures, str):
            architectures = (architectures,)
        values.extend(architecture for architecture in architectures if isinstance(architecture, str))
    architectures = getattr(model_config, "architectures", None) or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    values.extend(architecture for architecture in architectures if isinstance(architecture, str))
    return tuple(dict.fromkeys(values))


def _unsupported_architectures(architectures: Iterable[str]) -> tuple[str, ...]:
    if get_current_hardware_profile().supports(HardwareCapability.TRITON_KERNELS):
        return ()
    return tuple(
        dict.fromkeys(
            architecture
            for architecture in architectures
            if isinstance(architecture, str) and architecture in TRITON_ONLY_ARCHITECTURES
        )
    )


def validate_architectures_triton_compatibility(architectures: Iterable[str]) -> None:
    """Reject Triton-only architectures before the lazy registry imports them."""
    unsupported = _unsupported_architectures(architectures)
    if not unsupported:
        return

    models = ", ".join(unsupported)
    raise ValueError(
        "The selected model requires Triton-Ascend, but the current hardware "
        "does not support Triton kernels (including 310P). "
        f"Unsupported architecture(s): {models}. Select a model with a native "
        "310P implementation or run this model on A2/A3/A5 hardware."
    )


def install_model_registry_gate(registry) -> None:
    """Gate vLLM's lazy model registry before it imports a model module.

    ``ModelConfig`` may inspect the registry while it is being constructed,
    before ``NPUPlatform.check_and_update_config`` runs.  Wrap the registry's
    inspection method at platform registration time so Triton-only model
    modules cannot reach their module-level imports on unsupported hardware.
    Model registration also installs this gate for callers constructing a
    ``ModelConfig`` directly without going through ``EngineArgs``.
    Remove this wrapper when vLLM's inspection path invokes the platform's
    ``verify_model_arch`` hook before importing a lazy model.
    """
    registry_type = type(registry)
    original = registry_type.inspect_model_cls
    if getattr(original, "_ascend_triton_gate", False) is True:
        return

    @wraps(original)
    def inspect_model_cls(self, architectures, model_config):
        values = (architectures,) if isinstance(architectures, str) else architectures
        validate_architectures_triton_compatibility(values)
        return original(self, architectures, model_config)

    inspect_model_cls._ascend_triton_gate = True  # type: ignore[attr-defined]
    registry_type.inspect_model_cls = inspect_model_cls


def validate_model_triton_compatibility(model_config) -> None:
    """Reject a Triton-only model before its implementation module is loaded.

    Hardware capability is intentionally checked here rather than package
    availability. An installed ``triton`` wheel cannot enable Triton-Ascend
    on 310P, and the error points users at a model path that can run there.
    """
    validate_architectures_triton_compatibility(_architectures(model_config))
