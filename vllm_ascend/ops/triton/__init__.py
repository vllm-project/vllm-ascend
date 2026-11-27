"""Triton implementation namespace.

Keep the hardware and backend gate at the package boundary as a final safety
net for model modules that import a Triton implementation directly. Callers
should still prefer a lazy import guarded by ``supports_triton()`` so native
fallbacks remain selectable; this boundary prevents an unsupported model from
silently loading a Triton-only implementation later in initialization.
"""

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

if not get_current_hardware_profile().supports(HardwareCapability.TRITON_KERNELS):
    raise RuntimeError(
        "Triton-Ascend implementations are unavailable on this hardware or installation. "
        "Select a native/fallback model path, or use hardware with Triton-Ascend support."
    )
