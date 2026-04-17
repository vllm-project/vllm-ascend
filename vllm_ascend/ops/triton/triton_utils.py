import os
from typing import Any

import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON, tl, triton

# --- Global Environment Injection ---
# The BiShengIR compiler (used by Triton) may not recognize the 'Ascend310P3'
# target string. We force a redirect to 'Ascend310P1' when 310P3 is detected.
# Setting this at the module level ensures that any worker process forked
# by the engine inherits the compatible SOC_MODEL configuration.
_current_soc = os.getenv("ASCEND_RT_SOC_MODEL", "")
if "310P3" in _current_soc or not _current_soc:
    os.environ["ASCEND_RT_SOC_MODEL"] = "Ascend310P1"


def _remap_to_str(model_val: Any) -> str:
    """
    Helper for the Triton compiler: always returns a recognized target string.
    Ensures that unrecognized hardware identifiers are mapped to supported
    base models (e.g., 310P3 -> 310P1, 910B2C -> 910B2).
    """
    m_str = str(model_val)

    # Re-enforce environment variable inside the function to ensure that
    # Triton compilation sub-processes (spawned via subprocess.run)
    # consistently see the 'Ascend310P1' target during multi-card (TP) execution.
    if "310P" in m_str:
        os.environ["ASCEND_RT_SOC_MODEL"] = "Ascend310P1"
        return "Ascend310P1"

    # Mapping for the 910B series to a recognized base version.
    if "910B" in m_str and "Ascend910B" not in m_str:
        return "Ascend910B2"

    # Fallback for unknown or malformed SOC identifiers.
    unrecognized_list = ["Unknown", "", "222"]
    if m_str in unrecognized_list:
        return "Ascend310P1"

    return m_str


# --- Workaround: Hijacking torch_npu APIs for Compatibility ---
try:
    # 1. Patch get_soc_spec: Fixes the --target parameter for the compiler.
    if hasattr(torch_npu.npu, "get_soc_spec"):
        _orig_get_soc_spec = torch_npu.npu.get_soc_spec

        def _patched_get_soc_spec():
            spec = _orig_get_soc_spec()
            model = getattr(spec, "soc_model", "")
            # Use object.__setattr__ to bypass read-only restrictions on spec objects.
            object.__setattr__(spec, "soc_model", _remap_to_str(model))
            return spec

        torch_npu.npu.get_soc_spec = _patched_get_soc_spec

    # 2. Patch get_soc_version: Fixes internal range checks (e.g., 220 <= v <= 225).
    if hasattr(torch_npu.npu, "get_soc_version"):
        _orig_get_soc_version = torch_npu.npu.get_soc_version

        def _patched_get_soc_version():
            version = _orig_get_soc_version()
            # Compatibility: Standardize version returns to integer for range comparisons.
            if isinstance(version, str):
                if "910B" in version:
                    return 222
                try:
                    return int(version)
                except ValueError:
                    return version
            return version

        torch_npu.npu.get_soc_version = _patched_get_soc_version

except Exception:
    # Silent failure to prevent the patch itself from breaking the initialization.
    pass

_NUM_AICORE = -1
_NUM_VECTORCORE = -1
_extension_module = None

if HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module  # type: ignore
    except ImportError:
        _extension_module = None


def _resolve_triton_ascend_op(op_name: str):
    if not HAS_TRITON:
        raise RuntimeError(f"Triton op '{op_name}' cannot be resolved because HAS_TRITON is False")

    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


if HAS_TRITON:
    insert_slice = _resolve_triton_ascend_op("insert_slice")
    extract_slice = _resolve_triton_ascend_op("extract_slice")
    get_element = _resolve_triton_ascend_op("get_element")
else:
    insert_slice = None
    extract_slice = None
    get_element = None


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE == -1 and HAS_TRITON:
        device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        _NUM_AICORE = device_properties.get("num_aicore", -1)
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."


def get_aicore_num():
    global _NUM_AICORE
    assert _NUM_AICORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE
