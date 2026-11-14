# SPDX-License-Identifier: Apache-2.0

"""Runtime eligibility shared by FLA GDN metadata and execution paths."""

from functools import lru_cache
from importlib import metadata

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

# No supporting wheel is validated yet, including 26.9.1+dev153268a.
# Git-hash version suffixes cannot be ordered; future additions require NPU validation.
_FLA_GDN_CHUNK32_VERSIONS: frozenset[str] = frozenset()


@lru_cache(maxsize=1)
def _get_fla_a5_version() -> str | None:
    """Inspect the installed distribution once per worker, never once per layer."""
    try:
        return metadata.version("flash-linear-attention-npu-a5")
    except (metadata.PackageNotFoundError, OSError):
        return None


def is_fla_gdn_chunk32_supported() -> bool:
    """Require both the target hardware and an approved FLA build."""
    return (
        get_current_hardware_profile().supports(HardwareCapability.FLA_GDN_PREFILL_CHUNK32)
        and _get_fla_a5_version() in _FLA_GDN_CHUNK32_VERSIONS
    )
