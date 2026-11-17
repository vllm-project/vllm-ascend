# SPDX-License-Identifier: Apache-2.0

"""Hardware eligibility shared by FLA GDN metadata and execution paths."""

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


def is_fla_gdn_chunk32_supported() -> bool:
    """Enable chunk32 on A5."""
    return get_current_hardware_profile().supports(HardwareCapability.FLA_GDN_PREFILL_CHUNK32)
