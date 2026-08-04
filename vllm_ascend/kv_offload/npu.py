# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compatibility exports for the native NPU KV-cache offload spec.

New configurations should import from :mod:`vllm_ascend.kv_offload.native.npu`.
This module keeps configurations created before the package reorganization
working without duplicating the implementation.
"""

from vllm_ascend.kv_offload.native.npu import (
    CPUOffloadingSpec,
    NPUOffloadingSpec,
)

__all__ = ["CPUOffloadingSpec", "NPUOffloadingSpec"]
