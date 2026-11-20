# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in A3 RecoverSSM operators for GDN decode and single-layer PLE.

Pass the model's reference verify and commit functions to the wrappers.
The specialized path requires BF16-rounded gates and forward-order FP32
replay. Unsupported layouts delegate to the supplied functions.
TileLang-Ascend is loaded on the first supported specialization.
"""

from .dispatch import compact_conv_state as compact_ple_conv_state
from .dispatch import wrap_commit, wrap_verify

__all__ = ["compact_ple_conv_state", "wrap_commit", "wrap_verify"]
