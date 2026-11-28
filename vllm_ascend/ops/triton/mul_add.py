# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of the muls_add Triton kernel.

The mul_add and muls_add Triton kernels are identical. Keep a single
implementation in ``muls_add.py`` and re-export it here so the
deepseek_v4 / deepseek_v41 callers (which import from ``mul_add``) and the
compiler pass / custom-op registration (which import from ``muls_add``)
share one kernel and one set of input validation, instead of maintaining
two copies of the same guard.
"""

from vllm_ascend.ops.triton.muls_add import muls_add_kernel, muls_add_triton  # noqa: F401
