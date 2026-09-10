# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the ``_C_ascend`` torch extension for tests that call the ops.

``vllm_ascend`` loads ``vllm_ascend_C`` lazily (``enable_custom_op`` is only
invoked during engine startup), so a plain ``import vllm_ascend`` does NOT
register ``torch.ops._C_ascend``. Tests that call the compiled ops must load
the extension explicitly.

This module itself only imports torch; the vllm_ascend import happens inside
the function so pure-torch reference tests stay dependency-free.
"""

from __future__ import annotations


def ensure_c_ascend_loaded(required_op: str) -> None:
    """Load ``vllm_ascend_C`` and register ``torch.ops._C_ascend``.

    Uses the same bootstrap path as the engine (``enable_custom_op``). When
    the engine gate skips the import (Ascend 950 / batch-invariant mode), the
    extension is still imported directly so the compiled ops remain testable.
    """
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

    if not enable_custom_op():
        bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C  # noqa: F401

    import torch

    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, required_op):
        raise RuntimeError(
            f"torch.ops._C_ascend.{required_op} is not registered. Rebuild the "
            "torch extension (COMPILE_CUSTOM_KERNELS=1 pip install -e . "
            "--no-build-isolation) and verify with "
            "'python -c \"import vllm_ascend.vllm_ascend_C\"'."
        )
