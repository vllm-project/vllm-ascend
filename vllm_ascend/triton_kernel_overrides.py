# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Suffix overrides using the dispatcher registration pattern from PR #19153.

Keys name the defining module so the dispatcher also rebinds from-import
copies. Values are plain launch adapters: their grids are reconstructed from
tensor metadata, and observation includes a second launch to clear markers.
Keep this table limited to suffix until the remaining overrides migrate in
#19153. Remove entries when upstream kernels support Ascend compilation and
the bucket-reduction observer no longer provides a performance benefit.
"""

from collections.abc import Callable
from typing import Any

from vllm_ascend.ops.triton.spec_decode.suffix_corpus import launch_suffix_ingest
from vllm_ascend.ops.triton.spec_decode.suffix_finalize import launch_suffix_finalize
from vllm_ascend.ops.triton.spec_decode.suffix_local_scan import launch_suffix_local_scan
from vllm_ascend.ops.triton.spec_decode.suffix_observe import launch_suffix_observe

TRITON_KERNEL_OVERRIDES: dict[str, Callable[..., Any]] = {
    "vllm.v1.worker.gpu.spec_decode.suffix.speculator._suffix_local_scan_kernel": launch_suffix_local_scan,
    "vllm.v1.worker.gpu.spec_decode.suffix.speculator._suffix_finalize_kernel": launch_suffix_finalize,
    "vllm.v1.worker.gpu.spec_decode.suffix.speculator._suffix_ingest_kernel": launch_suffix_ingest,
    "vllm.v1.worker.gpu.spec_decode.suffix.speculator._suffix_observe_kernel": launch_suffix_observe,
}

# Worker initialization and bare operator tests can register in the same
# process. Latch only after success so a failed registration can be retried.
_kernels_registered = False


def register_triton_kernels() -> None:
    """Install suffix overrides once per process before any kernel launches."""
    global _kernels_registered
    if _kernels_registered:
        return
    from vllm.triton_utils.dispatcher import register_kernels

    register_kernels(TRITON_KERNEL_OVERRIDES)
    _kernels_registered = True
