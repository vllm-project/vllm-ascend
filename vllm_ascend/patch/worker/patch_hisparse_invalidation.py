# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Replace A5 HiSparse invalidation's CANN searchsorted chain with one Triton kernel.

Upstream HiSparseRuntime currently performs invalidation directly in Torch.
Install this wrapper only when constructing the Ascend cache backend. Other
backends and unsupported layouts retain the original method. Remove this
patch when upstream exposes a backend invalidation hook; keep the native
implementation behind that hook. No upstream source file is modified.
"""

from functools import wraps


def install() -> None:
    from vllm.v1.hisparse.runtime import HiSparseRuntime

    original = HiSparseRuntime.invalidate_sorted_slots
    if getattr(original, "_ascend_invalidation_patch", False):
        return

    @wraps(original)
    def invalidate(self, sorted_slots, state_indices):
        from vllm_ascend.ops.triton.hisparse_invalidation import try_invalidate

        if try_invalidate(self.cache_ops, self.index_group.device_global_indices, sorted_slots, state_indices):
            return
        return original(self, sorted_slots, state_indices)

    invalidate._ascend_invalidation_patch = True
    HiSparseRuntime.invalidate_sorted_slots = invalidate
