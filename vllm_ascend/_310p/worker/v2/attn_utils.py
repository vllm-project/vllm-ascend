# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P-only KV cache view helpers.

310P paged attention stores K/V in ACL FRACTAL_NZ. The backend reports a 5D
shape ``(2, num_blocks, hidden/16, block_size, 16)`` rather than the ND layout
``(2, num_blocks, block_size, num_kv_heads, head_size)`` that shared
``vllm_ascend.worker.v2.attn_utils._reshape_kv_cache_v2`` assumes.

Do not fold this into the shared worker: other NPU platforms keep the mainline
reshape (last dim = head size, optional asymmetric ``head_size_v``). 310P V2
allocates NZ tensors in ``NPUModelRunner310V2._allocate_kv_cache_tensors_310p``
and never calls the shared reshape path.
"""

from __future__ import annotations

from typing import Any


def get_310p_non_mla_kv_cache_shapes(
    kv_cache_shape: tuple[int, ...],
    kv_cache_spec: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return K/V view shapes for a non-MLA cache.

    For 310P NZ (trailing dim 16) K and V share the 4D view ``shape[1:]``.
    For ND / asymmetric-head layouts, V's last dim uses ``head_size_v``.
    """
    k_shape = kv_cache_shape[1:]
    if len(kv_cache_shape) == 5 and kv_cache_shape[-1] == 16:
        # FRACTAL_NZ: last dim is the 16-aligned tile, not head_size.
        return k_shape, k_shape
    v_shape = (
        *kv_cache_shape[1:-1],
        getattr(kv_cache_spec, "head_size_v", kv_cache_spec.head_size),
    )
    return k_shape, v_shape
