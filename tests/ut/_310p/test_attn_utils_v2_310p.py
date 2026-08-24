# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P-focused unit tests for FRACTAL_NZ KV cache view helpers."""

from types import SimpleNamespace

from vllm_ascend._310p.worker.v2.attn_utils import get_310p_non_mla_kv_cache_shapes


def test_symmetric_head_size_preserves_310p_nz_layout():
    kv_cache_shape = (2, 572, 64, 128, 16)
    kv_cache_spec = SimpleNamespace(head_size=128, head_size_v=128)

    k_shape, v_shape = get_310p_non_mla_kv_cache_shapes(kv_cache_shape, kv_cache_spec)

    assert k_shape == (572, 64, 128, 16)
    assert v_shape == k_shape


def test_asymmetric_head_size_keeps_existing_v_shape_behavior():
    kv_cache_shape = (2, 10, 16, 8, 128)
    kv_cache_spec = SimpleNamespace(head_size=128, head_size_v=64)

    k_shape, v_shape = get_310p_non_mla_kv_cache_shapes(kv_cache_shape, kv_cache_spec)

    assert k_shape == (10, 16, 8, 128)
    assert v_shape == (10, 16, 8, 64)
