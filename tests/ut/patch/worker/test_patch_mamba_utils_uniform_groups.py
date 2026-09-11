# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.mamba_utils import MambaCopyBuffers

from vllm_ascend.patch.worker.patch_mamba_utils import _get_mamba_groups


def test_uniform_mamba_groups_are_visible_to_all_mamba_buffers() -> None:
    mamba_spec = MambaSpec(
        block_size=384,
        shapes=((10, 2304), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=488448,
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )
    groups = []
    for group_id in range(3):
        layer_specs = {f"mamba.{group_id}.{layer_id}": mamba_spec for layer_id in range(23)}
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(layer_specs)
        assert uniform_spec is not None
        groups.append(
            KVCacheGroupSpec(
                layer_names=list(layer_specs),
                kv_cache_spec=uniform_spec,
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )

    group_ids, resolved_spec = _get_mamba_groups(kv_cache_config)
    assert group_ids == [0, 1, 2]
    assert resolved_spec == mamba_spec

    def make_buffer(n: int, dtype: torch.dtype) -> SimpleNamespace:
        return SimpleNamespace(n=n, dtype=dtype)

    copy_bufs = MambaCopyBuffers.create(
        max_num_reqs=2,
        kv_cache_config=kv_cache_config,
        copy_funcs=(object(), object()),
        make_buffer=make_buffer,
    )
    assert copy_bufs.mamba_group_ids == [0, 1, 2]
    assert copy_bufs.mamba_spec == mamba_spec
    assert copy_bufs.src_ptrs.n == 2 * 69 * 2
    assert copy_bufs.src_ptrs.dtype == torch.int64
    assert copy_bufs.dst_ptrs.dtype == torch.int64
    assert copy_bufs.sizes.dtype == torch.int32
