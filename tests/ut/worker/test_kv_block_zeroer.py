# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.worker.utils import AscendKVBlockZeroer


class _PackedKVBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 1


class _SingleTensorBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 0


def test_init_meta_supports_mixed_kv_cache_layouts_and_page_sizes():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    packed_kv = torch.zeros((2, 4, 2, 3), dtype=torch.float32)
    index_k = torch.zeros((4, 2, 4), dtype=torch.uint8)
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    groups = [
        SimpleNamespace(
            backend=_PackedKVBackend,
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            layer_names=["main_attn"],
        ),
        SimpleNamespace(
            backend=_SingleTensorBackend,
            kv_cache_spec=spec,
            kv_cache_group_id=1,
            layer_names=["indexer"],
        ),
    ]

    zeroer.init_meta(
        attn_groups_iter=groups,
        kernel_block_sizes=[[2], [2]],
        cache_dtype="fp8",
        runner_only_attn_layers=set(),
        static_forward_context={
            "main_attn": SimpleNamespace(kv_cache=packed_kv),
            "indexer": SimpleNamespace(kv_cache=index_k),
        },
    )

    assert zeroer._meta is not None
    seg_addrs, seg_page_sizes, max_page_size_el, block_size, num_segments = zeroer._meta
    assert seg_addrs.tolist() == [
        packed_kv.data_ptr(),
        packed_kv[1].data_ptr(),
        index_k.data_ptr(),
    ]
    assert seg_page_sizes.tolist() == [12, 12, 4]
    assert max_page_size_el == 12
    assert block_size == 4
    assert num_segments == 3
