#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.worker.utils import AscendKVBlockZeroer


def test_init_meta_supports_non_uniform_page_sizes() -> None:
    """MLA K/V cache segments may have different sizes per logical block."""
    device = torch.device("cpu")
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    k_cache = torch.zeros((4, 16384), dtype=torch.int32, device=device)
    v_cache = torch.zeros((4, 4096), dtype=torch.int32, device=device)
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
        ),
        kv_cache_group_id=0,
        layer_names=["mla_layer"],
    )

    zeroer.init_meta(
        attn_groups_iter=[group],
        kernel_block_sizes=[[128]],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={"mla_layer": SimpleNamespace(kv_cache=(k_cache, v_cache))},
    )

    assert zeroer._meta is not None
    _, seg_page_sizes, max_chunks, block_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == [16384, 4096]
    assert block_size == 4096
    assert max_chunks == 4
    assert n_segs == 2


def test_init_meta_preserves_uniform_page_size_behavior() -> None:
    """The per-segment metadata path must retain the former uniform layout."""
    device = torch.device("cpu")
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    cache = torch.zeros((4, 16384), dtype=torch.int32, device=device)
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
        ),
        kv_cache_group_id=0,
        layer_names=["attn_layer"],
    )

    zeroer.init_meta(
        attn_groups_iter=[group],
        kernel_block_sizes=[[128]],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={"attn_layer": SimpleNamespace(kv_cache=(cache, cache.clone()))},
    )

    assert zeroer._meta is not None
    _, seg_page_sizes, max_chunks, block_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == [16384, 16384]
    assert block_size == 8192
    assert max_chunks == 2
    assert n_segs == 2
