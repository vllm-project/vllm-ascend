# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""The worker side of SWA bounded replay: padding replayed slots.

A replayed request keeps its prefix hit's blocks, which stay shared with every
other request that hit the same prefix. The cacheable groups must therefore not
write the replayed positions: their KV already exists, and rewriting it would
put a different value in a block other requests read. The group that owns the
replay (``prefix_cacheable=False``) does write them, and that write is what
rebuilds its sliding window.

What runs on CPU is the bookkeeping around the kernel -- which group is padded,
which pointers and window reach the launch, and how the per-request replay start
is built. The kernel itself needs an NPU and is covered by the e2e tests.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    register_ascend_kv_cache_specs,
)

# The prefix-replay geometry upstream's own test uses: a 100-token prompt with a
# 96-token hit and a 32-token window recomputes [64, 96).
WINDOW = 32
REPLAY_START = 64
BLOCK_SIZE = 16
COMPRESS_RATIO = 4


def _cacheable_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)


def _non_cacheable_spec() -> AscendIndexerKPoolTailSpec:
    """A real group that opts out of prefix caching.

    The other one is the bounded-replay sliding-window group, which upstream
    #56227 makes non-cacheable; this stands in for it so these cases do not
    depend on that spec gaining a replay window.
    """
    register_ascend_kv_cache_specs()
    return AscendIndexerKPoolTailSpec(
        block_size=BLOCK_SIZE,
        sliding_window=COMPRESS_RATIO,
        compress_ratio=COMPRESS_RATIO,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
    )


def _cacheable_group(name: str = "kv") -> KVCacheGroupSpec:
    return KVCacheGroupSpec([name], _cacheable_spec())


def _replaying_group(name: str = "swa") -> KVCacheGroupSpec:
    return KVCacheGroupSpec([name], _non_cacheable_spec())


def _dcp_group_patch(dcp_world_size: int = 1, dcp_rank: int = 0):
    mock_group = SimpleNamespace(world_size=dcp_world_size, rank_in_group=dcp_rank)
    return patch("vllm_ascend.worker.block_table.get_dcp_group", return_value=mock_group)


def _block_table(groups, *, max_num_reqs: int = 4, max_num_batched_tokens: int = 512, dcp_world_size: int = 1):
    with _dcp_group_patch(dcp_world_size=dcp_world_size):
        from vllm_ascend.worker.block_table import MultiGroupBlockTable

        return MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=1024,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[BLOCK_SIZE] * len(groups),
            kernel_sizes=[BLOCK_SIZE] * len(groups),
            max_num_blocks=[64] * len(groups),
            kv_cache_groups=groups,
        )


# --- which group is padded, which group writes -------------------------------


@pytest.mark.parametrize("cacheable", [True, False])
def test_block_table_reads_cacheability_off_the_spec(cacheable):
    group = _cacheable_group("g") if cacheable else _replaying_group("g")
    with _dcp_group_patch():
        from vllm_ascend.worker.block_table import BlockTable

        table = BlockTable(
            block_size=BLOCK_SIZE,
            max_num_reqs=4,
            max_num_blocks_per_req=64,
            max_num_batched_tokens=512,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[BLOCK_SIZE],
            kv_cache_group=group,
        )

    assert table.is_prefix_cacheable is cacheable


def test_group_without_a_spec_counts_as_cacheable():
    """The tables built for the pre-grouping call sites pass no group at all."""
    with _dcp_group_patch():
        from vllm_ascend.worker.block_table import BlockTable

        table = BlockTable(
            block_size=BLOCK_SIZE,
            max_num_reqs=4,
            max_num_blocks_per_req=64,
            max_num_batched_tokens=512,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[BLOCK_SIZE],
        )

    assert table.is_prefix_cacheable


def test_cacheability_resolves_through_a_uniform_wrapper():
    """A sliding-window group can be wrapped in a uniform collection, so the
    opt-out only shows up if every spec inside agrees."""
    replaying = _non_cacheable_spec()
    wrapped = UniformTypeKVCacheSpecs.from_specs({"a": replaying, "b": replace(replaying)})

    table = _block_table([KVCacheGroupSpec(["swa"], wrapped)])

    assert not table.block_tables[0].is_prefix_cacheable
    assert table._replay_cacheable_groups.tolist() == []


# --- the indices and pointers handed to the PAD pass -------------------------


def test_only_cacheable_groups_are_padded():
    table = _block_table([_cacheable_group("kv"), _replaying_group("swa"), _cacheable_group("idx")])

    assert table._replay_cacheable_groups.tolist() == [0, 2]


def test_addresses_are_each_groups_own_slot_mapping():
    """A group's slot mapping is patched in place through its own pointer, so
    the array has to be per group and not the packed one the fused kernel uses."""
    table = _block_table([_cacheable_group("kv"), _replaying_group("swa")])

    assert table._replay_slot_mapping_addrs.dtype == torch.uint64
    assert table._replay_slot_mapping_addrs.tolist() == [
        block_table.slot_mapping.gpu.data_ptr() for block_table in table.block_tables
    ]


def test_addresses_exist_even_when_the_groups_cannot_fuse():
    """The fused kernel is skipped whenever DCP is on; the PAD pass must not be."""
    groups = [_cacheable_group("kv"), _cacheable_group("idx")]
    table = _block_table(groups, dcp_world_size=2)

    assert not table._can_fuse_slot_mapping
    assert table._replay_cacheable_groups.tolist() == [0, 1]
    assert len(table._replay_slot_mapping_addrs) == len(groups)


def test_mamba_groups_are_not_addressed():
    """A mamba group advances state, not paged KV, so there is no slot mapping
    of its own to pad -- the fused path skips it for the same reason."""
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4, 8),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = [_cacheable_group("kv"), KVCacheGroupSpec(["mamba"], mamba_spec)]
    table = _block_table(groups)

    assert table.block_tables[1].is_mamba_group
    assert len(table._replay_slot_mapping_addrs) == 1
    assert table._replay_slot_mapping_addrs.tolist() == [table.block_tables[0].slot_mapping.gpu.data_ptr()]


def test_pad_pass_forwards_its_own_arrays():
    table = _block_table([_cacheable_group("kv"), _replaying_group("swa")])
    positions = torch.arange(REPLAY_START, REPLAY_START + 4, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    replay_start = torch.tensor([REPLAY_START, 0], dtype=torch.int32)

    with patch("vllm_ascend.worker.block_table.pad_replayed_slot_mapping") as pad:
        table.pad_replayed_slots(2, query_start_loc, positions, replay_start, WINDOW)

    (num_reqs, qsl, pos, starts, addrs, cacheable, window), kwargs = pad.call_args
    assert num_reqs == 2
    assert qsl is query_start_loc
    assert pos is positions
    assert starts is replay_start
    assert addrs is table._replay_slot_mapping_addrs
    assert cacheable is table._replay_cacheable_groups
    assert window == WINDOW
    assert kwargs == {"pad_id": PAD_SLOT_ID}


# --- the host function's guards ---------------------------------------------


def test_host_function_returns_before_launching_when_nothing_replays():
    from vllm_ascend.ops.triton.compute_slot_mapping import pad_replayed_slot_mapping

    empty = torch.tensor([], dtype=torch.int32)
    # No cacheable group, and no request: both have to be non-launches rather
    # than a kernel over an empty grid.
    pad_replayed_slot_mapping(4, empty, empty, empty, empty, empty, WINDOW, pad_id=-1)
    pad_replayed_slot_mapping(0, empty, empty, empty, empty, torch.tensor([0], dtype=torch.int32), WINDOW, pad_id=-1)
