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

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

from tests.ut.attention.utils import BatchSpec, create_common_attn_metadata
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

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


def _ascend_config_patch(block_table_no_commit_optimize: int = 0):
    """``MultiGroupBlockTable`` picks its block-table class off the Ascend
    config, which a unit test never initializes; 0 is the production default
    and selects ``OptimizedBlockTable``."""
    return patch(
        "vllm_ascend.worker.block_table.get_ascend_config",
        return_value=SimpleNamespace(block_table_no_commit_optimize=block_table_no_commit_optimize),
    )


def _block_table(groups, *, max_num_reqs: int = 4, max_num_batched_tokens: int = 512, dcp_world_size: int = 1):
    with _dcp_group_patch(dcp_world_size=dcp_world_size), _ascend_config_patch():
        from vllm_ascend.worker.block_table import MultiGroupBlockTable

        return MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=1024,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[BLOCK_SIZE] * len(groups),
            kernel_sizes=[[BLOCK_SIZE]] * len(groups),
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


# --- the per-request replay start the runner builds --------------------------


def _runner(prefix_replay_tokens: int, max_num_reqs: int = 4) -> NPUModelRunner:
    """A runner with only what the replay-start builder touches. The buffer
    copies for real, so a value the device would still be holding is visible."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.prefix_replay_tokens = prefix_replay_tokens
    buffer = SimpleNamespace(
        np=np.zeros(max_num_reqs, dtype=np.int32),
        gpu=torch.zeros(max_num_reqs, dtype=torch.int32),
    )
    buffer.copy_to_gpu = lambda num=None: buffer.gpu.copy_(torch.from_numpy(buffer.np))
    runner.replay_start = buffer
    runner.input_batch = SimpleNamespace(req_id_to_index={"req-0": 0, "req-1": 1})
    # Set by __init__ in production; _fill_replay_start owns it afterwards.
    runner._replay_active = False
    return runner


def _scheduler_output(new_reqs=(), cached_replay_start=None):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=SimpleNamespace(replay_start=cached_replay_start or {}),
    )


def test_no_replay_window_means_no_work_at_all():
    """A run where no group declares a replay window pays nothing here, which is
    every run except DeepSeek-V4.1 with the switch on."""
    runner = _runner(prefix_replay_tokens=0)

    assert runner._fill_replay_start(_scheduler_output(), num_reqs=2) is None


def test_newly_admitted_request_carries_its_replay_start():
    runner = _runner(prefix_replay_tokens=WINDOW)
    new_req = SimpleNamespace(req_id="req-1", replay_start=REPLAY_START)

    replay_start = runner._fill_replay_start(_scheduler_output(new_reqs=[new_req]), num_reqs=2)

    assert replay_start.tolist() == [0, REPLAY_START]


def test_resumed_request_carries_its_replay_start():
    """The V1 path: a preempted request comes back through CachedRequestData."""
    runner = _runner(prefix_replay_tokens=WINDOW)

    replay_start = runner._fill_replay_start(_scheduler_output(cached_replay_start={"req-0": REPLAY_START}), num_reqs=2)

    assert replay_start.tolist() == [REPLAY_START, 0]


def test_a_step_with_no_replay_reports_none():
    runner = _runner(prefix_replay_tokens=WINDOW)

    assert runner._fill_replay_start(_scheduler_output(), num_reqs=2) is None


def test_decode_step_never_inherits_an_earlier_replay_start():
    """Replay is confined to admission steps, so every later step -- the whole
    steady-state decode -- has to read all zeros, on the device too: the
    attention metadata of that step is built from this buffer. Pinning this is
    what replaces upstream's ``is_prefilling`` filter.

    The padded rows are asserted as well, and they are why the buffer is
    cleared and copied whole rather than up to ``num_reqs``: a graph row must
    not inherit a write start from the step that did replay.
    """
    runner = _runner(prefix_replay_tokens=WINDOW, max_num_reqs=4)
    new_req = SimpleNamespace(req_id="req-0", replay_start=REPLAY_START)
    runner._fill_replay_start(_scheduler_output(new_reqs=[new_req]), num_reqs=2)
    assert runner.replay_start.gpu.tolist() == [REPLAY_START, 0, 0, 0]

    # The next step names nobody.
    assert runner._fill_replay_start(_scheduler_output(), num_reqs=2) is None
    assert runner.replay_start.np.tolist() == [0, 0, 0, 0]
    assert runner.replay_start.gpu.tolist() == [0, 0, 0, 0]


def test_requests_outside_the_batch_are_ignored():
    """A request can leave the batch between the scheduler's decision and this
    step; it must not land its value on another request's row."""
    runner = _runner(prefix_replay_tokens=WINDOW, max_num_reqs=2)
    gone = SimpleNamespace(req_id="req-gone", replay_start=REPLAY_START)
    admitted = SimpleNamespace(req_id="req-1", replay_start=WINDOW)

    replay_start = runner._fill_replay_start(_scheduler_output(new_reqs=[gone, admitted]), num_reqs=2)

    assert replay_start.tolist() == [0, WINDOW]


# --- what the attention metadata carries -------------------------------------


@pytest.mark.parametrize(
    ("new_reqs", "cached_replay_start", "expected"),
    [
        ((), None, (False, None)),
        ((SimpleNamespace(req_id="req-0", replay_start=REPLAY_START),), None, (True, [REPLAY_START, 0])),
        ((), {"req-1": REPLAY_START}, (True, [0, REPLAY_START])),
        # A request that leaves the batch leaves nothing behind: no row was
        # written, so the step is not a replay step at all.
        ((SimpleNamespace(req_id="req-gone", replay_start=REPLAY_START),), None, (False, None)),
    ],
)
def test_attention_metadata_reports_a_replay_only_when_the_step_replays(
    new_reqs,
    cached_replay_start,
    expected,
):
    runner = _runner(prefix_replay_tokens=WINDOW)
    runner._fill_replay_start(
        _scheduler_output(new_reqs=new_reqs, cached_replay_start=cached_replay_start),
        num_reqs=2,
    )

    active, values = expected
    replay_start = runner._attn_replay_start(num_reqs_padded=2)

    assert runner._replay_active is active
    if values is None:
        assert replay_start is None
    else:
        assert replay_start.tolist() == values


def test_no_replay_window_never_reports_a_replay():
    """A run where no group declares a window cannot replay, whatever the
    scheduler payload happens to hold."""
    runner = _runner(prefix_replay_tokens=0)

    runner._fill_replay_start(
        _scheduler_output(new_reqs=[SimpleNamespace(req_id="req-0", replay_start=REPLAY_START)]),
        num_reqs=2,
    )

    assert runner._attn_replay_start(num_reqs_padded=2) is None


# --- the metadata field the attention backends consume -----------------------


def _metadata(*, replay_start):
    batch = BatchSpec(seq_lens=[1, 2, 3], query_lens=[1, 1, 1], name="replay")
    return replace(
        create_common_attn_metadata(batch, BLOCK_SIZE, torch.device("cpu")),
        replay_start=replay_start,
    )


def test_unpadded_slices_the_replay_start_with_the_batch():
    """The unpadded view is what speculative decoding builds from, so a per-req
    tensor that is not sliced there would be read against the wrong request."""
    metadata = _metadata(replay_start=torch.tensor([REPLAY_START, 0, 0], dtype=torch.int32))

    unpadded = metadata.unpadded(num_actual_tokens=2, num_actual_reqs=2)

    assert unpadded.replay_start.tolist() == [REPLAY_START, 0]


def test_replay_start_defaults_to_none():
    """Dummy/profiling/capture batches never fill it; a consumer has to read
    None as "nothing replays", the same as all zeros."""
    metadata = _metadata(replay_start=None)

    assert metadata.replay_start is None
    assert metadata.unpadded(num_actual_tokens=1, num_actual_reqs=1).replay_start is None
