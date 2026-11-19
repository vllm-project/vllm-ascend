# SPDX-License-Identifier: Apache-2.0
"""Endpoint fingerprints preserve full-prefix identity without replaying chains."""

import hashlib
import math
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh


def make_request(tokens, unit=2, salt=b""):
    digest = hashlib.sha256(salt).digest()
    hashes = []
    for start in range(0, len(tokens) - unit + 1, unit):
        digest = hashlib.sha256(digest + b"".join(struct.pack(">q", t) for t in tokens[start : start + unit])).digest()
        hashes.append(digest)
    return SimpleNamespace(prompt_token_ids=tokens, block_hashes=hashes)


def make_scheduler(sizes=(8,), windows=(0,), unit=2, cp=1, state=False):
    scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
    scheduler._decode_hash_block_size = unit
    scheduler.pcp_size = cp
    scheduler.dcp_size = 1
    scheduler.group_transfer_info = [
        d2rh.GroupTransferInfo(tokens_per_block=size, blocks_per_window=window, is_state_group=state)
        for size, window in zip(sizes, windows)
    ]
    return scheduler


def test_shared_prefix_keys_ignore_request_length_and_physical_block_ids():
    scheduler = make_scheduler()
    original = make_request(list(range(27)))
    longer = make_request(list(range(27)) + [99] * 16)
    before = scheduler._d2rh_get_decode_block_hashes(original, ([10, 20, 30, 40],))[0]
    after = scheduler._d2rh_get_decode_block_hashes(longer, ([5, 6, 7, 8],))[0]
    assert before[:3] == after[:3]
    assert before[3] != after[3]


@pytest.mark.parametrize("changed_index", [0, 8, 26])
def test_key_covers_earlier_prefix_and_partial_tail(changed_index):
    scheduler = make_scheduler()
    tokens = list(range(27))
    before = scheduler._d2rh_get_decode_block_hashes(make_request(tokens), ([1, 2, 3, 4],))[0]
    changed = list(tokens)
    changed[changed_index] += 100
    after = scheduler._d2rh_get_decode_block_hashes(make_request(changed), ([1, 2, 3, 4],))[0]
    first = changed_index // 8
    assert before[:first] == after[:first]
    assert all(a != b for a, b in zip(before[first:], after[first:]))


def test_cache_identity_in_request_hash_is_preserved():
    scheduler = make_scheduler()
    first = scheduler._d2rh_get_decode_block_hashes(make_request(list(range(24)), salt=b"a"), ([1, 2, 3],))
    other = scheduler._d2rh_get_decode_block_hashes(make_request(list(range(24)), salt=b"b"), ([1, 2, 3],))
    assert all(a != b for a, b in zip(first[0], other[0]))


def test_runtime_request_hasher_preserves_prefix_salt_and_partial_tail():
    init_none_hash(sha256)
    scheduler = make_scheduler()

    def hashes(tokens, salt):
        request = Request(
            request_id="endpoint-hash-test",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            cache_salt=salt,
            block_hasher=get_request_block_hasher(2, sha256),
        )
        assert len(request.block_hashes) == len(tokens) // 2
        return scheduler._d2rh_get_decode_block_hashes(request, ([1, 2, 3, 4],))[0]

    tokens = list(range(27))
    reference = hashes(tokens, "first")
    assert reference == hashes(tokens, "first")
    assert all(a != b for a, b in zip(reference, hashes(tokens, "second")))
    different_tail = hashes(tokens[:-1] + [99], "first")
    assert reference[:-1] == different_tail[:-1]
    assert reference[-1] != different_tail[-1]
    longer = hashes(tokens + [99] * 10, "first")
    assert reference[:3] == longer[:3]
    assert reference[-1] != longer[-1]


def test_sliding_group_uses_tail_endpoints_and_empty_group_stays_empty():
    full = make_scheduler()
    sliding = make_scheduler(windows=(2,))
    request = make_request(list(range(27)))
    expected = full._d2rh_get_decode_block_hashes(request, ([1, 2, 3, 4],))[0][-2:]
    assert sliding._d2rh_get_decode_block_hashes(request, ([9, 10],)) == (expected,)
    assert sliding._d2rh_get_decode_block_hashes(request, ([],)) == ([],)


def test_context_parallel_token_coverage_matches_logical_page():
    request = make_request(list(range(35)))
    cp = make_scheduler(sizes=(8,), cp=2)
    logical = make_scheduler(sizes=(16,))
    assert cp._d2rh_get_decode_block_hashes(request, ([1, 2, 3],)) == logical._d2rh_get_decode_block_hashes(
        request, ([10, 11, 12],)
    )


def test_groups_and_state_slots_have_distinct_keys():
    request = make_request(list(range(25)))
    scheduler = make_scheduler(sizes=(8, 8), windows=(0, 0))
    values = scheduler._d2rh_get_decode_block_hashes(request, ([1, 2], [1, 2]))
    assert all(a != b for a, b in zip(*values))
    state = make_scheduler(state=True)
    values = state._d2rh_get_decode_block_hashes(request, ([1, 2, 3],))[0]
    assert len(set(values)) == 3


@pytest.mark.parametrize("mode", ["missing_unit", "missing_hashes", "short_hashes", "short_prompt"])
def test_missing_hash_information_retains_legacy_fallback(mode):
    scheduler = make_scheduler()
    request = make_request(list(range(25)))
    if mode == "missing_unit":
        scheduler._decode_hash_block_size = None
    elif mode == "missing_hashes":
        del request.block_hashes
    elif mode == "short_hashes":
        request.block_hashes = request.block_hashes[:1]
    else:
        request = make_request([1])
    assert scheduler._d2rh_get_decode_endpoint_hashes(request, ([1],)) is None
    with patch.object(scheduler, "_d2rh_get_decode_endpoint_hashes", return_value=None):
        expected = scheduler._d2rh_get_decode_block_hashes(request, ([1],))
    assert scheduler._d2rh_get_decode_block_hashes(request, ([1],)) == expected


def test_mismatched_remote_count_disables_group_cache():
    scheduler = make_scheduler()
    assert scheduler._d2rh_get_decode_block_hashes(make_request([1, 2, 3]), ([1, 2],)) == ([None, None],)


def test_128k_hash_work_scales_with_transferred_blocks():
    sizes = (128, 4096, 32, 32, 2, 4)
    counts = (1025, 33, 5, 5, 4, 33)
    scheduler = make_scheduler(sizes=sizes, windows=(0, 0, 5, 5, 5, 33))
    request = make_request(list(range(131155)))
    ids = tuple(list(range(count)) for count in counts)
    sha256 = hashlib.sha256
    with patch.object(d2rh.hashlib, "sha256", wraps=sha256) as hashing:
        result = scheduler._d2rh_get_decode_block_hashes(request, ids)
    assert tuple(map(len, result)) == counts
    assert hashing.call_count == sum(counts) == 1105
    assert sum(math.ceil(131155 / size) for size in sizes) == 107623
