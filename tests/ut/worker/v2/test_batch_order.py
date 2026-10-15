# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.worker.v2.batch import sort_batch_req_ids


def test_prefill_order_matches_scheduler_instead_of_length():
    scheduled = {"decode": 1, "short": 21, "long": 2717, "last": 17}
    assert sort_batch_req_ids(scheduled, {}, 1) == list(scheduled)


def test_decode_and_verification_keep_their_priority():
    scheduled = {"prefill": 64, "short_extend": 2, "decode": 4, "draft_b": 8, "draft_a": 6}
    drafts = {"draft_b": [1, 2], "draft_a": [3]}
    assert sort_batch_req_ids(scheduled, drafts, 4) == ["draft_a", "draft_b", "decode", "short_extend", "prefill"]


def test_prefill_permutation_is_preserved_after_decode_moves_first():
    scheduled = {"long": 1000, "small": 20, "decode": 1, "medium": 400}
    assert sort_batch_req_ids(scheduled, {}, 1) == ["decode", "long", "small", "medium"]


def test_empty_batch():
    assert sort_batch_req_ids({}, {}, 1) == []
