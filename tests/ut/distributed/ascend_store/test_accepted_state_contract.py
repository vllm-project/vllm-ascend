# SPDX-License-Identifier: Apache-2.0
"""Committed native boundaries and GLM MTP input identity."""

from types import SimpleNamespace
from unittest.mock import Mock

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.model_adapters.glm import describe_key_policy
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.state_handoff import (
    bind_cache_key_hashes,
    limit_save_prefixes,
    select_boundary_puts,
)


def test_decode_checkpoint_requires_native_offer_and_committed_hash():
    options = dict(
        store_group_ids={7: 2},
        state_group_ids=frozenset({2}),
        hash_block_size=16,
        committed_hash_count=4,
        prefill_end_tokens=32,
    )
    offers = [(7, 3, 32), (7, 4, 48), (7, 5, 80), (9, 6, 48)]
    assert select_boundary_puts(offers, save_decode_cache=False, **options) == (BoundaryPut(2, 3, 32),)
    # Boundary 64 has a committed hash but no native offer; 80 is uncommitted.
    assert select_boundary_puts(offers, save_decode_cache=True, **options) == (
        BoundaryPut(2, 3, 32),
        BoundaryPut(2, 4, 48),
    )
    assert select_boundary_puts([], save_decode_cache=True, **options) == ()


def test_mtp_checkpoint_waits_for_committed_extra_token():
    policy = describe_key_policy(
        SimpleNamespace(num_nextn_predict_layers=1),
        SimpleNamespace(num_speculative_tokens=3, method="mtp"),
    )
    puts = (BoundaryPut(2, 9, 4), BoundaryPut(7, 11, 4), BoundaryPut(2, 14, 8))
    tokens = list(range(8))
    selected, keys = bind_cache_key_hashes(puts, [b"h4", b"h8"], tokens, hash_block_size=4, policy=policy)
    # token[8] is not committed: keep both groups at 4 and defer boundary 8.
    assert selected == puts[:2]
    assert set(keys) == {4}
    assert keys[4] != b"h4"
    assert policy.bind(b"h8", 8, tokens) is None

    selected, committed_keys = bind_cache_key_hashes(
        puts, [b"h4", b"h8"], tokens + [99], hash_block_size=4, policy=policy
    )
    assert selected == puts
    assert set(committed_keys) == {4, 8}
    assert committed_keys[4] == keys[4]
    assert committed_keys[8] != b"h8"


def test_mtp_identity_depends_on_extra_token_not_draft_length():
    keys = []
    for draft_length in (1, 7):
        policy = describe_key_policy(
            SimpleNamespace(num_nextn_predict_layers=1),
            SimpleNamespace(num_speculative_tokens=draft_length, method="mtp"),
        )
        key = policy.bind(b"engine-hash", 4, [10, 11, 12, 13, 99])
        assert key is not None
        assert key != policy.bind(b"engine-hash", 4, [10, 11, 12, 13, 98])
        assert policy.bind(b"engine-hash", 4, [10, 11, 12, 13]) is None
        keys.append(key)
    assert keys[0] == keys[1]


def test_draft_limit_keeps_job_and_aligns_page_save_progress():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread

    request = ReqMeta(
        req_id="limited",
        token_len_chunk=32,
        can_save=True,
        block_ids_by_group=[[1, 2]],
        block_hashes=[b"h16", b"h32"],
        cache_key_hashes={16: b"k16", 32: b"k32"},
        boundary_puts=[BoundaryPut(1, 3, 32)],
        event_id=7,
        store_job_id=7,
    )
    metadata = AscendConnectorMetadata(set())
    metadata.add_request(request)

    limit_save_prefixes(metadata, {"limited": 19})

    assert len(metadata.requests) == 1 and metadata.requests[0] is request
    assert request.can_save and request.store_job_id == request.event_id == 7
    assert request.token_len_chunk == request.save_end_token == 19
    assert request.cache_key_hashes == {16: b"k16"}
    assert request.boundary_puts == []

    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None)
    database.cache_coordinator = SimpleNamespace(lcm_block_size=16, cacheable_group_ids={0})
    sender = Mock(spec=KVCacheStoreSendingThread)
    sender.token_database = database
    sender.request_queue = Mock()
    sender.is_live_store_job.return_value = True
    page_ends = []

    def save_pages(job, *, source_ready):
        for _, end, _, _, _ in database.process_token_key_strings_with_block_ids(
            job.token_len_chunk, job.block_hashes, job.block_ids_by_group[0]
        ):
            assert end in job.cache_key_hashes
            page_ends.append(end)
        return True

    sender._handle_stored_request.side_effect = save_pages
    KVCacheStoreSendingThread._handle_store_job(sender, request)

    assert page_ends == [16]
    assert request.token_len_chunk == request.save_end_token == 16
    sender.record_saved_offset.assert_called_once_with(request, 16)
    sender._finish_state_job.assert_called_once_with(request)
    sender.request_queue.task_done.assert_called_once_with()
