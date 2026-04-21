import unittest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
    RequestTracker,
    normalize_block_ids_by_group,
)


class TestAscendStoreGroupAwareConfig(unittest.TestCase):
    def test_normalize_block_ids_by_group(self):
        self.assertEqual(normalize_block_ids_by_group([1, 2]), [[1, 2]])
        self.assertEqual(
            normalize_block_ids_by_group(([1, 2], [3, 4])),
            [[1, 2], [3, 4]],
        )
        self.assertEqual(
            normalize_block_ids_by_group([[5, 6], [7, 8]]),
            [[5, 6], [7, 8]],
        )

    def test_group_id_is_part_of_cache_key(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        key_group0 = next(token_db.process_tokens(16, ["hash0"], kv_cache_group_id=0))[2]
        key_group1 = next(token_db.process_tokens(16, ["hash0"], kv_cache_group_id=1))[2]

        self.assertNotEqual(key_group0.to_string(), key_group1.to_string())
        self.assertIn("@group:0@", key_group0.to_string())
        self.assertIn("@group:1@", key_group1.to_string())

    def test_prepare_value_uses_group_specific_buffers(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        token_db.set_kv_caches_base_addr([1000])
        token_db.set_block_len([64])
        token_db.set_group_buffers(
            {1: [2000, 3000]},
            {1: [128, 256]},
        )

        addrs, sizes, block_id = token_db.prepare_value(
            start=16,
            end=32,
            block_ids=[10, 11],
            kv_cache_group_id=1,
        )

        self.assertEqual(block_id, 11)
        self.assertEqual(addrs, [2000 + 11 * 128, 3000 + 11 * 256])
        self.assertEqual(sizes, [128, 256])

    def test_process_tokens_with_block_ids_skips_null_blocks(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)

        chunks = list(
            token_db.process_tokens_with_block_ids(
                token_len=48,
                block_hashes=["h0", "h1", "h2"],
                block_ids=[0, 0, 9],
                kv_cache_group_id=1,
                skip_null_blocks=True,
            )
        )

        self.assertEqual(len(chunks), 1)
        start, end, key, block_id = chunks[0]
        self.assertEqual((start, end, block_id), (32, 48, 9))
        self.assertIn("@group:1@", key.to_string())

    def test_req_meta_tracks_all_groups(self):
        tracker = RequestTracker(
            req_id="req-1",
            token_len=32,
            allocated_block_ids_by_group=[[1, 2], [10, 11]],
            token_ids=[1] * 32,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            block_hashes=["h0", "h1"],
            skip_save=False,
            discard_partial_chunks=True,
        )

        assert req_meta is not None
        self.assertEqual(req_meta.block_ids_by_group, [[1, 2], [10, 11]])
        self.assertEqual(req_meta.kv_cache_group_ids, [0, 1])


if __name__ == "__main__":
    unittest.main()
