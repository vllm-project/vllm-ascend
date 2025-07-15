import unittest

from vllm_ascend.multistream.base import (MSAttentionMetadataSplitConfig,
                                          MSEventKey)


class TestBase(unittest.TestCase):
    def test_ms_event_key(self):
        self.assertEqual(MSEventKey.ATTN_COM_FINISH, 0)
        self.assertEqual(MSEventKey.ATTN_AR_FINISH, 1)
        self.assertEqual(MSEventKey.FFN_COM_FINISH, 2)
        self.assertEqual(MSEventKey.FFN_AR_FINISH, 3)
        self.assertEqual(MSEventKey.MOE_BEFORE_COMM, 4)
        self.assertEqual(MSEventKey.MOE_AFTER_COMM, 5)
        self.assertEqual(MSEventKey.MOE_SE_COMM_FINISH, 6)
        self.assertEqual(MSEventKey.MOE_SE_COMP_FINISH, 7)
        self.assertEqual(MSEventKey.MOE_GATE_FINISH, 8)

    def test_ms_attention_metadata_split_config_default(self):
        config = MSAttentionMetadataSplitConfig()
        self.assertEqual(config.num_micro_batches, 2)
        self.assertEqual(config.min_total_tokens_to_split, 256)
        self.assertEqual(config.min_prefill_tokens_to_split, 64)

    def test_ms_attention_metadata_split_config_custom(self):
        config = MSAttentionMetadataSplitConfig(
            num_micro_batches=4,
            min_total_tokens_to_split=512,
            min_prefill_tokens_to_split=128)
        self.assertEqual(config.num_micro_batches, 4)
        self.assertEqual(config.min_total_tokens_to_split, 512)
        self.assertEqual(config.min_prefill_tokens_to_split, 128)
