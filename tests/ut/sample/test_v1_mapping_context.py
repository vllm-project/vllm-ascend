import unittest

import numpy as np
import torch

from tests.ut.base import TestBase


class TestV1MappingContext(TestBase):
    def test_from_decode_builds_identity_request_mapping(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        positions = torch.tensor([10, 11, 12, 999], dtype=torch.int64)
        input_ids = torch.tensor([101, 102, 103, 9999], dtype=torch.int32)

        ctx = V1MappingContext.from_v1_logits(
            num_reqs=3,
            positions_at_logits=positions[:3],
            input_ids_at_logits=input_ids[:3],
            req_indices_at_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req0", "req1", "req2"),
        )

        self.assertEqual(ctx.num_reqs, 3)
        self.assertFalse(ctx.expanded_logits)
        self.assertEqual(ctx.num_logits, 3)
        self.assertTrue(ctx.is_identity_request_mapping)
        self.assertIsNone(ctx.cu_num_logits_np)
        self.assertEqual(ctx.expanded_idx_mapping.dtype, torch.int32)
        self.assertEqual(ctx.expanded_idx_mapping.device.type, "cpu")
        self.assertEqual(ctx.expanded_idx_mapping.tolist(), [0, 1, 2])
        np.testing.assert_array_equal(ctx.idx_mapping_np, np.array([0, 1, 2], dtype=np.int32))

        self.assertEqual(ctx.pos.dtype, torch.int64)
        self.assertEqual(ctx.pos.tolist(), [10, 11, 12])
        self.assertEqual(ctx.input_ids.tolist(), [101, 102, 103])
        self.assertEqual(ctx.expanded_local_pos.dtype, torch.int64)
        self.assertEqual(ctx.expanded_local_pos.tolist(), [0, 0, 0])
        self.assertEqual(ctx.req_ids, ("req0", "req1", "req2"))

    def test_from_decode_slices_to_active_request_count(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        positions = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int64)
        input_ids = torch.tensor([13, 14, 15, 16, 17], dtype=torch.int64)

        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=positions[:2],
            input_ids_at_logits=input_ids[:2],
            req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        self.assertEqual(ctx.expanded_idx_mapping.tolist(), [0, 1])
        self.assertEqual(ctx.idx_mapping_np.tolist(), [0, 1])
        self.assertEqual(ctx.pos.tolist(), [3, 4])
        self.assertEqual(ctx.input_ids.tolist(), [13, 14])
        self.assertEqual(ctx.expanded_local_pos.tolist(), [0, 0])

    def test_from_decode_handles_empty_batch(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        ctx = V1MappingContext.from_v1_logits(
            num_reqs=0,
            positions_at_logits=torch.empty(0, dtype=torch.int64),
            input_ids_at_logits=torch.empty(0, dtype=torch.int64),
            req_indices_at_logits=torch.empty(0, dtype=torch.int32),
            device=torch.device("cpu"),
        )

        self.assertEqual(ctx.num_reqs, 0)
        self.assertEqual(ctx.expanded_idx_mapping.numel(), 0)
        self.assertEqual(ctx.idx_mapping_np.size, 0)
        self.assertEqual(ctx.pos.numel(), 0)
        self.assertEqual(ctx.input_ids.numel(), 0)
        self.assertEqual(ctx.expanded_local_pos.numel(), 0)

    def test_from_logits_keeps_request_mapping_separate_from_logits_rows(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=torch.tensor([3, 4, 7], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([13, 14, 17], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req0", "req1"),
        )

        self.assertEqual(ctx.num_logits, 3)
        self.assertTrue(ctx.expanded_logits)
        self.assertFalse(ctx.is_identity_request_mapping)
        self.assertEqual(ctx.expanded_idx_mapping.tolist(), [0, 0, 1])
        self.assertEqual(ctx.expanded_local_pos.tolist(), [0, 1, 0])
        np.testing.assert_array_equal(ctx.num_logits_per_req_np, np.array([2, 1], dtype=np.int32))
        np.testing.assert_array_equal(ctx.cu_num_logits_np, np.array([0, 2, 3], dtype=np.int32))

    def test_from_logits_rejects_mismatched_row_tensors(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        with self.assertRaisesRegex(ValueError, "positions_at_logits"):
            V1MappingContext.from_v1_logits(
                num_reqs=2,
                positions_at_logits=torch.tensor([3], dtype=torch.int64),
                input_ids_at_logits=torch.tensor([13, 14], dtype=torch.int64),
                req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
                device=torch.device("cpu"),
            )

        with self.assertRaisesRegex(ValueError, "input_ids_at_logits"):
            V1MappingContext.from_v1_logits(
                num_reqs=2,
                positions_at_logits=torch.tensor([3, 4], dtype=torch.int64),
                input_ids_at_logits=torch.tensor([13], dtype=torch.int64),
                req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
                device=torch.device("cpu"),
            )

    def test_from_logits_rejects_out_of_range_request_indices(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            V1MappingContext.from_v1_logits(
                num_reqs=2,
                positions_at_logits=torch.tensor([3, 4], dtype=torch.int64),
                input_ids_at_logits=torch.tensor([13, 14], dtype=torch.int64),
                req_indices_at_logits=torch.tensor([0, 2], dtype=torch.int32),
                device=torch.device("cpu"),
            )

    def test_from_logits_requires_expanded_rows_grouped_by_request(self):
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        with self.assertRaisesRegex(ValueError, "grouped by request"):
            V1MappingContext.from_v1_logits(
                num_reqs=2,
                positions_at_logits=torch.tensor([3, 7, 4], dtype=torch.int64),
                input_ids_at_logits=torch.tensor([13, 17, 14], dtype=torch.int64),
                req_indices_at_logits=torch.tensor([0, 1, 0], dtype=torch.int32),
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
