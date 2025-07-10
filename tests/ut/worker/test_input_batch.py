import unittest

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_ascend.worker.npu_input_batch import CachedRequestState, InputBatch


def mock_cached_request_state(req_id="1", prompt=[1, 2, 3], output=[4, 5, 6]):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_inputs=[],
        mm_positions=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([], ),
        num_computed_tokens=0,
        output_token_ids=output,
    )


class TestInputBatch(unittest.TestCase):

    def setUp(self):
        self.max_num_reqs = 10
        self.max_model_len = 32
        self.max_num_batched_tokens = 132
        self.vocab_size = 1000
        self.device = torch.device("cpu")
        self.block_sizes = [128]

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            device=self.device,
            pin_memory=False,
            vocab_size=self.vocab_size,
            block_sizes=self.block_sizes,
        )
        self.cached_request_state = mock_cached_request_state()

    def test_shapes_and_defaults(self):
        # torch tensor shape assertions
        self.assertEqual(self.input_batch.token_ids_cpu_tensor.shape,
                         (self.max_num_reqs, self.max_model_len))
        self.assertEqual(self.input_batch.temperature.shape,
                         (self.max_num_reqs, ))
        self.assertEqual(self.input_batch.top_k.shape, (self.max_num_reqs, ))
        self.assertEqual(self.input_batch.min_p_cpu_tensor.shape,
                         (self.max_num_reqs, ))

        # numpy shape assertions
        self.assertEqual(self.input_batch.token_ids_cpu.shape,
                         (self.max_num_reqs, self.max_model_len))
        self.assertEqual(self.input_batch.num_tokens.shape,
                         (self.max_num_reqs, ))
        self.assertEqual(self.input_batch.num_tokens.shape,
                         (self.max_num_reqs, ))

        # type assertions
        self.assertIsInstance(self.input_batch.greedy_reqs, set)
        self.assertIsInstance(self.input_batch.req_id_to_index, dict)
        self.assertIsInstance(self.input_batch.sampling_metadata,
                              SamplingMetadata)
        self.assertIsInstance(self.input_batch.block_table,
                              MultiGroupBlockTable)
        self.assertIsNone(self.input_batch.allowed_token_ids_mask)
        self.assertIsNone(self.input_batch.allowed_token_ids_mask_cpu_tensor)

    def test_add_request(self):
        # case1: add a new req
        self.input_batch.add_request(self.cached_request_state)
        self.assertIn(self.cached_request_state.req_id,
                      self.input_batch.req_id_to_index)
        req_index = self.input_batch.req_id_to_index[
            self.cached_request_state.req_id]
        self.assertEqual(self.input_batch.num_prompt_tokens[req_index],
                         len(self.cached_request_state.prompt_token_ids))
        self.assertEqual(self.input_batch.num_tokens[req_index],
                         self.cached_request_state.num_tokens)

        # case2: add an existing req, maybe need update
        self.cached_request_state.output_token_ids.extend([7, 8, 9])
        self.cached_request_state.num_computed_tokens += 3
        cached_index = self.input_batch.req_id_to_index[
            self.cached_request_state.req_id]
        self.input_batch.add_request(self.cached_request_state, cached_index)
        # check if this index in the input_batch is updated
        # This np arrat "token_ids_cpu" should be filled with prompt_token_ids + output_token_ids
        self.assertTrue(
            np.all(self.input_batch.token_ids_cpu[
                cached_index, :self.cached_request_state.num_tokens]),
            msg=f"Token IDs at index {cached_index} did not update correctly.")

        # case3: add req that greater than max_num_reqs
        with self.assertRaises(AssertionError):
            self.input_batch.add_request(self.cached_request_state,
                                         req_index=self.max_num_reqs)

        # case4: add req that out of max_model_len
        long_prompt = list(range(self.max_model_len + 1))
        long_request = mock_cached_request_state(req_id="2",
                                                 prompt=long_prompt,
                                                 output=[10])
        with self.assertRaises(ValueError) as cm:
            self.input_batch.add_request(long_request)
        self.assertIn("could not broadcast", str(cm.exception))

    def test_remove_request(self):
        self.input_batch.add_request(self.cached_request_state)
        req_index = self.input_batch.remove_request(
            self.cached_request_state.req_id)
        self.assertIsNotNone(req_index)
        self.assertNotIn(self.cached_request_state.req_id,
                         self.input_batch.req_id_to_index)
        self.assertIsNone(self.input_batch._req_ids[req_index])

    def test_condense(self):
        # Let's say we have some requests like below
        # Index     Req ID
        #   0         1
        #   1         2
        #   2         3
        #   3         4
        for i in range(4):
            request = mock_cached_request_state(req_id=str(i + 1))
            self.input_batch.add_request(request)
        removed_req_indices = []
        id_to_remove = ["2", "4"]  # IDs to remove
        for req_id in id_to_remove:
            removed_index = self.input_batch.remove_request(req_id)
            if removed_index is not None:
                removed_req_indices.append(removed_index)
        self.assertEqual(len(removed_req_indices), len(id_to_remove))
        self.input_batch.condense(sorted(removed_req_indices, reverse=True))

        # Check if the remaining requests are condensed correctly
        indices = [
            self.input_batch.req_id_to_index[req_id] for req_id in ["1", "3"]
        ]
        self.assertTrue(all(idx < self.input_batch.num_reqs
                            for idx in indices))

        for i in range(self.input_batch.num_reqs):
            self.assertIsNotNone(self.input_batch._req_ids[i])
        for i in range(self.input_batch.num_reqs,
                       len(self.input_batch._req_ids)):
            self.assertIsNone(self.input_batch._req_ids[i])

        for req_id in ["1", "3"]:
            idx = self.input_batch.req_id_to_index[req_id]
            tokens = self.input_batch.token_ids_cpu[idx]
            self.assertTrue(
                tokens.any(),
                f"Tokens at index {idx} for req {req_id} should not be all zero"
            )
