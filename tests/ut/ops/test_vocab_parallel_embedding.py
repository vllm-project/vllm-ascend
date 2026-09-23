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
# Adapted from vllm/tests/lora/test_layers.py

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm.config.vllm import set_current_vllm_config

from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.vocab_parallel_embedding import (
    AscendLogitsProcessor,
    AscendParallelLMHead,
    AscendVocabParallelEmbedding,
)

VOCAB_PARALLEL_EMBEDDING_TEST_NUM_RANDOM_SEEDS = 128


class TestCustomVocabParallelEmbedding(unittest.TestCase):
    def setUp(self):
        self.num_embeddings = 50
        self.embedding_dim = 10
        self.org_num_embeddings = 40
        self.padding_size = 8

        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0
        self.mock_group.unique_name = "test_tp_group"

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {}
        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.embedding_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.utils.get_ascend_config", return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_lmhead_tp_group", return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_tp_group",
                return_value=self.mock_group,
            ),
        ]

        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def _create_layer(self):
        # Patch methods and dependencies for VocabParallelEmbedding
        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_group.rank_in_group = 0
        with (
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=mock_group),
            patch("vllm.model_executor.layers.vocab_parallel_embedding.pad_vocab_size", side_effect=lambda x, y: x + y),
            patch("vllm.model_executor.layers.vocab_parallel_embedding.divide", side_effect=lambda x, y: x // y),
        ):
            # Create an instance of VocabParallelEmbedding
            layer = AscendVocabParallelEmbedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                org_num_embeddings=self.org_num_embeddings,
                padding_size=self.padding_size,
                quant_config=None,  # Mock quantization config
                prefix="",
            )

            layer.shard_indices = MagicMock()
            layer.shard_indices.org_vocab_start_index = 10
            layer.shard_indices.org_vocab_end_index = 20
            layer.shard_indices.num_org_vocab_padding = 5
            layer.shard_indices.added_vocab_start_index = 30
            layer.shard_indices.added_vocab_end_index = 40

            # Mock the quantization method
            layer.quant_method.embedding = MagicMock(
                side_effect=lambda _, x: torch.randn(x.shape[0], self.embedding_dim)
            )
            return layer

    def test_mask_input_for_vocab_range(self):
        """Test the mask and offset calculation helper function."""
        layer = self._create_layer()

        input_ = torch.tensor([5, 15, 25, 35, 45])

        masked_input, mask = layer._mask_input_for_vocab_range(
            input_,
            org_vocab_start_index=10,
            org_vocab_end_index=20,
            num_org_vocab_padding=5,
            added_vocab_start_index=30,
            added_vocab_end_index=40,
        )

        expected_mask = torch.tensor([True, False, True, False, True])
        self.assertTrue(torch.equal(mask, expected_mask), f"Mask mismatch. Expected {expected_mask}, got {mask}")

        expected_masked = torch.tensor([0, 5, 0, 20, 0])
        self.assertTrue(
            torch.equal(masked_input, expected_masked),
            f"Masked input mismatch. Expected {expected_masked}, got {masked_input}",
        )

    def test_forward_with_tp_size_1(self):
        """Test forward pass without tensor parallelism."""
        # Create a fresh mock embedding with tp_size=1
        layer = self._create_layer()
        layer.tp_size = 1
        self.mock_group.world_size = 1
        layer.quant_method.embedding = MagicMock(return_value=torch.randn(3, layer.embedding_dim))

        input_ = torch.tensor([1, 2, 3])

        with patch("torch.ops.vllm.all_reduce", side_effect=lambda x, _: x) as mock_reduce_tp1:
            output = layer.forward(input_)

        # Should just pass through without masking
        layer.quant_method.embedding.assert_called_once_with(layer, input_.long())
        self.assertEqual(output.shape, (3, layer.embedding_dim))

        # A tp_size==1 layer already holds the full output locally (e.g. the
        # replicated DSpark Markov head), so the reduce must be skipped.
        mock_reduce_tp1.assert_not_called()

    def test_forward_with_tp(self):
        layer = self._create_layer()
        layer.tp_size = 2

        input_ = torch.tensor([15, 35])  # one org vocab, one added vocab

        with patch("torch.ops.vllm.all_reduce", side_effect=lambda x, _: x) as mock_reduce_tp:
            # Call the forward method
            output = layer.forward(input_)

        # Check that masking was applied correctly
        layer.quant_method.embedding.assert_called_once()
        called_input = layer.quant_method.embedding.call_args[0][1]
        expected_input = torch.tensor([5, 20])  # after offset calculation
        self.assertTrue(torch.all(called_input == expected_input))

        # Check that all reduce was called
        mock_reduce_tp.assert_called_once()
        self.assertEqual(output.shape, (2, self.embedding_dim))

    def test_sequence_parallel_moe_keeps_complete_embedding(self):
        layer = self._create_layer()
        input_ = torch.tensor([15, 35, 16, 36])
        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config.use_sequence_parallel_moe = True

        with (
            set_current_vllm_config(mock_vllm_config),
            patch("torch.ops.vllm.all_reduce", side_effect=lambda x, _: x) as mock_all_reduce,
            patch("torch.ops.vllm.reduce_scatter") as mock_reduce_scatter,
        ):
            output = layer.forward(input_)

        self.assertEqual(output.shape, (input_.shape[0], self.embedding_dim))
        mock_all_reduce.assert_called_once()
        mock_reduce_scatter.assert_not_called()

    def test_forward_with_invalid_vocab(self):
        """Test that invalid vocab indices are properly masked out."""
        # Create a fresh embedding layer
        layer = self._create_layer()
        input_ = torch.tensor([5, 15, 25, 35, 45])  # includes invalid cases
        # Create predictable mock output
        mock_output = torch.randn(5, self.embedding_dim)
        layer.quant_method.embedding = MagicMock(return_value=mock_output.clone())

        # Patch tensor_model_parallel_all_reduce to mock its behavior
        with patch("torch.ops.vllm.all_reduce", side_effect=lambda x, _: x):
            # Call the forward method
            output = layer.forward(input_)
        # Check that invalid positions (0, 2, 4) were zeroed out
        self.assertTrue(torch.all(output[0] == 0))
        self.assertTrue(torch.all(output[2] == 0))
        self.assertTrue(torch.all(output[4] == 0))
        self.assertTrue(torch.all(output[1] == mock_output[1]))
        self.assertTrue(torch.all(output[3] == mock_output[3]))
        self.assertEqual(output.shape, (5, self.embedding_dim))

    def test_output_shape(self):
        """Test that output shape is correct."""
        # Create a fresh embedding layer
        layer = self._create_layer()

        test_cases = [
            (torch.tensor([15]), (1, self.embedding_dim)),
            (torch.tensor([15, 35]), (2, self.embedding_dim)),
            (torch.tensor([15, 35, 16, 36]), (4, self.embedding_dim)),
        ]

        for input_, expected_shape in test_cases:
            with self.subTest(input=input_):
                with patch("torch.ops.vllm.all_reduce", side_effect=lambda x, _: x):
                    # Call the forward method
                    output = layer.forward(input_)
                self.assertEqual(output.shape, expected_shape)

    def test_disable_tp(self):
        layer = AscendVocabParallelEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            org_num_embeddings=self.org_num_embeddings,
            padding_size=self.padding_size,
            quant_config=None,
            prefix="",
            disable_tp=True,
        )

        self.assertTrue(layer.disable_tp)
        self.assertIs(layer.comm_group, parallel_state.get_replicated_group())
        self.assertEqual(layer.tp_size, 1)
        self.assertEqual(layer.tp_rank, 0)

    def test_dspark_markov_lm_head_replicated(self):
        """The DSpark markov lm_head is replicated on every rank (vllm#49731).

        vllm's DSparkMarkovHead constructs markov_w2 as a ParallelLMHead with
        disable_tp=True; its prefix ("layers.N.markov_head.markov_w2")
        contains "head", so disable_tp must win over the lmhead prefix match
        even when lmhead_tp is enabled — setUp makes lmhead_tp_enable()
        return True — and pin the layer to the world_size=1 ReplicatedGroup
        so every rank holds the full table and forward skips all
        communication.
        """
        markov_group = MagicMock()
        markov_group.world_size = 1
        markov_group.rank_in_group = 0
        with (
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_replicated_group", return_value=markov_group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=MagicMock()),
            patch(
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm.model_executor.layers.vocab_parallel_embedding.pad_vocab_size",
                side_effect=lambda x, y: x + y,
            ),
            patch("vllm.model_executor.layers.vocab_parallel_embedding.divide", side_effect=lambda x, y: x // y),
        ):
            layer = AscendVocabParallelEmbedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                org_num_embeddings=self.org_num_embeddings,
                padding_size=self.padding_size,
                quant_config=None,
                prefix="layers.0.markov_head.markov_w2",
                disable_tp=True,
            )

        self.assertIs(layer.comm_group, markov_group)
        self.assertEqual(layer.tp_size, 1)
        self.assertEqual(layer.tp_rank, 0)
        self.assertIsNone(layer.forward_type)

        # tp_size==1: shard indices cover the full padded vocab, so each rank
        # holds the entire markov table (no padding rows reserved for peers).
        self.assertEqual(layer.num_embeddings_per_partition, layer.num_embeddings_padded)
        self.assertEqual(layer.num_org_embeddings_per_partition, layer.org_vocab_size_padded)
        self.assertEqual(layer.num_added_embeddings_per_partition, layer.num_added_embeddings)


class TestPCPVocabParallelEmbedding(unittest.TestCase):
    def setUp(self):
        self.module = "vllm_ascend.ops.vocab_parallel_embedding"
        self.config = MagicMock()
        self.config.parallel_config.prefill_context_parallel_size = 4
        self.config.model_config.hf_text_config.tie_word_embeddings = False
        self.config.scheduler_config.max_num_batched_tokens = 4
        self.config.compilation_config.max_cudagraph_capture_size = 2
        self.tp_group = MagicMock(world_size=1, rank_in_group=0)
        self.pcp_group = MagicMock(world_size=4, rank_in_group=0)
        for name, value in (
            ("get_current_vllm_config_or_none", self.config),
            ("get_tp_group", self.tp_group),
            ("get_pcp_group", self.pcp_group),
            ("embedding_tp_enable", False),
            ("lmhead_tp_enable", False),
            ("get_potential_max_tokens", 2),
            ("get_ascend_config", MagicMock(enable_reduce_sample=False)),
            ("is_forward_context_available", False),
        ):
            patcher = patch(f"{self.module}.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_layer(self, cls=AscendVocabParallelEmbedding, **kwargs):
        return cls(num_embeddings=70, embedding_dim=8, params_dtype=torch.float32, **kwargs)

    def test_pcp_weight_loading_and_local_outputs(self):
        self._check_local_outputs(([0, 32, 69], [31], [64, 32, 32, 1], []))

    def test_replicated_decode_outputs(self):
        self._check_local_outputs(([69], [69], [69], [69]))

    def test_mixed_prefill_decode_outputs(self):
        self._check_local_outputs(([5, 0, 69], [5, 31], [5, 32, 64], [5]))

    def _check_local_outputs(self, sequences):
        # Different sequences, repeated IDs, vocab boundaries, padding and an
        # empty rank exercise the gather/scatter layout, not just group choice.
        weight = torch.arange(70 * 8, dtype=torch.float32).reshape(70, 8)
        inputs = [torch.tensor(ids, dtype=torch.long) for ids in sequences]
        gathered = torch.zeros(4, 4, dtype=torch.long)
        for rank, ids in enumerate(inputs):
            gathered[rank, : len(ids)] = ids
        partials, destinations, outputs = [], [], []

        def gather(out, inp, **kwargs):
            torch.testing.assert_close(inp, gathered[self.pcp_group.rank_in_group])
            out.copy_(gathered.flatten())
            self.assertIs(kwargs["group"], self.pcp_group.device_group)

        def scatter(out, inp, **kwargs):
            partials.append(inp.clone())
            destinations.append(out)
            self.assertIs(kwargs["group"], self.pcp_group.device_group)

        with (
            patch(f"{self.module}.dist.all_gather_into_tensor", side_effect=gather),
            patch(f"{self.module}.dist.reduce_scatter_tensor", side_effect=scatter),
        ):
            for rank, ids in enumerate(inputs):
                self.pcp_group.rank_in_group = rank
                layer = self.make_layer()
                self.assertIs(layer.comm_group, self.pcp_group)
                self.assertEqual(layer.weight.shape, (32, 8))
                layer.weight_loader(layer.weight, weight)
                outputs.append(layer(ids))
        reduced = torch.stack(partials).sum(0).reshape(4, 4, 8)
        for rank, (out, ids) in enumerate(zip(outputs, inputs)):
            destinations[rank].copy_(reduced[rank])
            torch.testing.assert_close(out, weight[ids])

    def test_other_layouts_keep_original_group(self):
        self.assertIs(self.make_layer(AscendParallelLMHead).comm_group, self.pcp_group)
        self.assertEqual(self.make_layer(disable_tp=True).comm_group.world_size, 1)
        self.tp_group.world_size = 2
        self.assertIs(self.make_layer().comm_group, self.tp_group)
        self.tp_group.world_size = 1
        self.config.parallel_config.prefill_context_parallel_size = 1
        self.assertIs(self.make_layer().comm_group, self.tp_group)

    def test_tied_weights_share_pcp_partition(self):
        self.config.model_config.hf_text_config.tie_word_embeddings = True
        embed = self.make_layer()
        head = self.make_layer(AscendParallelLMHead)
        head.tie_weights(embed)
        self.assertIs(head.weight, embed.weight)
        self.assertEqual(head.shard_indices, embed.shard_indices)
        self.assertEqual(head.forward_type, "lmhead_pcp")

    def test_reduce_sample_rejected(self):
        with (
            patch(f"{self.module}.get_ascend_config", return_value=MagicMock(enable_reduce_sample=True)),
            self.assertRaisesRegex(ValueError, "enable_reduce_sample"),
        ):
            self.make_layer(AscendParallelLMHead)

    def test_lmhead_restored_prefill_decode_and_mixed_rows(self):
        # PCP restores token rows before compute_logits for every batch type.
        for num_rows in (1, 3, 7):
            for skip_gather in (False, True):
                with self.subTest(num_rows=num_rows, skip_gather=skip_gather):
                    self._check_logits(num_rows, skip_gather)

    def _check_logits(self, num_rows, skip_gather):
        weight = torch.arange(70 * 8, dtype=torch.float32).reshape(70, 8) / 100
        bias = torch.arange(70, dtype=torch.float32) / 10
        hidden = torch.arange(num_rows * 8, dtype=torch.float32).reshape(num_rows, 8) / 10
        processor = object.__new__(AscendLogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.org_vocab_size = 70
        processor.head_dtype = None
        partials, gathered, outputs = [], [], []

        def gather(logits, dim):
            self.assertEqual(dim, -1)
            self.assertEqual(logits.shape, (num_rows, 32))
            partials.append(logits.clone())
            result = torch.empty(num_rows, 128)
            gathered.append(result)
            return result

        with patch.object(self.pcp_group, "all_gather", side_effect=gather):
            for rank in range(4):
                self.pcp_group.rank_in_group = rank
                head = self.make_layer(AscendParallelLMHead, bias=True)
                head.weight_loader(head.weight, weight)
                head.weight_loader(head.bias, bias)
                outputs.append(processor._get_logits(hidden, head, head.bias, skip_gather))
        full_logits = torch.cat(partials, dim=-1)
        expected = torch.nn.functional.linear(hidden, weight, bias)
        for buffer, output in zip(gathered, outputs):
            buffer.copy_(full_logits)
            self.assertEqual(output.shape, (num_rows, 128 if skip_gather else 70))
            torch.testing.assert_close(output[:, :70], expected)
            if skip_gather:
                torch.testing.assert_close(output[:, 70:], torch.zeros(num_rows, 58))

    def test_capacity_overflow(self):
        with self.assertRaisesRegex(ValueError, "static capacity"):
            self.make_layer()(torch.zeros(5, dtype=torch.long))

    def test_pcp_forward_dispatch(self):
        layer = self.make_layer()
        ids = torch.tensor([0, 32, 69])
        expected = torch.empty(3, 8)
        with (
            patch.object(layer, "_forward_partitioned_inputs", return_value=expected) as forward,
            patch.object(layer, "_forward_origin", side_effect=AssertionError("PCP must not use TP all-reduce")),
            patch.object(layer, "_forward_embed_tp", side_effect=AssertionError("PCP needs prefill capacity")),
        ):
            self.assertIs(layer(ids), expected)
            forward.assert_called_once_with(ids, 4, active_tokens=None)

    def test_pcp_layout_dispatch(self):
        layer = self.make_layer()
        ids = torch.tensor([0, 32])
        for replicated in (True, False):
            context = SimpleNamespace(attn_metadata=SimpleNamespace(pcp_inputs_replicated=replicated))
            with (
                patch(f"{self.module}.is_forward_context_available", return_value=True),
                patch(f"{self.module}.get_forward_context", return_value=context),
                patch.object(layer, "_forward_replicated_pcp") as decode,
                patch.object(layer, "_forward_partitioned_inputs") as partitioned,
            ):
                layer(ids)
                if replicated:
                    decode.assert_called_once_with(ids)
                    partitioned.assert_not_called()
                else:
                    partitioned.assert_called_once_with(ids, 4, active_tokens=2)
                    decode.assert_not_called()

    def test_decode_uses_one_all_reduce_without_id_gather(self):
        weight = torch.arange(70 * 8, dtype=torch.float32).reshape(70, 8)
        ids = torch.tensor([1, 69])
        partials, outputs = [], []

        def reduce(output, **kwargs):
            self.assertEqual(output.shape, (2, 8))
            self.assertIs(kwargs["group"], self.pcp_group.device_group)
            partials.append(output.clone())

        with (
            patch(f"{self.module}.dist.all_reduce", side_effect=reduce) as all_reduce,
            patch(f"{self.module}.dist.all_gather_into_tensor") as gather,
            patch(f"{self.module}.dist.reduce_scatter_tensor") as scatter,
        ):
            for rank in range(4):
                self.pcp_group.rank_in_group = rank
                layer = self.make_layer()
                layer.weight_loader(layer.weight, weight)
                outputs.append(layer._forward_replicated_pcp(ids))
            self.assertEqual(all_reduce.call_count, 4)
            gather.assert_not_called()
            scatter.assert_not_called()
        result = torch.stack(partials).sum(0)
        torch.testing.assert_close(result, weight[ids])

    def test_prefill_communicates_current_padded_length(self):
        layer = self.make_layer()
        ids = torch.tensor([0, 1])
        layer.weight.data.fill_(1)

        def gather(out, inp, **kwargs):
            self.assertEqual(inp.shape, (2,))
            self.assertEqual(out.shape, (8,))
            out.copy_(inp.repeat(4))

        def scatter(out, inp, **kwargs):
            self.assertEqual(inp.shape, (8, 8))
            self.assertEqual(out.shape, (2, 8))
            out.copy_(inp[:2])

        with (
            patch(f"{self.module}.dist.all_gather_into_tensor", side_effect=gather),
            patch(f"{self.module}.dist.reduce_scatter_tensor", side_effect=scatter),
        ):
            output = layer._forward_partitioned_inputs(ids, 4, active_tokens=2)
            self.assertEqual(output.shape, (2, 8))
            address = layer._embed_rs_in_buf.data_ptr()
            layer._forward_partitioned_inputs(ids, 4, active_tokens=2)
            self.assertEqual(layer._embed_rs_in_buf.data_ptr(), address)


class TestAscendLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.compilation_config.custom_ops = ["all"]
        self.mock_vllm_config.model_config = None

        from vllm.config.vllm import set_current_vllm_config

        self.config_context = set_current_vllm_config(self.mock_vllm_config)
        self.config_context.__enter__()
        self.addCleanup(self.config_context.__exit__, None, None, None)
        self.vocab_size = 50
        self.num_embeddings = 50
        self.embedding_dim = 10
        self.org_num_embeddings = 40
        self.padding_size = 8

        self.mock_group = MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0
        self.mock_ascend_config = MagicMock()
        # enable_reduce_sample must be explicitly False so _get_logits_lmheadtp
        # reaches the lmhead_all_to_all branch (a MagicMock attribute is truthy
        # and would silently skip it).
        self.mock_ascend_config.enable_reduce_sample = False
        self.mock_quant_method = MagicMock()
        # 2 rows so lmhead_all_to_all's equal split (world_size=2) holds.
        self.mock_quant_method.apply = MagicMock(return_value=torch.randn(2, self.vocab_size))
        self.mock_all_to_all_single = MagicMock(side_effect=lambda out, inp, **kwargs: out.copy_(inp))
        self.patches = [
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_ascend_config", return_value=self.mock_ascend_config),
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable", return_value=True),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.dist.all_to_all_single",
                self.mock_all_to_all_single,
            ),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group.all_gather",
                return_value=torch.randn(2, self.vocab_size),
            ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_create_processor(self):
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        self.assertEqual(processor.vocab_size, self.vocab_size)

    def test_get_logits(self):
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        lmhead = AscendParallelLMHead(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, prefix="lm_head"
        )
        lmhead.quant_method = self.mock_quant_method
        lmhead.quant_method.apply = self.mock_quant_method.apply
        hidden_state = torch.randn(1, self.org_num_embeddings)
        logits = processor._get_logits(hidden_state, lmhead)
        self.mock_quant_method.apply.assert_called_once()
        # The lmhead-TP path must actually reach the collective; a missing
        # assertion here would silently regress to not exercising it.
        self.mock_all_to_all_single.assert_called_once()
        # [N/P, V] after redistribution, then truncated to org_vocab_size.
        self.assertEqual(logits.shape, (1, self.vocab_size))

    def test_get_logits_replicated_head_takes_normal_path(self):
        """A replicated head (tp_size==1, e.g. the DSpark Markov w2) must not
        join the lmhead_tp logits exchange even when lmhead_tp is enabled:
        it holds the full table locally, so gathering/scattering across the
        finegrained group would be wrong."""
        hidden_states = torch.randn(1, 4)
        replicated_head = MagicMock()
        replicated_head.tp_size = 1
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        with (
            patch.object(processor, "_get_logits_normal", return_value="normal") as mock_normal,
            patch.object(processor, "_get_logits_lmheadtp") as mock_lmheadtp,
        ):
            result = processor._get_logits(hidden_states, replicated_head, None)

        self.assertEqual(result, "normal")
        mock_normal.assert_called_once_with(hidden_states, replicated_head, None)
        mock_lmheadtp.assert_not_called()

    def test_get_logits_sharded_head_takes_lmheadtp_path(self):
        """A tp_size>1 lm_head keeps the lmhead_tp path (guard precision)."""
        hidden_states = torch.randn(1, 4)
        sharded_head = MagicMock()
        sharded_head.tp_size = 2
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        with (
            patch.object(processor, "_get_logits_normal") as mock_normal,
            patch.object(processor, "_get_logits_lmheadtp", return_value="lmheadtp") as mock_lmheadtp,
        ):
            result = processor._get_logits(hidden_states, sharded_head, None)

        self.assertEqual(result, "lmheadtp")
        mock_lmheadtp.assert_called_once_with(hidden_states, sharded_head, None)
        mock_normal.assert_not_called()
