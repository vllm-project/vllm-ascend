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
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm.config.vllm import set_current_vllm_config

from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.vocab_parallel_embedding import (
    AscendLogitsProcessor,
    AscendParallelLMHead,
    AscendVocabParallelEmbedding,
    VocabParallelMode,
    _resolve_vocab_parallel_plan,
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

        self.mock_pcp_group = mock.MagicMock()
        self.mock_pcp_group.world_size = 1
        self.mock_pcp_group.rank_in_group = 0

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
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_pcp_group",
                return_value=self.mock_pcp_group,
            ),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.enable_pcp_embedding_lmhead_weight_sharding",
                return_value=False,
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
        self.assertIs(layer.parallel_mode, VocabParallelMode.REPLICATED)
        self.assertIsNone(layer.token_exchange_group)
        self.assertIsNone(layer.output_reduce_group)
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
        with (
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

        self.assertIs(layer.parallel_mode, VocabParallelMode.REPLICATED)
        self.assertIsNone(layer.token_exchange_group)
        self.assertIsNone(layer.output_reduce_group)
        self.assertEqual(layer.tp_size, 1)
        self.assertEqual(layer.tp_rank, 0)

        # tp_size==1: shard indices cover the full padded vocab, so each rank
        # holds the entire markov table (no padding rows reserved for peers).
        self.assertEqual(layer.num_embeddings_per_partition, layer.num_embeddings_padded)
        self.assertEqual(layer.num_org_embeddings_per_partition, layer.org_vocab_size_padded)
        self.assertEqual(layer.num_added_embeddings_per_partition, layer.num_added_embeddings)

    def test_pcp_embedding_communication_order(self):
        layer = self._create_layer()
        pcp_group = MagicMock(world_size=2, device_group="pcp")
        tp_group = MagicMock(world_size=2, unique_name="tp")
        layer.token_exchange_group = pcp_group
        layer.output_reduce_group = tp_group
        layer.embedding_tp_capacity = 2
        layer.params_dtype = torch.float32
        events = []

        def all_gather(output, input_, *, group):
            events.append(("pcp_all_gather", group))
            output.copy_(input_.repeat(2))

        def embedding(_, input_):
            events.append(("embedding", None))
            return input_.unsqueeze(-1).expand(-1, layer.embedding_dim).float()

        def reduce_scatter(output, input_, *, group):
            events.append(("pcp_reduce_scatter", group))
            output.copy_(input_[: output.shape[0]])

        def all_reduce(output, group_name):
            events.append(("tp_all_reduce", group_name))
            return output

        layer.quant_method.embedding = MagicMock(side_effect=embedding)
        with (
            patch("vllm_ascend.ops.vocab_parallel_embedding.dist.all_gather_into_tensor", side_effect=all_gather),
            patch("vllm_ascend.ops.vocab_parallel_embedding.dist.reduce_scatter_tensor", side_effect=reduce_scatter),
            patch("torch.ops.vllm.all_reduce", side_effect=all_reduce),
        ):
            output = layer(torch.tensor([15, 16]))

        self.assertEqual(
            events,
            [
                ("pcp_all_gather", "pcp"),
                ("embedding", None),
                ("pcp_reduce_scatter", "pcp"),
                ("tp_all_reduce", "tp"),
            ],
        )
        self.assertEqual(output.shape, (2, self.embedding_dim))


class TestVocabParallelPlan(unittest.TestCase):
    def test_resolve_vocab_parallel_plan(self):
        """Cover layout selection, communication groups, and conflicts."""

        def group(rank, world_size):
            return MagicMock(rank_in_group=rank, world_size=world_size)

        tp_group = group(1, 2)
        pcp_group = group(1, 2)
        embed_group = group(2, 4)
        lmhead_group = group(3, 4)
        pcp_config = MagicMock()
        pcp_config.parallel_config.prefill_context_parallel_size = 2
        cases: tuple[tuple[str, str, bool, bool, bool, bool, VocabParallelMode, int, int], ...] = (
            ("replicated", "layers.0.markov_head", True, False, False, False, VocabParallelMode.REPLICATED, 0, 1),
            ("standard", "model.norm", False, False, False, False, VocabParallelMode.STANDARD, 1, 2),
            (
                "fine-grained embedding",
                "model.embed_tokens",
                False,
                False,
                True,
                False,
                VocabParallelMode.FINE_GRAINED,
                2,
                4,
            ),
            (
                "fine-grained lm head",
                "lm_head",
                False,
                False,
                False,
                True,
                VocabParallelMode.FINE_GRAINED,
                3,
                4,
            ),
            (
                "pcp embedding",
                "model.embed_tokens",
                False,
                True,
                False,
                False,
                VocabParallelMode.PCP_X_TP,
                3,
                4,
            ),
            (
                "pcp lm head",
                "lm_head",
                False,
                True,
                False,
                False,
                VocabParallelMode.PCP_X_TP,
                3,
                4,
            ),
        )

        for name, prefix, disable_tp, pcp, embed_tp, lmhead_tp, mode, rank, world_size in cases:
            with (
                self.subTest(name=name),
                patch(
                    "vllm_ascend.ops.vocab_parallel_embedding.enable_pcp_embedding_lmhead_weight_sharding",
                    return_value=pcp,
                ),
                patch("vllm_ascend.ops.vocab_parallel_embedding.embedding_tp_enable", return_value=embed_tp),
                patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable", return_value=lmhead_tp),
                patch(
                    "vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config",
                    return_value=pcp_config,
                ),
                patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=tp_group),
                patch("vllm_ascend.ops.vocab_parallel_embedding.get_pcp_group", return_value=pcp_group),
                patch("vllm_ascend.ops.vocab_parallel_embedding.get_embed_tp_group", return_value=embed_group),
                patch("vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group", return_value=lmhead_group),
            ):
                plan = _resolve_vocab_parallel_plan(prefix=prefix, disable_tp=disable_tp)

            self.assertIs(plan.mode, mode)
            self.assertEqual((plan.shard_rank, plan.shard_world_size), (rank, world_size))
            if name == "pcp embedding":
                self.assertIs(plan.token_exchange_group, pcp_group)
                self.assertIs(plan.output_reduce_group, tp_group)
            elif name == "fine-grained embedding":
                self.assertIs(plan.token_exchange_group, embed_group)
                self.assertIsNone(plan.output_reduce_group)
            elif mode is VocabParallelMode.STANDARD:
                self.assertIsNone(plan.token_exchange_group)
                self.assertIs(plan.output_reduce_group, tp_group)
            else:
                self.assertIsNone(plan.token_exchange_group)
                self.assertIsNone(plan.output_reduce_group)

        non_pcp_config = MagicMock()
        non_pcp_config.parallel_config.prefill_context_parallel_size = 1
        with (
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.enable_pcp_embedding_lmhead_weight_sharding",
                return_value=True,
            ),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config",
                return_value=non_pcp_config,
            ),
            patch("vllm_ascend.ops.vocab_parallel_embedding.embedding_tp_enable", return_value=False),
            patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable", return_value=False),
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=tp_group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_pcp_group") as get_pcp_group,
        ):
            plan = _resolve_vocab_parallel_plan(prefix="model.embed_tokens", disable_tp=False)

        self.assertIs(plan.mode, VocabParallelMode.STANDARD)
        get_pcp_group.assert_not_called()

        for prefix, embed_tp, lmhead_tp, error in (
            ("model.embed_tokens", True, False, "embedding_tensor_parallel_size"),
            ("lm_head", False, True, "lmhead_tensor_parallel_size"),
        ):
            with (
                self.subTest(prefix=prefix, error=error),
                patch(
                    "vllm_ascend.ops.vocab_parallel_embedding.enable_pcp_embedding_lmhead_weight_sharding",
                    return_value=True,
                ),
                patch("vllm_ascend.ops.vocab_parallel_embedding.embedding_tp_enable", return_value=embed_tp),
                patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable", return_value=lmhead_tp),
                patch(
                    "vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config",
                    return_value=pcp_config,
                ),
                patch("vllm_ascend.ops.vocab_parallel_embedding.get_pcp_group", return_value=pcp_group),
                self.assertRaisesRegex(ValueError, error),
            ):
                _resolve_vocab_parallel_plan(prefix=prefix, disable_tp=False)


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
        self.mock_pcp_group = MagicMock(world_size=1, rank_in_group=0)
        self.mock_ascend_config = MagicMock()

        self.mock_quant_method = MagicMock()
        # 2 rows so lmhead_all_to_all's equal split (world_size=2) holds.
        self.mock_quant_method.apply = MagicMock(return_value=torch.randn(2, self.vocab_size))
        self.mock_all_to_all_single = MagicMock(side_effect=lambda out, inp, **kwargs: out.copy_(inp))
        self.patches = [
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable", return_value=True),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_pcp_group",
                return_value=self.mock_pcp_group,
            ),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.enable_pcp_embedding_lmhead_weight_sharding",
                return_value=False,
            ),
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
        replicated_head.parallel_mode = VocabParallelMode.REPLICATED
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
        sharded_head.parallel_mode = VocabParallelMode.FINE_GRAINED
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        with (
            patch.object(processor, "_get_logits_normal") as mock_normal,
            patch.object(processor, "_get_logits_lmheadtp", return_value="lmheadtp") as mock_lmheadtp,
        ):
            result = processor._get_logits(hidden_states, sharded_head, None)

        self.assertEqual(result, "lmheadtp")
        mock_lmheadtp.assert_called_once_with(hidden_states, sharded_head, None)
        mock_normal.assert_not_called()

    def test_get_logits_pcp_reconstructs_vocab_in_group_order(self):
        processor = AscendLogitsProcessor(vocab_size=7)
        lm_head = MagicMock(parallel_mode=VocabParallelMode.PCP_X_TP)
        events = []

        pcp_group = MagicMock(world_size=2)
        tp_group = MagicMock(world_size=2)

        def pcp_all_gather(logits, *, dim):
            events.append(("pcp", dim))
            return torch.cat((logits, logits + 10), dim=dim)

        def tp_all_gather(logits, *, dim):
            events.append(("tp", dim))
            return torch.cat((logits, logits + 20), dim=dim)

        pcp_group.all_gather.side_effect = pcp_all_gather
        tp_group.all_gather.side_effect = tp_all_gather
        with (
            patch.object(processor, "_apply_head", return_value=torch.tensor([[0.0, 1.0]])),
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_pcp_group", return_value=pcp_group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=tp_group),
        ):
            logits = processor._get_logits(torch.randn(1, 4), lm_head)

        self.assertEqual(events, [("pcp", -1), ("tp", -1)])
        self.assertTrue(torch.equal(logits, torch.tensor([[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0]])))
