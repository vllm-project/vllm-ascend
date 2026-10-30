# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


class CommonMetadata(SimpleNamespace):
    def replace(self, **kwargs):
        return CommonMetadata(**(vars(self) | kwargs))


def make_builder(cls, ratio=0):
    """Use real builder methods with CPU buffers and mocked device operators."""
    builder = cls.__new__(cls)
    builder.compressor_ratio = ratio
    builder.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=2)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            sliding_window=4,
            num_attention_heads=8,
            index_topk=16,
            index_n_heads=8,
            index_head_dim=8,
        ),
        get_head_size=lambda: 8,
    )
    builder.vllm_config = SimpleNamespace(
        model_config=builder.model_config,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
    )
    builder.use_pcp = True
    builder._device_metadata_enabled = False
    builder.compressor_metadata_buffers = None
    builder._sas_variant_buffers = {}
    builder._step_rope_buffers = {}
    builder.decode_threshold = 2
    builder.num_actual_tokens = 2
    builder.num_decodes, builder.num_prefills = 1, 0
    builder.num_decode_tokens, builder.num_prefill_tokens = 2, 0
    builder.seq_lens = torch.tensor([6], dtype=torch.int32)
    builder.block_table = torch.tensor([[3, 4]], dtype=torch.int32)
    builder.slot_mapping = torch.zeros(8, dtype=torch.int32)
    builder.start_pos_prefill = torch.full((4,), -1, dtype=torch.int32)
    builder.qli_seqused_k = torch.full((4,), -1, dtype=torch.int32)
    builder.qli_cmp_residual_k = torch.full((4,), -1, dtype=torch.int32)
    builder.sas_metadata_buffer = torch.zeros(dsa_v1.DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    builder.qli_metadata_buffer = torch.zeros_like(builder.sas_metadata_buffer)
    builder.seqused_q = torch.zeros(4, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.zeros(4, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.zeros(4, dtype=torch.int32)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32)
    builder.dspark_swa_indices_buffer = torch.empty((8, 1, 8), dtype=torch.int32)
    builder.storage_block_size = 4
    builder.cache_group_key = "cache"
    builder.hadamard = None
    builder.common_ratio_to_sas_metadata = {}
    builder.metadata_cls = dsa_v1.AscendDSAMetadata
    builder._num_compressor_metadata_rows = Mock(return_value=2)
    builder.set_num_actual_tokens = Mock()
    return builder


class TestDSAMetadataBuilders(TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(
                dsa_v1.AscendDSAMetadataBuilder,
                "_dspark_swa_indices_fast_path_eligible",
                return_value=False,
            )
        )
        stack.enter_context(patch.object(dsa_v1, "_draft_uses_sparse_flash_mla", return_value=False))
        self.builder = make_builder(dsa_v1.AscendDSASWAMetadataBuilder)
        self.common = CommonMetadata(
            num_reqs=1,
            num_actual_tokens=2,
            num_input_tokens=2,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=self.builder.seq_lens,
            _seq_lens_cpu=self.builder.seq_lens,
            block_table_tensor=self.builder.block_table,
            slot_mapping=torch.tensor([12, 13], dtype=torch.int32),
            positions=torch.tensor([4, 5]),
            causal=False,
            attn_state=None,
        )
        self.sas_op = Mock(
            side_effect=lambda **kw: torch.full(
                (dsa_v1.DSA_METADATA_BUFFER_SIZE,),
                2 if kw["ori_mask_mode"] == 0 else 1,
                dtype=torch.int32,
            )
        )
        self.qli_op = stack.enter_context(
            patch.object(
                torch.ops._C_ascend,
                "npu_quant_lightning_indexer_v2_metadata",
                create=True,
                return_value=torch.ones(dsa_v1.DSA_METADATA_BUFFER_SIZE, dtype=torch.int32),
            )
        )
        plan = SimpleNamespace(
            layout_kv="PA_BSND",
            get_dsa_sparse_attn_metadata_op=lambda: self.sas_op,
            get_dsa_sparse_attn_metadata_kwargs=lambda device: {},
            format_dsa_slot_mapping=lambda slots, block_size: slots,
        )
        stack.enter_context(patch.object(dsa_v1, "get_dsa_attn_kv_plan", return_value=plan))
        stack.enter_context(patch.object(dsa_v1, "_dsa_swa_only_cmp_ratio", return_value=1))
        stack.enter_context(patch.object(dsa_v1, "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(dsa_v1, "is_turboquant", return_value=False))
        stack.enter_context(patch.object(dsa_v1, "get_dspark_sparse_sas_window", return_value=(5, 0)))
        stack.enter_context(patch.object(dsa_v1, "get_full_cos_and_sin_dsa", return_value=(None, None)))
        self.rope_op = stack.enter_context(patch.object(dsa_v1, "get_cos_and_sin_dsa", return_value=(None, None)))
        self.indices_op = stack.enter_context(
            patch.object(
                dsa_v1,
                "build_dspark_swa_indices",
                return_value=(torch.ones((2, 1, 8), dtype=torch.int32), None),
            )
        )
        stack.enter_context(
            patch.object(
                dsa_v1,
                "DeviceOperator",
                SimpleNamespace(
                    get_dsa_decode_cu_seqlens_ori_kv=lambda *args: None,
                    get_dsa_decode_cu_seqlens_cmp_kv=lambda *args: None,
                    get_dsa_indexer_quant_mode=lambda: 0,
                ),
            )
        )
        stack.enter_context(patch.object(dsa_v1, "split_decodes_and_prefills", return_value=(1, 0, 2, 0)))

    def build_req(self, builder=None, **kwargs):
        builder = builder or self.builder
        return builder.build_req_metadata(
            self.common,
            self.common._seq_lens_cpu,
            1,
            None,
            None,
            **kwargs,
        )

    def test_backend_selects_role_builder_in_both_modes(self):
        roles = [
            (dsa_v1.AscendDSAC4StateBackend, "State"),
            (dsa_v1.AscendDSAC128StateBackend, "State"),
            (dsa_v1.AscendDSASWABackend, "SWA"),
            (dsa_v1.AscendDSAC4Backend, "Attention"),
            (dsa_v1.AscendDSAC128Backend, "Attention"),
            (dsa_v1.AscendDSAIndexerBackend, "Indexer"),
        ]
        for base, module, prefix in [
            (dsa_v1.AscendDSAMetadataBuilder, dsa_v1, "AscendDSA"),
            (dsa_cp.AscendDSAPCPMetadataBuilder, dsa_cp, "AscendDSAPCP"),
        ]:
            with patch.object(dsa_v1.AscendDSABackend, "get_builder_cls", return_value=base):
                for backend, role in roles:
                    cls = backend.get_builder_cls()
                    self.assertIs(cls, getattr(module, f"{prefix}{role}MetadataBuilder"))
                    builder = make_builder(cls)
                    self.assertTrue(builder._use_specialized_metadata())

    def test_state_skips_query_and_compressor_preparation(self):
        builder = make_builder(dsa_v1.AscendDSAStateMetadataBuilder, ratio=4)
        result = builder.build(0, self.common, common_ratio_to_sas_metadata={})
        self.assertIsNone(result.req_metadata.sas_metadata)
        self.assertIsNone(result.req_metadata.qli_metadata)
        self.assertIsNone(result.req_metadata.cos)
        self.assertIsNone(result.req_metadata.start_pos)
        self.rope_op.assert_not_called()
        self.sas_op.assert_not_called()
        self.qli_op.assert_not_called()

    def test_state_zeros_padding_including_empty_rank(self):
        for actual_reqs in (0, 1):
            builder = make_builder(dsa_v1.AscendDSAStateMetadataBuilder, ratio=4)
            common = self.common.replace(
                num_reqs=2,
                block_table_tensor=torch.ones((2, 2), dtype=torch.int32),
                seq_lens=torch.tensor([6, 0], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32),
            )
            shared = {
                "num_decodes": 2,
                "num_prefills": 0,
                "num_decode_tokens": 2,
                "num_prefill_tokens": 0,
                "seq_lens": common.seq_lens,
            }
            req = builder.build(
                0,
                common,
                num_actual_reqs=actual_reqs,
                common_ratio_to_sas_metadata=shared,
            ).req_metadata
            self.assertEqual(req.num_actual_reqs, actual_reqs)
            self.assertTrue(torch.all(req.block_table[actual_reqs:] == 0))

    def test_swa_dspark_keeps_indices_and_window(self):
        req = self.build_req(update_cache=False)
        self.assertIs(req.dspark_swa_indices, self.indices_op.return_value[0])
        self.assertEqual((req.ori_win_left, req.ori_win_right), (5, 0))
        self.assertEqual(self.sas_op.call_args.kwargs["ori_mask_mode"], 0)
        self.assertIsNone(req.qli_metadata)
        self.assertIsNone(req.start_pos)
        self.assertIsNone(req.slot_mapping)
        self.qli_op.assert_not_called()

    def test_sparse_flash_draft_preserves_logical_indices_and_lengths(self):
        self.builder.dspark_swa_topk_lengths_buffer = torch.zeros(8, dtype=torch.int32)
        indices = torch.ones((2, 1, 8), dtype=torch.int32)
        lengths = self.builder.dspark_swa_topk_lengths_buffer[:2]
        self.indices_op.return_value = (indices, lengths)
        with (
            patch.object(dsa_v1, "_draft_uses_sparse_flash_mla", return_value=True),
            patch.object(
                self.builder,
                "_build_dspark_sparse_flash_mla_metadata",
                return_value=torch.full_like(self.builder.sas_metadata_buffer, 7),
            ) as sparse,
        ):
            req = self.build_req()
        self.assertIs(req.dspark_swa_indices, indices)
        self.assertIs(req.dspark_swa_topk_lengths, lengths)
        self.assertTrue(self.indices_op.call_args.kwargs["use_logical_indices"])
        self.assertIs(sparse.call_args.kwargs["ori_topk_length"], lengths)
        self.assertEqual(sparse.call_args.kwargs["ori_topk"], 8)
        self.assertTrue(torch.all(req.sas_metadata == 7))
        self.sas_op.assert_not_called()

    def test_cache_only_skips_sparse_flash_preparation(self):
        with (
            patch.object(dsa_v1, "_draft_uses_sparse_flash_mla", return_value=True),
            patch.object(self.builder, "_build_dspark_sparse_flash_mla_metadata") as sparse,
        ):
            req = self.build_req(build_query=False)
        self.assertIsNone(req.sas_metadata)
        self.assertIsNone(req.dspark_swa_indices)
        sparse.assert_not_called()
        self.indices_op.assert_not_called()

    def test_causal_and_noncausal_sas_do_not_alias(self):
        self.common.causal = True
        causal = self.build_req()
        self.common.causal = False
        draft = self.build_req()
        self.assertNotEqual(causal.sas_metadata.data_ptr(), draft.sas_metadata.data_ptr())
        self.assertTrue(torch.all(causal.sas_metadata == 1))
        self.assertTrue(torch.all(draft.sas_metadata == 2))
        self.assertEqual(self.sas_op.call_count, 2)

    def test_sas_shared_but_physical_indices_are_per_group(self):
        other = make_builder(dsa_v1.AscendDSASWAMetadataBuilder)
        other.block_table = torch.tensor([[9, 10]], dtype=torch.int32)
        other.common_ratio_to_sas_metadata = self.builder.common_ratio_to_sas_metadata
        first = self.build_req()
        second = self.build_req(other)
        self.assertIs(first.sas_metadata, second.sas_metadata)
        self.sas_op.assert_called_once()
        self.assertEqual(self.indices_op.call_count, 2)
        torch.testing.assert_close(self.indices_op.call_args.args[0], other.block_table)

    def test_sas_recomputed_next_step_at_stable_address(self):
        first = self.build_req()
        self.builder.common_ratio_to_sas_metadata = {}
        second = self.build_req()
        self.assertIs(first.sas_metadata, second.sas_metadata)
        self.assertEqual(self.sas_op.call_count, 2)

    def test_prefill_skips_dspark_decode_indices(self):
        self.builder.num_prefills = 1
        req = self.build_req()
        self.assertIsNone(req.dspark_swa_indices)
        self.assertEqual(req.ori_win_left, 3)
        self.indices_op.assert_not_called()

    def test_indexer_shares_qli_output_and_auxiliary_buffers(self):
        first_builder = make_builder(dsa_v1.AscendDSAIndexerMetadataBuilder, ratio=4)
        second_builder = make_builder(dsa_v1.AscendDSAIndexerMetadataBuilder, ratio=4)
        second_builder.common_ratio_to_sas_metadata = first_builder.common_ratio_to_sas_metadata
        first = self.build_req(first_builder, update_cache=False)
        second = self.build_req(second_builder, update_cache=False)
        self.assertIs(first.qli_metadata, second.qli_metadata)
        self.assertIs(first.qli_seqused_k, second.qli_seqused_k)
        self.assertIs(first.qli_cmp_residual_k, second.qli_cmp_residual_k)
        torch.testing.assert_close(first.qli_seqused_k, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(first.qli_cmp_residual_k, torch.tensor([2], dtype=torch.int32))
        self.assertTrue(torch.all(second_builder.qli_seqused_k == -1))
        self.qli_op.assert_called_once()
        self.sas_op.assert_not_called()
        self.indices_op.assert_not_called()

    def test_compressed_attention_skips_qli_and_shares_start_positions(self):
        first_builder = make_builder(dsa_v1.AscendDSAAttentionMetadataBuilder, ratio=4)
        second_builder = make_builder(dsa_v1.AscendDSAAttentionMetadataBuilder, ratio=128)
        second_builder.common_ratio_to_sas_metadata = first_builder.common_ratio_to_sas_metadata
        first = self.build_req(first_builder, build_query=False)
        second = self.build_req(second_builder, build_query=False)
        self.assertIs(first.start_pos, second.start_pos)
        torch.testing.assert_close(first.start_pos, torch.tensor([4], dtype=torch.int32))
        self.assertEqual(first.num_compressed_tokens, 2)
        self.assertIsNone(first.sas_metadata)
        self.assertIsNone(first.qli_metadata)
        self.indices_op.assert_not_called()
        self.sas_op.assert_not_called()
        self.qli_op.assert_not_called()

    def test_compressed_query_builds_sas_only(self):
        builder = make_builder(dsa_v1.AscendDSAAttentionMetadataBuilder, ratio=4)
        self.common.causal = True
        req = self.build_req(builder, update_cache=False)
        self.assertIsNotNone(req.sas_metadata)
        self.assertIsNone(req.qli_metadata)
        self.assertIsNone(req.start_pos)
        self.assertIsNone(req.num_compressed_tokens)
        self.qli_op.assert_not_called()

    def test_global_compressor_needs_neither_cpu_lengths_nor_query_limits(self):
        builder = make_builder(dsa_v1.AscendDSAAttentionMetadataBuilder, ratio=4)
        common = self.common.replace(_seq_lens_cpu=None, seq_lens_cpu=None, num_input_tokens=8)
        req = builder.build(
            0,
            common,
            common_ratio_to_sas_metadata={},
            build_query=False,
            full_graph_mode=True,
        ).req_metadata
        self.assertEqual(req.num_compressed_tokens, 3)
        self.assertNotIn("seq_lens_cpu", builder.common_ratio_to_sas_metadata)
        self.assertNotIn("req_preamble", builder.common_ratio_to_sas_metadata)
        self.rope_op.assert_not_called()
        self.sas_op.assert_not_called()
        self.qli_op.assert_not_called()

    def test_dspark_rope_is_staged_once_per_invocation(self):
        other = make_builder(dsa_v1.AscendDSASWAMetadataBuilder)
        proxy = Mock()
        proxy.copy_to_buffers.return_value = (object(), object())
        self.rope_op.return_value = (proxy, object())
        shared: dict[Any, Any] = {}
        results = []
        for builder in (self.builder, other):
            builder.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs = lambda: True
            results.append(
                builder.build_for_cudagraph_capture(
                    self.common,
                    common_ratio_to_sas_metadata=shared,
                    full_graph_mode=True,
                )
            )
        self.rope_op.assert_called_once()
        self.assertFalse(self.rope_op.call_args.kwargs["use_cache"])
        proxy.copy_to_buffers.assert_called_once()
        self.assertIs(results[0].req_metadata.cos, results[1].req_metadata.cos)

    def test_device_metadata_and_vision_keep_full_builder(self):
        self.builder._device_metadata_enabled = True
        self.assertFalse(self.builder._use_specialized_metadata())
        self.builder._device_metadata_enabled = False
        self.builder.model_config.hf_config.vision_n_layers = 1
        self.assertFalse(self.builder._use_specialized_metadata())
        self.builder.model_config.hf_config.vision_n_layers = 0
        self.assertTrue(self.builder._use_specialized_metadata())

    def test_role_fields_for_query_cache_and_combined_builds(self):
        for role, ratio in (
            ("State", 4),
            ("State", 128),
            ("SWA", 0),
            ("Attention", 4),
            ("Attention", 128),
            ("Indexer", 4),
        ):
            for query, cache in ((True, True), (True, False), (False, True)):
                for full_graph in (False, True):
                    with self.subTest(role=role, ratio=ratio, query=query, cache=cache, graph=full_graph):
                        builder = make_builder(getattr(dsa_v1, f"AscendDSA{role}MetadataBuilder"), ratio)
                        rope = torch.ones(2)
                        self.rope_op.return_value = (rope, rope)
                        req = builder.build(
                            0,
                            self.common.replace(causal=True),
                            common_ratio_to_sas_metadata={},
                            build_query=query,
                            update_cache=cache,
                            full_graph_mode=full_graph,
                        ).req_metadata
                        self.assertEqual(req.sas_metadata is not None, query and role in ("SWA", "Attention"))
                        self.assertEqual(req.qli_metadata is not None, query and role == "Indexer")
                        self.assertEqual(req.start_pos is not None, cache and role in ("Attention", "Indexer"))
                        self.assertEqual(req.slot_mapping is not None, cache and role == "SWA")
                        self.assertEqual(req.num_compressed_tokens is not None, cache and role != "State")
                        self.assertEqual(req.cos is not None, role != "State" and (query or role == "SWA"))
                        if role == "State":
                            self.assertNotIn("seq_lens_cpu", builder.common_ratio_to_sas_metadata)
                        if req.slot_mapping is not None:
                            torch.testing.assert_close(req.slot_mapping, self.common.slot_mapping)

    def test_role_builders_keep_deferred_tasks_and_output_ownership(self):
        common = self.common.replace(causal=True)
        for role, ratio in (("State", 4), ("SWA", 0), ("Attention", 4), ("Attention", 128), ("Indexer", 4)):
            with self.subTest(role=role, ratio=ratio):
                cls = getattr(dsa_v1, f"AscendDSA{role}MetadataBuilder")
                builders = [make_builder(cls, ratio), make_builder(cls, ratio)]
                shared: dict[Any, Any] = {}
                results = []
                for builder in builders:
                    builder._device_metadata_enabled = True
                    if ratio > 1:
                        builder.compressor_metadata_buffers = tuple(torch.zeros(4) for _ in range(3))
                    req = builder.build(
                        0,
                        common,
                        common_ratio_to_sas_metadata=shared,
                        build_query=False,
                        update_cache=False,
                    ).req_metadata
                    tasks = builder.take_device_metadata_tasks()
                    stages = [DeviceMetadataStage.COMPRESSOR] if ratio > 1 else []
                    if ratio == 4:
                        stages.append(DeviceMetadataStage.INDEXER)
                    stages.append(DeviceMetadataStage.ATTENTION)
                    self.assertEqual([task.stage for task in tasks], stages)
                    self.assertIs(req.sas_metadata, builder.sas_metadata_buffer)
                    self.assertEqual(req.qli_metadata is not None, ratio == 4)
                    for task in tasks:
                        with patch.object(dsa_v1, "build_compressor_metadata_out"):
                            task.run()
                    self.assertTrue(torch.all(req.sas_metadata == 1))
                    results.append(req)
                self.assertIsNot(results[0].sas_metadata, results[1].sas_metadata)
                if ratio == 4:
                    self.assertIsNot(results[0].qli_metadata, results[1].qli_metadata)

    def test_vision_role_builders_keep_complete_metadata(self):
        for role in ("State", "Attention", "Indexer"):
            with self.subTest(role=role):
                builder = make_builder(getattr(dsa_v1, f"AscendDSA{role}MetadataBuilder"), 4)
                builder.model_config.hf_config.vision_n_layers = 1
                builder.model_config.hf_config.vision_max_n_token = 8
                common = self.common.replace(causal=True, mm_req_doc_ranges={0: [(0, 1)]})
                indices = torch.ones((2, 1, 8), dtype=torch.int32)
                with (
                    patch.object(dsa_v1, "split_decodes_and_prefills", return_value=(0, 1, 0, 2)),
                    patch.object(dsa_v1, "build_vision_bidirectional_swa_indices", return_value=(indices, None)),
                ):
                    req = builder.build(0, common, common_ratio_to_sas_metadata={}, build_query=False).req_metadata
                self.assertIsNotNone(req.sas_metadata)
                self.assertIsNotNone(req.qli_metadata)
                self.assertIsNotNone(req.start_pos)
                self.assertIs(req.vision_swa_indices, indices)

    def test_pcp_role_outputs_and_sharing_across_real_builders(self):
        for role, ratio in (("State", 4), ("SWA", 0), ("Attention", 4), ("Attention", 128), ("Indexer", 4)):
            for prefill, sharded in ((True, False), (False, True), (False, False)):
                with self.subTest(role=role, ratio=ratio, prefill=prefill, sharded=sharded):
                    shared: dict[Any, Any] = {}
                    results = []
                    global_common = self.common.replace(causal=True, seq_lens=torch.tensor([10], dtype=torch.int32))
                    context = SimpleNamespace(
                        global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([prefill]), num_reqs=1),
                        global_block_tables=[torch.tensor([[10, 11]]), torch.tensor([[20, 21]])],
                        global_slot_mappings=[torch.tensor([40, 41]), torch.tensor([80, 81])],
                        hidden_restore_idx=torch.tensor([0, 1]),
                    )
                    for group in range(2):
                        builder = make_builder(getattr(dsa_cp, f"AscendDSAPCP{role}MetadataBuilder"), ratio)
                        builder._global_metadata_builder = make_builder(builder._global_metadata_builder_cls, ratio)
                        builder._is_decode_sharded = sharded
                        builder._prepare_graph_pcp_context = Mock(return_value=context)
                        builder._build_global_common_attn_metadata = Mock(return_value=global_common)
                        builder._build_graph_common_attn_metadata = Mock(side_effect=lambda common, actual: common)
                        builder._build_local_common_attn_metadata = Mock(return_value=self.common.replace(causal=True))
                        result = builder.build(
                            0,
                            self.common.replace(causal=True).replace(pcp_context=context, pcp_cache_group_idx=group),
                            num_actual_reqs=1,
                            common_ratio_to_sas_metadata=shared,
                        )
                        self.assertIs(builder.common_ratio_to_sas_metadata, shared)
                        local = result.req_metadata
                        self.assertEqual(local.sas_metadata is not None, role in ("SWA", "Attention"))
                        self.assertEqual(local.qli_metadata is not None, role == "Indexer")
                        if prefill or sharded:
                            global_req = result.global_dsa_metadata.req_metadata
                            self.assertIsNone(global_req.sas_metadata)
                            self.assertIsNone(global_req.qli_metadata)
                            self.assertIsNone(local.start_pos)
                            self.assertIsNone(local.slot_mapping)
                            torch.testing.assert_close(global_req.block_table, context.global_block_tables[group])
                            if role in ("Attention", "Indexer"):
                                torch.testing.assert_close(global_req.start_pos, torch.tensor([8], dtype=torch.int32))
                            if role == "SWA":
                                torch.testing.assert_close(
                                    global_req.slot_mapping, context.global_slot_mappings[group].int()
                                )
                        else:
                            self.assertNotIsInstance(result, dsa_cp.AscendDSAPCPMetadata)
                            self.assertEqual(local.start_pos is not None, role in ("Attention", "Indexer"))
                        results.append(result)
                    self.assertIsNot(shared["pcp_global"], shared["pcp_local"])
                    self.assertIs(results[0].req_metadata.sas_metadata, results[1].req_metadata.sas_metadata)
                    self.assertIs(results[0].req_metadata.qli_metadata, results[1].req_metadata.qli_metadata)
                    if (prefill or sharded) and role in ("Attention", "Indexer"):
                        self.assertIs(
                            results[0].global_dsa_metadata.req_metadata.start_pos,
                            results[1].global_dsa_metadata.req_metadata.start_pos,
                        )

    def test_pcp_shares_preparation_but_preserves_group_addresses(self):
        shared: dict[Any, Any] = {}
        context = SimpleNamespace(
            global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([True]), num_reqs=1),
            global_block_tables=[torch.tensor([[10]]), torch.tensor([[20]])],
            global_slot_mappings=[torch.tensor([1, 2]), torch.tensor([3, 4])],
            hidden_restore_idx=torch.tensor([0, 1]),
        )
        builders = []
        for cls in (dsa_cp.AscendDSAPCPStateMetadataBuilder, dsa_cp.AscendDSAPCPSWAMetadataBuilder):
            builder = make_builder(cls)
            builder._is_decode_sharded = False
            builder._prepare_graph_pcp_context = Mock(return_value=context)
            builder._build_global_common_attn_metadata = Mock(return_value=self.common)
            builder._build_graph_common_attn_metadata = Mock(side_effect=lambda common, actual: common)
            builder._build_local_common_attn_metadata = Mock(return_value=self.common)
            builder._build_local_dsa_metadata = Mock(return_value=SimpleNamespace())
            builder._global_metadata_builder = SimpleNamespace(build=Mock(return_value=SimpleNamespace()))
            builders.append(builder)
        with patch.object(dsa_cp.AscendDSAPCPMetadata, "from_local_metadata"):
            for index, builder in enumerate(builders):
                builder.build(
                    0,
                    self.common.replace(pcp_context=context, pcp_cache_group_idx=index),
                    num_actual_reqs=1,
                    common_ratio_to_sas_metadata=shared,
                )
        builders[0]._prepare_graph_pcp_context.assert_called_once()
        builders[1]._prepare_graph_pcp_context.assert_not_called()
        builders[1]._build_graph_common_attn_metadata.assert_not_called()
        for index, builder in enumerate(builders):
            call = builder._global_metadata_builder.build.call_args
            self.assertIs(call.args[1].block_table_tensor, context.global_block_tables[index])
            self.assertIs(call.args[1].slot_mapping, context.global_slot_mappings[index])
            self.assertIs(call.kwargs["common_ratio_to_sas_metadata"], shared["pcp_global"])
            self.assertFalse(call.kwargs["build_query"])
            self.assertFalse(builder._build_local_dsa_metadata.call_args.kwargs["update_cache"])

    def test_sharded_decode_stages_global_rope_once_without_local_aliasing(self):
        proxy = Mock()
        global_cos, global_sin = object(), object()
        proxy.copy_to_buffers.return_value = (global_cos, global_sin)
        local_cos, local_sin = object(), object()
        local_proxy = Mock()
        local_proxy.copy_to_buffers.return_value = (local_cos, local_sin)
        self.rope_op.side_effect = [(proxy, object()), (local_proxy, object())]
        context = SimpleNamespace(
            global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([False]), num_reqs=1),
            global_block_tables=[torch.tensor([[10, 11]]), torch.tensor([[20, 21]])],
            global_slot_mappings=[torch.tensor([40, 41]), torch.tensor([80, 81])],
            hidden_restore_idx=torch.tensor([0, 1]),
        )
        shared: dict[Any, Any] = {}
        results = []
        for group in range(2):
            builder = make_builder(dsa_cp.AscendDSAPCPSWAMetadataBuilder)
            builder._global_metadata_builder = make_builder(dsa_v1.AscendDSASWAMetadataBuilder)
            builder._is_decode_sharded = True
            builder._global_rope_buffers = {}
            builder._hidden_restore_idx_buffer = torch.arange(8)
            builder.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs = lambda: True
            builder._prepare_graph_pcp_context = Mock(return_value=context)
            builder._build_global_common_attn_metadata = Mock(return_value=self.common)
            builder._build_graph_common_attn_metadata = Mock(side_effect=lambda common, actual: common)
            builder._build_local_common_attn_metadata = Mock(return_value=self.common)
            results.append(
                builder.build(
                    0,
                    self.common.replace(pcp_context=context, pcp_cache_group_idx=group),
                    num_actual_reqs=1,
                    common_ratio_to_sas_metadata=shared,
                    full_graph_mode=True,
                )
            )
        proxy.copy_to_buffers.assert_called_once()
        local_proxy.copy_to_buffers.assert_called_once()
        self.assertEqual(self.rope_op.call_count, 2)
        for result in results:
            self.assertIs(result.global_dsa_metadata.req_metadata.cos, global_cos)
            self.assertIs(result.req_metadata.cos, local_cos)

    def test_pcp_empty_local_rank_preserves_global_cache_and_root_namespace(self):
        builder = make_builder(dsa_cp.AscendDSAPCPAttentionMetadataBuilder, 4)
        builder._global_metadata_builder = make_builder(dsa_v1.AscendDSAAttentionMetadataBuilder, 4)
        builder._is_decode_sharded = True
        context = SimpleNamespace(
            global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([False]), num_reqs=1),
            global_block_tables=[self.common.block_table_tensor],
            global_slot_mappings=[self.common.slot_mapping],
            hidden_restore_idx=torch.tensor([0, 1]),
        )
        builder._prepare_graph_pcp_context = Mock(return_value=context)
        builder._build_global_common_attn_metadata = Mock(return_value=self.common)
        builder._build_graph_common_attn_metadata = Mock(side_effect=lambda common, actual: common)
        builder._build_local_common_attn_metadata = Mock(
            return_value=self.common.replace(
                num_actual_tokens=0,
                num_input_tokens=0,
                query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
                query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
            )
        )
        shared: dict[Any, Any] = {}
        result = builder.build(
            0,
            self.common.replace(pcp_context=context, pcp_cache_group_idx=0),
            num_actual_reqs=0,
            common_ratio_to_sas_metadata=shared,
        )
        self.assertIsNone(result.req_metadata)
        self.assertIsNotNone(result.global_dsa_metadata.req_metadata.start_pos)
        self.assertIs(builder.common_ratio_to_sas_metadata, shared)
        self.assertNotIn("req_preamble", shared["pcp_global"])
        self.assertEqual(shared["pcp_local"], {})
        for offsets in shared["pcp_local_offsets"]:
            torch.testing.assert_close(offsets, torch.tensor([0, 0], dtype=torch.int32))

    def test_full_pcp_builders_share_base_cache_without_sharing_graph_preparation(self):
        shared: dict[Any, Any] = {}
        context = SimpleNamespace(
            global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([True]), num_reqs=1),
            global_block_tables=[torch.tensor([[10]]), torch.tensor([[20]])],
            global_slot_mappings=[torch.tensor([1, 2]), torch.tensor([3, 4])],
            hidden_restore_idx=torch.tensor([0, 1]),
        )
        builders = []
        with patch.object(dsa_cp.AscendDSAPCPMetadata, "from_local_metadata"):
            for index in range(2):
                builder = make_builder(dsa_cp.AscendDSAPCPMetadataBuilder)
                builder._is_decode_sharded = False
                builder._prepare_graph_pcp_context = Mock(return_value=context)
                builder._build_global_common_attn_metadata = Mock(return_value=self.common)
                builder._build_graph_common_attn_metadata = Mock(side_effect=lambda common, actual: common)
                builder._build_local_common_attn_metadata = Mock(return_value=self.common)
                builder._global_metadata_builder = SimpleNamespace(build=Mock(return_value=SimpleNamespace()))

                def build_local(*args, _builder=builder, **kwargs):
                    _builder.common_ratio_to_sas_metadata = kwargs["common_ratio_to_sas_metadata"]
                    return SimpleNamespace()

                builder._build_local_dsa_metadata = Mock(side_effect=build_local)
                builder.build(
                    0,
                    self.common.replace(pcp_context=context, pcp_cache_group_idx=index),
                    num_actual_reqs=1,
                    common_ratio_to_sas_metadata=shared,
                )
                builders.append(builder)
        self.assertIsNot(shared["pcp_global"], shared["pcp_local"])
        for index, builder in enumerate(builders):
            builder._prepare_graph_pcp_context.assert_called_once()
            global_call = builder._global_metadata_builder.build.call_args
            local_call = builder._build_local_dsa_metadata.call_args
            self.assertIs(global_call.kwargs["common_ratio_to_sas_metadata"], shared["pcp_global"])
            self.assertIs(local_call.kwargs["common_ratio_to_sas_metadata"], shared["pcp_local"])
            self.assertIs(builder.common_ratio_to_sas_metadata, shared)
            self.assertIs(global_call.args[1].block_table_tensor, context.global_block_tables[index])
            self.assertTrue(global_call.kwargs["build_query"])
            self.assertTrue(local_call.kwargs["update_cache"])
