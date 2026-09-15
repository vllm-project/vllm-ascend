# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for MRv1 replicated GQA tables, without loading NPU modules."""

import ast
import copy
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def load_proposer(**overrides):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/utils.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DCPReplicatedDraftMixin")
    namespace = {"torch": torch, "copy": copy, "replace": replace, "VllmConfig": object, **overrides}
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"uses_dcp_replicated_gqa_draft", "draft_additional_config"}
    ]
    exec(compile(ast.Module(body=[*helpers, cls], type_ignores=[]), str(source), "exec"), namespace)
    proposer_source = source.with_name("dspark_proposer.py")
    proposer_tree = ast.parse(proposer_source.read_text())
    proposer = next(
        node for node in proposer_tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDSparkProposer"
    )
    assert isinstance(proposer.bases[0], ast.Name) and proposer.bases[0].id == "DCPReplicatedDraftMixin"
    proposer.bases = proposer.bases[:1]
    proposer.body = [
        node
        for node in proposer.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_per_group_attn_metadata"
    ]
    assert len(proposer.body) == 1
    exec(compile(ast.Module(body=[proposer], type_ignores=[]), str(proposer_source), "exec"), namespace)
    return namespace["AscendDSparkProposer"]


class TestReplicatedBlockTable(unittest.TestCase):
    def test_batch_transitions_preserve_storage_and_target(self):
        for replication in (2, 8):
            for subblocks in (1, 3):
                with self.subTest(replication=replication, subblocks=subblocks):
                    proposer = load_proposer()()
                    proposer.device = torch.device("cpu")
                    proposer.max_batch_size = 4
                    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=24 * replication))
                    proposer._per_group_replication_sizes = {0: replication}
                    proposer._per_group_manager_block_sizes = {0: 4 * subblocks}
                    proposer._per_group_kernel_block_sizes = {0: 4}
                    proposer._replicated_block_table_storage = {}
                    proposer._replicated_block_table_arange = {}
                    proposer._per_group_block_tables = {}
                    proposer._per_group_slot_mappings = {}
                    proposer._per_group_block_table_buffers = {}
                    target = torch.arange(12, 36, dtype=torch.int32).reshape(4, 6)
                    original = target.clone()
                    slots = torch.full((16,), 777, dtype=torch.int32)
                    proposer.set_per_group_attn_metadata(0, target, slots)
                    buffer = proposer._per_group_block_table_buffers[0]
                    pointer = buffer.data_ptr()
                    self.assertEqual(buffer.shape, (5, 6 * replication))
                    self.assertEqual(torch.count_nonzero(buffer).item(), 0)

                    for batch in (1, 4, 2, 0, 3):
                        lengths = torch.ones(batch, dtype=torch.int32)
                        if batch > 1:
                            lengths[-1] = 0
                        refreshed = proposer._get_draft_block_table(0, batch, lengths)
                        actual = refreshed[:batch]
                        expected = torch.zeros_like(actual)
                        for row in range(batch):
                            if lengths[row] == 0:
                                continue
                            expanded = []
                            for start in range(0, 6, subblocks):
                                for lane in range(replication):
                                    for block in target[row, start : start + subblocks].tolist():
                                        physical, offset = divmod(block, subblocks)
                                        expanded.append((physical * replication + lane) * subblocks + offset)
                            expected[row] = torch.tensor(expanded, dtype=torch.int32)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(proposer._replicated_block_table_storage[0].data_ptr(), pointer)
                        torch.testing.assert_close(buffer[:batch], expected)
                        self.assertEqual(torch.count_nonzero(buffer[batch:]).item(), 0)

                    # A runner metadata refresh must reuse the allocation too.
                    proposer.set_per_group_attn_metadata(0, target, slots)
                    self.assertEqual(proposer._per_group_block_table_buffers[0].data_ptr(), pointer)
                    torch.testing.assert_close(target, original)
                    self.assertTrue(torch.all(slots == 777))


class TestReplicatedDraftHooks(unittest.TestCase):
    def test_loading_restores_target_context_on_success_and_failure(self):
        @dataclass
        class Config:
            model_config: object
            parallel_config: object
            cache_config: object
            kv_transfer_config: object
            speculative_config: object
            additional_config: object = None

        for replicated in (False, True):
            for fail in (False, True):
                with self.subTest(replicated=replicated, fail=fail):
                    draft_model = SimpleNamespace(
                        hf_config=SimpleNamespace(
                            model_type="qwen3" if replicated else "mla",
                            architectures=["DSparkDraftModel"],
                        )
                    )
                    spec = SimpleNamespace(
                        draft_model_config=draft_model,
                        draft_parallel_config=SimpleNamespace(rank=0, decode_context_parallel_size=2),
                    )
                    target = Config(
                        SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
                        SimpleNamespace(rank=3, decode_context_parallel_size=2),
                        SimpleNamespace(block_size=384),
                        object(),
                        spec,
                    )
                    current = [target]

                    @contextmanager
                    def set_config(config, current=current):
                        previous = current[0]
                        current[0] = config
                        try:
                            yield
                        finally:
                            current[0] = previous

                    @lru_cache(maxsize=1)
                    def enable_dcp(current=current):
                        return current[0].parallel_config.decode_context_parallel_size > 1

                    class Base:
                        def _create_draft_vllm_config(self):
                            return self.vllm_config

                        def _get_model(self, fail=fail):
                            config = self._create_draft_vllm_config()
                            self.loaded_config = config
                            with set_config(config):
                                self.loaded_dcp = enable_dcp()
                                if fail:
                                    raise RuntimeError("load failed")
                            return "model"

                    mixin = load_proposer(enable_dcp=enable_dcp, set_current_vllm_config=set_config)

                    class Proposer(mixin, Base):
                        pass

                    proposer = Proposer()
                    proposer.vllm_config = target
                    proposer.speculative_config = spec
                    proposer.runner = object()
                    proposer.dcp_size, proposer.dcp_rank = 2, 1
                    proposer._init_dcp_replicated_draft()
                    self.assertTrue(enable_dcp())
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "load failed"):
                            proposer._get_model()
                    else:
                        self.assertEqual(proposer._get_model(), "model")
                    self.assertIs(current[0], target)
                    self.assertTrue(enable_dcp())
                    self.assertEqual(proposer.loaded_dcp, not replicated)
                    if replicated:
                        self.assertEqual((proposer.dcp_size, proposer.dcp_rank), (1, 0))
                        self.assertIs(proposer.loaded_config.model_config, draft_model)
                        self.assertEqual(proposer.loaded_config.parallel_config.rank, 3)
                        self.assertIsNone(proposer.loaded_config.kv_transfer_config)
                        proposer.loaded_config.cache_config.block_size = 128
                        self.assertEqual(target.cache_config.block_size, 384)
                        self.assertEqual(spec.draft_parallel_config.decode_context_parallel_size, 2)
                    else:
                        self.assertEqual((proposer.dcp_size, proposer.dcp_rank), (2, 1))
                        self.assertIs(proposer.loaded_config, target)

    def test_draft_disables_target_pd_recompute_before_validation(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/platform.py"
        tree = ast.parse(source.read_text())
        check = next(node for node in tree.body if getattr(node, "name", None) == "_check_ascend_config")
        recompute_check = next(
            node
            for node in check.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "recompute_scheduler_enable"
        )
        validation = compile(ast.Module(body=[recompute_check], type_ignores=[]), str(source), "exec")

        @dataclass
        class Config:
            model_config: object
            parallel_config: object
            cache_config: object
            kv_transfer_config: object
            additional_config: object
            scheduler_config: object = None
            compilation_config: object = None

            def __post_init__(self):
                if self.kv_transfer_config is None:
                    options = self.additional_config or {}
                    scheduler_options = options.get("scheduler_config") or {}
                    enabled = scheduler_options.get(
                        "recompute_scheduler_enable", options.get("recompute_scheduler_enable", False)
                    )
                    exec(
                        validation,
                        {
                            "scheduler_extension_config": SimpleNamespace(recompute_scheduler_enable=enabled),
                            "vllm_config": self,
                        },
                    )

        class Base:
            def _create_draft_vllm_config(self):
                return self.vllm_config

        mixin = load_proposer()

        class Proposer(mixin, Base):
            def _uses_dcp_replicated_draft_kv(self):
                return True

        for options in (
            None,
            {},
            {"recompute_scheduler_enable": True},
            {"multistream_overlap_shared_expert": True, "recompute_scheduler_enable": True},
            {"scheduler_config": {"recompute_scheduler_enable": True}},
            {"recompute_scheduler_enable": True, "scheduler_config": {"recompute_scheduler_enable": False}},
            {"recompute_scheduler_enable": False, "scheduler_config": {"recompute_scheduler_enable": True}},
        ):
            with self.subTest(options=options):
                additional = copy.deepcopy(options)
                if additional is not None:
                    additional["other_option"] = {"values": [1, 2]}
                original = copy.deepcopy(additional)
                connector = SimpleNamespace(
                    kv_connector="MooncakeConnectorV2",
                    kv_role="kv_consumer",
                    kv_port="30300",
                    engine_id="3",
                    kv_connector_extra_config={
                        "prefill": {"dp_size": 4, "tp_size": 8, "dcp_size": 1},
                        "decode": {"dp_size": 4, "tp_size": 8, "dcp_size": 8},
                        "ascend_local_comm_res_path": "/etc/hixlep",
                    },
                )
                target = Config(
                    SimpleNamespace(
                        model="/mnt/share/kimik3_0726/Kimi-K3-w4a8-mxfp-flex-quarot-0729", max_model_len=200000
                    ),
                    SimpleNamespace(
                        rank=3,
                        data_parallel_size=4,
                        tensor_parallel_size=8,
                        decode_context_parallel_size=8,
                        cp_kv_cache_interleave_size=768,
                    ),
                    SimpleNamespace(block_size=768, enable_prefix_caching=True),
                    connector,
                    additional,
                    SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=768),
                    SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
                )
                proposer = Proposer()
                proposer.vllm_config = target
                proposer.speculative_config = SimpleNamespace(
                    method="dspark",
                    num_speculative_tokens=3,
                    draft_tensor_parallel_size=8,
                    enforce_eager=False,
                    draft_sample_method="greedy",
                    draft_parallel_config=SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=8),
                    draft_model_config=SimpleNamespace(model="/mnt/share/weights/Kimi-K3-DSpark", max_model_len=4096),
                )
                draft = proposer._create_draft_vllm_config()
                self.assertIsNone(draft.kv_transfer_config)
                self.assertEqual(draft.parallel_config.decode_context_parallel_size, 1)
                self.assertEqual(draft.parallel_config.tensor_parallel_size, 8)
                self.assertEqual(target.parallel_config.decode_context_parallel_size, 8)
                self.assertEqual(draft.model_config.max_model_len, 4096)
                self.assertEqual(target.model_config.max_model_len, 200000)
                self.assertEqual(draft.cache_config.block_size, 768)
                self.assertTrue(draft.cache_config.enable_prefix_caching)
                self.assertEqual(draft.compilation_config.cudagraph_mode, "FULL_DECODE_ONLY")
                self.assertEqual(draft.scheduler_config.max_num_batched_tokens, 768)
                self.assertEqual(draft.scheduler_config.max_num_seqs, 32)
                self.assertEqual(
                    draft.additional_config.get("multistream_overlap_shared_expert"),
                    (additional or {}).get("multistream_overlap_shared_expert"),
                )
                self.assertFalse(draft.additional_config["scheduler_config"]["recompute_scheduler_enable"])
                self.assertFalse(draft.additional_config.get("recompute_scheduler_enable", False))
                self.assertEqual(target.additional_config, original)
                self.assertIs(target.kv_transfer_config, connector)
                if additional is not None:
                    draft.additional_config["other_option"]["values"].append(3)
                    self.assertEqual(target.additional_config, original)

    def test_metadata_builder_proxy_preserves_group_spec(self):
        builder_cls = MagicMock(side_effect=lambda *args: SimpleNamespace(args=args))
        proposer_cls = load_proposer(AscendAttentionMetadataBuilder=builder_cls)
        config = object()
        device = torch.device("cpu")
        for kernel_block_size in (None, 128):
            for num_builders in (1, 3):
                with self.subTest(kernel_block_size=kernel_block_size, num_builders=num_builders):
                    original_spec = MagicMock(block_size=384)
                    copied_spec = SimpleNamespace(block_size=128)
                    original_spec.copy_with_new_block_size.return_value = copied_spec
                    group = SimpleNamespace(kv_cache_spec=original_spec, layer_names=["draft.0"])
                    proxy = proposer_cls.MetadataBuilderProxy(group)
                    proxy.create_metadata_builders(config, device, kernel_block_size, num_builders)
                    self.assertIs(group.kv_cache_spec, original_spec)
                    self.assertEqual(original_spec.block_size, 384)
                    self.assertEqual(len(group.metadata_builders), num_builders)
                    self.assertEqual(len({id(builder) for builder in group.metadata_builders}), num_builders)
                    expected_spec = original_spec if kernel_block_size is None else copied_spec
                    for builder in group.metadata_builders:
                        self.assertEqual(builder.args, (expected_spec, group.layer_names, config, device))
                    if kernel_block_size is None:
                        original_spec.copy_with_new_block_size.assert_not_called()
                    else:
                        original_spec.copy_with_new_block_size.assert_called_once_with(kernel_block_size)

    def test_nonreplicated_table_is_forwarded(self):
        proposer = load_proposer()()
        table = torch.tensor([[3, 7]], dtype=torch.int32)
        proposer._per_group_replication_sizes = {}
        proposer._per_group_block_tables = {0: table}
        self.assertIs(proposer._get_draft_block_table(0, 1, torch.tensor([8])), table)

    def test_context_slots_follow_expanded_pages(self):
        proposer = load_proposer()()
        proposer.device = torch.device("cpu")
        proposer._per_group_kernel_block_sizes = {0: 4}
        proposer._per_group_replication_sizes = {0: 2}
        proposer._per_group_context_slot_mapping_buffers = {0: torch.empty(8, dtype=torch.int32)}
        table = torch.tensor([[6, 7, 14, 15], [10, 11, 18, 19]], dtype=torch.int32)
        slots = proposer._build_replicated_context_slot_mapping(
            0, table, torch.tensor([0, 4, 9, 3, 8]), torch.tensor([0, 3, 5]), 2, 5
        )
        torch.testing.assert_close(slots, torch.tensor([24, 28, 57, 43, 72, -1, -1, -1], dtype=torch.int32))
        proposer._build_replicated_context_slot_mapping(0, table, torch.empty(0), torch.tensor([0]), 0, 0)
        self.assertTrue(torch.all(slots == -1))


if __name__ == "__main__":
    unittest.main()
