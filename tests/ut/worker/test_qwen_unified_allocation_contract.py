# SPDX-License-Identifier: Apache-2.0
"""Dependency-free checks of production dispatch and byte placement.

Execute with python -m unittest discover -s tests/ut/worker -p
 test_qwen_unified_allocation_contract.py. Tensor view coverage lives in
 test_qwen4_exp_cache_config.py.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class TestQwenUnifiedAllocationContract(unittest.TestCase):
    def test_entry_point_uses_nested_and_outer_config(self):
        for model_type in ("qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp"):
            for field in ("hf_config", "hf_text_config"):
                with self.subTest(model_type=model_type, field=field):
                    config = SimpleNamespace(**{field: SimpleNamespace(model_type=model_type)})
                    ns = load_functions(
                        "vllm_ascend/worker/v2/attn_utils.py",
                        {"allocate_kv_cache_main"},
                        {
                            "get_current_vllm_config": lambda config=config: SimpleNamespace(model_config=config),
                            "unified_allocate_kv_cache": lambda *args: args,
                        },
                    )
                    self.assertEqual(
                        ns["allocate_kv_cache_main"]("cache", "device", "layout", [128]),
                        ("cache", "device", "layout", [128]),
                    )

    def test_unlisted_types_use_legacy_path(self):
        class LegacyPathReached(Exception):
            pass

        def legacy_layers(*args):
            raise LegacyPathReached

        for model_type in (
            "qwen3",
            "qwen3_8",
            "qwen3_8_text",
            "qwen4",
            "qwen4_max",
            "qwen40",
            "qwen4_unknown",
            None,
            4,
        ):
            with self.subTest(model_type=model_type):
                config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type))
                ns = load_functions(
                    "vllm_ascend/worker/v2/attn_utils.py",
                    {"allocate_kv_cache_main"},
                    {
                        "get_current_vllm_config": lambda config=config: SimpleNamespace(model_config=config),
                        "get_layers_from_vllm_config": legacy_layers,
                        "AttentionLayerBase": object,
                    },
                )
                with self.assertRaises(LegacyPathReached):
                    ns["allocate_kv_cache_main"]("cache", "device", "layout", [128])

    def test_different_groups_and_block_ids_have_disjoint_bytes(self):
        groups = [
            SimpleNamespace(layer_names=list(members), members=members)
            for members in (
                {"main": SimpleNamespace(page_size_bytes=65536)},
                {"main.indexer.raw_key_cache": SimpleNamespace(page_size_bytes=512)},
                {"gdn": SimpleNamespace(page_size_bytes=33792), "ple": SimpleNamespace(page_size_bytes=1280)},
            )
        ]
        ns = load_functions(
            "vllm_ascend/models/qwen4_exp/cache_config.py",
            {"get_qwen4_exp_pool_bytes_per_block", "get_qwen4_exp_kv_cache_config"},
            {
                "_group_member_specs": lambda group: group.members,
                "may_override_num_blocks": lambda config, count: count,
                "KVCacheTensor": lambda **kwargs: SimpleNamespace(**kwargs),
                "KVCacheConfig": lambda **kwargs: SimpleNamespace(**kwargs),
            },
        )
        page_bytes = 65536 + 512 + 33792 + 1280
        self.assertEqual(ns["get_qwen4_exp_pool_bytes_per_block"](groups, unified=True), page_bytes)
        for count in (2, 3, 7):
            with self.subTest(count=count):
                config = SimpleNamespace(cache_config=SimpleNamespace(prefix_cache_retention_interval=None))
                plan = ns["get_qwen4_exp_kv_cache_config"](config, groups, count * page_bytes + 1, unified=True)
                self.assertEqual(plan.num_blocks, count)
                backing = bytearray(count * page_bytes)
                spans = []
                for owner, descriptor in enumerate(plan.kv_cache_tensors, 1):
                    for block in range(count):
                        start = descriptor.offset + block * descriptor.block_stride
                        end = start + descriptor.block_stride
                        spans.append((start, end, owner))
                        backing[start:end] = bytes([owner]) * (end - start)
                for start, end, owner in spans:
                    self.assertEqual(backing[start:end], bytes([owner]) * (end - start))
                ordered = sorted(spans)
                self.assertTrue(all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])))
                self.assertEqual(ordered[-1][1], len(backing))


if __name__ == "__main__":
    unittest.main()
