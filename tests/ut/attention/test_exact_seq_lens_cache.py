# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-light contracts for same-build exact sequence-length sharing.

Production methods are AST-loaded because the local CPU test environment need
not contain TorchNPU/vLLM. These tests do not validate device synchronization
timing; the optional Torch case checks actual contiguous/strided tensor values.
"""

import ast
import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


class _Tensor:
    def __init__(self, values, stats=None):
        self.values = list(values)
        self.stats = {"conversions": 0, "slices": 0} if stats is None else stats

    def __getitem__(self, index):
        self.stats["slices"] += 1
        return _Tensor(self.values[index], self.stats)

    def tolist(self):
        self.stats["conversions"] += 1
        return self.values.copy()


def _load_contracts():
    root = Path(__file__).resolve().parents[3]
    namespace = {
        "dataclass": dataclass,
        "torch": SimpleNamespace(Tensor=_Tensor, from_numpy=_Tensor),
        "AscendCommonAttentionMetadata": lambda **kwargs: SimpleNamespace(**kwargs),
        "CrossAttentionSpec": type("CrossAttentionSpec", (), {}),
        "AscendDSAMetadataBuilder": type("AscendDSAMetadataBuilder", (), {}),
        "AscendSFAMetadataBuilder": type("AscendSFAMetadataBuilder", (), {}),
    }
    source = root / "vllm_ascend/attention/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cache = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExactSeqLensListCache")
    exec(compile(ast.Module(body=[cache], type_ignores=[]), str(source), "exec"), namespace)

    source = root / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    builder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionMetadataBuilder"
    )
    method = next(
        node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "_get_seq_lens_list"
    )
    for argument in method.args.args:
        argument.annotation = None
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    builder_type = type("AscendAttentionMetadataBuilder", (), {method.name: namespace[method.name]})
    namespace["AscendAttentionMetadataBuilder"] = builder_type

    source = root / "vllm_ascend/worker/v2/attn_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_attn_metadata")
    for argument in build.args.args + build.args.kwonlyargs:
        argument.annotation = None
    build.returns = None
    exec(compile(ast.Module(body=[build], type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _make_builder(namespace, builder_type=None):
    builder = (builder_type or namespace["AscendAttentionMetadataBuilder"])()
    builder.vllm_config = SimpleNamespace(use_v2_model_runner=True)
    builder.speculative_config = SimpleNamespace(parallel_drafting=True)
    builder.pcp_enabled = False
    builder.kv_cache_spec = None
    return builder


class TestExactSeqLensListCache(unittest.TestCase):
    def setUp(self):
        self.namespace = _load_contracts()
        self.cache_type = self.namespace["ExactSeqLensListCache"]

    def test_converts_once_and_returns_independent_group_lists(self):
        tensor = _Tensor([17, 2047, 110000])
        cache = self.cache_type()
        first = cache.get_list(tensor)
        first.append(1)
        first[0] = -1
        self.assertEqual(cache.get_list(tensor), [17, 2047, 110000])
        self.assertEqual(tensor.stats["conversions"], 1)
        self.assertEqual(tensor.values, [17, 2047, 110000])

    def test_different_slice_identity_is_not_reused(self):
        tensor = _Tensor([10, 20, 30, 40])
        contiguous, strided = tensor[:2], tensor[::2]
        cache = self.cache_type()
        self.assertEqual(cache.get_list(contiguous), [10, 20])
        self.assertEqual(cache.get_list(strided), [10, 30])
        self.assertEqual(tensor.stats["conversions"], 2)

    def test_exact_device_values_not_host_upper_bounds(self):
        tensor = _Tensor([1535, 2047])
        metadata = SimpleNamespace(
            exact_seq_lens_list_cache=self.cache_type(),
            seq_lens_cpu_upper_bound=_Tensor([1543, 2055]),
            seq_lens_cpu=_Tensor([9999, 9999]),
        )
        for _ in range(5):
            self.assertEqual(_make_builder(self.namespace)._get_seq_lens_list(metadata, tensor), [1535, 2047])
        self.assertEqual(tensor.stats["conversions"], 1)
        self.assertEqual(metadata.seq_lens_cpu_upper_bound.stats["conversions"], 0)

    def test_excluded_builders_keep_existing_conversion(self):
        base = self.namespace["AscendAttentionMetadataBuilder"]
        subclass = type("OtherBuilder", (base,), {})
        for mode in ("v1", "pcp", "cross", "no_spec", "not_parallel", "subclass", "no_cache"):
            with self.subTest(mode=mode):
                tensor = _Tensor([11, 23])
                builder = _make_builder(self.namespace, subclass if mode == "subclass" else base)
                metadata = SimpleNamespace(exact_seq_lens_list_cache=self.cache_type())
                if mode == "v1":
                    builder.vllm_config.use_v2_model_runner = False
                elif mode == "pcp":
                    builder.pcp_enabled = True
                elif mode == "cross":
                    builder.kv_cache_spec = self.namespace["CrossAttentionSpec"]()
                elif mode == "no_spec":
                    builder.speculative_config = None
                elif mode == "not_parallel":
                    builder.speculative_config.parallel_drafting = False
                elif mode == "no_cache":
                    metadata.exact_seq_lens_list_cache = None
                for _ in range(2):
                    self.assertEqual(builder._get_seq_lens_list(metadata, tensor), [11, 23])
                self.assertEqual(tensor.stats["conversions"], 2)

    def test_v2_build_shares_one_view_and_resets_cache_next_step(self):
        builders = [_make_builder(self.namespace) for _ in range(5)]
        seen_caches = []
        for builder in builders:

            def build(*, common_prefix_len, common_attn_metadata, builder=builder):
                seen_caches.append(common_attn_metadata.exact_seq_lens_list_cache)
                return SimpleNamespace(
                    seq_lens=common_attn_metadata.seq_lens,
                    seq_lens_list=builder._get_seq_lens_list(common_attn_metadata, common_attn_metadata.seq_lens),
                )

            builder.build = build
        groups = [
            [SimpleNamespace(get_metadata_builder=lambda _, builder=builder: builder, layer_names=[str(i)])]
            for i, builder in enumerate(builders)
        ]
        tensor = _Tensor([17, 25, 777])
        kwargs = dict(
            attn_groups=groups,
            num_reqs=2,
            num_tokens=16,
            query_start_loc_gpu=object(),
            query_start_loc_cpu=object(),
            max_query_len=8,
            seq_lens=tensor,
            max_seq_len=9999,
            block_tables=[object()] * 5,
            slot_mappings=[object()] * 5,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object()] * 5),
            seq_lens_np=[9999, 9999],
            positions=object(),
        )
        first = self.namespace["build_attn_metadata"](**kwargs)
        self.assertEqual(tensor.stats, {"conversions": 1, "slices": 1})
        self.assertTrue(all(item.seq_lens is first["0"].seq_lens for item in first.values()))
        self.assertTrue(all(cache is seen_caches[0] for cache in seen_caches))
        self.assertTrue(all(item.seq_lens_list == [17, 25] for item in first.values()))
        tensor.values[:2] = [18, 31]
        second = self.namespace["build_attn_metadata"](**kwargs)
        self.assertEqual(tensor.stats, {"conversions": 2, "slices": 2})
        self.assertIsNot(seen_caches[0], seen_caches[5])
        self.assertTrue(all(item.seq_lens_list == [18, 31] for item in second.values()))
        self.assertTrue(all(item.seq_lens_list == [17, 25] for item in first.values()))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Requires Torch for real tensor checks")
    def test_real_torch_views_and_next_build(self):
        torch = importlib.import_module("torch")
        tensor = torch.tensor([1535, 1536, 2047, 2048, 110000], dtype=torch.int32)
        cache = self.cache_type()
        for view in (tensor, tensor[1:], tensor[::2], tensor[:0]):
            values = cache.get_list(view)
            self.assertEqual(values, view.tolist())
            values.append(-1)
            self.assertEqual(cache.get_list(view), view.tolist())
        tensor.add_(7)
        self.assertEqual(self.cache_type().get_list(tensor), tensor.tolist())


if __name__ == "__main__":
    unittest.main()
