# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU metadata-contract tests, not vLLM/NPU runtime integration tests.

Execute the production functions with lightweight spec fixtures so these
regressions can also run with the standard library on a machine without torch.
Only external imports are substituted; the layout and planner logic is real.
"""

import ast
import math
import runpy
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def _dtype_size(dtype):
    return {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}[dtype]


@dataclass(frozen=True)
class _FullAttentionSpec:
    block_size: int = 128
    num_kv_heads: int = 2
    head_size: int = 256
    head_size_v: int = 256
    dtype: str = "torch.bfloat16"
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return self.page_size_padded or (
            self.block_size * self.num_kv_heads * (self.head_size + self.head_size_v) * _dtype_size(self.dtype)
        )


@dataclass(frozen=True)
class _SlidingWindowSpec(_FullAttentionSpec):
    sliding_window: int = 2048


@dataclass(frozen=True)
class _MambaSpec:
    shapes: tuple = ((5120, 10), (24, 128, 128))
    dtypes: tuple = ("torch.bfloat16", "torch.float32")
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return self.page_size_padded or sum(
            math.prod(shape) * _dtype_size(dtype) for shape, dtype in zip(self.shapes, self.dtypes)
        )


@dataclass(frozen=True)
class _UniformTypeKVCacheSpecs:
    kv_cache_specs: dict


def _load_production():
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_ascend/core/dflash_cache.py"
    layout = runpy.run_path(str(root / "vllm_ascend/core/dflash_cache_layout.py"))
    namespace = dict(
        logger=Mock(),
        get_dtype_size=_dtype_size,
        FullAttentionSpec=_FullAttentionSpec,
        SlidingWindowSpec=_SlidingWindowSpec,
        MambaSpec=_MambaSpec,
        UniformTypeKVCacheSpecs=_UniformTypeKVCacheSpecs,
        get_dflash_aligned_block_size=layout["get_dflash_aligned_block_size"],
        get_dflash_fia_safe_num_blocks=layout["get_dflash_fia_safe_num_blocks"],
    )
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    module.body = [
        node for node in module.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm"))
    ]
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


class TestDFlashCache(unittest.TestCase):
    def setUp(self):
        self.impl = _load_production()
        self.conv = 5120 * 10 * 2
        self.ssm = 24 * 128 * 128 * 4
        self.page = self.conv + 2 * self.ssm
        self.pool_per_block = 2 * self.page
        self.config = SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(
                method="dflash",
                draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
                    layer_types=["sliding_attention"] * 4 + ["full_attention"],
                )),
            ),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        )

    def _specs(self):
        return {
            "target.full": _FullAttentionSpec(page_size_padded=self.page),
            "target.mamba": _MambaSpec(page_size_padded=self.page),
            "draft.full": _FullAttentionSpec(num_kv_heads=4, head_size=128, head_size_v=128),
            **{
                f"draft.swa.{i}": _SlidingWindowSpec(num_kv_heads=4, head_size=128, head_size_v=128)
                for i in range(4)
            },
        }

    def _plan(self, count, *, legacy=False):
        specs = self.impl.align_dflash_cache_specs(self.config, self._specs())
        full_names = [name for name, spec in specs.items() if type(spec) is _FullAttentionSpec]
        groups = [
            SimpleNamespace(
                layer_names=full_names,
                kv_cache_spec=_UniformTypeKVCacheSpecs({name: specs[name] for name in full_names}),
            ),
            SimpleNamespace(layer_names=["target.mamba"], kv_cache_spec=specs["target.mamba"]),
            SimpleNamespace(
                layer_names=[name for name in specs if ".swa." in name],
                kv_cache_spec=_UniformTypeKVCacheSpecs({
                    name: spec for name, spec in specs.items() if ".swa." in name
                }),
            ),
        ]
        if legacy:
            tensors = [SimpleNamespace(size=count * self.page, shared_by=[name]) for name in full_names]
        else:
            # Both descriptors reference the same backing; do not sum them.
            tensors = [SimpleNamespace(size=count * self.pool_per_block, layers=[name]) for name in full_names]
        return SimpleNamespace(num_blocks=count, kv_cache_groups=groups, kv_cache_tensors=tensors)

    def _planner(self, counts=(9000,), *, legacy=False):
        calls = []

        def original(config, specs, budgets):
            calls.append(config)
            override = config.cache_config.num_gpu_blocks_override
            return [self._plan(override if override is not None else count, legacy=legacy) for count in counts]

        return self.impl.wrap_dflash_cache_planner(original), calls

    def test_mixed_alignment_preserves_all_four_windows_and_spec_types(self):
        original = self._specs()
        aligned = self.impl.align_dflash_cache_specs(self.config, original)
        for name, spec in aligned.items():
            with self.subTest(name=name):
                self.assertIs(type(spec), type(original[name]))
                self.assertEqual(spec.page_size_bytes, self.page)
                if not isinstance(spec, _MambaSpec):
                    self.assertEqual(spec.block_size, 1536)
                    self.assertEqual(original[name].block_size, 128)
                if isinstance(spec, _SlidingWindowSpec):
                    self.assertEqual(spec.sliding_window, 2048)
        self.assertEqual(self.impl.align_dflash_cache_specs(self.config, aligned), aligned)

    def test_scope_leaves_unmixed_and_target_only_specs_unchanged(self):
        original = self._specs()
        draft_config = self.config.speculative_config.draft_model_config.hf_config
        for layer_types in (["full_attention"] * 5, ["sliding_attention"] * 5):
            with self.subTest(layer_types=layer_types):
                draft_config.layer_types = layer_types
                self.assertIs(self.impl.align_dflash_cache_specs(self.config, original), original)
        self.config.speculative_config = None
        self.assertIs(self.impl.align_dflash_cache_specs(self.config, original), original)

    def test_scope_leaves_v1_and_non_dflash_unchanged(self):
        original = self._specs()
        self.config.use_v2_model_runner = False
        self.assertIs(self.impl.align_dflash_cache_specs(self.config, original), original)
        self.config.use_v2_model_runner = True
        self.config.speculative_config.method = "eagle"
        self.assertIs(self.impl.align_dflash_cache_specs(self.config, original), original)

    def test_incompatible_page_layout_is_rejected(self):
        specs = self._specs()
        specs["target.full"] = replace(specs["target.full"], page_size_padded=self.page + 128)
        with self.assertRaisesRegex(ValueError, "common page bytes"):
            self.impl.align_dflash_cache_specs(self.config, specs)

    def test_pipeline_parallel_is_rejected(self):
        self.config.parallel_config.pipeline_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "pipeline_parallel_size=1"):
            self.impl.align_dflash_cache_specs(self.config, self._specs())

    def test_smaller_explicit_override_is_preserved(self):
        self.config.cache_config.num_gpu_blocks_override = 4096
        planner, calls = self._planner()
        result = planner(self.config, [{}], [9000 * self.pool_per_block])
        self.assertEqual(result[0].num_blocks, 4096)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], self.config)

    def test_fia_boundary_replans_9000_to_5461_without_mutating_config(self):
        self.config.cache_config.num_gpu_blocks_override = 9000
        planner, calls = self._planner()
        result = planner(self.config, [{}], [9000 * self.pool_per_block])
        self.assertEqual(result[0].num_blocks, 5461)
        self.assertEqual(len(calls), 2)
        self.assertIsNot(calls[1], self.config)
        self.assertIsNot(calls[1].cache_config, self.config.cache_config)
        self.assertEqual(self.config.cache_config.num_gpu_blocks_override, 9000)

    def test_override_is_also_limited_by_real_profiled_memory(self):
        self.config.cache_config.num_gpu_blocks_override = 9000
        planner, calls = self._planner()
        result = planner(self.config, [{}], [3000 * self.pool_per_block + 1])
        self.assertEqual(result[0].num_blocks, 3000)
        self.assertEqual(calls[1].cache_config.num_gpu_blocks_override, 3000)

    def test_minimum_rank_budget_applies_to_all_workers(self):
        planner, _ = self._planner(counts=(9000, 8000))
        result = planner(self.config, [{}, {}], [7000 * self.pool_per_block, 3500 * self.pool_per_block])
        self.assertEqual([plan.num_blocks for plan in result], [3500, 3500])

    def test_main_shared_descriptors_are_counted_once(self):
        plan = self._plan(9000)
        self.assertEqual(self.impl._pool_bytes_per_block(plan), self.pool_per_block)
        planner, _ = self._planner()
        result = planner(self.config, [{}], [6000 * self.pool_per_block])
        self.assertEqual(result[0].num_blocks, 5461)

    def test_legacy_separate_descriptors_are_summed(self):
        plan = self._plan(9000, legacy=True)
        self.assertEqual(self.impl._pool_bytes_per_block(plan), self.pool_per_block)
        planner, _ = self._planner(legacy=True)
        result = planner(self.config, [{}], [3500 * self.pool_per_block])
        self.assertEqual(result[0].num_blocks, 3500)

    def test_replanner_ignoring_safe_override_is_rejected(self):
        planner = self.impl.wrap_dflash_cache_planner(lambda *_: [self._plan(9000)])
        with self.assertRaisesRegex(ValueError, "Replanned.*limit"):
            planner(self.config, [{}], [9000 * self.pool_per_block])

    def test_replanner_exceeding_memory_budget_is_rejected(self):
        def original(config, *_):
            count = config.cache_config.num_gpu_blocks_override or 9000
            plan = self._plan(count)
            if config.cache_config.num_gpu_blocks_override is not None:
                for tensor in plan.kv_cache_tensors:
                    tensor.size *= 4
            return [plan]

        planner = self.impl.wrap_dflash_cache_planner(original)
        with self.assertRaisesRegex(ValueError, "Replanned.*limit"):
            planner(self.config, [{}], [9000 * self.pool_per_block])

    def test_replanner_losing_aligned_layout_is_rejected(self):
        def original(config, *_):
            count = config.cache_config.num_gpu_blocks_override or 9000
            plan = self._plan(count)
            if config.cache_config.num_gpu_blocks_override is not None:
                specs = plan.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
                specs["target.full"] = replace(specs["target.full"], block_size=128)
            return [plan]

        planner = self.impl.wrap_dflash_cache_planner(original)
        with self.assertRaisesRegex(ValueError, "lost its aligned layout"):
            planner(self.config, [{}], [9000 * self.pool_per_block])

    def test_initial_plan_missing_worker_budget_is_rejected(self):
        planner, _ = self._planner(counts=(9000, 9000))
        with self.assertRaisesRegex(ValueError, "every worker"):
            planner(self.config, [{}, {}], [9000 * self.pool_per_block])

    def test_replanner_missing_worker_plan_is_rejected(self):
        def original(config, *_):
            count = config.cache_config.num_gpu_blocks_override
            return [] if count is not None else [self._plan(9000)]

        planner = self.impl.wrap_dflash_cache_planner(original)
        with self.assertRaisesRegex(ValueError, "missing worker plans"):
            planner(self.config, [{}], [9000 * self.pool_per_block])

    def test_insufficient_memory_for_null_and_live_block_is_rejected(self):
        planner, _ = self._planner()
        with self.assertRaisesRegex(ValueError, "No safe"):
            planner(self.config, [{}], [self.pool_per_block])

    def test_planner_is_passthrough_outside_scope(self):
        expected = [SimpleNamespace(num_blocks=9000)]
        self.config.speculative_config = None
        original = Mock(return_value=expected)
        planner = self.impl.wrap_dflash_cache_planner(original)
        self.assertIs(planner(self.config, [], [1]), expected)
        original.assert_called_once_with(self.config, [], [1])


if __name__ == "__main__":
    unittest.main()
