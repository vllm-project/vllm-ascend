# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for manager tails and their scheduler boundary producer.

Hardware cache/CoW correctness is a separate gate.
"""

import ast
import functools
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

# Execute the owner modules' actual function bodies without importing the NPU plugin.
platform = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform"
source = (platform / "patch_mamba_manager.py").read_text()
scheduler_source = (platform / "patch_mamba_block_aligned_split.py").read_text()
scope = dict(
    functools=functools,
    Scheduler=NS,
    Request=NS,
    _original_mamba_block_aligned_split=lambda s, r, n, *a: n,
    _cache_tail=lambda *a: None,
    _attention_tail=lambda *a: None,
    _skips_eagle_block_drop=lambda c: c is None or not c.is_kv_consumer,
    _get_sparse_index_kpool=lambda c: None,
    get_mamba_prefill_checkpoint_position=lambda n, u, d: max((n - 1) // u * u - (u if d else 0), 0),
    is_mamba_prefill_checkpoint_valid=lambda **kw: False,
)
for module_source, names in (
    (scheduler_source, {"_split_reusable_tail", "_mamba_block_aligned_split"}),
    (source, {"cache_tail", "attention_tail"}),
):
    tree = ast.parse(module_source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, "<patch-owner>", "exec"), scope)
scope["split"] = scope["_mamba_block_aligned_split"]


class Tests(unittest.TestCase):
    def scheduler(self, consumer=False):
        return NS(
            mamba_partial_cache_hit=True,
            hash_block_size=16,
            vllm_config=NS(kv_transfer_config=NS(is_kv_consumer=True) if consumer else None, model_config=NS()),
            mamba_has_prefill_checkpoint_blocks=False,
            mamba_prefill_checkpoint_alignment=None,
            cache_config=NS(block_size=896),
        )

    def req(self, n, start=0):
        return NS(num_prompt_tokens=n, num_tokens=n, num_computed_tokens=start, request_id="r")

    def test_exact_multiples(self):
        for n in [32, 128, 512, 592, 896, 1024, 1792]:
            self.assertEqual(scope["split"](self.scheduler(), self.req(n), n), n - 16)

    def test_no_zero(self):
        self.assertEqual(scope["split"](self.scheduler(), self.req(16), 16), 16)
        self.assertEqual(scope["split"](self.scheduler(), self.req(16, 1), 15), 15)

    def test_other_units(self):
        for unit in [32, 64, 128]:
            scheduler = self.scheduler()
            scheduler.hash_block_size = unit
            self.assertEqual(scope["split"](scheduler, self.req(512), 512), 512 - unit)

    def test_pmu_disabled(self):
        scheduler = self.scheduler()
        scheduler.mamba_partial_cache_hit = False
        self.assertEqual(scope["split"](scheduler, self.req(512), 512), 512)

    def test_nonmultiples(self):
        self.assertEqual(scope["split"](self.scheduler(), self.req(593), 593), 593)

    def test_resume_after_boundary(self):
        self.assertEqual(scope["split"](self.scheduler(), self.req(128, 112), 16), 16)

    def test_consumer_verifier(self):
        self.assertEqual(scope["split"](self.scheduler(True), self.req(128, 127), 4), 4)

    def test_decode(self):
        r = self.req(128, 128)
        r.num_tokens = 129
        self.assertEqual(scope["split"](self.scheduler(), r, 4), 4)

    def manager(self):
        block = NS(is_null=False)
        return NS(
            _checkpoints={},
            mamba_cache_mode="align",
            use_eagle=False,
            block_size=896,
            block_pool=NS(hash_block_size=16, cache_partial_block=lambda **kw: "hash"),
            req_to_blocks={"r": [block]},
            kv_cache_group_id=1,
            _partial_hit_reqs={},
            num_cached_block={},
            _producer_partial_tail_reqs={},
        )

    def test_register_exact_boundary(self):
        manager = self.manager()
        self.assertEqual(scope["cache_tail"](manager, self.req(128), 112, None), "hash")
        self.assertEqual(manager._producer_partial_tail_reqs["r"][1], 112)

    def test_not_arbitrary_boundary(self):
        self.assertIsNone(scope["cache_tail"](self.manager(), self.req(128), 96, None))

    def test_existing_checkpoint(self):
        manager = self.manager()
        manager._checkpoints["r"] = (112, 0)
        self.assertIsNone(scope["cache_tail"](manager, self.req(128), 112, None))

    def test_dense_only(self):
        self.assertIsNone(scope["cache_tail"](self.manager(), self.req(128), 112, 0))

    def test_internal_checkpoint_inherited(self):
        scheduler = self.scheduler()
        scheduler.mamba_has_prefill_checkpoint_blocks = True
        original = scope["is_mamba_prefill_checkpoint_valid"]
        scope["is_mamba_prefill_checkpoint_valid"] = lambda **kw: True
        try:
            self.assertEqual(scope["split"](scheduler, self.req(128), 128), 128)
        finally:
            scope["is_mamba_prefill_checkpoint_valid"] = original

    def test_attention_common_boundary(self):
        manager = self.manager()
        calls = []
        manager.block_pool.cache_partial_block = lambda **kw: calls.append(kw["num_tokens"])
        scope["attention_tail"](manager, self.req(128), 112)
        scope["attention_tail"](manager, self.req(128), 128)
        self.assertEqual(calls, [112, 112])

    def test_attention_nonmultiple_inherited(self):
        manager = self.manager()
        calls = []
        manager.block_pool.cache_partial_block = lambda **kw: calls.append(kw["num_tokens"])
        scope["attention_tail"](manager, self.req(129), 128)
        self.assertEqual(calls, [])

    def test_attention_does_not_register_future_kv(self):
        manager = self.manager()
        calls = []
        manager.block_pool.cache_partial_block = lambda **kw: calls.append(kw["num_tokens"])
        scope["attention_tail"](manager, self.req(128), 96)
        self.assertEqual(calls, [])

    def test_registry_not_module_alias(self):
        class RegisteredMamba:
            _cache_partial_tail_block = staticmethod(lambda *a: "registered-mamba")

        class RegisteredAttention:
            _cache_partial_tail_block = staticmethod(lambda *a: "registered-attention")

        class MambaSpec:
            pass

        class FullAttentionSpec:
            pass

        registry = NS(
            get_manager_class=lambda spec: RegisteredMamba if isinstance(spec, MambaSpec) else RegisteredAttention
        )
        local = dict(
            Scheduler=NS(_mamba_block_aligned_split=lambda *a: 1),
            KVCacheSpecRegistry=registry,
            MambaSpec=MambaSpec,
            FullAttentionSpec=FullAttentionSpec,
        )
        local.update(scope)
        body = source[source.index("# v0.30 resolves") :]
        exec(body, local)
        self.assertIs(local["_registered_mamba_manager"], RegisteredMamba)
        self.assertEqual(local["_cache_tail"](), "registered-mamba")
        self.assertIs(RegisteredMamba._cache_partial_tail_block, local["cache_tail"])
        self.assertIs(RegisteredAttention._cache_partial_tail_block, local["attention_tail"])
        self.assertEqual(RegisteredMamba._cache_partial_tail_block(None, None, 0, None), "registered-mamba")

    def test_sparse_path_does_not_apply_reusable_split(self):
        scheduler = self.scheduler()
        scheduler.block_size = 896
        scheduler.use_eagle = True
        original = scope["_get_sparse_index_kpool"]
        scope["_get_sparse_index_kpool"] = lambda c: 4
        try:
            self.assertEqual(scope["split"](scheduler, self.req(128), 128), 128)
        finally:
            scope["_get_sparse_index_kpool"] = original

    def test_split_restores_drop_flags_on_upstream_error(self):
        scheduler = self.scheduler()
        scheduler.use_eagle = True
        original = scope["_original_mamba_block_aligned_split"]

        def fail(*args):
            raise RuntimeError("upstream error")

        scope["_original_mamba_block_aligned_split"] = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "upstream error"):
                scope["split"](scheduler, self.req(128), 128)
        finally:
            scope["_original_mamba_block_aligned_split"] = original
        self.assertTrue(scheduler.use_eagle)

    def test_split_restores_drop_flags_before_fallback(self):
        scheduler = self.scheduler()
        scheduler.use_eagle = True
        scheduler.use_eagle_block_drop = True
        observed = []
        original = scope["_original_mamba_block_aligned_split"]
        scope["_original_mamba_block_aligned_split"] = lambda s, r, n, *a: (
            observed.append((s.use_eagle, s.use_eagle_block_drop)) or n
        )
        try:
            self.assertEqual(scope["split"](scheduler, self.req(128), 128), 112)
        finally:
            scope["_original_mamba_block_aligned_split"] = original
        self.assertEqual(observed, [(False, False)])
        self.assertTrue(scheduler.use_eagle)
        self.assertTrue(scheduler.use_eagle_block_drop)

    def test_attention_alias_registered_after_native_tail(self):
        manager = self.manager()
        calls = []
        original = scope["_attention_tail"]
        scope["_attention_tail"] = lambda *a: calls.append("native")
        manager.block_pool.cache_partial_block = lambda **kw: calls.append(kw["num_tokens"])
        try:
            scope["attention_tail"](manager, self.req(128), 128)
        finally:
            scope["_attention_tail"] = original
        self.assertEqual(calls, ["native", 112])


if __name__ == "__main__":
    unittest.main()
