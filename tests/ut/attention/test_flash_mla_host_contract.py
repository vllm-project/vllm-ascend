# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly on CPU; mocked operators do not establish NPU correctness."""

import ast
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_definitions(path, names=None, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)
    ]
    module = types.ModuleType("_flash_host_test")
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(ROOT / path), "exec"), module.__dict__)
    return module


class FlashMLAHostContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        package = types.ModuleType("cann_ops_transformer.ops")

        def metadata(lengths, heads, kv_heads, **kwargs):
            self.calls.append(("metadata", lengths, heads, kv_heads, kwargs))
            return torch.full((lengths.numel() * 8,), len(self.calls), dtype=torch.int32)

        def main(query, cache, **kwargs):
            self.calls.append(("main", query, cache, kwargs))
            return torch.zeros(query.shape[1], query.shape[0], 512, dtype=query.dtype), torch.empty(0)

        package.flash_mla_with_kvcache_metadata = metadata
        package.flash_mla_with_kvcache = main
        self.modules = patch.dict(sys.modules, {"cann_ops_transformer.ops": package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.api = load_definitions(
            "vllm_ascend/attention/flash_mla.py",
            torch=torch,
            dataclass=dataclass,
            MLA_FLASH_SUPPORTED_Q_HEADS={8, 12, 64, 96},
            FLASH_MLA_BLOCK_SIZE=128,
            FLASH_MLA_QK_DIM=576,
            FLASH_MLA_V_DIM=512,
            FLASH_MLA_MASK_SIZE=2048,
        )

    def builder(self, heads=8):
        builder = NS(device="cpu", decode_threshold=1)
        self.api.init_flash_mla_metadata(builder, NS(num_heads=heads, num_kv_heads=1))
        return builder

    @staticmethod
    def common(batch=1, query=2, padding=2, causal=True, kv=128000):
        tokens = batch * query
        return NS(
            num_reqs=batch,
            num_actual_tokens=tokens,
            num_input_tokens=tokens + padding,
            max_query_len=query,
            causal=causal,
            block_table_tensor=torch.arange(batch * 1000, dtype=torch.int32).view(batch, 1000),
            query_start_loc=torch.arange(0, tokens + 1, query, dtype=torch.int32),
            seq_lens=torch.full((batch,), kv, dtype=torch.int32),
            slot_mapping=torch.arange(tokens, dtype=torch.int64),
            positions=torch.arange(tokens),
        )

    @staticmethod
    def cache():
        backing = torch.full((4 * 128 * 2 * 576 + 37,), -9, dtype=torch.bfloat16)
        return backing, torch.as_strided(backing, (2, 128, 1, 576), (2 * 128 * 2 * 576, 2 * 576, 576, 1), 37)

    def test_head_batch_length_matrix(self):
        for heads in (8, 12, 64, 96):
            builder = self.builder(heads)
            for batch, query, kv in (
                (1, 1, 100000),
                (29, 2, 128000),
                (30, 4, 100000),
                (31, 8, 128000),
                (32, 16, 128000),
            ):
                with self.subTest(heads=heads, batch=batch, query=query, kv=kv):
                    b = self.api.build_flash_mla_metadata(builder, self.common(batch, query, kv=kv))
                    self.assertEqual(b.num_tokens, batch * query + 2)
                    self.assertEqual(b.cu[-1], b.num_tokens)
                    self.assertEqual(b.used_q[-1], 0)
                    self.assertTrue(torch.equal(b.cache_lens[:-1], torch.full((batch,), kv)))
                    self.assertTrue((b.slots[-2:] == -1).all())
                    call = self.calls[-1]
                    self.assertEqual(call[2:4], (heads, 1))
                    self.assertEqual((call[-1]["max_seqlen_q"], call[-1]["max_seqlen_kv"]), (-1, -1))

    def test_fresh_schedule_and_metadata_each_step(self):
        builder, common = self.builder(), self.common()
        first = self.api.build_flash_mla_metadata(builder, common)
        common.seq_lens.fill_(129)
        common.block_table_tensor.add_(7)
        second = self.api.build_flash_mla_metadata(builder, common)
        self.assertIsNot(first.schedule, second.schedule)
        self.assertEqual(first.cache_lens[0], 128000)
        self.assertEqual(second.cache_lens[0], 129)
        self.assertEqual(first.block_table[0, 0], 0)
        self.assertEqual(second.block_table[0, 0], 7)
        self.assertFalse(hasattr(builder, "_flash_buffers"))

    def test_main_consumes_exact_metadata_and_original_cache(self):
        for causal in (True, False):
            b = self.api.build_flash_mla_metadata(self.builder(), self.common(causal=causal))
            metadata_call = self.calls[-1]
            backing, cache = self.cache()
            before = backing.clone()
            q = torch.zeros(b.num_tokens, b.num_heads, 576, dtype=torch.bfloat16)
            output, lse = self.api.run_flash_mla(q, cache, b, 0.125)
            _, actual_q, actual_cache, args = self.calls[-1]
            self.assertIs(actual_q, q)
            self.assertIs(actual_cache, cache)
            self.assertEqual(cache.storage_offset(), 37)
            self.assertTrue(torch.equal(before, backing))
            self.assertEqual(output.shape, (8, 4, 512))
            self.assertEqual(lse.numel(), 0)
            self.assertEqual((args["layout_q"], args["layout_kv"], args["layout_out"]), ("TND", "PA_BBND", "NTD"))
            self.assertEqual((args["max_seqlen_q"], args["max_seqlen_kv"]), (-1, -1))
            self.assertIs(args["metadata"], b.schedule)
            self.assertIs(args["cache_seqlens"], metadata_call[1])
            for name in ("cu_seqlens_q", "seqused_q"):
                self.assertIs(args[name], metadata_call[-1][name])
            self.assertEqual(args["mask_mode"], metadata_call[-1]["mask_mode"])
            self.assertFalse(args["return_softmax_lse"])
            if causal:
                self.assertEqual(b.attn_mask.shape, (2048, 2048))
                self.assertEqual(b.attn_mask[0, 1], 1)
                self.assertEqual(b.attn_mask[1, 0], 0)
                self.assertEqual(b.attn_mask[0, 0], 0)
            else:
                self.assertIsNone(args["attn_mask"])

    def test_inactive_request_padding_and_variable_queries(self):
        common = self.common(batch=3, query=2)
        common.query_start_loc = torch.tensor([0, 1, 4, 6], dtype=torch.int32)
        common.seq_lens[1] = 0
        b = self.api.build_flash_mla_metadata(self.builder(), common)
        self.assertEqual(b.used_q.tolist(), [1, 0, 2, 0])
        self.assertEqual(b.slots.tolist(), [0, -1, -1, -1, 4, 5, -1, -1])
        self.assertEqual(b.token_live.tolist(), [True, False, False, False, True, True, False, False])

    def test_uses_device_lengths_not_cpu_upper_bounds(self):
        common = self.common()
        common.seq_lens_cpu = torch.tensor([17])
        common.seq_lens_cpu_upper_bound = torch.tensor([128007])
        b = self.api.build_flash_mla_metadata(self.builder(), common)
        self.assertEqual(b.cache_lens[0], 128000)

    def test_reject_invalid_cache_heads_or_query(self):
        for cache in (torch.empty(2, 1, 128, 576), torch.empty(2, 16, 1, 576), torch.empty(2, 128, 1, 576)):
            with self.assertRaises(ValueError):
                self.api.validate_flash_cache(cache)
        with self.assertRaises(ValueError):
            self.builder(heads=24)
        b = self.api.build_flash_mla_metadata(self.builder(), self.common())
        _, cache = self.cache()
        with self.assertRaises(ValueError):
            self.api.run_flash_mla(torch.zeros(4, 12, 576, dtype=torch.bfloat16), cache, b, 1)

    def test_no_external_package_fails_without_fallback(self):
        with patch.dict(sys.modules, {"cann_ops_transformer.ops": None}), self.assertRaises(ModuleNotFoundError):
            self.api.build_flash_mla_metadata(self.builder(), self.common())

    def test_builder_routes_before_legacy_cpu_length_handling(self):
        tree = ast.parse((ROOT / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAMetadataBuilder")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
        method.body = [method.body[0]]  # Execute the real opt-in builder branch.
        method.args.args = [ast.arg(arg=a.arg) for a in method.args.args]
        method.returns = None
        scope = dict(
            envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            split_decodes_and_prefills=lambda *a, **kw: (1, 0, 1, 0),
            build_flash_mla_metadata=self.api.build_flash_mla_metadata,
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "builder", "exec"), scope)
        builder = self.builder()
        builder.metadata_cls = NS
        common = self.common(query=1)
        common.attn_state = "decode"
        result = scope["build"](builder, 0, common)
        self.assertEqual(result.num_decode_tokens, 1)
        self.assertIsNone(result.seq_lens_cpu)
        self.assertIs(result.seq_lens, common.seq_lens)
        self.assertEqual(result.flash.num_tokens, 3)

    def test_empty_padding_only_metadata_and_overlapping_cache(self):
        common = self.common()
        common.seq_lens.zero_()
        b = self.api.build_flash_mla_metadata(self.builder(), common)
        self.assertFalse(b.token_live.any())
        self.assertTrue((b.slots == -1).all())
        backing, cache = self.cache()
        overlap = torch.as_strided(backing, cache.shape, (576, 576, 576, 1))
        with self.assertRaises(ValueError):
            self.api.validate_flash_cache(overlap)

    def test_forward_writer_rope_nope_and_output(self):
        tree = ast.parse((ROOT / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_flash")
        for use_rope in (False, True):
            for gate in (False, True):
                common = self.common()
                common.slot_mapping = torch.tensor([127, 128])
                b = self.api.build_flash_mla_metadata(self.builder(), common)
                backing, cache = self.cache()
                expected = backing.clone()
                expected_cache = torch.as_strided(expected, cache.shape, cache.stride(), cache.storage_offset())
                data = torch.arange(4 * 576).to(torch.bfloat16).view(4, 576) / 1024
                expected_cache[0, 127, 0] = data[0]
                expected_cache[1, 0, 0] = data[1]
                if use_rope:
                    expected_cache[0, 127, 0, 512:] += 1
                    expected_cache[1, 0, 0, 512:] += 1
                events = []

                def scatter(*, events=events, cache=cache, **kwargs):
                    events.append("write")
                    self.assertEqual(kwargs["key_cache"].stride(), cache.stride())
                    for index, slot in enumerate(kwargs["slot_mapping"].tolist()):
                        if slot >= 0:
                            kwargs["key_cache"][slot // 128, slot % 128] = kwargs["key"][index]
                            kwargs["value_cache"][slot // 128, slot % 128] = kwargs["value"][index]

                def run(q, kv, flash, scale, events=events, backing=backing, expected=expected, cache=cache):
                    events.append("attention")
                    self.assertEqual(events, ["write", "notify", "attention"])
                    self.assertTrue(torch.equal(backing, expected))
                    self.assertIs(kv, cache)
                    return torch.ones(8, 4, 512, dtype=torch.bfloat16), None

                def project(latent):
                    self.assertEqual(latent.shape, (8, 4, 512))
                    projected = torch.ones(4, 8, dtype=torch.bfloat16)
                    projected[2:] = float("nan")
                    return projected

                scope = dict(
                    torch=torch,
                    torch_npu=NS(npu_scatter_pa_kv_cache=scatter),
                    validate_flash_cache=self.api.validate_flash_cache,
                    run_flash_mla=run,
                    get_cos_and_sin_mla=lambda *a, **k: (None, None),
                    notify_kv_cache_written=lambda name, events=events: events.append("notify"),
                )
                exec(compile(ast.Module(body=[method], type_ignores=[]), "forward", "exec"), scope)
                impl = NS(
                    layerwise_kv_cache_hook=None,
                    fused_qkv_a_proj=None,
                    kv_a_proj_with_mqa=lambda x, data=data: (data, None),
                    kv_a_layernorm=lambda x: x,
                    _q_proj_and_k_up_proj=lambda x: (torch.zeros(4, 8, 512), torch.zeros(4, 8, 64)),
                    use_mla_rope=use_rope,
                    rope_single=lambda x, cos, sin: x + 1,
                    scale=1,
                    _v_up_proj=project,
                    use_output_gate=gate,
                    g_proj=lambda x: (torch.zeros(4, 8), None),
                    o_proj=lambda x, **kwargs: (x + 2, None),
                )
                output = torch.full((6, 8), -99, dtype=torch.bfloat16)
                scope["_forward_flash"](impl, "layer", data, cache, NS(flash=b), output)
                torch.testing.assert_close(output[:2], torch.full((2, 8), 2.5 if gate else 3, dtype=torch.bfloat16))
                self.assertTrue((output[2:] == 0).all())

    def test_scope_guards_and_disabled_default(self):
        hardware = types.ModuleType("vllm_ascend.device.device_config")
        hardware.is_950 = lambda: True
        enabled = NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
        api = load_definitions(
            "vllm_ascend/platform.py",
            {"_validate_flash_mla_config"},
            VllmConfig=object,
            torch=torch,
            envs=enabled,
            model_uses_sfa_sparse=lambda model: False,
            KVPPConfig=NS(from_vllm_config=lambda cfg: NS(size=1)),
        )
        cfg = NS(
            model_config=NS(enforce_eager=True, use_mla=True, dtype=torch.bfloat16),
            use_v2_model_runner=True,
            cache_config=NS(cache_dtype="auto"),
            parallel_config=NS(prefill_context_parallel_size=1, decode_context_parallel_size=1),
            speculative_config=None,
            kv_transfer_config=None,
        )
        with patch.dict(sys.modules, {hardware.__name__: hardware}):
            api._validate_flash_mla_config(cfg)
            for obj, field, value in (
                (cfg.model_config, "enforce_eager", False),
                (cfg, "speculative_config", NS(method="dspark")),
                (cfg.parallel_config, "decode_context_parallel_size", 2),
                (cfg.parallel_config, "prefill_context_parallel_size", 2),
                (cfg.cache_config, "cache_dtype", "fp8"),
                (cfg, "use_v2_model_runner", False),
            ):
                previous = getattr(obj, field)
                setattr(obj, field, value)
                with self.assertRaises(ValueError):
                    api._validate_flash_mla_config(cfg)
                setattr(obj, field, previous)
            enabled.VLLM_ASCEND_ENABLE_FLASH_MLA = False
            api._validate_flash_mla_config(None)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
