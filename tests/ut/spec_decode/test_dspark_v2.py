#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for the dspark_v2 head modules + checkpoint name remap.

Pure-tensor + pure-Python checks; no NPU device required. We exercise:

* ``DSparkMarkovHead.embed``   — vocab embedding lookup at rank=256
* ``DSparkConfidenceHead``     — fp32 score regardless of input dtype
* ``DSparkDeepseekV4ForCausalLM._remap_dspark_name``
   — ``mtp.<i>.<head>``   → ``model.<head>``
   — ``mtp.<i>.<rest>``   → ``model.layers.<i>.<rest>``
   — non-``mtp.*``         → None  (target-owned, skipped)

These tests lock the head math contract and the loader's prefix-rewriting
rule so a future refactor of the upstream reference (vllm PR #46995) can be
back-ported without silently breaking acceptance rates.
"""

import unittest

import torch


class TestDSparkV2Heads(unittest.TestCase):
    """DSparkMarkovHead + DSparkConfidenceHead numerical contract."""

    def setUp(self):
        # Lazy import so we don't pull vllm at module-import time on CI runners
        # that don't have it installed (mac dev box).
        from vllm_ascend.models.deepseek_v4_dspark import (
            DSparkConfidenceHead,
            DSparkMarkovHead,
        )

        self._markov_cls = DSparkMarkovHead
        self._confidence_cls = DSparkConfidenceHead

    def test_markov_embed_shape(self):
        vocab, rank, bsz = 64, 8, 3
        head = self._markov_cls(vocab_size=vocab, markov_rank=rank)
        token_ids = torch.randint(0, vocab, (bsz,), dtype=torch.long)
        embed = head.embed(token_ids)
        self.assertEqual(embed.shape, (bsz, rank))

    def test_confidence_shape_and_fp32(self):
        bsz, block, hidden, rank = 2, 5, 16, 4
        head = self._confidence_cls(input_dim=hidden + rank)
        h = torch.randn(bsz, block, hidden, dtype=torch.bfloat16)
        m = torch.randn(bsz, block, rank, dtype=torch.bfloat16)
        score = head(h, m)
        self.assertEqual(score.shape, (bsz, block))
        # Score must be fp32 regardless of input dtype — paper §3.2.1.
        self.assertEqual(score.dtype, torch.float32)

    def test_confidence_equals_concat_then_linear(self):
        """No normalization or activation between concat and linear."""
        hidden, rank = 8, 4
        head = self._confidence_cls(input_dim=hidden + rank)
        h = torch.randn(3, hidden)
        m = torch.randn(3, rank)
        score = head(h, m)
        manual = torch.nn.functional.linear(
            torch.cat([h, m], dim=-1).float(),
            head.proj.weight,
        ).squeeze(-1)
        torch.testing.assert_close(score, manual)


class TestDSparkRemapName(unittest.TestCase):
    """_remap_dspark_name correctness — mirrors upstream PR #46995 dspark.py L459."""

    def setUp(self):
        from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM

        self._remap = DSparkDeepseekV4ForCausalLM._remap_dspark_name

    def test_head_stack_lands_at_model_level(self):
        # head-stack params live at model.<name> (regardless of stage id).
        cases = {
            "mtp.0.main_proj.weight": "model.main_proj.weight",
            "mtp.0.main_norm.weight": "model.main_norm.weight",
            "mtp.2.norm.weight": "model.norm.weight",
            "mtp.2.hc_head_fn": "model.hc_head_fn",
            "mtp.2.hc_head_base": "model.hc_head_base",
            "mtp.2.hc_head_scale": "model.hc_head_scale",
            "mtp.2.markov_head.markov_w1.weight": "model.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight": "model.markov_head.markov_w2.weight",
            "mtp.2.confidence_head.proj.weight": "model.confidence_head.proj.weight",
        }
        for src, want in cases.items():
            self.assertEqual(self._remap(src), want, msg=src)

    def test_layer_params_land_at_layer_slot(self):
        # Non-head params go into model.layers.<stage>.<rest>.
        cases = {
            "mtp.0.attn.wkv.weight": "model.layers.0.attn.wkv.weight",
            "mtp.1.attn.attn_sink": "model.layers.1.attn.attn_sink",
            "mtp.2.ffn.experts.0.w1.weight": "model.layers.2.ffn.experts.0.w1.weight",
        }
        for src, want in cases.items():
            self.assertEqual(self._remap(src), want, msg=src)

    def test_non_mtp_returns_none(self):
        # Target-owned weights — must be skipped by the draft loader.
        for src in (
            "embed.weight",
            "head.weight",
            "model.layers.0.attn.wkv.weight",
            "norm.weight",
        ):
            self.assertIsNone(self._remap(src), msg=src)


if __name__ == "__main__":
    unittest.main()
