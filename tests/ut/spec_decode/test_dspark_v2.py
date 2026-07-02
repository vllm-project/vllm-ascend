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


class TestDSparkSequentialSample(unittest.TestCase):
    """Lightweight contract test for ``_sample_sequential`` math.

    We exercise the loop body directly with stub model hooks so we don't need
    a real DSpark checkpoint or NPU device. Verifies:

    * Loop produces ``[num_reqs, N]`` tokens.
    * Step k's input ``prev`` token equals step k-1's sampled output (the
      sequential dependency that defines DSpark's "semi-autoregressive"
      property — paper §3.1).
    * Markov bias from step k-1's sample affects step k's logits (otherwise
      the head wouldn't have any inter-token signal).
    * Greedy path returns argmax(base_logits + bias) exactly.
    """

    def setUp(self):
        from vllm_ascend.spec_decode.dspark.speculator import AscendDsparkSpeculator

        # We construct an instance via ``__new__`` to bypass the full DFlash
        # __init__ chain (which needs a real VllmConfig).
        self._cls = AscendDsparkSpeculator

    def test_sample_sequential_greedy_argmax_with_bias(self):
        vocab = 8
        num_reqs = 2
        n_spec = 3
        rank = 4
        hidden = 5

        spec = self._cls.__new__(self._cls)
        spec.num_speculative_tokens = n_spec
        spec._markov_embeds_buffer = []
        spec.last_confidence = None

        # Build a stub draft model whose embed / bias / compute_logits /
        # compute_confidence we control deterministically.
        torch.manual_seed(0)
        base_logits_per_pos = torch.randn(num_reqs * n_spec, vocab)
        markov_w1_lookup = torch.randn(vocab, rank)
        markov_w2 = torch.randn(rank, vocab)

        class StubModel:
            def __init__(self):
                self.bias_calls = []
                self.embed_calls = []

            def compute_logits(self, h):
                # h shape is [num_reqs * n_spec, hidden] — we ignore content
                # and return our deterministic logits.
                return base_logits_per_pos

            def markov_embed(self, tok):
                self.embed_calls.append(tok.clone())
                return markov_w1_lookup[tok]

            def markov_bias(self, embed):
                self.bias_calls.append(embed.clone())
                return embed @ markov_w2

            def compute_confidence(self, hidden, markov_embeds):
                # Return per-position score = 0.5 regardless.
                return torch.full((hidden.shape[0], hidden.shape[1]), 0.5, dtype=torch.float32)

        spec._dspark_model = StubModel()

        head_hidden = torch.randn(num_reqs * n_spec, hidden)
        anchor_tokens = torch.tensor([3, 5], dtype=torch.int64)
        draft = spec._sample_sequential(
            num_reqs=num_reqs,
            head_hidden=head_hidden,
            anchor_tokens=anchor_tokens,
        )

        # Shape contract.
        self.assertEqual(draft.shape, (num_reqs, n_spec))
        self.assertEqual(draft.dtype, torch.int64)

        # Step 0 prev token must equal anchor; subsequent prev = previous step's sample.
        self.assertTrue(torch.equal(spec._dspark_model.embed_calls[0], anchor_tokens))
        for k in range(1, n_spec):
            self.assertTrue(torch.equal(spec._dspark_model.embed_calls[k], draft[:, k - 1]))

        # Greedy path: step k's sample MUST equal argmax(base_logits[k] + bias_k).
        base_view = base_logits_per_pos.view(num_reqs, n_spec, vocab)
        prev = anchor_tokens
        for k in range(n_spec):
            embed_k = markov_w1_lookup[prev]
            bias_k = embed_k @ markov_w2
            expected = (base_view[:, k] + bias_k).argmax(dim=-1)
            self.assertTrue(torch.equal(draft[:, k], expected), msg=f"step {k}")
            prev = draft[:, k]

        # Confidence was computed.
        self.assertIsNotNone(spec.last_confidence)
        self.assertEqual(spec.last_confidence.shape, (num_reqs, n_spec))


class TestDSparkModelWiring(unittest.TestCase):
    """Regression guard for the load hook.

    The framework loads draft models via ``_get_model`` (base ``load_model``
    does ``self.model = self._get_model()``); there is NO ``load_draft_model``
    hook. A previous version defined ``load_draft_model`` — which nobody calls —
    so ``self._dspark_model`` stayed ``None`` and ``_propose`` silently fell
    back to plain DFlash, making the DSpark algorithm inert. These tests lock
    the correct wiring so that regression cannot come back.
    """

    def setUp(self):
        import vllm_ascend.spec_decode.dspark.speculator as spec_mod

        self._mod = spec_mod
        self._cls = spec_mod.AscendDsparkSpeculator

    def test_no_dead_load_draft_model_hook(self):
        # There must be no ``load_draft_model`` (dead hook); the real hooks are
        # ``load_model`` + ``_get_model``.
        self.assertFalse(
            hasattr(self._cls, "load_draft_model"),
            "load_draft_model is a dead hook nobody calls; use _get_model.",
        )
        self.assertTrue(hasattr(self._cls, "_get_model"))
        self.assertTrue(hasattr(self._cls, "load_model"))

    def test_get_model_caches_dspark_model(self):
        spec = self._cls.__new__(self._cls)
        sentinel_target = object()
        sentinel_draft = object()
        spec._target_model = sentinel_target
        spec.vllm_config = object()
        spec._dspark_model = None

        calls = {}

        def fake_load_dspark_model(target, vllm_config):
            calls["target"] = target
            calls["vllm_config"] = vllm_config
            return sentinel_draft

        orig = self._mod.load_dspark_model
        self._mod.load_dspark_model = fake_load_dspark_model
        try:
            returned = self._cls._get_model(spec)
        finally:
            self._mod.load_dspark_model = orig

        # _get_model must (a) route through load_dspark_model with the target,
        # (b) cache the concrete draft model, (c) return it for base load_model.
        self.assertIs(calls["target"], sentinel_target)
        self.assertIs(spec._dspark_model, sentinel_draft)
        self.assertIs(returned, sentinel_draft)


class TestDSparkOnehotLogits(unittest.TestCase):
    """``_onehot_logits`` — argmax must recover sampled tokens; padding dropped."""

    def setUp(self):
        from vllm_ascend.spec_decode.dspark.speculator import AscendDsparkSpeculator

        self._fn = AscendDsparkSpeculator._onehot_logits

    def test_argmax_recovers_tokens_exact(self):
        vocab = 16
        flat = torch.tensor([3, 15, 0, 7], dtype=torch.int64)
        fake = self._fn(flat, num_sample=4, vocab_size=vocab, device=torch.device("cpu"))
        self.assertEqual(fake.shape, (4, vocab))
        self.assertTrue(torch.equal(fake.argmax(dim=-1), flat))

    def test_padded_rows_are_ignored_but_present(self):
        # lmhead_tp path: num_sample (6) > real (4). First 4 rows must recover
        # the tokens; the base slices logits[:num_indices] so the 2 pad rows
        # only need to exist (their content is irrelevant).
        vocab = 16
        flat = torch.tensor([3, 15, 0, 7], dtype=torch.int64)
        fake = self._fn(flat, num_sample=6, vocab_size=vocab, device=torch.device("cpu"))
        self.assertEqual(fake.shape, (6, vocab))
        self.assertTrue(torch.equal(fake[:4].argmax(dim=-1), flat))

    def test_num_sample_smaller_than_real_asserts(self):
        flat = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            self._fn(flat, num_sample=2, vocab_size=8, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
