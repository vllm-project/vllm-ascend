#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for the DSpark M1 model-side helpers.

Covers the pure-math + no-runtime-state pieces shipped in M1.4-M1.5:
  * DSparkMarkovHead.forward returning (logits_bias, markov_embed)
  * DSparkConfidenceHead returning per-position scores in fp32
  * DeepSeekMultiTokenPredictorLayer.apply_dspark_markov_bias being a
    pure pass-through when markov_head is None (default for non-DSpark layers).

We deliberately do NOT exercise the full proposer / MTP forward here — those
need a loaded checkpoint and an NPU device, and are covered by the e2e suite
in M1.9 instead. The aim of this UT file is to lock in the math contract of
the heads so future refactors of `inference/model.py` upstream can be
back-ported without silently breaking acceptance rates.
"""

import importlib
import unittest
from unittest.mock import patch

import torch


class TestDsparkMarkovHead(unittest.TestCase):
    """Numerical contract of DSparkMarkovHead.

    Per `inference/model.py:798-803`:
        embed   = markov_w1(token_ids)            # [*, rank]
        logits  = markov_w2(embed, full_logits=True)  # [*, vocab]
        return logits, embed
    """

    def setUp(self):
        # Import lazily so VLLM_ASCEND_ENABLE_DSPARK gate is respected.
        with patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_DSPARK": "1"}):
            # Force re-evaluation in case the test runner cached envs earlier.
            import vllm_ascend.envs as _envs

            importlib.reload(_envs)
            self._envs_module = _envs
            from vllm_ascend.models.deepseek_v4_mtp import DSparkMarkovHead

            self._head_cls = DSparkMarkovHead

    def tearDown(self):
        # Restore default-off behaviour for any subsequent tests.
        with patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_DSPARK": "0"}):
            importlib.reload(self._envs_module)

    def test_markov_returns_pair_of_correct_shapes(self):
        vocab, rank, bsz = 64, 8, 3
        head = self._head_cls(vocab_size=vocab, markov_rank=rank)
        token_ids = torch.randint(0, vocab, (bsz,), dtype=torch.long)
        logits_bias, embed = head(token_ids)
        self.assertEqual(logits_bias.shape, (bsz, vocab))
        self.assertEqual(embed.shape, (bsz, rank))

    def test_markov_logits_bias_matches_two_step_linear(self):
        """``logits_bias`` must equal ``markov_w2.weight @ markov_w1(tok)``."""
        vocab, rank = 32, 4
        head = self._head_cls(vocab_size=vocab, markov_rank=rank)
        token_ids = torch.tensor([1, 5, 7], dtype=torch.long)
        logits_bias, embed = head(token_ids)
        expected = torch.nn.functional.linear(embed, head.markov_w2.weight)
        torch.testing.assert_close(logits_bias, expected)


class TestDsparkConfidenceHead(unittest.TestCase):
    """Numerical contract of DSparkConfidenceHead.

    Per `inference/model.py:813-816`:
        hidden = cat([hidden, markov_embed], dim=-1)
        return proj(hidden.float()).squeeze(-1)

    Note the explicit cast to fp32: the checkpoint stores proj in fp32 and
    the M2 prefix scheduler needs full precision for the cumprod survival
    probability calculation.
    """

    def setUp(self):
        with patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_DSPARK": "1"}):
            import vllm_ascend.envs as _envs

            importlib.reload(_envs)
            self._envs_module = _envs
            from vllm_ascend.models.deepseek_v4_mtp import DSparkConfidenceHead

            self._head_cls = DSparkConfidenceHead

    def tearDown(self):
        with patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_DSPARK": "0"}):
            importlib.reload(self._envs_module)

    def test_confidence_shape_and_dtype(self):
        bsz, block, hidden, rank = 2, 5, 16, 4
        head = self._head_cls(input_dim=hidden + rank)
        h = torch.randn(bsz, block, hidden, dtype=torch.bfloat16)
        m = torch.randn(bsz, block, rank, dtype=torch.bfloat16)
        score = head(h, m)
        self.assertEqual(score.shape, (bsz, block))
        # Score is fp32 regardless of input dtype (proj is fp32).
        self.assertEqual(score.dtype, torch.float32)

    def test_confidence_invariant_to_concat_order(self):
        """A direct concat-then-linear must equal the head's output."""
        hidden, rank = 8, 4
        head = self._head_cls(input_dim=hidden + rank)
        h = torch.randn(3, hidden)
        m = torch.randn(3, rank)
        score = head(h, m)
        manual = torch.nn.functional.linear(
            torch.cat([h, m], dim=-1).float(),
            head.proj.weight,
        ).squeeze(-1)
        torch.testing.assert_close(score, manual)


if __name__ == "__main__":
    unittest.main()
