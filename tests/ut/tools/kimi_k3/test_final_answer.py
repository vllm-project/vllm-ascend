# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest

from tools.kimi_k3.final_answer import final_message
from tools.kimi_k3.final_answer import gsm8k_final_postprocess as score


class FinalAnswerTests(unittest.TestCase):
    def test_no_reasoning_fallback(self):
        self.assertEqual(final_message({"content": None, "reasoning": "The answer is 42"}, "stop"), "")
        self.assertEqual(score("We have 4 items and need 42"), "NULL")

    def test_truncation_rejected(self):
        self.assertEqual(final_message({"content": "The answer is 42"}, "length"), "")

    def test_final_channel(self):
        raw = "<|open|>response<|sep|>The answer is 42.<|close|>response<|sep|><|close|>message<|sep|>"
        self.assertEqual(score(final_message({"reasoning": raw}, "stop")), "42")
        self.assertEqual(score(final_message({"content": "4</think>The answer is 42."}, "stop")), "42")

    def test_supported_answers(self):
        for text, expected in [
            ("#### 1,234", "1234"),
            ("The answer is $42.", "42"),
            ("\\boxed{-3}", "-3"),
            ("42", "42"),
        ]:
            self.assertEqual(score(text), expected)


if __name__ == "__main__":
    unittest.main()
