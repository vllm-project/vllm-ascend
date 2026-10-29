# SPDX-License-Identifier: Apache-2.0
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE_PATH = "vllm_ascend/worker/v1/model_runner.py"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / SOURCE_PATH).is_file())
TREE = ast.parse((ROOT / SOURCE_PATH).read_text(encoding="utf-8"))


class TestSamplerMode(unittest.TestCase):
    def test_model_mode_reaches_constructor(self):
        calls = [
            n
            for n in ast.walk(TREE)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute)
                and t.attr == "sampler"
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                for t in n.targets
            )
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "AscendSampler"
        ]
        self.assertEqual(len(calls), 1)
        statement = compile(ast.Module(body=calls, type_ignores=[]), SOURCE_PATH, "exec")
        for mode in ("raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"):
            with self.subTest(mode=mode):
                constructor = Mock()
                instance = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode=mode))
                exec(statement, {"self": instance, "AscendSampler": constructor})
                constructor.assert_called_once_with(logprobs_mode=mode)
                self.assertIs(instance.sampler, constructor.return_value)


if __name__ == "__main__":
    unittest.main()
