# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "prepare_deepseek_distill_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("prepare_distill_tokenizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class TestPrepareDistillTokenizer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.output = self.root / "tokenizer"
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            TOOL.PROBES,
            trainer=trainers.BpeTrainer(vocab_size=512, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        tokenizer.save(str(self.checkpoint / "tokenizer.json"))
        (self.checkpoint / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
        self.original_metadata = {"tokenizer_class": "LlamaTokenizerFast", "chat_template": "{{ messages }}"}
        (self.checkpoint / "tokenizer_config.json").write_text(json.dumps(self.original_metadata), encoding="utf-8")

    def invoke(self):
        with patch.object(sys, "argv", [str(SCRIPT), str(self.checkpoint), str(self.output)]):
            TOOL.main()

    def test_preserves_pipeline_metadata_and_source(self):
        self.invoke()
        manifest = json.loads((self.output / "compatibility-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["probes"]), len(TOOL.PROBES))
        self.assertEqual(
            (self.output / "tokenizer.json").read_bytes(), (self.checkpoint / "tokenizer.json").read_bytes()
        )
        self.assertEqual(
            json.loads((self.checkpoint / "tokenizer_config.json").read_text(encoding="utf-8")),
            self.original_metadata,
        )

    def test_refuses_existing_output(self):
        self.output.mkdir()
        marker = self.output / "existing.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_refuses_wrong_architecture(self):
        (self.checkpoint / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_refuses_non_bytelevel_definition(self):
        definition = json.loads((self.checkpoint / "tokenizer.json").read_text(encoding="utf-8"))
        definition["decoder"] = {"type": "Fuse"}
        (self.checkpoint / "tokenizer.json").write_text(json.dumps(definition), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
