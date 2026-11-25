# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fresh-load masquerade must not mutate the shared LoadConfig."""

import unittest
from unittest.mock import patch

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm_ascend.model_loader.bigtensorloader.bigtensordefaultloader import (
    BigTensorDefaultLoader,
    _copy_load_config,
)


class TestBigTensorDefaultLoader(unittest.TestCase):
    def test_prepare_weights_masquerades_without_mutating_shared_config(self):
        config = LoadConfig(load_format="bigtensorloader")
        loader = BigTensorDefaultLoader(config)

        seen = {}

        def fake_prepare(self, *args, **kwargs):
            seen["format"] = self.load_config.load_format
            return "prepared"

        with patch.object(DefaultModelLoader, "_prepare_weights", fake_prepare):
            result = loader._prepare_weights()

        self.assertEqual(result, "prepared")
        # the delegated loader saw the masqueraded format...
        self.assertEqual(seen["format"], "safetensors")
        # ...while the shared config object was never mutated (draft/target
        # models can load concurrently off one LoadConfig)
        self.assertEqual(config.load_format, "bigtensorloader")

    def test_copy_load_config_copies_without_mutation(self):
        config = LoadConfig(load_format="bigtensorloader")
        copied = _copy_load_config(config, "safetensors")
        self.assertEqual(copied.load_format, "safetensors")
        self.assertIsNot(copied, config)
        self.assertEqual(config.load_format, "bigtensorloader")

    def test_prepare_weights_passthrough_for_other_formats(self):
        loader = BigTensorDefaultLoader(LoadConfig(load_format="safetensors"))

        with patch.object(DefaultModelLoader, "_prepare_weights", return_value="ok") as fake:
            result = loader._prepare_weights()

        self.assertEqual(result, "ok")
        # no masquerade: the same loader instance (its own config) is used
        fake.assert_called_once()
        self.assertEqual(loader.load_config.load_format, "safetensors")


if __name__ == "__main__":
    unittest.main()
