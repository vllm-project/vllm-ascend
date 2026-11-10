# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for immutable MiniMax M3 Union policy configuration."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_ascend.models.minimax_m3.union_policy import UnionPrefillConfig, _load_policy


class TestUnionPolicy(unittest.TestCase):
    def setUp(self):
        _load_policy.cache_clear()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "policy.json"
        self.policy = {
            "valid": True,
            "routing": "frozen-layer-selector-head",
            "per_token_dynamic": False,
            "selector_modified": False,
            "decisions": {
                "0:0": {"path": "union"},
                "0:1": {"path": "q-gather-kv"},
                "1:0": {"path": "q-gather-kv"},
                "1:1": {"path": "union"},
            },
        }

    def config(self):
        content = json.dumps(self.policy).encode()
        self.path.write_bytes(content)
        return UnionPrefillConfig.from_dict(
            {
                "mode": "policy",
                "policy_path": str(self.path),
                "policy_sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    def test_default_off_and_pure_union_do_not_read_files(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected policy IO")):
            self.assertFalse(UnionPrefillConfig.from_dict(None).use_union("unused", 0, 2, 2))
            self.assertFalse(UnionPrefillConfig.from_dict({}).use_union("unused", 0, 2, 2))
            self.assertTrue(UnionPrefillConfig.from_dict({"mode": "union"}).use_union("unused", 0, 2, 2))

    def test_routes_by_layer_and_selector_head(self):
        config = self.config()
        for layer in range(2):
            for head in range(2):
                with self.subTest(layer=layer, head=head):
                    self.assertEqual(
                        config.use_union(f"model.layers.{layer}.self_attn.attn", head, 2, 2), layer == head
                    )

    def test_snapshot_is_immutable_and_loaded_once(self):
        config = self.config()
        self.assertTrue(config.use_union("layers.0.attn", 0, 2, 2))
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected second read")):
            self.assertFalse(config.use_union("layers.0.attn", 1, 2, 2))
        routes = _load_policy(str(self.path), config.policy_sha256, 2, 2)
        self.assertIsInstance(routes, frozenset)

    def test_hash_mismatch_fails_before_routing(self):
        config = self.config()
        self.path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            config.use_union("layers.0.attn", 0, 2, 2)

    def test_invalid_configuration(self):
        invalid = [
            [],
            {"unknown": True},
            {"mode": "dynamic"},
            {"mode": "policy"},
            {"mode": "policy", "policy_path": "x", "policy_sha256": "bad"},
            {"mode": "off", "policy_path": "x"},
            {"mode": "union", "policy_sha256": "0" * 64},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                UnionPrefillConfig.from_dict(config)

    def test_rejects_dynamic_or_modified_selector_policies(self):
        for key, value in (
            ("valid", False),
            ("routing", "per-token"),
            ("per_token_dynamic", True),
            ("selector_modified", True),
        ):
            with self.subTest(key=key):
                original = self.policy[key]
                self.policy[key] = value
                config = self.config()
                with self.assertRaisesRegex(ValueError, "frozen layer/selector-head"):
                    config.use_union("layers.0.attn", 0, 2, 2)
                self.policy[key] = original

    def test_rejects_incomplete_extra_or_invalid_routes(self):
        for change in ("missing", "extra", "invalid"):
            with self.subTest(change=change):
                decisions = dict(self.policy["decisions"])
                if change == "missing":
                    del self.policy["decisions"]["0:0"]
                elif change == "extra":
                    self.policy["decisions"]["2:0"] = {"path": "union"}
                else:
                    self.policy["decisions"]["0:0"] = {"path": "both"}
                config = self.config()
                with self.assertRaises(ValueError):
                    config.use_union("layers.0.attn", 0, 2, 2)
                self.policy["decisions"] = decisions

    def test_rejects_incompatible_topology_and_layer_names(self):
        config = self.config()
        for layer, head, layers, heads in (
            ("unknown", 0, 2, 2),
            ("layers.2.attn", 0, 2, 2),
            ("layers.0.attn", -1, 2, 2),
            ("layers.0.attn", 2, 2, 2),
            ("layers.0.attn", 0, 3, 2),
        ):
            with self.subTest(layer=layer, head=head, layers=layers), self.assertRaises(ValueError):
                config.use_union(layer, head, layers, heads)


if __name__ == "__main__":
    unittest.main()
