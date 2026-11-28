# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU tests for the Prefill/Decode speculative-decoding config parity check.

issue #22659: when speculative decoding is enabled on one side only (or with
different layout-relevant knobs), Prefill and Decode register different KV cache
tensor sets, and the KV transfer pairs addresses that do not describe the same
layers. The handshake metadata carries a small fingerprint of the layout-relevant
config, and the receiving engine rejects a mismatch instead of serving output
decoded from KV cache it never received.
"""

import importlib
import threading
import unittest
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.utils.spec_config import (
    KVTransferSpecConfigMismatch,
    SpecConfigFingerprint,
    SpecConfigParityGuard,
    fingerprint_from_vllm_config,
    peer_fingerprint,
)

CONNECTOR_MODULES = (
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
)


def _vllm_config(spec=None):
    return SimpleNamespace(speculative_config=spec)


def _spec(method, num_tokens, draft_layers=None):
    draft_model_config = None
    if draft_layers is not None:
        draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=draft_layers), dtype=None
        )
    return SimpleNamespace(method=method, num_speculative_tokens=num_tokens, draft_model_config=draft_model_config)


def _draft_spec(hidden_size, model="org/draft", dtype="bfloat16"):
    hf_config = SimpleNamespace(
        architectures=["EagleDraftModel"],
        hidden_size=hidden_size,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=1,
    )
    draft_model_config = SimpleNamespace(hf_config=hf_config, dtype=dtype, model=model)
    return SimpleNamespace(method="eagle", num_speculative_tokens=2, draft_model_config=draft_model_config)


class TestFingerprint(unittest.TestCase):
    def test_disabled_config_is_reported_as_disabled(self):
        fingerprint = SpecConfigFingerprint.decode(fingerprint_from_vllm_config(_vllm_config()))
        self.assertIsNotNone(fingerprint)
        self.assertFalse(fingerprint.enabled)
        self.assertEqual(fingerprint.method, "none")

    def test_mtp_config_round_trips(self):
        raw = fingerprint_from_vllm_config(_vllm_config(_spec("mtp", 1)))
        self.assertEqual(SpecConfigFingerprint.decode(raw).encode(), raw)

    def test_draft_layers_are_part_of_the_fingerprint(self):
        one = fingerprint_from_vllm_config(_vllm_config(_spec("eagle", 2, draft_layers=1)))
        two = fingerprint_from_vllm_config(_vllm_config(_spec("eagle", 2, draft_layers=2)))
        self.assertNotEqual(one, two)
        self.assertEqual(SpecConfigFingerprint.decode(one).draft_layers, 1)

    def test_draft_model_geometry_is_part_of_the_fingerprint(self):
        small = fingerprint_from_vllm_config(_vllm_config(_draft_spec(hidden_size=2048)))
        large = fingerprint_from_vllm_config(_vllm_config(_draft_spec(hidden_size=4096)))
        self.assertNotEqual(small, large)
        # Same geometry, different path must not differ: the digest is layout,
        # not identity, so moving a checkpoint does not trip the check.
        self.assertEqual(small, fingerprint_from_vllm_config(_vllm_config(_draft_spec(hidden_size=2048))))

    def test_unknown_and_foreign_payloads_decode_to_none(self):
        self.assertIsNone(SpecConfigFingerprint.decode(""))
        self.assertIsNone(SpecConfigFingerprint.decode("mtp:1"))
        # A peer on another encoding version must not be compared by accident.
        self.assertIsNone(SpecConfigFingerprint.decode("v0|enabled=1|method=mtp|n=1|draft_layers=0"))

    def test_peer_fingerprint_reads_the_metadata_field(self):
        metadata = SimpleNamespace(spec_config_fingerprint="v1|enabled=0|method=none|n=0|draft_layers=0")
        self.assertTrue(peer_fingerprint(metadata).startswith("v1|"))
        self.assertEqual(peer_fingerprint(SimpleNamespace()), "")


class TestGuard(unittest.TestCase):
    def test_equal_configs_pass(self):
        guard = SpecConfigParityGuard(_vllm_config(_spec("mtp", 1)), role="decode")
        guard.check(guard.local_fingerprint, "10.0.0.7:31002")

    def test_legacy_peer_without_fingerprint_is_skipped(self):
        guard = SpecConfigParityGuard(_vllm_config(_spec("mtp", 1)), role="decode")
        guard.check("", "10.0.0.7:31002")
        guard.check(None, "10.0.0.7:31002")

    def test_peer_on_another_encoding_version_is_skipped(self):
        guard = SpecConfigParityGuard(_vllm_config(_spec("mtp", 1)), role="decode")
        guard.check("v0|enabled=0|method=none|n=0|draft_layers=0", "10.0.0.7:31002")

    def test_mismatch_raises_with_both_sides_and_a_fix(self):
        guard = SpecConfigParityGuard(_vllm_config(_spec("mtp", 1)), role="decode")
        peer = fingerprint_from_vllm_config(_vllm_config())
        with self.assertRaises(KVTransferSpecConfigMismatch) as raised:
            guard.check(peer, "10.0.0.7:31002")
        message = str(raised.exception)
        self.assertIn("prefill=", message)
        self.assertIn("decode=", message)
        # The operator gets the exact flag to copy to the other side.
        self.assertIn('--speculative-config \'{"method": "mtp", "num_speculative_tokens": 1}\'', message)

    def test_mismatch_keeps_raising_on_repeated_calls(self):
        guard = SpecConfigParityGuard(_vllm_config(), role="decode")
        peer = fingerprint_from_vllm_config(_vllm_config(_spec("mtp", 1)))
        for _ in range(2):
            with self.assertRaises(KVTransferSpecConfigMismatch):
                guard.check(peer, "10.0.0.7:31002")


class TestPeerIdentity(unittest.TestCase):
    """A peer is identified by engine id plus port, not by port alone.

    Two Prefill engines on different hosts can use the same handshake port. If
    the cache were keyed by port only, the second engine would silently reuse
    the first engine's fingerprint and its mismatch would pass the check.
    """

    @classmethod
    def setUpClass(cls):
        cls.modules = []
        for name in CONNECTOR_MODULES:
            try:
                cls.modules.append((name, importlib.import_module(name)))
            except ImportError as exc:  # pragma: no cover - reduced test env
                raise unittest.SkipTest("connector import unavailable: %s" % exc)

    @staticmethod
    def _make_recv_thread(thread_cls, fingerprints):
        thread = thread_cls.__new__(thread_cls)  # bypass the NPU-bound __init__
        thread.kv_caches_base_addr = {}
        thread.peer_spec_config = {}
        thread.remote_metadata_lock = threading.Lock()
        thread.peer_metadata_fetch_lock = threading.Lock()
        handshakes = []

        def fetch(host, port):
            handshakes.append((host, port))
            engine_id = "engine-" + host
            thread.kv_caches_base_addr.setdefault(engine_id, {})[port] = []
            thread.peer_spec_config[(engine_id, port)] = fingerprints[host]

        thread._get_remote_metadata = fetch
        return thread, handshakes

    def test_two_peers_sharing_a_port_are_checked_separately(self):
        good = fingerprint_from_vllm_config(_vllm_config(_spec("mtp", 1)))
        bad = fingerprint_from_vllm_config(_vllm_config(_spec("mtp", 3)))
        fingerprints = {"10.0.0.7": good, "10.0.0.8": bad}

        for name, module in self.modules:
            with self.subTest(connector=name):
                thread, handshakes = self._make_recv_thread(module.KVCacheRecvingThread, fingerprints)
                guard = SpecConfigParityGuard(_vllm_config(_spec("mtp", 1)), role="decode")

                peer_a = thread.ensure_peer_spec_config("engine-10.0.0.7", "10.0.0.7", 31002)
                guard.check(peer_a, "10.0.0.7:31002")
                self.assertEqual(len(handshakes), 1)

                # Same port, another host and engine: must handshake again and
                # expose its own (different) config instead of reusing A's.
                peer_b = thread.ensure_peer_spec_config("engine-10.0.0.8", "10.0.0.8", 31002)
                self.assertEqual(len(handshakes), 2)
                self.assertEqual(peer_b, bad)
                with self.assertRaises(KVTransferSpecConfigMismatch):
                    guard.check(peer_b, "10.0.0.8:31002")

                # A stays cached and is not handshaken again.
                self.assertTrue(thread._cached_peer_spec_config("engine-10.0.0.7", 31002)[0])
                self.assertEqual(len(handshakes), 2)


if __name__ == "__main__":
    unittest.main()
