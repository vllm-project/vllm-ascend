# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the INT2 launcher-bridge mixin.

Pins the fail-closed loader contract (no vendor OPP path -> no load, no
raise) and the class-level once-latch semantics. The r135 arm is excluded:
its base-path side effects start before the failure point, so it is covered
by the device battery rather than a host-only unit test.
"""

from typing import Any

from vllm_ascend.attention.int2.bridge import Int2BridgeMixin


class _BridgeSelf(Int2BridgeMixin):
    def __init__(self):
        self.head_size = 128
        self.num_heads = 32
        self.num_kv_heads = 8
        self.key_cache = None
        self.value_cache = None
        self.k_scale_cache = None
        self.k_zero_cache = None
        self.v_scale_cache = None
        self.v_zero_cache = None


class TestLoaderFailClosed:
    def test_r91_fails_closed_without_vendor_opp(self, monkeypatch):
        for var in ("R144_OPP_PATH", "OSCAR_OPP_PATH", "R157_OPP"):
            monkeypatch.delenv(var, raising=False)
        self_obj = _BridgeSelf()
        assert self_obj._r91_ensure_loaded("npu:0") is False
        assert self_obj._r91_loaded is True

    def test_r91_once_latch_skips_retry(self, monkeypatch):
        for var in ("R144_OPP_PATH", "OSCAR_OPP_PATH", "R157_OPP"):
            monkeypatch.delenv(var, raising=False)
        self_obj = _BridgeSelf()
        assert self_obj._r91_ensure_loaded("npu:0") is False
        # second call: the once-latch holds even if the env changes
        monkeypatch.setenv("R144_OPP_PATH", "/nonexistent-opp")
        assert self_obj._r91_ensure_loaded("npu:0") is False

    def test_r112_fails_closed_without_vendor_opp(self, monkeypatch):
        for var in ("R144_OPP_PATH", "OSCAR_OPP_PATH", "R157_OPP"):
            monkeypatch.delenv(var, raising=False)
        self_obj = _BridgeSelf()
        assert self_obj._r112_ensure_fast("npu:0") is False
        assert self_obj._r112_loaded is True
        assert self_obj._r112_mod is None


class TestClassStateDefaults:
    def test_latches_default_none_counters_default_zero(self):
        for name in (
            "_r91_loaded",
            "_r91_mod",
            "_r112_loaded",
            "_r112_mod",
            "_r135_loaded",
            "_r135_mod",
        ):
            assert getattr(Int2BridgeMixin, name, None) is None
        for name in (
            "_r91_entry_count",
            "_r91_fallback_count",
            "_r91_getws_count",
            "_r91_launch_count",
            "_r91_sync_count",
            "_r91_output_consumed_count",
        ):
            # counters must start at 0: they are incremented on the class
            # before any init path runs (None + 1 raised TypeError)
            assert getattr(Int2BridgeMixin, name) == 0
        # subclass state never leaks back to the mixin
        _B: Any = type("_B", (Int2BridgeMixin,), {})
        _B._r91_loaded = True
        assert Int2BridgeMixin._r91_loaded is None

    def test_bridge_decode_counters_increment_on_fail_closed(self):
        # regression: with None counter defaults the exception handler itself
        # raised TypeError and crashed instead of falling back
        _B: Any = type("_B", (Int2BridgeMixin,), {})
        self_obj = _B()
        # key_cache is None -> shape access raises -> fallback path
        assert self_obj._r91_bridge_decode(None, None, None, None, None, None) is False
        assert _B._r91_entry_count == 1
        assert _B._r91_fallback_count == 1

    def test_r135_fails_closed_without_vendor(self, monkeypatch):
        for var in ("R106_VENDOR", "R133_VENDOR", "R133_GRAPH_V1", "R135_BASE", "R157_OPP", "R144_OPP_PATH"):
            monkeypatch.delenv(var, raising=False)
        self_obj = _BridgeSelf()
        assert self_obj._r135_ensure_v1("npu:0") is False
        assert self_obj._r135_loaded is True
        assert self_obj._r135_mod is None
