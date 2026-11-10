# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the recurrent GDN capability merge.

No NPU and no ``fla_npu`` install are required: the module under test imports
both lazily, and these tests supply their own capability records.
"""

from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.ops import gdn_capabilities as caps


def _capability(max_verify_len, *, published=True, source="opp-capabilities"):
    return SimpleNamespace(
        arch="arch22",
        max_verify_len=max_verify_len,
        published=published,
        source=source,
        path=None,
    )


class TestBuildLimits:
    def test_missing_api_is_the_legacy_eight_token_window(self):
        limits = caps.build_limits(None, 9)
        assert limits.max_verify_len == 8
        assert limits.published is False
        assert limits.source == "unavailable"
        assert not limits.requires_upstream_recurrent(8, adaptive=False)
        assert limits.requires_upstream_recurrent(9, adaptive=False)

    def test_old_opp_header_keeps_the_profile_workaround(self):
        limits = caps.build_limits(_capability(8, source="legacy-kernel-header"), 9)
        assert limits.max_verify_len == 8
        assert limits.legacy_fallback_min_query_len == 9
        assert not limits.requires_upstream_recurrent(8, adaptive=False)
        assert limits.requires_upstream_recurrent(9, adaptive=False)

    def test_v2_opp_drops_the_legacy_numerical_workaround(self):
        # The V2 window-loop kernel re-validated clean at 9+, so the profile's
        # old eight-token workaround must not keep routing A2 to Triton.
        limits = caps.build_limits(_capability(None), 9)
        assert limits.max_verify_len is None
        assert limits.legacy_fallback_min_query_len is None
        assert not limits.requires_upstream_recurrent(9, adaptive=False)
        assert not limits.requires_upstream_recurrent(64, adaptive=False)

    def test_published_hard_limit_is_exact(self):
        limits = caps.build_limits(_capability(16), None)
        assert not limits.requires_upstream_recurrent(16, adaptive=False)
        assert limits.requires_upstream_recurrent(17, adaptive=False)

    def test_unpublished_overlay_stays_conservative(self):
        limits = caps.build_limits(_capability(None, published=False, source="unpublished"), 9)
        assert limits.max_verify_len == 8
        assert limits.requires_upstream_recurrent(9, adaptive=False)

    def test_adaptive_always_uses_the_strided_kernel(self):
        limits = caps.build_limits(_capability(None), None)
        assert limits.requires_upstream_recurrent(1, adaptive=True)
        assert limits.requires_upstream_recurrent(64, adaptive=True)


class TestGetLimits:
    def setup_method(self):
        caps._cached_limits.cache_clear()

    def teardown_method(self):
        caps._cached_limits.cache_clear()

    def test_merges_the_profiled_device(self):
        with (
            patch.object(caps, "_fla_capability", return_value=_capability(None)),
            patch.object(caps, "_profile_fallback_min_query_len", return_value=9),
        ):
            limits = caps.get_recurrent_gdn_limits(0)
        assert limits.max_verify_len is None
        assert not limits.requires_upstream_recurrent(9, adaptive=False)

    def test_old_opp_wins_over_the_profile(self):
        with (
            patch.object(
                caps,
                "_fla_capability",
                return_value=_capability(8, source="legacy-kernel-header"),
            ),
            patch.object(caps, "_profile_fallback_min_query_len", return_value=9),
        ):
            limits = caps.get_recurrent_gdn_limits(0)
        assert limits.max_verify_len == 8
        assert limits.requires_upstream_recurrent(9, adaptive=False)


def _fake_ascendc(**attributes) -> ModuleType:
    module = ModuleType("fla_npu.ops.ascendc")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


class TestFlaCapabilityLookup:
    def _modules(self, ascendc):
        fla_npu = ModuleType("fla_npu")
        ops = ModuleType("fla_npu.ops")
        # ``ModuleType`` has no typed attributes for the synthetic package tree.
        vars(ops)["ascendc"] = ascendc
        vars(fla_npu)["ops"] = ops
        return {"fla_npu": fla_npu, "fla_npu.ops": ops, "fla_npu.ops.ascendc": ascendc}

    def test_old_wheel_without_the_api(self):
        with patch.dict("sys.modules", self._modules(_fake_ascendc())):
            assert caps._fla_capability(0) is None

    @pytest.mark.parametrize(
        "error",
        [
            LookupError("architecture not published"),
            RuntimeError("driver not initialised"),
            ValueError("unexpected topology"),
        ],
        ids=["lookup", "runtime", "value"],
    )
    def test_vendor_errors_are_unknown_not_unbounded(self, error):
        def resolve(device_index):
            raise error

        with patch.dict(
            "sys.modules",
            self._modules(_fake_ascendc(recurrent_gdn_capability=resolve)),
        ):
            limits = caps.build_limits(caps._fla_capability(0), 9)
        assert limits.max_verify_len == 8
        assert limits.requires_upstream_recurrent(9, adaptive=False)

    def test_reads_the_published_record(self):
        record = _capability(None)
        with patch.dict(
            "sys.modules",
            self._modules(_fake_ascendc(recurrent_gdn_capability=lambda device_index: record)),
        ):
            assert caps._fla_capability(0) is record
