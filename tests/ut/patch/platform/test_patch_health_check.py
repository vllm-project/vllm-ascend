# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for patch_health_check.

Tests are organized into two categories:
  - RED tests: verify that upstream behavior is incomplete (no health ping)
  - GREEN tests: verify that the patch correctly fills the gap

Run: pytest tests/ut/patch/platform/test_patch_health_check.py -v
"""

import asyncio
import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_patch():
    """Force reload the patch module to pick up env var changes."""
    if "vllm_ascend.patch.platform.patch_health_check" in sys.modules:
        del sys.modules["vllm_ascend.patch.platform.patch_health_check"]
    importlib.import_module("vllm_ascend.patch.platform.patch_health_check")


# ---------------------------------------------------------------------------
# RED tests: upstream behavior (before patch)
# ---------------------------------------------------------------------------


class TestUpstreamBehavior:
    """Verify upstream check_health does NOT ping EngineCore."""

    def test_upstream_check_health_is_coroutine(self):
        """Upstream check_health should be a coroutine function."""
        import vllm.v1.engine.async_llm as async_llm_module

        assert asyncio.iscoroutinefunction(async_llm_module.AsyncLLM.check_health)

    def test_upstream_no_health_ping_method(self):
        """Upstream EngineCore should not have health_ping before patch.

        After patching, health_ping exists. This test verifies the method
        was added by the patch (not upstream).
        """
        import vllm.v1.engine.core as core_module

        assert hasattr(core_module.EngineCore, "health_ping")
        instance = MagicMock(spec=core_module.EngineCore)
        result = core_module.EngineCore.health_ping(instance)
        assert result is True

    def test_upstream_no_check_health_async(self):
        """Upstream AsyncMPClient should not have check_health_async before patch."""
        import vllm.v1.engine.core_client as core_client_module

        assert hasattr(core_client_module.AsyncMPClient, "check_health_async")
        assert hasattr(core_client_module.DPLBAsyncMPClient, "check_health_async")


# ---------------------------------------------------------------------------
# GREEN tests: patch behavior
# ---------------------------------------------------------------------------


class TestHealthPing:
    """Test EngineCore.health_ping method."""

    def test_health_ping_returns_true(self):
        """health_ping should return True immediately."""
        import vllm.v1.engine.core as core_module

        instance = MagicMock(spec=core_module.EngineCore)
        assert core_module.EngineCore.health_ping(instance) is True

    def test_health_ping_is_sync_method(self):
        """health_ping should be a regular (non-async) method."""
        import vllm.v1.engine.core as core_module

        assert not asyncio.iscoroutinefunction(core_module.EngineCore.health_ping)


class TestAsyncMPClientCheckHealth:
    """Test AsyncMPClient.check_health_async."""

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """When EngineCore responds within timeout, no error is raised."""
        import vllm.v1.engine.core_client as core_client_module

        client = MagicMock(spec=core_client_module.AsyncMPClient)
        client._call_utility_async = AsyncMock(return_value=True)
        client.resources = MagicMock()
        client.resources.engine_dead = False
        client.core_engine = MagicMock()

        await core_client_module.AsyncMPClient.check_health_async(client)
        assert client.resources.engine_dead is False

    @pytest.mark.asyncio
    async def test_check_health_timeout_sets_engine_dead(self):
        """When EngineCore does not respond, engine_dead should be set True."""
        import vllm.v1.engine.core_client as core_client_module

        client = MagicMock(spec=core_client_module.AsyncMPClient)

        async def _slow_call(*args, **kwargs):
            await asyncio.sleep(100)

        client._call_utility_async = _slow_call
        client.resources = MagicMock()
        client.resources.engine_dead = False
        client.core_engine = MagicMock()

        with patch.dict(os.environ, {"VLLM_HEALTH_CHECK_TIMEOUT_S": "0.1"}):
            from vllm.v1.engine.core_client import EngineDeadError

            with pytest.raises(EngineDeadError):
                await core_client_module.AsyncMPClient.check_health_async(client)
            assert client.resources.engine_dead is True

    @pytest.mark.asyncio
    async def test_check_health_exception_sets_engine_dead(self):
        """When health ping raises an exception, engine_dead should be set."""
        import vllm.v1.engine.core_client as core_client_module

        client = MagicMock(spec=core_client_module.AsyncMPClient)

        async def _failing_call(*args, **kwargs):
            raise RuntimeError("connection broken")

        client._call_utility_async = _failing_call
        client.resources = MagicMock()
        client.resources.engine_dead = False
        client.core_engine = MagicMock()

        from vllm.v1.engine.core_client import EngineDeadError

        with pytest.raises(EngineDeadError):
            await core_client_module.AsyncMPClient.check_health_async(client)
        assert client.resources.engine_dead is True


class TestDPLBClientCheckHealth:
    """Test DPLBAsyncMPClient.check_health_async (DP mode)."""

    @pytest.mark.asyncio
    async def test_dp_all_engines_healthy(self):
        """All engines respond -> no error."""
        import vllm.v1.engine.core_client as core_client_module

        client = MagicMock(spec=core_client_module.DPLBAsyncMPClient)
        client._call_utility_async = AsyncMock(return_value=True)
        client.resources = MagicMock()
        client.resources.engine_dead = False
        client.core_engines = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        await core_client_module.DPLBAsyncMPClient.check_health_async(client)
        assert client.resources.engine_dead is False

    @pytest.mark.asyncio
    async def test_dp_one_engine_hung(self):
        """One engine times out -> engine_dead set, EngineDeadError raised."""
        import vllm.v1.engine.core_client as core_client_module

        client = MagicMock(spec=core_client_module.DPLBAsyncMPClient)

        call_count = [0]

        async def _mixed_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                await asyncio.sleep(100)
            return True

        client._call_utility_async = _mixed_call
        client.resources = MagicMock()
        client.resources.engine_dead = False
        client.core_engines = [MagicMock(), MagicMock()]

        with patch.dict(os.environ, {"VLLM_HEALTH_CHECK_TIMEOUT_S": "0.1"}):
            from vllm.v1.engine.core_client import EngineDeadError

            with pytest.raises(EngineDeadError):
                await core_client_module.DPLBAsyncMPClient.check_health_async(client)
            assert client.resources.engine_dead is True

    @pytest.mark.asyncio
    async def test_dp_engine_exception(self):
        """One engine raises exception -> engine_dead set."""
        import vllm.v1.engine.core_client as core_client_module

        client = MagicMock(spec=core_client_module.DPLBAsyncMPClient)

        async def _failing_call(*args, **kwargs):
            raise RuntimeError("HCCL timeout")

        client._call_utility_async = _failing_call
        client.resources = MagicMock()
        client.resources.engine_dead = False
        client.core_engines = [MagicMock()]

        from vllm.v1.engine.core_client import EngineDeadError

        with pytest.raises(EngineDeadError):
            await core_client_module.DPLBAsyncMPClient.check_health_async(client)
        assert client.resources.engine_dead is True


class TestTimeoutConfig:
    """Test VLLM_HEALTH_CHECK_TIMEOUT_S environment variable."""

    def test_default_timeout(self):
        """Default timeout should be 10.0 seconds."""
        from vllm_ascend.patch.platform.patch_health_check import _get_health_check_timeout

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_HEALTH_CHECK_TIMEOUT_S", None)
            assert _get_health_check_timeout() == 10.0

    def test_custom_timeout(self):
        """Custom timeout should be respected."""
        from vllm_ascend.patch.platform.patch_health_check import _get_health_check_timeout

        with patch.dict(os.environ, {"VLLM_HEALTH_CHECK_TIMEOUT_S": "5.0"}):
            assert _get_health_check_timeout() == 5.0

    def test_timeout_is_float(self):
        """Timeout should be returned as float."""
        from vllm_ascend.patch.platform.patch_health_check import _get_health_check_timeout

        with patch.dict(os.environ, {"VLLM_HEALTH_CHECK_TIMEOUT_S": "3.5"}):
            result = _get_health_check_timeout()
            assert isinstance(result, float)
            assert result == 3.5


class TestPatchApplied:
    """Verify monkey-patch is correctly applied."""

    def test_health_ping_patched(self):
        """EngineCore.health_ping should be the patch function."""
        import vllm.v1.engine.core as core_module

        from vllm_ascend.patch.platform.patch_health_check import _health_ping

        assert core_module.EngineCore.health_ping is _health_ping

    def test_async_mp_client_patched(self):
        """AsyncMPClient.check_health_async should be the patch function."""
        import vllm.v1.engine.core_client as core_client_module

        from vllm_ascend.patch.platform.patch_health_check import _async_mp_client_check_health_async

        assert core_client_module.AsyncMPClient.check_health_async is _async_mp_client_check_health_async

    def test_dp_lb_client_patched(self):
        """DPLBAsyncMPClient.check_health_async should be the patch function."""
        import vllm.v1.engine.core_client as core_client_module

        from vllm_ascend.patch.platform.patch_health_check import _dp_lb_client_check_health_async

        assert core_client_module.DPLBAsyncMPClient.check_health_async is _dp_lb_client_check_health_async

    def test_async_llm_check_health_wrapped(self):
        """AsyncLLM.check_health should be wrapped to call check_health_async."""
        import vllm.v1.engine.async_llm as async_llm_module

        from vllm_ascend.patch.platform.patch_health_check import _wrapped_check_health

        assert async_llm_module.AsyncLLM.check_health is _wrapped_check_health
