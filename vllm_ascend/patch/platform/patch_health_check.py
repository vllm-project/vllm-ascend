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
"""Patch: Add health ping to detect hung EngineCore processes on Ascend NPU.

Problem
-------
On Ascend NPU, HCCL collective communication operations can hang indefinitely
without crashing the process. The upstream vLLM ``check_health`` only inspects
the ``errored`` flag (set when the EngineCore process dies via ``is_alive()``).
A hung-but-alive EngineCore will keep returning HTTP 200 on ``/health``,
causing load balancers to continue routing requests to a dead instance.

Solution
--------
This patch monkey-patches four upstream vllm classes:

1. ``EngineCore.health_ping()`` — a trivial method that returns ``True``.
   If the EngineCore event loop is blocked (e.g. by an HCCL hang), the
   caller's ``asyncio.wait_for`` will timeout.

2. ``AsyncMPClient.check_health_async()`` — sends a health ping with a
   configurable timeout (``VLLM_HEALTH_CHECK_TIMEOUT_S``, default 10s).
   On timeout, sets ``engine_dead = True`` and raises ``EngineDeadError``.

3. ``DPLBAsyncMPClient.check_health_async()`` — same as above but pings
   ALL EngineCores concurrently (DP mode), fails if any is unresponsive.

4. ``AsyncLLM.check_health()`` — wrapped to call ``check_health_async()``
   after the original ``errored`` check.

No vllm source files are modified. All changes are applied via runtime
monkey-patching, consistent with the existing vllm-ascend patch mechanism.
"""

import asyncio
import os

import vllm.v1.engine.async_llm as async_llm_module
import vllm.v1.engine.core as core_module
import vllm.v1.engine.core_client as core_client_module
from vllm.logger import logger


def _get_health_check_timeout() -> float:
    return float(os.environ.get("VLLM_HEALTH_CHECK_TIMEOUT_S", "10.0"))


def _health_ping(self) -> bool:
    """A trivial health check method.

    If the EngineCore event loop is blocked (e.g. HCCL hang), this method
    cannot be called, causing the caller's ``asyncio.wait_for`` to timeout.
    """
    return True


async def _async_mp_client_check_health_async(self) -> None:
    from vllm.v1.engine.core_client import EngineDeadError

    timeout = _get_health_check_timeout()
    logger.debug("Sending health ping to EngineCore (timeout=%ss)", timeout)
    try:
        await asyncio.wait_for(
            self._call_utility_async("health_ping", engine=self.core_engine),
            timeout=timeout,
        )
        logger.debug("Health ping succeeded")
    except asyncio.TimeoutError:
        logger.error("EngineCore did not respond to health ping within %ss - marking engine as dead", timeout)
        self.resources.engine_dead = True
        raise EngineDeadError(f"EngineCore did not respond to health ping within {timeout}s")
    except EngineDeadError:
        raise
    except Exception:
        logger.exception("Health ping to EngineCore failed")
        self.resources.engine_dead = True
        raise EngineDeadError("Health ping to EngineCore failed")


async def _dp_lb_client_check_health_async(self) -> None:
    from vllm.v1.engine.core_client import EngineDeadError

    timeout = _get_health_check_timeout()
    num_engines = len(self.core_engines)
    logger.debug("Sending health ping to %d EngineCores (timeout=%ss)", num_engines, timeout)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *[self._call_utility_async("health_ping", engine=engine) for engine in self.core_engines],
                return_exceptions=True,
            ),
            timeout=timeout,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Engine %d health ping failed: %s", i, result)
                self.resources.engine_dead = True
                raise EngineDeadError(f"Engine {i} health ping failed: {result}")
        logger.debug("Health ping succeeded for all %d engines", num_engines)
    except asyncio.TimeoutError:
        logger.error(
            "One or more EngineCores did not respond to health ping within %ss - marking engine as dead", timeout
        )
        self.resources.engine_dead = True
        raise EngineDeadError(f"One or more EngineCores did not respond to health ping within {timeout}s")
    except EngineDeadError:
        raise
    except Exception:
        logger.exception("Health ping to EngineCores failed")
        self.resources.engine_dead = True
        raise EngineDeadError("Health ping to EngineCores failed")


_original_check_health = async_llm_module.AsyncLLM.check_health


async def _wrapped_check_health(self) -> None:
    await _original_check_health(self)
    await self.engine_core.check_health_async()


# Monkey-patch: inject health_ping into EngineCore
core_module.EngineCore.health_ping = _health_ping

# Monkey-patch: inject check_health_async into AsyncMPClient and DPLBAsyncMPClient
core_client_module.AsyncMPClient.check_health_async = _async_mp_client_check_health_async
core_client_module.DPLBAsyncMPClient.check_health_async = _dp_lb_client_check_health_async

# Monkey-patch: wrap AsyncLLM.check_health to call check_health_async
async_llm_module.AsyncLLM.check_health = _wrapped_check_health

logger.info("patch_health_check: applied health ping (timeout=%ss)", _get_health_check_timeout())
