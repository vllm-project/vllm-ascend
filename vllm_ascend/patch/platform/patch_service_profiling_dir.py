#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Env-driven ``--profiler-config`` for the Ascend PD deployment.

Deployments with a fixed ``vllm serve`` command line cannot append
``--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`` without
editing their launch scripts. This patch takes the same settings from the
profiling directory environment variable defined in ``vllm_ascend.envs``
instead, and is a no-op when that variable is unset.

Why ``EngineArgs.from_cli_args`` is the hook:
    Every entry point funnels through it, and the ``Namespace`` it receives is
    the very object ``build_app`` stores as ``app.state.args`` (api_server
    entry: ``AsyncEngineArgs.from_cli_args(args)`` ... ``build_app(args, ...)``).
    Its ``.profiler_config`` is what the profiling router inspects in
    ``attach_router``, and it is also what ``EngineArgs.create_vllm_config``
    hands to ``VllmConfig(profiler_config=self.profiler_config)``, which then
    reaches every worker, where ``AscendWorker.profile`` builds the
    ``TorchNPUProfilerWrapper`` (torch_npu.profiler, ProfilerLevel.Level1,
    AiCMetrics.PipeUtilization). Mutating that one object therefore covers both
    halves of a profiling session:

      * router registration -- ``attach_profile_router`` calls
        ``app.include_router`` only when ``args.profiler_config.profiler`` is
        set, which is what makes ``POST /start_profile`` / ``/stop_profile``
        exist at all;
      * worker side -- the same object is already inside the engine config.

    Platform hooks run earlier (``pre_register_and_update`` fires from
    ``create_parser`` / ``EngineArgs.__post_init__``), so no earlier hook sees
    a parsed Namespace. This is the earliest point where the config object
    exists, and it is still before ``build_app``.

The field is mutated in place rather than replaced: ``from_cli_args`` copies
attributes off the Namespace into ``EngineArgs(**{...})`` without cloning, so
keeping the identity of ``args.profiler_config`` is what guarantees the API
process and the workers observe the same settings. ``ignore_frontend`` stays
True because the trace we need is the device-side one carrying the MegaMoe
operator rows; the frontend CPU trace would only add a second full capture per
request path.

Nothing happens unless the environment variable is set, so a normal launch is
untouched.
"""

import os

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import logger

from vllm_ascend import envs

PATCH_FLAG = "_vllm_ascend_profiler_dir_patched"


def _apply_profiler_dir(profiler_config) -> bool:
    """Fill an unset ProfilerConfig from the environment. True if changed."""
    directory = (envs.VLLM_ASCEND_PROFILER_DIR or "").strip()
    if not directory:
        return False
    # An explicit --profiler-config on the command line always wins: that is
    # the documented path, and this patch exists only because the deployment
    # scripts cannot pass it.
    if getattr(profiler_config, "profiler", None) is not None:
        return False
    if getattr(profiler_config, "torch_profiler_dir", ""):
        return False
    profiler_config.profiler = "torch"
    profiler_config.torch_profiler_dir = os.path.abspath(os.path.expanduser(directory))
    # Keep the capture on the workers only (device trace with the operator
    # rows); the API-process CPU trace is not part of this measurement.
    profiler_config.ignore_frontend = True
    logger.info(
        "Profiler enabled from VLLM_ASCEND_PROFILER_DIR: profiler=torch dir=%s ignore_frontend=True. "
        "POST /start_profile and /stop_profile are now available on this server.",
        profiler_config.torch_profiler_dir,
    )
    return True


def _maybe_enable_profiling(args) -> None:
    """Accept both the real Namespace and anything that lacks the field."""
    if not hasattr(args, "profiler_config"):
        return
    profiler_config = args.profiler_config
    if profiler_config is None:
        from vllm.config import ProfilerConfig

        profiler_config = ProfilerConfig()
        args.profiler_config = profiler_config
    _apply_profiler_dir(profiler_config)


def _install() -> None:
    """Wrap ``EngineArgs.from_cli_args`` exactly once, idempotently.

    The pristine original is parked on the class instead of in a module
    global: reloading this module (or importing it twice through different
    paths) would otherwise make the new copy's ``_original`` point at the
    already-wrapped function and recurse into itself forever.
    """
    if getattr(EngineArgs, PATCH_FLAG, None):
        return
    attr = EngineArgs.__dict__["from_cli_args"]
    if isinstance(attr, (classmethod, staticmethod)):
        original, kind = attr.__func__, "staticmethod" if isinstance(attr, staticmethod) else "classmethod"
    else:  # plain function object; keep the same shape back
        original, kind = attr, "function"
    EngineArgs._ascend_unpatched_from_cli_args = original

    def _from_cli_args(cls, args, *rest, **kwargs):
        _maybe_enable_profiling(args)
        return original(cls, args, *rest, **kwargs)

    if kind == "classmethod":
        EngineArgs.from_cli_args = classmethod(_from_cli_args)
    elif kind == "staticmethod":
        EngineArgs.from_cli_args = staticmethod(lambda args, *a, **k: _from_cli_args(EngineArgs, args, *a, **k))
    else:
        EngineArgs.from_cli_args = _from_cli_args
    setattr(EngineArgs, PATCH_FLAG, True)


_install()
