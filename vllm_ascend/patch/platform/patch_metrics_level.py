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

"""Install ``POST /set_metrics_level`` and the EngineCore method it calls.

The HTTP route is attached when the API server registers its routers. The
EngineCore method runs in the engine process: it sets that process's level,
then ``collective_rpc`` reaches every worker, including pipeline stages that
do not run the wave-head file reload.
"""

import sys

_PATCHED = False


def _install_engine_method() -> None:
    from vllm.v1.engine.core import EngineCore

    existing = getattr(EngineCore, "set_metrics_level", None)
    if getattr(existing, "_ascend_metrics_level", False):
        return

    def set_metrics_level(self, level: str) -> str:
        from vllm_ascend.observability.ms_metrics.control import apply_engine_metrics_level

        return apply_engine_metrics_level(self, level)

    set_metrics_level._ascend_metrics_level = True  # type: ignore[attr-defined]
    EngineCore.set_metrics_level = set_metrics_level  # type: ignore[method-assign]


def _install_route_hook() -> None:
    from vllm_ascend.observability.runtime_config.dist import _is_distributed_worker_process

    # Workers and EngineCore processes do not serve HTTP. Skip the API import there.
    if _is_distributed_worker_process():
        return

    import vllm.entrypoints.serve as serve_mod

    current = serve_mod.register_vllm_serve_api_routers
    if getattr(current, "_ascend_metrics_level", False):
        return

    def register_vllm_serve_api_routers(app):
        current(app)
        from vllm_ascend.observability.ms_metrics.api import attach_router

        attach_router(app)

    register_vllm_serve_api_routers._ascend_metrics_level = True  # type: ignore[attr-defined]
    serve_mod.register_vllm_serve_api_routers = register_vllm_serve_api_routers

    # `routers.py` binds the name at import. Replace that binding too when the
    # module is already loaded.
    routers_mod = sys.modules.get("vllm.entrypoints.launchers.api_server.routers")
    if routers_mod is not None and getattr(routers_mod, "register_vllm_serve_api_routers", None) is current:
        routers_mod.register_vllm_serve_api_routers = register_vllm_serve_api_routers


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _install_engine_method()
    _install_route_hook()


_apply_patch()
