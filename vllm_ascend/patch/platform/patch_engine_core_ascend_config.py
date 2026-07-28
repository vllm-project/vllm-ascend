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
"""
Ensure the AscendConfig singleton is initialized inside every EngineCore
process.

``NPUPlatform.check_and_update_config`` (which calls ``init_ascend_config``)
only runs in the process that builds ``VllmConfig``. EngineCore subprocesses
spawn mode, or any launcher that ships a pickled ``VllmConfig`` to the child
never re-run it, so the process-global ``AscendConfig`` singleton is missing
there. Components instantiated during ``EngineCore.__init__`` (e.g. the
scheduler's score encoder cache manager) rely on ``get_ascend_config()`` and
would fail with "Ascend config is not initialized".

This module is imported as part of the global platform patches, which vLLM
loads in engine-core subprocesses through the general plugin entry points
before any ``EngineCore`` is created. Wrapping ``EngineCore.__init__`` here
therefore guarantees the singleton exists in every DP rank / engine-core
process regardless of the multiprocessing start method.
"""


from vllm.v1.engine.core import EngineCore

_ORIGINAL_ENGINE_CORE_INIT = EngineCore.__init__


def _patched_engine_core_init(self, vllm_config, *args, **kwargs):
    from vllm_ascend.ascend_config import init_ascend_config

    # Idempotent: returns the cached singleton if it was already initialized
    # for this vllm_config in this process.
    init_ascend_config(vllm_config)
    return _ORIGINAL_ENGINE_CORE_INIT(self, vllm_config, *args, **kwargs)


if not getattr(EngineCore.__init__, "_vllm_ascend_config_patched", False):
    _patched_engine_core_init._vllm_ascend_config_patched = True
    EngineCore.__init__ = _patched_engine_core_init
