#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import inspect

import vllm.transformers_utils.utils as _target_utils

from modelscope.hub.api import HubApi as _HubApi

# Modelscope v1.38.0 removed the `revision` parameter from
# LegacyHubApi.get_model_files(). Detect the signature once at
# import time so the patch is zero-overhead at runtime.
try:
    _sig = inspect.signature(_HubApi.get_model_files)
    _HAS_REVISION = "revision" in _sig.parameters
except (ValueError, TypeError):
    _HAS_REVISION = True  # assume old API on inspect failure


def _patched_modelscope_list_repo_files(
    repo_id: str,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    kwargs = {"model_id": repo_id, "recursive": True}
    if _HAS_REVISION:
        kwargs["revision"] = revision
    raw_files = api.get_model_files(**kwargs)
    return [
        f.get("Path", f.get("path"))
        for f in raw_files
        if f.get("Type", f.get("type")) == "blob"
    ]


_target_utils.modelscope_list_repo_files = _patched_modelscope_list_repo_files
