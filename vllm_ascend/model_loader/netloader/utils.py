#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import socket
from typing import Any

import regex as re
from vllm.logger import logger

_DEFAULT_LOADER_EXTRA_KEYS = (
    "enable_multithread_load",
    "num_threads",
    "enable_weights_track",
)


def disk_fallback_loader_extra_config(existing_extra: Any) -> dict[str, Any]:
    """Extra config DefaultModelLoader accepts on a local-disk fallback.

    Netloader extras carry SOURCE / LISTEN_PORT, which DefaultModelLoader
    rejects, so the fallback used to drop the whole dict and silently lose the
    user's multithread-load settings. Keep the keys DefaultModelLoader knows
    and drop the rest. Defaults are left to vLLM.
    """
    kept: dict[str, Any] = {}
    if isinstance(existing_extra, dict):
        for key in _DEFAULT_LOADER_EXTRA_KEYS:
            if key in existing_extra:
                kept[key] = existing_extra[key]
    return kept


def find_free_port():
    """
    Finds a free port on the local machine.

    Returns:
    - A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def is_valid_path_prefix(path_prefix):
    """
    Checks if the provided path prefix is valid.

    Parameters:
    - path_prefix: The path prefix to validate.

    Returns:
    - True if the path prefix is valid, otherwise False.
    """
    if not path_prefix:
        return False

    if re.search(r'[<>:"|?*]', path_prefix):
        logger.warning("The path prefix %s contains illegal characters.", path_prefix)
        return False

    if path_prefix.startswith("/") or path_prefix.startswith("\\"):
        if not os.path.exists(os.path.dirname(path_prefix)):
            logger.warning("The directory for the path prefix %s does not exist.", os.path.dirname(path_prefix))
            return False
    else:
        if not os.path.exists(os.path.dirname(os.path.abspath(path_prefix))):
            logger.warning(
                "The directory for the path prefix %s does not exist.", os.path.dirname(os.path.abspath(path_prefix))
            )
            return False
    return True
