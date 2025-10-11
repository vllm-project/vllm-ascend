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
import re
import socket

from vllm.logger import logger


def find_free_port():
    """
    Finds a free port on the local machine.

    Returns:
    - A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
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
        logger.warning(
            f'The path prefix {path_prefix} contains illegal characters.')
        return False

    if path_prefix.startswith('/') or path_prefix.startswith('\\'):
        if not os.path.exists(os.path.dirname(path_prefix)):
            logger.warning(
                f'The directory for the path prefix {os.path.dirname(path_prefix)} does not exist.'
            )
            return False
    else:
        if not os.path.exists(os.path.dirname(os.path.abspath(path_prefix))):
            logger.warning(
                f'The directory for the path prefix {os.path.dirname(os.path.abspath(path_prefix))} does not exist.'
            )
            return False
    return True