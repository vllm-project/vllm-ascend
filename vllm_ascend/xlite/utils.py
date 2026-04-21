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
"""Utility functions for xlite."""

from typing import Any

_MISSING = object()
"""Unique sentinel for missing attributes in this module."""


def _get_nested_attr(obj: Any, /, *attrs: str, default: Any = None) -> Any:
    """Get/collect a nested attribute from an object.

    The attribute path is specified as a sequence of attribute names. If any attribute in the path is missing, the
    function returns the specified default value (which is None by default).

    Args:
        obj (Any): Root object.
        *attrs (str): Sequence of attribute names to traverse.
        default (Any, keyword-only, default=None): Default value to return if any attribute is missing.

    Returns:
        Any: The resolved nested attribute.
    """
    current = obj
    for attr in attrs:
        if (current := getattr(current, attr, _MISSING)) is _MISSING:
            return default
    return current
