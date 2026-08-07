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

"""Install the breakable ACL graph decorator before attention modules load.

Modules such as ``attention`` and ``sparse_attn_indexer`` use
``from vllm.compilation.breakable_cudagraph import
eager_break_during_capture``. This creates a local binding at import time, and
``attention`` immediately uses that binding to decorate and register its
custom ops. Replace the function in the original module before global patches
or model-runner imports so those bindings resolve to the Ascend implementation.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from vllm.compilation import breakable_cudagraph

F = TypeVar("F", bound=Callable[..., Any])


def eager_break_during_capture(fn: F) -> F:
    """Lazily delegate decoration to the ACL graph implementation."""
    from vllm_ascend.compilation.breakable_aclgraph import (
        eager_break_during_capture as acl_eager_break_during_capture,
    )

    return acl_eager_break_during_capture(fn)


breakable_cudagraph.eager_break_during_capture = eager_break_during_capture
