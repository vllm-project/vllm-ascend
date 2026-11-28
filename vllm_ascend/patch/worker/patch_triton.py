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
"""Triton swaps that the kernel dispatcher cannot express.

The kernel overrides that used to live in this file are registered with
the official vLLM Triton kernel dispatcher --
``vllm_ascend/triton_kernel_overrides.py``. This file keeps only the
``triton.next_power_of_2`` host-helper shim: upstream kernels
(fused_gdn_prefill_post_conv, fused_recurrent, packed_qk_rope, ...)
evaluate it inside their ``@triton.jit`` bodies at kernel-compile time,
and the dispatcher's ``KernelOverride`` is not a compile-time-recognizable
callable. The attribute must stay a plain swap, with a value packaged for
the installed Triton.

Retirement: remove this file once Triton-Ascend ships a
``triton.next_power_of_2`` that accepts ``tl.constexpr`` arguments and
passes JIT dependency tracking on every supported release, or vLLM
registers the host helper centrally in ``vllm/triton_utils``.
"""

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import next_power_of_2


# TODO: keep this helper beside the swap it serves, not in a utils
# module: the swap below changes global Triton state at import time,
# and the helper has no other caller. Not every supported
# Triton-Ascend release ships a next_power_of_2 that kernel bodies
# can call with tl.constexpr arguments (older ones ship a plain
# function), so this shim must stay until they all do. The shim is
# all this file holds, so its removal removes the file. Removal
# checklist: delete this file, drop its import and the naming comment
# in patch/worker/__init__.py, close its manifest entry in
# patch/__init__.py, and re-check the 310P placeholder injection in
# _310p/worker/v2/sample/dispatch.py.
def _next_power_of_2(n):
    # vLLM #58651's packed_qk_rope kernel evaluates
    # triton.next_power_of_2(rotary_dim) with a tl.constexpr argument.
    # The host helper only understands plain ints, so unwrap first.
    if HAS_TRITON and isinstance(n, tl.constexpr):
        n = n.value
    return next_power_of_2(n)


# Newer Triton tracks JIT kernel dependencies and rejects plain Python
# callables. Keep the host helper callable from constexpr code as well.
if HAS_TRITON and hasattr(triton, "constexpr_function"):
    triton.next_power_of_2 = triton.constexpr_function(next_power_of_2)
else:
    triton.next_power_of_2 = _next_power_of_2
