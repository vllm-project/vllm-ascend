# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply Ascend-specific Mamba chunk-boundary handling.

Chunk boundaries and consumer verifier padding are handled by vLLM.

Producer and standalone instances use the same EAGLE block-drop policy as
the cache coordinator; other instances retain the configured upstream policy.
"""

import functools
import inspect

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    _skips_eagle_block_drop,
)

_EXPECTED_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)

_original_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split


@functools.wraps(_original_mamba_block_aligned_split)
def _mamba_block_aligned_split(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Delegate boundary selection to vLLM with the coordinator role policy.

    Producer/standalone split decisions must follow the coordinator's EAGLE
    block-drop exemption. Temporarily clear the relevant bit while calling the
    original method, then restore it. Scheduling is single-threaded per
    scheduler instance. Consumers and kv_both retain the upstream drop policy.
    """
    kv_transfer_config = self.vllm_config.kv_transfer_config
    if not _skips_eagle_block_drop(kv_transfer_config):
        return _original_mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
    # vLLM 0.28.x and newer revisions use different names.  Some
    # transitional scheduler implementations expose both and different
    # wrapper layers consult different attributes, so clear every
    # attribute that exists and restore all of them after the call.
    drop_attrs = tuple(name for name in ("use_eagle", "use_eagle_block_drop") if hasattr(self, name))
    original_drop_values = {name: getattr(self, name) for name in drop_attrs}
    for name in drop_attrs:
        setattr(self, name, False)
    try:
        return _original_mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
    finally:
        for name, value in original_drop_values.items():
            setattr(self, name, value)


current_parameters = tuple(inspect.signature(_original_mamba_block_aligned_split).parameters)
if current_parameters != _EXPECTED_PARAMETERS:
    raise RuntimeError(
        "Cannot apply the PD consumer Mamba split patch: unexpected "
        "Scheduler._mamba_block_aligned_split signature "
        f"{current_parameters}"
    )

Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
