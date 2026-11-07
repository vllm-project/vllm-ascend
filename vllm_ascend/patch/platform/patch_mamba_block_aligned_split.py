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

A newly admitted KV-consumer request can have one prompt token left after the
external cache hit. The waiting scheduler pads that token to a ``1 + K``
speculative verifier window before Mamba alignment is applied. If the window
starts mid-block, the alignment split can shorten its physical width while the
request still advertises all ``K`` speculative placeholders.

Decode consumers preserve the complete verifier window. Sparse index-kpool
producers align against the resolved common cache-group boundary because their
physical indexer-state block is smaller than the Mamba checkpoint interval.
Other models retain the upstream behavior.

On a pure PD prefill producer or a standalone instance the EAGLE one-block
backoff of the last cacheable position is suppressed: matched content-hash
blocks are always verified prompt blocks there, while the backoff would leave
the final full mamba-align state page unmaterialized across the chunk boundary
(copy-on-write in MambaManager.allocate_new_blocks), pinning hybrid prefix
hits one full page (or entirely) short.
"""

import copy
import functools
import inspect

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    _skips_eagle_block_drop,
)
from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
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
    """Use upstream prefill boundaries while preserving the final PD window."""
    kv_transfer_config = self.vllm_config.kv_transfer_config
    # A consumer only needs the unsplit verifier window after some prefix has
    # already been computed locally or loaded from the connector.  ``kv_both``
    # also handles cold prefills; bypassing alignment for those requests means
    # no reusable Mamba state is ever materialized, so neither HBM nor the KV
    # pool can cache the prefix.
    has_computed_prefix = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens > 0
    if kv_transfer_config is not None and kv_transfer_config.is_kv_consumer and has_computed_prefix:
        return num_new_tokens

    # Sparse indexer groups have a smaller physical block than their common
    # state boundary. Give upstream a scheduler-local config view without
    # changing the configuration shared with other components.
    cache_config = self.cache_config
    sparse_index_kpool = _get_sparse_index_kpool(self.vllm_config.model_config) is not None
    # vLLM 0.28.x and newer revisions use different names.  Some
    # transitional scheduler implementations expose both and different
    # wrapper layers consult different attributes, so clear every
    # attribute that exists and restore all of them after the call.
    drop_attrs = (
        tuple(name for name in ("use_eagle", "use_eagle_block_drop") if hasattr(self, name))
        if _skips_eagle_block_drop(kv_transfer_config)
        else ()
    )
    original_drop_values = {name: getattr(self, name) for name in drop_attrs}
    for name in drop_attrs:
        setattr(self, name, False)
    try:
        if sparse_index_kpool:
            self.cache_config = copy.copy(cache_config)
            self.cache_config.block_size = self.block_size
        return _original_mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
    finally:
        if sparse_index_kpool:
            self.cache_config = cache_config
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
