# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the graph-external target preparation helpers."""

from vllm_ascend.target_prep_masks import (
    _HOST_MASK_WRITES as _HOST_MASK_WRITES,
)
from vllm_ascend.target_prep_masks import (
    _enabled as _enabled,
)
from vllm_ascend.target_prep_masks import (
    _scope_owner as _scope_owner,
)
from vllm_ascend.target_prep_masks import (
    mask_write_host_scope as mask_write_host_scope,
)
from vllm_ascend.target_prep_masks import (
    try_prepare_positions_with_padding as try_prepare_positions_with_padding,
)
from vllm_ascend.target_prep_masks import (
    try_write_prefix_mask as try_write_prefix_mask,
)
