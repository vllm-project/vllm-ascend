# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_ascend.distributed.kv_transfer.utils.state_layout import mamba_conv_projection_widths


def test_shared_projection_geometry_preserves_v2_mamba_and_gdn_storage_order():
    assert mamba_conv_projection_widths(MambaAttentionBackendEnum.MAMBA1, 16, None) == (16,)
    assert mamba_conv_projection_widths(MambaAttentionBackendEnum.MAMBA2, 16, (2, 4, 4)) == (8, 4, 4)
    assert mamba_conv_projection_widths(MambaAttentionBackendEnum.GDN_ATTN, 16, (2, 4, 4)) == (4, 4, 8)
    # GLM KDA's q/k/v widths are equal; speculative temporal rows have no
    # effect on projection boundaries or TP head ownership.
    assert mamba_conv_projection_widths(MambaAttentionBackendEnum.GDN_ATTN, 12, (2, 2, 2)) == (4, 4, 4)


@pytest.mark.parametrize("shape,width", [(None, 12), ((2,), 12), ((2, 4, 4), 8), ((2, 4, 4), 11)])
def test_shared_projection_geometry_rejects_incomplete_or_indivisible_state(shape, width):
    with pytest.raises(ValueError):
        mamba_conv_projection_widths(MambaAttentionBackendEnum.GDN_ATTN, width, shape)


def test_shared_projection_geometry_rejects_unimplemented_mamba_type():
    with pytest.raises(NotImplementedError):
        mamba_conv_projection_widths(MambaAttentionBackendEnum.SHORT_CONV, 12, (2, 2, 2))
