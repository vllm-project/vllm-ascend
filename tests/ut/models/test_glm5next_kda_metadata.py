# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend._310p.ops.gdn_attn_builder_310 import GDNAttentionMetadataBuilder310
from vllm_ascend.models.glm5next import kda_metadata
from vllm_ascend.models.glm5next.kda_metadata import Glm5NextKDAMetadataBuilder
from vllm_ascend.ops import gdn_attn_builder
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder


@pytest.mark.parametrize(
    "builder_cls",
    [
        AscendGDNAttentionMetadataBuilder,
        AscendGDNHostMetadataBuilder,
        GDNAttentionMetadataBuilder310,
        Glm5NextKDAMetadataBuilder,
    ],
)
@pytest.mark.parametrize("device_type", ["cpu", "npu"])
@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_fused_aligned_lookup_is_only_selected_by_glm_kda(builder_cls, device_type, cache_mode):
    builder = builder_cls.__new__(builder_cls)
    builder.kv_cache_spec = SimpleNamespace(block_size=16, num_speculative_blocks=3)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode=cache_mode))
    common = SimpleNamespace(
        block_table_tensor=SimpleNamespace(device=SimpleNamespace(type=device_type)),
        seq_lens=torch.tensor([17], dtype=torch.int32),
    )
    with (
        patch.object(gdn_attn_builder, "mamba_get_block_table_tensor", return_value=torch.tensor([1])) as original,
        patch.object(kda_metadata, "aligned_mamba_block_table", return_value=torch.tensor([2])) as fused,
    ):
        result = builder._get_mamba_block_table_tensor(common)

    use_fused = builder_cls is Glm5NextKDAMetadataBuilder and device_type == "npu" and cache_mode == "align"
    assert result.item() == (2 if use_fused else 1)
    if use_fused:
        fused.assert_called_once_with(common.block_table_tensor, common.seq_lens, 16, 3)
        original.assert_not_called()
    else:
        original.assert_called_once_with(common.block_table_tensor, common.seq_lens, builder.kv_cache_spec, cache_mode)
        fused.assert_not_called()
    assert (builder_cls is Glm5NextKDAMetadataBuilder) == builder._USE_SPEC_SEQUENCE_VIEWS
