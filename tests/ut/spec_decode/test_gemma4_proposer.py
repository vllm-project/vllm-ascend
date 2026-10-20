#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.spec_decode as spec_decode
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer


def test_routes_gemma4_mtp_to_ascend_proposer():
    speculative_config = MagicMock()
    speculative_config.use_gemma4_mtp.return_value = True
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    expected = object()
    with patch.object(
        spec_decode,
        "AscendGemma4Proposer",
        return_value=expected,
    ) as proposer_cls:
        result = spec_decode.get_spec_decode_method(
            "mtp",
            vllm_config,
            device="npu",
            runner=object(),
        )
    assert result is expected
    proposer_cls.assert_called_once()


def test_sync_kv_sharing_target_to_impl():
    proposer = AscendGemma4Proposer.__new__(AscendGemma4Proposer)
    proposer.vllm_config = MagicMock()
    proposer._draft_attn_layer_names = {"draft.attn"}
    impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    attn = SimpleNamespace(
        impl=impl,
        kv_sharing_target_layer_name="target.attn",
    )
    with patch(
        "vllm_ascend.spec_decode.gemma4_proposer.get_layers_from_vllm_config",
        return_value={"draft.attn": attn},
    ):
        proposer._sync_kv_sharing_target_to_impl()
    assert impl.kv_sharing_target_layer_name == "target.attn"


def test_keeps_draft_lm_head():
    proposer = AscendGemma4Proposer.__new__(AscendGemma4Proposer)
    draft_lm_head = object()
    proposer.model = SimpleNamespace(lm_head=draft_lm_head)
    proposer.method = "mtp"
    proposer.use_cuda_graph = False
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                has_full_cudagraphs=lambda: False,
            )
        ),
    )
    proposer._maybe_share_lm_head(SimpleNamespace(lm_head=object()))
    assert proposer.model.lm_head is draft_lm_head


def test_build_draft_attn_metadata_uses_per_group_block_tables():
    proposer = AscendGemma4Proposer.__new__(AscendGemma4Proposer)
    block_tables = {
        0: torch.arange(12).view(3, 4),
        1: torch.arange(12, 24).view(3, 4),
    }
    proposer._per_group_block_tables = block_tables
    proposer.runner = SimpleNamespace(get_model=MagicMock(return_value=object()))
    metadata = [
        SimpleNamespace(attn_state=None, causal=True),
        SimpleNamespace(attn_state=None, causal=False, attn_mask=object()),
    ]
    builders = [MagicMock(), MagicMock()]
    for builder, group_metadata in zip(builders, metadata):
        builder.build.return_value = group_metadata
    proposer.draft_attn_groups = [
        SimpleNamespace(
            kv_cache_group_id=gid,
            layer_names=[f"draft.attn.{gid}"],
            get_metadata_builder=MagicMock(return_value=builders[gid]),
        )
        for gid in range(2)
    ]
    common_metadata = SimpleNamespace(
        num_reqs=2,
        block_table_tensor=torch.zeros(2, 4),
    )
    multi_steps, first_metadata = proposer.build_draft_attn_metadata(
        common_metadata,
        num_input_tokens=2,
        num_actual_tokens=2,
    )
    assert first_metadata is metadata[0]
    assert multi_steps == [
        {
            "draft.attn.0": metadata[0],
            "draft.attn.1": metadata[1],
        }
    ]
    for gid, builder in enumerate(builders):
        group_common_metadata = builder.build.call_args.args[1]
        assert group_common_metadata is not common_metadata
        assert torch.equal(
            group_common_metadata.block_table_tensor,
            block_tables[gid][:2],
        )
        assert metadata[gid].attn_state == AscendAttentionState.SpecDecoding
    assert metadata[1].attn_mask is None
