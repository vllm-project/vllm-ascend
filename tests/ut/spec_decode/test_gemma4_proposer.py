# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.spec_decode as spec_decode
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


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


def test_gemma_config_override_delegates_to_vllm(monkeypatch):
    hf_config = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="gemma4_assistant",
    )
    expected = object()
    original_override = MagicMock(return_value=expected)
    monkeypatch.setattr(
        patch_speculative_config,
        "_orig_hf_config_override",
        original_override,
    )

    result = patch_speculative_config.hf_config_override(hf_config)

    assert result is expected
    original_override.assert_called_once_with(hf_config)


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

    graph_metadata = [object(), object()]
    for builder, graph_item in zip(builders, graph_metadata):
        builder.build_for_graph_capture.return_value = graph_item
    graph_result = proposer._build_multi_group_graph_capture_metadata(
        common_metadata,
        draft_index=0,
    )

    assert graph_result == {
        "draft.attn.0": graph_metadata[0],
        "draft.attn.1": graph_metadata[1],
    }
    for gid, builder in enumerate(builders):
        call_args = builder.build_for_graph_capture.call_args.args
        assert torch.equal(call_args[0].block_table_tensor, block_tables[gid][:2])
        assert call_args[1] == AscendAttentionState.SpecDecoding


def test_attn_update_uses_only_active_group_block_table():
    proposer = AscendGemma4Proposer.__new__(AscendGemma4Proposer)
    proposer._per_group_block_tables = {1: torch.arange(12).view(3, 4)}
    common_metadata = SimpleNamespace(
        num_reqs=2,
        block_table_tensor=torch.zeros(2, 4),
    )
    attn_group = SimpleNamespace(kv_cache_group_id=1)
    expected = object()

    with patch.object(
        AscendSpecDecodeBaseProposer,
        "attn_update_stack_num_spec_norm",
        return_value=expected,
    ) as base_update:
        result = proposer.attn_update_stack_num_spec_norm(
            1,
            common_metadata,
            2,
            2,
            torch.tensor([3, 4]),
            "none",
            attn_group=attn_group,
        )

    assert result is expected
    group_common_metadata = base_update.call_args.args[1]
    assert group_common_metadata is not common_metadata
    assert torch.equal(
        group_common_metadata.block_table_tensor,
        proposer._per_group_block_tables[1][:2],
    )
    assert base_update.call_args.kwargs["attn_group"] is attn_group
