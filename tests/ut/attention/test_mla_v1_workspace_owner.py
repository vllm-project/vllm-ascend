# SPDX-License-Identifier: Apache-2.0
"""Keep FIA workspaces owned by the graph that selected them in MLA decode."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    DecodeMLAPreprocessResult,
)
from vllm_ascend.compilation import acl_graph


@pytest.mark.parametrize("draft,prefill", [(False, False), (True, False), (True, True)])
def test_fia_workspace_is_owned_by_selected_graph(draft, prefill):
    def registry(workspace=None):
        return SimpleNamespace(
            workspaces={12: workspace},
            events={12: []},
            attn_params={12: []},
            handles={12: []},
        )

    target = registry()
    draft_decode = registry()
    draft_prefill = registry()
    selected = (draft_prefill if prefill else draft_decode) if draft else target
    workspace = torch.empty(11, dtype=torch.uint8)

    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.num_heads = 2
    impl.num_heads_padded = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.qk_nope_head_dim = 3
    impl.head_padding = 0
    impl.scale = 1.0
    impl.enable_kv_nz = False
    impl.fa_quant_layer = False
    impl.support_fp8_attention = False
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl._v_up_proj = lambda output: output

    decode = AscendMLADecodeMetadata(
        input_positions=torch.arange(12),
        block_table=torch.ones(4, 2, dtype=torch.int32),
        seq_lens=torch.tensor([20] * 4),
        max_seq_lens=20,
        seq_lens_list=[20] * 4,
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=12,
        slot_mapping=torch.arange(12),
        query_start_loc=torch.arange(0, 13, 3),
        seq_lens=torch.tensor([20] * 4),
        seq_lens_cpu=torch.tensor([20] * 4),
        block_tables=torch.ones(4, 2, dtype=torch.int32),
        num_decodes=4,
        num_decode_tokens=12,
        num_prefills=0,
        query_lens=[3] * 4,
        attn_state=AscendAttentionState.SpecDecoding,
        decode=decode,
    )
    metadata.causal = False
    inputs = DecodeMLAPreprocessResult(
        torch.zeros(12, 2, 3),
        torch.zeros(12, 2, 2),
        torch.zeros(2, 1, 4, 3),
        torch.zeros(2, 1, 4, 2),
    )

    def fill_output(*args, **kwargs):
        for tensor in kwargs["out"]:
            tensor.zero_()

    with (
        patch.object(
            mla_v1,
            "_EXTRA_CTX",
            SimpleNamespace(
                capturing=True,
                is_draft_model=draft,
                is_draft_model_prefill=prefill,
            ),
        ),
        patch.object(acl_graph, "_graph_params", target),
        patch.object(acl_graph, "_draft_graph_params", draft_decode),
        patch.object(acl_graph, "_draft_graph_prefill_params", draft_prefill),
        patch.object(mla_v1, "weak_ref_tensors", side_effect=lambda tensor: tensor),
        patch.object(mla_v1.torch_npu.npu, "current_stream", return_value=Mock()),
        patch.object(mla_v1.torch.npu, "ExternalEvent", return_value=Mock()),
        patch.object(mla_v1.torch.npu, "graph_task_group_begin"),
        patch.object(mla_v1.torch.npu, "graph_task_group_end", return_value=Mock()),
        patch.object(
            mla_v1.torch_npu,
            "npu_fused_infer_attention_score_v2",
            SimpleNamespace(out=fill_output),
        ),
        patch.object(
            mla_v1.torch_npu,
            "_npu_fused_infer_attention_score_v2_get_max_workspace",
            return_value=workspace,
            create=True,
        ) as allocate,
    ):
        impl._forward_decode(inputs, 4, metadata)
        assert selected.workspaces[12] is workspace
        if draft:
            assert target.workspaces[12] is None
        if prefill:
            assert draft_decode.workspaces[12] is None
        elif draft:
            assert draft_prefill.workspaces[12] is None

        impl._forward_decode(inputs, 4, metadata)
        assert allocate.call_count == 1
