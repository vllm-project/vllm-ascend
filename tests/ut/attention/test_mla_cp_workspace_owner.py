# SPDX-License-Identifier: Apache-2.0
"""Keep FIA workspaces owned by each target/draft graph and reused within that graph."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel import mla_cp
from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata, DecodeMLAPreprocessResult
from vllm_ascend.compilation import updatable_graph


@pytest.mark.parametrize("draft,prefill", [(False, False), (True, False), (True, True)])
def test_fia_workspace_is_owned_and_reused_by_selected_graph(draft, prefill):
    def registry():
        graph = SimpleNamespace(capture_resources={})
        graph.get_capture_resource = lambda key, factory, use_max_workspace=False: (
            updatable_graph.UpdatableGraph.get_capture_resource(graph, key, factory, use_max_workspace)
        )
        graph.register_task = lambda operation, kwargs, provider: operation(**kwargs)
        return graph

    target_marker = torch.empty(7, dtype=torch.uint8)
    target = registry()
    if draft:
        target.capture_resources["other"] = target_marker
    draft_decode = registry()
    draft_prefill = registry()
    selected = (draft_prefill if prefill else draft_decode) if draft else target
    workspace = torch.empty(11, dtype=torch.uint8)

    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.dcp_size = 8
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.scale = 1.0
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl._merge_dcp_attention_output = lambda output, lse: output
    impl._v_up_proj_batch_major = lambda output: output

    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(12),
        block_table=torch.ones(4, 2, dtype=torch.int32),
        seq_lens=torch.tensor([20] * 4),
        max_seq_lens=20,
        seq_lens_list=[20] * 4,
        cp_seq_len=[10] * 4,
        dcp_mtp_attn_mask=None,
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
        attn_state=AscendAttentionState.DecodeOnly,
        decode=decode,
    )
    metadata.causal = False
    inputs = DecodeMLAPreprocessResult(
        torch.zeros(12, 16, 3),
        torch.zeros(12, 16, 2),
        torch.zeros(2, 1, 2, 3),
        torch.zeros(2, 1, 2, 2),
    )

    def fill_output(*args, **kwargs):
        for tensor in kwargs["out"]:
            tensor.zero_()

    with (
        patch.object(
            mla_cp,
            "_EXTRA_CTX",
            SimpleNamespace(
                capturing=True,
                is_draft_model=draft,
                is_draft_model_prefill=prefill,
            ),
        ),
        patch.object(updatable_graph, "_ACTIVE_GRAPH", SimpleNamespace(get=lambda: selected)),
        patch.object(
            mla_cp.torch_npu,
            "npu_fused_infer_attention_score_v2",
            SimpleNamespace(out=fill_output),
        ),
        patch.object(
            mla_cp.torch_npu,
            "_npu_fused_infer_attention_score_v2_get_max_workspace",
            return_value=workspace,
            create=True,
        ) as allocate,
    ):
        impl._forward_decode(inputs, 2, metadata)
        assert len(selected.capture_resources) == 1
        assert next(iter(selected.capture_resources.values())) is workspace
        if draft:
            assert target.capture_resources["other"] is target_marker
            assert len(target.capture_resources) == 1
        for graph in (draft_decode, draft_prefill):
            if graph is not selected:
                assert not graph.capture_resources

        impl._forward_decode(inputs, 2, metadata)
        assert allocate.call_count == 1
