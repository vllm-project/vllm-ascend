# SPDX-License-Identifier: Apache-2.0
"""Exercise the native DSA metadata refresh inside a real ACL graph."""

from types import SimpleNamespace

import torch

from vllm_ascend.attention import dsa_v1
from vllm_ascend.ops import rope_dsv4
from vllm_ascend.utils import enable_custom_op


def test_dsa_draft_metadata_replay_matches_fresh_build(monkeypatch):
    enable_custom_op()
    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    capacity, max_len = 3, 512
    table = torch.arange(max_len * 64, dtype=torch.float32, device="npu").view(max_len, 1, 1, 64)
    monkeypatch.setattr(
        rope_dsv4,
        "_ROPE_STATE",
        SimpleNamespace(
            full_rope_cache={"draft": (table, -table)},
            registry_summary={"draft": {"default"}},
            layer_info={"draft.layer": ("draft", ["default"])},
        ),
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(sliding_window=128, num_attention_heads=16, index_topk=512),
        get_head_size=lambda: 512,
        max_model_len=max_len,
    )
    config = SimpleNamespace(model_config=model_config, cache_config=SimpleNamespace(cache_dtype="bfloat16"))
    plan = dsa_v1.get_dsa_attn_kv_plan(config)
    metadata_size = (
        dsa_v1.SPARSE_FLASH_MLA_METADATA_BUFFER_SIZE
        if plan.get_dsa_sparse_attn_metadata_op() is dsa_v1.sparse_flash_mla_metadata
        else dsa_v1.SAS_METADATA_BUFFER_SIZE
    )

    def make_builder():
        builder = dsa_v1.AscendDSASWAMetadataBuilder.__new__(dsa_v1.AscendDSASWAMetadataBuilder)
        builder.model_config = model_config
        builder.vllm_config = config
        builder.metadata_cls = dsa_v1.AscendDSAMetadata
        builder.supports_draft_decode_metadata_update = True
        builder.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=2)
        builder.compressor_ratio = 1
        builder.layer_names = ["draft.layer"]
        builder.cache_group_key = "draft.layer"
        builder.storage_block_size = 128
        builder.decode_threshold = 3
        builder._device_metadata_enabled = False
        builder._step_rope_buffers = {}
        builder._sas_variant_buffers = {}
        shape = (capacity, 2) if plan.requires_block_offset_slots else (capacity,)
        builder.slot_mapping = torch.zeros(shape, dtype=torch.int32, device="npu")
        builder.sas_metadata_buffer_size = metadata_size
        builder.sas_metadata_buffer = torch.zeros(metadata_size, dtype=torch.int32, device="npu")
        builder.seqused_q = torch.empty(0, device="npu")
        builder.cu_seqlens_ori_kv = torch.empty(0, device="npu")
        builder.cu_seqlens_cmp_kv = torch.empty(0, device="npu")
        builder._zero_i32 = torch.zeros(1, dtype=torch.int32, device="npu")
        builder.hadamard = None
        return builder

    common = SimpleNamespace(
        num_reqs=capacity,
        num_actual_tokens=capacity,
        num_input_tokens=capacity,
        max_query_len=1,
        query_start_loc=torch.arange(capacity + 1, dtype=torch.int32, device="npu"),
        query_start_loc_cpu=torch.arange(capacity + 1, dtype=torch.int32),
        seq_lens=torch.tensor([127, 255, 383], dtype=torch.int32, device="npu"),
        # Use the same conservative bound for captured and fresh tiling.
        _seq_lens_cpu=torch.full((capacity,), max_len, dtype=torch.int32),
        positions=torch.tensor([126, 254, 382], dtype=torch.int64, device="npu"),
        slot_mapping=torch.tensor([254, 510, 766], dtype=torch.int64, device="npu"),
        block_table_tensor=torch.arange(1, 13, dtype=torch.int32, device="npu").view(capacity, 4),
        causal=True,
        attn_state=None,
    )
    builder = make_builder()
    metadata = builder.build_for_cudagraph_capture(common, common_ratio_to_sas_metadata={})
    req = metadata.req_metadata
    outputs = (req.cos["draft.layer"], req.sin["draft.layer"], req.slot_mapping, req.sas_metadata)
    pointers = [tensor.data_ptr() for tensor in outputs]
    builder.update_draft_decode_metadata(metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        builder.update_draft_decode_metadata(metadata)

    # Reuse the captured inputs: shrink to one request, then grow to three.
    # Consecutive substeps cross block/window boundaries without Python refresh.
    for lengths, positions, slots, starts in [
        ([128, 0, 0], [127, 0, 0], [255, -1, -1], [0, 1, 1, 1]),
        ([129, 257, 385], [128, 256, 384], [256, 512, 768], [0, 1, 2, 3]),
        ([130, 258, 386], [129, 257, 385], [257, 513, 769], [0, 1, 2, 3]),
    ]:
        common.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        common.positions.copy_(torch.tensor(positions, dtype=torch.int64, device="npu"))
        common.slot_mapping.copy_(torch.tensor(slots, dtype=torch.int64, device="npu"))
        common.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32, device="npu"))
        common.query_start_loc_cpu.copy_(torch.tensor(starts, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        expected = make_builder().build_for_cudagraph_capture(common, common_ratio_to_sas_metadata={}).req_metadata
        torch.npu.synchronize()
        references = (
            expected.cos["draft.layer"],
            expected.sin["draft.layer"],
            expected.slot_mapping,
            expected.sas_metadata,
        )
        for actual, reference in zip(outputs, references, strict=True):
            torch.testing.assert_close(actual, reference)
        assert [tensor.data_ptr() for tensor in outputs] == pointers
