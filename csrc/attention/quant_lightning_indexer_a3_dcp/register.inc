    // A3 DCP QLI computes each query's causal local prefix from global
    // sequence bounds. Keep the legacy local-candidate ABI for A/B and A5.
    ops.def(
        "npu_quant_lightning_indexer_dcp_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, "
        "int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
        "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int batch_size=-1, int max_seqlen_q=-1, "
        "int max_seqlen_k=-1, str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, "
        "int cmp_ratio=1, int enable_dcp=1, Tensor? causal_seqlens_kv_global=None, "
        "int cp_world=1, int cp_rank=0, int cp_kv_cache_interleave_size=1) -> Tensor"
    );
    ops.impl("npu_quant_lightning_indexer_dcp_metadata", torch::kPrivateUse1,
             &vllm_ascend::qli_v2_dcp::QuantLightningIndexerMetadata);
    ops.impl("npu_quant_lightning_indexer_dcp_metadata", torch::kMeta,
             &vllm_ascend::qli_v2_dcp::QuantLightningIndexerMetadata);
    ops.def(
        "npu_quant_lightning_indexer_dcp(Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, "
        "Tensor? candidate_topk_index=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
        "Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, "
        "int max_seqlen_q=-1, str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, "
        "int cmp_ratio=1, int candidate_mode=3, int candidate_topk_blocks=2048, "
        "int candidate_block_size=8, bool enable_dcp=True, int cp_world=1, int cp_rank=0, "
        "int cp_kv_cache_interleave_size=1, Tensor? causal_seqlens_kv_global=None, "
        "int return_value=1) -> (Tensor, Tensor, Tensor)"
    );
    ops.impl("npu_quant_lightning_indexer_dcp", torch::kPrivateUse1,
             &vllm_ascend::qli_v2_dcp::QuantLightningIndexerCandidate);
    ops.impl("npu_quant_lightning_indexer_dcp", torch::kMeta,
             &vllm_ascend::qli_v2_dcp::QuantLightningIndexerCandidate);
