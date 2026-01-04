from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


class BatchInvariantBackendImp(AscendAttentionBackendImpl):
    """Batch-invariant attention backend implementation for Ascend NPUs."""

    def forward_impl(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    ):
        return super().forward_impl(query, key, value, kv_cache, attn_metadata,
                                    output)
