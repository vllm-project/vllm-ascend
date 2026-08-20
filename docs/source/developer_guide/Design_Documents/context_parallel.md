# Context Parallel (CP)

Context Parallel distributes long-sequence work across devices. Prefill Context Parallel (PCP) adds ranks and splits prefill tokens to reduce time to first token, while Decode Context Parallel (DCP) reuses Tensor Parallel (TP) ranks and shards the KV cache to reduce duplicated storage. See the upstream [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) guide for the general design.

DSA-CP is a separate sparse-attention optimization. It uses TP/SP communication and is enabled by `additional_config.enable_dsa_cp`; it does not use the PCP process layout.

## Prefill Context Parallel (MRV2)

PCP divides the new tokens of each prefill request among a dedicated PCP group. Each rank computes rank-local query tokens while retaining access to the complete sequence state needed by attention. The current MRV2 implementations use the partial-query, full-key/value strategy described by upstream vLLM.

### Batch partitioning

The upstream `PCPManager` retains the scheduler-global batch and rewrites each attention step into rank-local DualChunkSwap rows. A prefill request is divided into `2 * pcp_size` chunks: rank 0 receives the first and last chunks, rank 1 receives the second and second-to-last chunks, and so on. Pairing the head and tail balances causal-attention work across ranks.

Each rank is padded to the same token count before collective communication. A restore index maps the gathered rank-major hidden states back to scheduler token order.

![PCP token partition](../../assets/cp/head-tail-style.png)

### GQA cache preparation

The GQA backend all-gathers the padded K/V tensors and expanded slot mappings. Each rank then writes the complete PCP KV-cache view before running attention for its rank-local query tokens. The temporary gathered tensors are layer-local and do not change the persistent cache-shard count.

### DSA cache preparation

DeepSeek-V4 DSA has three stateful cache paths: sliding-window attention (SWA), the main compressor, and the indexer compressor. Updating them independently from rank-local token order would produce different cache state on each rank.

The DSA PCP backend therefore:

1. All-gathers padded hidden states and restores scheduler token order once.
2. Builds canonical global DSA metadata from the scheduler-global batch.
3. Updates the SWA, main compressor, and indexer compressor caches once each on every rank.
4. Runs rank-local query and sparse attention with `cache_is_prepared=True`, which skips duplicate cache writes.

The replicated caches remain identical across the PCP group, while attention output remains rank-local. This path uses a single stream because cache preparation has already completed before local attention, leaving no cache-update work for the auxiliary stream to overlap.

DSA PCP is selected by `prefill_context_parallel_size > 1`. It is mutually exclusive with legacy DSA-CP, which is selected by `additional_config.enable_dsa_cp`.

## Decode Context Parallel (DCP)

DCP shards the KV cache along the sequence dimension across devices in a TP group. It eliminates redundant KV-cache storage without adding devices to the process world.

### KV-cache layout

DCP stores tokens in an interleaved layout across ranks. The interleaving granularity is controlled by `cp_kv_cache_interleave_size`, whose default value is `1`.

For a DCP size of `dcp_size`, a virtual block contains `block_size * dcp_size` tokens. For token `x`:

- `virtual_block_index = x // (block_size * dcp_size)`
- `offset_in_virtual_block = x % (block_size * dcp_size)`
- `local_block_index = offset_in_virtual_block // cp_kv_cache_interleave_size`
- `target_rank = local_block_index % dcp_size`

The slot-mapping calculation uses this layout so each DCP rank stores only its local sequence shard. The current implementation requires `block_size % cp_kv_cache_interleave_size == 0`.

![DCP block table](../../assets/cp/blocktable.png)

### Attention execution

#### Backend structure

DCP is implemented as a specialization of the corresponding v1 attention
backend rather than as a parallel copy of it:

- `DCPMetadataBuilderMixin` owns DCP group/rank discovery and access to the
  per-rank context-length matrix.
- `DCPImplMixin` owns DCP collectives and the common partial-output/LSE merge.
- GQA, MLA, and SFA DCP builders inherit their v1 metadata builders. They only
  add DCP metadata fields or temporarily expose the DCP-specific cache view.
- GQA, MLA, and SFA DCP implementations inherit their v1 implementations and
  override only the kernel stages whose communication or cache layout differs.

The normal v1 builders remain the source of truth for request classification,
padding, masks, graph metadata, and common KV-cache metadata. DCP-specific
metadata is kept out of the normal v1 metadata schemas.

#### Prefill and chunked prefill

During chunked or cached prefill, the local query must attend to KV-cache shards distributed across the DCP group.

- MLA gathers the context KV cache, restores request-contiguous order, and computes attention for the current query chunk.
- GQA gathers query heads across the DCP group, computes attention against each local KV shard, and combines partial outputs and LSE values.

![DCP prefill](../../assets/cp/dcp-prefill.png)

#### Decode

Decode gathers the query heads required by each DCP rank, computes attention against the local KV-cache shard, and combines partial outputs and LSE values across the DCP group.

![DCP decode](../../assets/cp/dcp-decode.png)

#### GLM-5.2 SFA DCP replicated indexer

GLM-5.2 uses Sparse Flash Attention (SFA) with a LightningIndexer. For DCP,
the indexer needs a full-sequence view to select the same sparse top-k blocks
as non-DCP SFA, while the much larger SFA KV cache should remain sharded to
retain DCP's memory benefit:

- The LightningIndexer cache is replicated on every DCP rank, so index
  selection uses the complete sequence.
- The SFA KV cache remains DCP-local. Global indices from the replicated
  indexer view are remapped to local KV indices before SFA runs.
- During prefill or a mixed batch, only KV blocks referenced by the sparse
  block table are compacted and all-gathered after the current layer writes its
  KV cache.
- Decode-only batches retain the DCP SFA Q-gather and result-merge path.

This mode is selected automatically for SFA sparse models when
`prefill_context_parallel_size=1` and `decode_context_parallel_size>1`. It
requires `decode_context_parallel_size == tensor_parallel_size`.

For a GLM-5.2 DSA-CP deployment, enable FlashComm1 and DSA-CP and keep the CP
interleave size equal to the KV-cache block size:

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve <glm-5.2-model> \
  --tensor-parallel-size <N> \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size <N> \
  --block-size <B> \
  --cp-kv-cache-interleave-size <B> \
  --additional-config '{"enable_dsa_cp": true}'
```

The replicated indexer increases indexer-cache memory in proportion to the
DCP world size; the SFA KV cache itself remains sharded.

#### SFA DSA-CP `o_proj` Path

SFA DSA-CP execution intentionally reuses the normal TP-sharded `o_proj`.
This applies both to a mixed-role instance and to a PD-disaggregated P node; it is not a standalone user-facing `o_proj` TP switch.
The runtime path supports two layouts:

- **Decode-only batches** keep the decode TP path.
  SFA outputs are exchanged with an all-to-all in the TP group, then the original TP-sharded `o_proj` runs normally.
- **Prefill or mixed batches**, including batches on a PD-disaggregated P node, produce SFA outputs that are not directly compatible with the TP-sharded `o_proj` input layout.
  Before `o_proj` forward, each rank all-gathers the TP-sharded `o_proj` weight and all input-sharded quantization parameters into temporary full-weight buffers.
  The full-weight `o_proj` forward runs once for that batch, and the module is then restored to the TP parameter aliases.

The storage invariant is that the original TP-sharded `o_proj` parameter remains the only persistent source of truth.
`o_proj_tp_*` tensors are aliases of the original parameter storage.
`o_proj_full_*` tensors are reusable communication buffers for prefill/mixed full-gather execution only.
They must not become a second persistent copy of the TP weight.

This coupling preserves the existing decode TP behavior, supports prefill/mixed DSA-CP batches on both mixed-role and P-only instances, and avoids a persistent full-weight copy on every TP rank.

## Related Files

- MRV2 PCP batch partitioning: `vllm_ascend/worker/v2/pcp_manager.py`
- MRV2 GQA PCP backend: `vllm_ascend/attention/attention_v1.py`
- MRV2 DSA PCP backend: `vllm_ascend/attention/context_parallel/dsa_cp.py`
- Slot mapping: `vllm_ascend/worker/block_table.py`
- Input and attention metadata: `vllm_ascend/worker/model_runner_v1.py`
- Shared DCP backend capabilities: `vllm_ascend/attention/context_parallel/common_cp.py`
- GQA DCP backend: `vllm_ascend/attention/context_parallel/attention_cp.py`
- MLA DCP backend: `vllm_ascend/attention/context_parallel/mla_cp.py`
- SFA DCP backend: `vllm_ascend/attention/context_parallel/sfa_cp.py`
- DSA-CP backend: `vllm_ascend/attention/context_parallel/dsa_cp.py`
