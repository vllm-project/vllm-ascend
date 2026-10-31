# Triton operator layout

V1-only runner operators and their documentation live in `v1/` and `v1/docs/`.
V2 operators use the main package, with their documentation in `docs/`.
Operators shared by both runners, model-layer kernels, and general utilities
keep their functional directories. A model supported by V2 alone does not make
all of its mathematical kernels V2 runner operators.

This inventory covers all 97 non-`__init__.py` Python modules under this package.
It includes launch wrappers and utilities as well as JIT definitions. Operators
defined under `worker/v2/`, `models/`, or other packages are outside this layout
change.

Warmup uses the shared upstream JIT registry. The generic
`ascend_jit_warmup.py` stays at the package root; V1 sampling wrappers register
only through `register_v1_sampling_warmup` during V1 runner construction.

## Adding or moving an operator

Paths below are relative to `vllm_ascend/ops/triton/`.

| Operator ownership | Implementation | Operator document |
| --- | --- | --- |
| V1-only runner operator | `v1/<functional-dir>/<operator>.py` | `v1/docs/<operator>.md` |
| V2 runner operator | `<functional-dir>/<operator>.py` | `docs/<operator>.md` |
| Shared runner, model-layer, or utility operator | Existing functional directory | `docs/<operator>.md` |

The functional subdirectory is optional for operators placed directly at the
package root. New V2 code and documents use the main package rather than a
`v2/` subdirectory. Choose ownership from the production callers; support for a
V2 model alone does not make an operator V2-only.

Use the [operator documentation template](docs/triton_description_specification.md)
for the operator document. When adding or moving a module, update this inventory,
production imports, warmup registrations, tests and mock targets, and documentation
examples and links together.

## V1-only operators

| Module | Runner ownership | Documentation |
| --- | --- | --- |
| [v1/bincount.py](v1/bincount.py) | Used by the V1 penalty implementation. | [apply_penalties.md](v1/docs/apply_penalties.md) |
| [v1/compute_slot_mapping.py](v1/compute_slot_mapping.py) | Used by `worker/block_table.py`, the V1 block-table path. | [compute_slot_mapping.md](v1/docs/compute_slot_mapping.md) |
| [v1/mamba/postprocess.py](v1/mamba/postprocess.py) | Patched into the legacy Mamba postprocess called by the V1 runner; V2 uses precopy. | [postprocess_mamba.md](v1/docs/postprocess_mamba.md) |
| [v1/penalty.py](v1/penalty.py) | Used by `sample/penalties.py` and V1 kernel warmup. | [apply_penalties.md](v1/docs/apply_penalties.md) |
| [v1/reject_sample.py](v1/reject_sample.py) | Used by `sample/rejection_sampler.py` and V1 kernel warmup. | [reject_sample.md](v1/docs/reject_sample.md) |
| [v1/spec_decode/ngram.py](v1/spec_decode/ngram.py) | Called by `worker/v1/model_runner.py`. | [spec_decode.md](v1/docs/spec_decode.md) |
| [v1/spec_decode/utils.py](v1/spec_decode/utils.py) | Used by the proposers under `worker/v1/spec_decode/`. | [spec_decode.md](v1/docs/spec_decode.md) |

## V2 runner operators

| Module | Runner ownership | Documentation |
| --- | --- | --- |
| [engram_lookback.py](engram_lookback.py) | `models/deepseek_v41/engram/model_state.py` | [engram.md](docs/engram.md) |
| [apply_grammar_bitmask.py](apply_grammar_bitmask.py) | `patch/worker/patch_v2/patch_triton.py` | [apply_grammar_bitmask.md](docs/apply_grammar_bitmask.md) |
| [block_table/compute_slot_mappings.py](block_table/compute_slot_mappings.py) | `worker/v2/block_table.py` | [compute_slot_mappings.md](docs/compute_slot_mappings.md) |
| [mamba/precopy.py](mamba/precopy.py) | `patch/worker/patch_v2/patch_triton.py` | [mamba_precopy.md](docs/mamba_precopy.md) |
| [metrics/num_nans.py](metrics/num_nans.py) | `patch/worker/patch_v2/patch_triton.py` | [num_nans.md](docs/num_nans.md) |
| [sample/categorical_sample.py](sample/categorical_sample.py) | `patch/worker/patch_v2/patch_triton.py` | [categorical_sample.md](docs/categorical_sample.md) |
| [sample/fill_logprob_token_idx.py](sample/fill_logprob_token_idx.py) | `patch/worker/patch_v2/patch_triton.py` | [fill_logprob_token_ids.md](docs/fill_logprob_token_ids.md) |
| [sample/sampling_mask.py](sample/sampling_mask.py) | `patch/worker/patch_v2/patch_triton.py` | [sampling_mask.md](docs/sampling_mask.md) |
| [sample/thinking_budget.py](sample/thinking_budget.py) | `patch/worker/patch_v2/patch_triton.py` | [thinking_budget.md](docs/thinking_budget.md) |
| [spec_decode/prepare_dflash_inputs.py](spec_decode/prepare_dflash_inputs.py) | `worker/v2/spec_decode/dflash/speculator.py` | [prepare_dflash_inputs.md](docs/prepare_dflash_inputs.md) |
| [spec_decode/resample.py](spec_decode/resample.py) | `worker/v2/spec_decode/rejection_sampler_utils.py` | [resample.md](docs/resample.md) |

## Shared runner support

These kernels support both runner versions and keep their current paths.

| Module | Shared use |
| --- | --- |
| [batch_memcpy.py](batch_memcpy.py) | Mamba copying and the KDA fallback copy plan installed by both runners. |
| [decoder_replay.py](decoder_replay.py) | Replay gathering in V1 and V2 model state. |
| [kda_state_copy_kernel.py](kda_state_copy_kernel.py) | KDA state-copy plans initialized by both runners. |

## Runner-independent modules

These model, attention, quantization, distributed, and utility modules retain
their existing paths. Their use depends on model, hardware, and feature
configuration rather than the V1/V2 runner input or sampling interface.

| Module | Functional area |
| --- | --- |
| [activation/swiglu_quant.py](activation/swiglu_quant.py) | Activation/quantization |
| [activation/swiglustep.py](activation/swiglustep.py) | Activation/quantization |
| [ascend_jit_warmup.py](ascend_jit_warmup.py) | Shared JIT warmup infrastructure |
| [attention_update.py](attention_update.py) | Model/attention primitive or shared utility |
| [batch_invariant/matmul.py](batch_invariant/matmul.py) | Batch-invariant operations |
| [batch_invariant/mean.py](batch_invariant/mean.py) | Batch-invariant operations |
| [batch_invariant/rmsnorm.py](batch_invariant/rmsnorm.py) | Batch-invariant operations |
| [batch_invariant/softmax.py](batch_invariant/softmax.py) | Batch-invariant operations |
| [build_window_indices.py](build_window_indices.py) | Model/attention primitive or shared utility |
| [c2_ring_metadata.py](c2_ring_metadata.py) | Model/attention primitive or shared utility |
| [compressor/compressor_triton.py](compressor/compressor_triton.py) | Model compression |
| [dcp/dcp_a2a.py](dcp/dcp_a2a.py) | Distributed attention |
| [dcp/dcp_a2a_batched.py](dcp/dcp_a2a_batched.py) | Distributed attention |
| [dsa_cp.py](dsa_cp.py) | Model/attention primitive or shared utility |
| [dspark_swa_indices.py](dspark_swa_indices.py) | Model/attention primitive or shared utility |
| [engram_gate.py](engram_gate.py) | Model/attention primitive or shared utility |
| [engram_hash.py](engram_hash.py) | Model/attention primitive or shared utility |
| [engram_lookup.py](engram_lookup.py) | Model/attention primitive or shared utility |
| [eplb.py](eplb.py) | Model/attention primitive or shared utility |
| [fla/chunk.py](fla/chunk.py) | Linear attention |
| [fla/chunk_delta_h.py](fla/chunk_delta_h.py) | Linear attention |
| [fla/chunk_delta_hupdate.py](fla/chunk_delta_hupdate.py) | Linear attention |
| [fla/chunk_o.py](fla/chunk_o.py) | Linear attention |
| [fla/chunk_scaled_dot_kkt.py](fla/chunk_scaled_dot_kkt.py) | Linear attention |
| [fla/cumsum.py](fla/cumsum.py) | Linear attention |
| [fla/fused_qkvzba_split_reshape.py](fla/fused_qkvzba_split_reshape.py) | Linear attention |
| [fla/l2norm.py](fla/l2norm.py) | Linear attention |
| [fla/layernorm_guard.py](fla/layernorm_guard.py) | Linear attention |
| [fla/solve_tril.py](fla/solve_tril.py) | Linear attention |
| [fla/utils.py](fla/utils.py) | Linear attention |
| [fla/wy_fast.py](fla/wy_fast.py) | Linear attention |
| [fold_indexer_cache.py](fold_indexer_cache.py) | Model/attention primitive or shared utility |
| [fused_gdn_gating.py](fused_gdn_gating.py) | Model/attention primitive or shared utility |
| [glm5_next_cann_keypool.py](glm5_next_cann_keypool.py) | Model/attention primitive or shared utility |
| [glm5_next_kpool_tail_compress.py](glm5_next_kpool_tail_compress.py) | Model/attention primitive or shared utility |
| [glm5_next_lightning_indexer.py](glm5_next_lightning_indexer.py) | Model/attention primitive or shared utility |
| [kda/chunk_delta_h.py](kda/chunk_delta_h.py) | KDA computation |
| [kda/conv_state.py](kda/conv_state.py) | KDA computation |
| [kda/cumsum.py](kda/cumsum.py) | KDA computation |
| [kda/fused_recurrent_kda.py](kda/fused_recurrent_kda.py) | KDA computation |
| [kda/gate.py](kda/gate.py) | KDA computation |
| [kda/kda.py](kda/kda.py) | KDA computation |
| [kda/l2norm.py](kda/l2norm.py) | KDA computation |
| [kda/output_writeback.py](kda/output_writeback.py) | KDA computation |
| [kda/recoverssm.py](kda/recoverssm.py) | Model state recovery |
| [kda/solve_tril.py](kda/solve_tril.py) | KDA computation |
| [kda/utils.py](kda/utils.py) | KDA computation |
| [layernorm_gated.py](layernorm_gated.py) | Model/attention primitive or shared utility |
| [linearnorm/fused_eh_norm.py](linearnorm/fused_eh_norm.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_mrope.py](linearnorm/split_qkv_rmsnorm_mrope.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope.py](linearnorm/split_qkv_rmsnorm_rope.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope_simt.py](linearnorm/split_qkv_rmsnorm_rope_simt.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope_vnorm.py](linearnorm/split_qkv_rmsnorm_rope_vnorm.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_tp_rmsnorm_rope.py](linearnorm/split_qkv_tp_rmsnorm_rope.py) | Normalization/rotary fusion |
| [mamba/causal_conv1d.py](mamba/causal_conv1d.py) | Model state/attention |
| [mamba/lightning_attn.py](mamba/lightning_attn.py) | Model state/attention |
| [mamba/state_index.py](mamba/state_index.py) | Model state/attention |
| [mamba/state_ops.py](mamba/state_ops.py) | Model state/attention |
| [msa/minimax_qknorm_rope_cache.py](msa/minimax_qknorm_rope_cache.py) | Model sparse attention |
| [msa/msa_m3_index_postprocess.py](msa/msa_m3_index_postprocess.py) | Model sparse attention |
| [mul_add.py](mul_add.py) | Model/attention primitive or shared utility |
| [muls_add.py](muls_add.py) | Model/attention primitive or shared utility |
| [mxfp_k_scale.py](mxfp_k_scale.py) | Model/attention primitive or shared utility |
| [packed_cache_slot_mapping.py](packed_cache_slot_mapping.py) | Model/attention primitive or shared utility |
| [pcp_kv_cache.py](pcp_kv_cache.py) | Model/attention primitive or shared utility |
| [prepare_indexer_indices.py](prepare_indexer_indices.py) | Model/attention primitive or shared utility |
| [prepare_nonempty_keypool_inputs.py](prepare_nonempty_keypool_inputs.py) | Model/attention primitive or shared utility |
| [quantize_indexer_query.py](quantize_indexer_query.py) | Model/attention primitive or shared utility |
| [quantize_mxfp4_indexer.py](quantize_mxfp4_indexer.py) | Model/attention primitive or shared utility |
| [query_gather_prep.py](query_gather_prep.py) | Model/attention primitive or shared utility |
| [rms_norm.py](rms_norm.py) | Model/attention primitive or shared utility |
| [rope.py](rope.py) | Model/attention primitive or shared utility |
| [sparse_index_remap.py](sparse_index_remap.py) | Model/attention primitive or shared utility |
| [spec_decode/dspark_swa_indices.py](spec_decode/dspark_swa_indices.py) | Model attention indices |
| [triton_utils.py](triton_utils.py) | Model/attention primitive or shared utility |
| [turboquant_finalize.py](turboquant_finalize.py) | Model/attention primitive or shared utility |
