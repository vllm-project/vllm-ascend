# Triton operator layout

Runner operators live in the main package, with their documentation in `docs/`.
Model-layer kernels and general utilities keep their functional directories. A
model supported by a runner does not make all of its mathematical kernels runner
operators.

This inventory covers every non-`__init__.py` Python module under this package.
It includes launch wrappers and utilities as well as JIT definitions. Operators
defined under `worker/v2/`, `models/`, or other packages are outside this layout
change.

Warmup uses the shared upstream JIT registry. The generic
`ascend_jit_warmup.py` stays at the package root.

## Adding or moving an operator

Paths below are relative to `vllm_ascend/ops/triton/`.

| Operator ownership | Implementation | Operator document |
| --- | --- | --- |
| V2 runner operator | `<functional-dir>/<operator>.py` | `docs/<operator>.md` |
| Shared runner, model-layer, or utility operator | Existing functional directory | `docs/<operator>.md` |

The functional subdirectory is optional for operators placed directly at the
package root. New code and documents use the main package rather than a `v2/`
subdirectory. Choose ownership from the production callers.

Use the [operator documentation template](docs/triton_description_specification.md)
for the operator document. When adding or moving a module, update this inventory,
production imports, warmup registrations, tests and mock targets, and documentation
examples and links together.

## Runner operators

| Module | Runner ownership | Documentation |
| --- | --- | --- |
| [engram_lookback.py](engram_lookback.py) | `models/deepseek_v41/engram/model_state.py` | [engram.md](docs/engram.md) |
| [apply_grammar_bitmask.py](apply_grammar_bitmask.py) | `patch/worker/patch_triton.py` | [apply_grammar_bitmask.md](docs/apply_grammar_bitmask.md) |
| [block_table/compute_slot_mappings.py](block_table/compute_slot_mappings.py) | `worker/v2/block_table.py` | [compute_slot_mappings.md](docs/compute_slot_mappings.md) |
| [mamba/precopy.py](mamba/precopy.py) | `patch/worker/patch_triton.py` | [mamba_precopy.md](docs/mamba_precopy.md) |
| [mamba/postprocess.py](mamba/postprocess.py) | `patch/worker/patch_mamba_utils.py` | [mamba_postprocess.md](docs/mamba_postprocess.md) |
| [metrics/num_nans.py](metrics/num_nans.py) | `patch/worker/patch_triton.py` | [num_nans.md](docs/num_nans.md) |
| [sample/categorical_sample.py](sample/categorical_sample.py) | `patch/worker/patch_triton.py` | [categorical_sample.md](docs/categorical_sample.md) |
| [sample/fill_logprob_token_idx.py](sample/fill_logprob_token_idx.py) | `patch/worker/patch_triton.py` | [fill_logprob_token_ids.md](docs/fill_logprob_token_ids.md) |
| [sample/sampling_mask.py](sample/sampling_mask.py) | `patch/worker/patch_triton.py` | [sampling_mask.md](docs/sampling_mask.md) |
| [sample/thinking_budget.py](sample/thinking_budget.py) | `patch/worker/patch_triton.py` | [thinking_budget.md](docs/thinking_budget.md) |
| [spec_decode/prepare_dflash_inputs.py](spec_decode/prepare_dflash_inputs.py) | `worker/v2/spec_decode/dflash/speculator.py` | [prepare_dflash_inputs.md](docs/prepare_dflash_inputs.md) |
| [spec_decode/resample.py](spec_decode/resample.py) | `worker/v2/spec_decode/rejection_sampler_utils.py` | [resample.md](docs/resample.md) |
| [mamba/gdn_prepare.py](mamba/gdn_prepare.py) | `attention/gdn_attn_builder.py` | [gdn_prepare.md](docs/gdn_prepare.md) |
| [spec_decode/entropy_verify.py](spec_decode/entropy_verify.py) | `worker/v2/spec_decode/rejection_sampler_utils.py` | [entropy_verify.md](docs/entropy_verify.md) |

## Shared runner support

These kernels are used on runner-neutral paths and keep their current paths.

| Module | Shared use |
| --- | --- |
| [batch_memcpy.py](batch_memcpy.py) | Mamba copying and the KDA fallback copy plan installed by the model runner. |
| [decoder_replay.py](decoder_replay.py) | Replay gathering in the V2 model state. |
| [kda_state_copy_kernel.py](kda_state_copy_kernel.py) | KDA state-copy plans initialized by the model runner. |

## Runner-independent modules

These model, attention, quantization, distributed, and utility modules retain
their existing paths. Their use depends on model, hardware, and feature
configuration rather than the runner input or sampling interface.

| Module | Functional area |
| --- | --- |
| [activation/swiglu_quant.py](activation/swiglu_quant.py) | Activation/quantization |
| [activation/swiglustep.py](activation/swiglustep.py) | Activation/quantization |
| [ascend_jit_warmup.py](ascend_jit_warmup.py) | Shared JIT warmup infrastructure |
| [dcp/dcp_jit.py](dcp/dcp_jit.py) | DCP compile-only JIT owners |
| [dcp/dcp_warmup.py](dcp/dcp_warmup.py) | SFA/MLA DCP owner registration and key coverage |
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
| [dcp/sfa_dcp_exchange.py](dcp/sfa_dcp_exchange.py) | Model/attention primitive or shared utility |
| [dcp/sfa_dcp_merge.py](dcp/sfa_dcp_merge.py) | Model/attention primitive or shared utility |
| [sharded_indexer.py](sharded_indexer.py) | DCP candidate transport and causal metadata |
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
| [gdn/recoverssm_gdn.py](gdn/recoverssm_gdn.py) | GDN RecoverSSM operators and commit context; [design](../../../docs/source/developer_guide/Design_Documents/qwen_gdn_recoverssm.md) |
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
| [kda/recoverssm.py](kda/recoverssm.py) | KDA RecoverSSM operators and commit context; [operator documentation](docs/kda_recoverssm.md) |
| [kda/solve_tril.py](kda/solve_tril.py) | KDA computation |
| [kda/utils.py](kda/utils.py) | KDA computation |
| [layernorm_gated.py](layernorm_gated.py) | Model/attention primitive or shared utility |
| [linearnorm/fused_eh_norm.py](linearnorm/fused_eh_norm.py) | Normalization/rotary fusion |
| [linearnorm/qsa_rmsnorm_mrope.py](linearnorm/qsa_rmsnorm_mrope.py) | Model/attention primitive or shared utility |
| [linearnorm/split_qkv_rmsnorm_mrope.py](linearnorm/split_qkv_rmsnorm_mrope.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope.py](linearnorm/split_qkv_rmsnorm_rope.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope_simt.py](linearnorm/split_qkv_rmsnorm_rope_simt.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope_vnorm.py](linearnorm/split_qkv_rmsnorm_rope_vnorm.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_tp_rmsnorm_rope.py](linearnorm/split_qkv_tp_rmsnorm_rope.py) | Normalization/rotary fusion |
| [mamba/causal_conv1d.py](mamba/causal_conv1d.py) | Model state/attention |
| [mamba/lightning_attn.py](mamba/lightning_attn.py) | Model state/attention |
| [mamba/recoverssm.py](mamba/recoverssm.py) | Shared RecoverSSM plan and convolution compaction; [operator documentation](docs/kda_recoverssm.md) |
| [mamba/state_index.py](mamba/state_index.py) | Model state/attention |
| [mamba/state_ops.py](mamba/state_ops.py) | Model state/attention |
| [msa/minimax_qknorm_rope_cache.py](msa/minimax_qknorm_rope_cache.py) | Model sparse attention |
| [msa/msa_m3_index_postprocess.py](msa/msa_m3_index_postprocess.py) | Model sparse attention |
| [msa/msa_m3_index_topk.py](msa/msa_m3_index_topk.py) | MiniMax-M3 A5 sparse-attention Triton fallback |
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
| [qwen4_exp/fia.py](qwen4_exp/fia.py) | Model/attention primitive or shared utility |
| [qwen4_exp/gbsa.py](qwen4_exp/gbsa.py) | Model/attention primitive or shared utility |
| [qwen4_exp/hc.py](qwen4_exp/hc.py) | Model/attention primitive or shared utility |
| [qwen4_exp/qsa.py](qwen4_exp/qsa.py) | Model/attention primitive or shared utility |
| [rms_norm.py](rms_norm.py) | Model/attention primitive or shared utility |
| [rope.py](rope.py) | Model/attention primitive or shared utility |
| [sfa_indexer_store.py](sfa_indexer_store.py) | Model/attention primitive or shared utility |
| [sparse_index_remap.py](sparse_index_remap.py) | Model/attention primitive or shared utility |
| [spec_decode/dspark_swa_indices.py](spec_decode/dspark_swa_indices.py) | Model attention indices |
| [triton_utils.py](triton_utils.py) | Model/attention primitive or shared utility |
| [turboquant_finalize.py](turboquant_finalize.py) | Model/attention primitive or shared utility |
