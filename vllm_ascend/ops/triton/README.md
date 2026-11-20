# Triton operator layout

Use this inventory to find the module and its owning runner or model. Implementations live at the package root or in functional subdirectories. The shared JIT registry and generic warmup code serve these paths.

Find callable interfaces in the [Triton operator reference](../../../docs/source/developer_guide/operator_reference/triton/index.md). The [contribution guide](../../../docs/source/developer_guide/contribution/triton_operators.md) describes how to document a wrapper and verify its production caller. Earlier [package notes](docs/) cover the remaining kernels and cross-component designs.

When adding or moving a module, update its inventory entry, production imports, warmup registrations, tests and documentation links.

## Runner operators

| Module | Runner ownership | Documentation |
| --- | --- | --- |
| [engram_lookback.py](engram_lookback.py) | `models/deepseek_v41/engram/model_state.py` | [engram.md](docs/engram.md) |
| [apply_grammar_bitmask.py](apply_grammar_bitmask.py) | `patch/worker/patch_triton.py` | [apply_grammar_bitmask.md](docs/apply_grammar_bitmask.md) |
| [block_table/compute_slot_mappings.py](block_table/compute_slot_mappings.py) | `worker/v2/block_table.py` | [`ascendblocktables.compute_slot_mappings`](../../../docs/source/developer_guide/operator_reference/triton/ascendblocktables.compute_slot_mappings.md) |
| [mamba/precopy.py](mamba/precopy.py) | `patch/worker/patch_triton.py` | [mamba_precopy.md](docs/mamba_precopy.md) |
| [mamba/postprocess.py](mamba/postprocess.py) | `patch/worker/patch_mamba_utils.py` | [mamba_postprocess.md](docs/mamba_postprocess.md) |
| [metrics/num_nans.py](metrics/num_nans.py) | `patch/worker/patch_triton.py` | [`get_num_nans`](../../../docs/source/developer_guide/operator_reference/triton/get_num_nans.md) |
| [sample/categorical_sample.py](sample/categorical_sample.py) | `patch/worker/patch_triton.py` | [`categorical_sample`](../../../docs/source/developer_guide/operator_reference/triton/categorical_sample.md) |
| [sample/fill_logprob_token_idx.py](sample/fill_logprob_token_idx.py) | `patch/worker/patch_triton.py` | [fill_logprob_token_ids.md](docs/fill_logprob_token_ids.md) |
| [sample/sampling_mask.py](sample/sampling_mask.py) | `patch/worker/patch_triton.py` | [`sampling_mask_from_logits_npu`](../../../docs/source/developer_guide/operator_reference/triton/sampling_mask_from_logits_npu.md) |
| [sample/thinking_budget.py](sample/thinking_budget.py) | `patch/worker/patch_triton.py` | [thinking_budget.md](docs/thinking_budget.md) |
| [spec_decode/prepare_dflash_inputs.py](spec_decode/prepare_dflash_inputs.py) | `worker/v2/spec_decode/dflash/speculator.py` | [`prepare_dflash_inputs_triton`](../../../docs/source/developer_guide/operator_reference/triton/prepare_dflash_inputs_triton.md) |
| [spec_decode/resample.py](spec_decode/resample.py) | `worker/v2/spec_decode/rejection_sampler_utils.py` | [`resample`](../../../docs/source/developer_guide/operator_reference/triton/resample.md) |
| [mamba/gdn_prepare.py](mamba/gdn_prepare.py) | `attention/gdn_attn_builder.py` | [`prepare_gdn`](../../../docs/source/developer_guide/operator_reference/triton/prepare_gdn.md) |
| [spec_decode/entropy_verify.py](spec_decode/entropy_verify.py) | `worker/v2/spec_decode/rejection_sampler_utils.py` | [`rejection_sample`](../../../docs/source/developer_guide/operator_reference/triton/rejection_sample.md) |

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
| [gdn/gdn_recoverssm_commit.py](gdn/gdn_recoverssm_commit.py) | GDN RecoverSSM commit. [Operator documentation](docs/gdn_recoverssm_commit.md) |
| [gdn/gdn_recoverssm_verify.py](gdn/gdn_recoverssm_verify.py) | GDN RecoverSSM verify. [Operator documentation](docs/gdn_recoverssm_verify.md) |
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
| [kimi_pixel_norm.py](kimi_pixel_norm.py) | Kimi image pixel lookup normalization; see [`kimi_pixel_norm_lut`](../../../docs/source/developer_guide/operator_reference/triton/kimi_pixel_norm_lut.md). |
| [kda/recoverssm.py](kda/recoverssm.py) | KDA RecoverSSM operators and commit context; [`kda_recoverssm_verify`](../../../docs/source/developer_guide/operator_reference/triton/kda_recoverssm_verify.md) |
| [kda/solve_tril.py](kda/solve_tril.py) | KDA computation |
| [kda/utils.py](kda/utils.py) | KDA computation |
| [hisparse.py](hisparse.py) | HiSparse mapped-Host refill, Resident gather, KV packing and invalidation; see [operator documentation](docs/hisparse.md). |
| [layernorm_gated.py](layernorm_gated.py) | Model/attention primitive or shared utility |
| [linearnorm/fused_eh_norm.py](linearnorm/fused_eh_norm.py) | Normalization/rotary fusion |
| [linearnorm/qsa_rmsnorm_mrope.py](linearnorm/qsa_rmsnorm_mrope.py) | Model/attention primitive or shared utility |
| [linearnorm/split_qkv_rmsnorm_mrope.py](linearnorm/split_qkv_rmsnorm_mrope.py) | Normalization/rotary fusion; [`triton_split_qkv_rmsnorm_mrope`](../../../docs/source/developer_guide/operator_reference/triton/triton_split_qkv_rmsnorm_mrope.md) |
| [linearnorm/split_qkv_rmsnorm_rope.py](linearnorm/split_qkv_rmsnorm_rope.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope_simt.py](linearnorm/split_qkv_rmsnorm_rope_simt.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_rmsnorm_rope_vnorm.py](linearnorm/split_qkv_rmsnorm_rope_vnorm.py) | Normalization/rotary fusion |
| [linearnorm/split_qkv_tp_rmsnorm_rope.py](linearnorm/split_qkv_tp_rmsnorm_rope.py) | Normalization/rotary fusion |
| [mamba/causal_conv1d.py](mamba/causal_conv1d.py) | Model state/attention |
| [mamba/compact_conv_state.py](mamba/compact_conv_state.py) | [RecoverSSM convolution compaction](docs/compact_conv_state.md) |
| [mamba/lightning_attn.py](mamba/lightning_attn.py) | Model state/attention |
| [mamba/prepare_commit_plan.py](mamba/prepare_commit_plan.py) | [RecoverSSM commit planning](docs/prepare_commit_plan.md) |
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
| [qwen4_exp/lightning_v2_prepare.py](qwen4_exp/lightning_v2_prepare.py) | Native QSA LightningIndexerV2 input preparation; [`prepare_qsa_lightning_v2_inputs`](../../../docs/source/developer_guide/operator_reference/triton/prepare_qsa_lightning_v2_inputs.md) |
| [qwen4_exp/qsa.py](qwen4_exp/qsa.py) | Model/attention primitive; [`qsa_compress_groups_with_ratio`](../../../docs/source/developer_guide/operator_reference/triton/qsa_compress_groups_with_ratio.md) |
| [rms_norm.py](rms_norm.py) | Model/attention primitive or shared utility |
| [rope.py](rope.py) | Model/attention primitive or shared utility |
| [sfa_indexer_store.py](sfa_indexer_store.py) | Indexer cache publication; see [`store_indexer_key_scale`](../../../docs/source/developer_guide/operator_reference/triton/store_indexer_key_scale.md). |
| [sparse_index_remap.py](sparse_index_remap.py) | Model/attention primitive or shared utility |
| [spec_decode/dspark_swa_indices.py](spec_decode/dspark_swa_indices.py) | Model attention indices |
| [step5/fused_add_rms_norm.py](step5/fused_add_rms_norm.py) | Fused optional FP32 residual add, RMSNorm, and BF16 output cast. [`fused_add_rms_norm_cast`](../../../docs/source/developer_guide/operator_reference/triton/fused_add_rms_norm_cast.md) |
| [step5/fused_qknorm_rope.py](step5/fused_qknorm_rope.py) | Step5 fused QK RMSNorm + NeoX RoPE with V passthrough. [`fused_qknorm_rope`](../../../docs/source/developer_guide/operator_reference/triton/fused_qknorm_rope.md) |
| [step5/summary_update.py](step5/summary_update.py) | Step5 CSA region summaries: [`update_decode`](../../../docs/source/developer_guide/operator_reference/triton/update_decode.md), [`update_verify`](../../../docs/source/developer_guide/operator_reference/triton/update_verify.md), [`update_prefill`](../../../docs/source/developer_guide/operator_reference/triton/update_prefill.md), [`build_prefill_groups`](../../../docs/source/developer_guide/operator_reference/triton/build_prefill_groups.md), [`append_current_region`](../../../docs/source/developer_guide/operator_reference/triton/append_current_region.md) |
| [triton_utils.py](triton_utils.py) | Model/attention primitive or shared utility |
| [turboquant_finalize.py](turboquant_finalize.py) | Model/attention primitive or shared utility |
