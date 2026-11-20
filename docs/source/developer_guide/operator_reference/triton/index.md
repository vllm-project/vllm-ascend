# Triton operators

Find an operator by its callable name. Each entry links to the wrapper, a call or registration site and the operator description. To add or update an entry, follow the [contribution guide](../../contribution/triton_operators.md).

| Operator | Purpose |
| --- | --- |
| [`append_current_region`](append_current_region.md) | Append the current region to the Step5 indexer output. |
| [`apply_min_p`](apply_min_p.md) | Filter logits in place using a per-request minimum probability. |
| [`AscendBlockTables.compute_slot_mappings`](ascendblocktables.compute_slot_mappings.md) | Translate scheduled token positions into paged KV slots. |
| [`bincount`](bincount.md) | Count prompt and output tokens for sampler penalties. |
| [`build_c2_ring_metadata`](build_c2_ring_metadata.md) | Derive C2 ring addresses and request controls. |
| [`build_dspark_swa_indices_triton`](build_dspark_swa_indices_triton.md) | Build request-scoped sliding windows for DSpark queries. |
| [`build_dspark_swa_slot_indices_triton`](build_dspark_swa_slot_indices_triton.md) | Build non-causal sliding-window slots for parallel drafting. |
| [`build_keypool_metadata`](build_keypool_metadata.md) | Build metadata for the GLM pooled key indexer. |
| [`build_local_key_lengths`](build_local_key_lengths.md) | Build local causal key lengths for DCP. |
| [`build_local_lengths`](build_local_lengths.md) | Compute rank-local lengths for a sharded index. |
| [`build_packed_cache_slot_mapping`](build_packed_cache_slot_mapping.md) | Convert physical slots to packed page and row coordinates. |
| [`build_prefill_groups`](build_prefill_groups.md) | Build region grouping metadata for Step5 prefill. |
| [`build_qsa_fia_mask`](build_qsa_fia_mask.md) | Build the FIA attention mask for selected QSA groups. |
| [`build_tail_inputs`](build_tail_inputs.md) | Prepare GLM KeyPool tail inputs. |
| [`build_window_indices_triton`](build_window_indices_triton.md) | Build causal sliding-window KV positions and valid lengths. |
| [`categorical_sample`](categorical_sample.md) | Sample one token per row, with greedy selection at zero temperature. |
| [`compressor_from_projected`](compressor_from_projected.md) | Pool projected C2 key/value pairs against ring state. |
| [`compute_token_logprobs`](compute_token_logprobs.md) | Compute log probabilities for selected token IDs. |
| [`compute_topk_logprobs`](compute_topk_logprobs.md) | Gather sampled and top token log probabilities and compute the sampled token rank. |
| [`copy_conv_state`](copy_conv_state.md) | Stage convolution state in contiguous storage or write it back. |
| [`copy_pcp_kv_cache`](copy_pcp_kv_cache.md) | Gather selected paged KV cache rows into contiguous storage. |
| [`copy_qsa_fia_output`](copy_qsa_fia_output.md) | Copy the FIA result to the QSA output buffer. |
| [`dcp_a2a_fused`](dcp_a2a_fused.md) | Exchange partial attention results across context-parallel ranks. |
| [`dequantize_packed_engram`](dequantize_packed_engram.md) | Decode packed Engram embedding weights. |
| [`distributed_topk`](distributed_topk.md) | Select global or owner-local winners across DCP ranks. |
| [`fold_indexer_cache_rows`](fold_indexer_cache_rows.md) | Fold updated cache rows into groups of eight. |
| [`fused_dcp_lse_combine`](fused_dcp_lse_combine.md) | Merge rank-local attention outputs using packed log-sum-exp values. |
| [`fused_eh_norm`](fused_eh_norm.md) | Mask embeddings and normalize two GLM MTP projection inputs. |
| [`fused_engram_gate`](fused_engram_gate.md) | Apply the fused Engram gate and residual update. |
| [`fused_gdn_gating_patch`](fused_gdn_gating_patch.md) | Compute the two gates used in Gated DeltaNet attention. |
| [`fused_qknorm_rope`](fused_qknorm_rope.md) | Split Step5 QKV projections, normalize Q/K and rotate them. |
| [`fused_qkvzba_split_reshape_cat`](fused_qkvzba_split_reshape_cat.md) | Unpack fused GDN projections into type-contiguous layouts. |
| [`gather_initial_states`](gather_initial_states.md) | Gather recurrent state and zero rows for fresh sequences. |
| [`gather_resident_rows`](gather_resident_rows.md) | Gather requested rows from resident KV storage. |
| [`gather_ssm_states`](gather_ssm_states.md) | Gather selected physical SSM cache rows. |
| [`get_num_nans`](get_num_nans.md) | Count NaN logits per request. |
| [`glm5_next_kpool_tail_compress_and_write_cache_triton`](glm5_next_kpool_tail_compress_and_write_cache_triton.md) | Compress completed KeyPools and update the request-owned tail. |
| [`glm5_next_lightning_indexer_triton`](glm5_next_lightning_indexer_triton.md) | Select KeyPools and expand them into causal token indices. |
| [`index_decode_topk_fused`](index_decode_topk_fused.md) | Select sparse logical blocks from MiniMax index scores. |
| [`kda_recoverssm_verify`](kda_recoverssm_verify.md) | Verify speculative KDA tokens against a paged checkpoint. |
| [`kda_state_gather`](kda_state_gather.md) | Gather KDA state into packed working storage. |
| [`kda_state_scatter`](kda_state_scatter.md) | Write packed KDA state to selected cache rows. |
| [`KDARecoverSSMCommitContext.commit`](kda_recoverssm_commit.md) | Reconstruct and commit accepted KDA state. |
| [`kimi_pixel_norm_lut`](kimi_pixel_norm_lut.md) | Normalize uint8 Kimi image patches for vision projection. |
| [`l2norm_fwd`](l2norm_fwd.md) | Normalize FLA tensor rows along the last dimension. |
| [`lightning_attention_npu`](lightning_attention_npu.md) | Run the diagonal and off-diagonal lightning attention pipeline. |
| [`linear_persistent`](linear_persistent.md) | Multiply activations by transposed linear weights with batch-invariant tiling. |
| [`lookup_quantized`](lookup_quantized.md) | Dequantize selected host embedding rows into a device output. |
| [`lora_linear`](lora_linear.md) | Accumulate a dense single-adapter LoRA update into the output. |
| [`matmul_persistent`](matmul_persistent.md) | Perform batch-invariant matrix multiplication with optional bias. |
| [`mean_dim`](mean_dim.md) | Compute a batch-invariant mean over a single dimension. |
| [`merge_packed_candidates`](merge_packed_candidates.md) | Merge gathered DCP candidate lists. |
| [`minimax_m3_index_decode`](minimax_m3_index_decode.md) | Order sparse attention candidates for MiniMax decode. |
| [`minimax_qknorm_rope_cache`](minimax_qknorm_rope_cache.md) | Normalize attention heads, apply rotary embeddings and populate KV cache. |
| [`normalize_dcp_candidates`](normalize_dcp_candidates.md) | Normalize candidate layout for DCP selection. |
| [`pack_attention_rows`](pack_attention_rows.md) | Pack attention rows for HiSparse staging. |
| [`pack_dcp_output_lse`](pack_dcp_output_lse.md) | Pack partial output and log-sum-exp into an exchange payload. |
| [`pack_keypool_indices`](pack_keypool_indices.md) | Pack selected GLM KeyPool token indices. |
| [`pack_local_candidates`](pack_local_candidates.md) | Pack local candidate IDs and scores for DCP exchange. |
| [`pack_qsa_group_indices_npu`](pack_qsa_group_indices_npu.md) | Pack complete QSA group IDs and visible residual tokens. |
| [`ple_conv_decode`](ple_conv_decode.md) | Advance PLE convolution state for a decode token. |
| [`ple_conv_prefill`](ple_conv_prefill.md) | Apply the PLE prefill convolution and update request state. |
| [`ple_conv_spec`](ple_conv_spec.md) | Apply speculative PLE convolution from the accepted state. |
| [`prep_query_head_major`](prep_query_head_major.md) | Assemble DCP query fragments in head-major order. |
| [`prepare_dflash_inputs_triton`](prepare_dflash_inputs_triton.md) | Prepare DFlash query and context buffers for speculative decoding. |
| [`prepare_gdn`](prepare_gdn.md) | Prepare aligned-cache GDN metadata and clear the convolution output. |
| [`prepare_indexer_indices`](prepare_indexer_indices.md) | Filter and order visible compressed indexer rows. |
| [`prepare_nonempty_keypool_inputs`](prepare_nonempty_keypool_inputs.md) | Build nonempty inputs for the pooled key indexer. |
| [`prepare_qsa_gbsa_inputs`](prepare_qsa_gbsa_inputs.md) | Prepare block-sparse attention inputs from selected QSA groups. |
| [`prepare_qsa_lightning_v2_inputs`](prepare_qsa_lightning_v2_inputs.md) | Prepare native LightningIndexerV2 inputs for QSA. |
| [`qsa_compress_groups_with_ratio`](qsa_compress_groups_with_ratio.md) | Compress completed groups of current and cached QSA keys. |
| [`quantize_mxfp4_indexer`](quantize_mxfp4_indexer.md) | Quantize BF16 indexer query groups into FP4 data and scales. |
| [`rejection_sample`](rejection_sample.md) | Verify speculative tokens with optional entropy-adjusted acceptance. |
| [`remap_sparse_indices_triton`](remap_sparse_indices_triton.md) | Remap sparse attention indices into rank-local coordinates. |
| [`resample`](resample.md) | Resample the rejected or bonus token during speculative decoding. |
| [`rms_norm`](rms_norm.md) | Normalize batch-invariant activations by their root mean square. |
| [`rope_forward_triton_siso`](rope_forward_triton_siso.md) | Apply rotary embeddings in place to one tensor. |
| [`rope_forward_triton`](rope_forward_triton.md) | Apply rotary embeddings to query and key tensors in place. |
| [`sampling_mask_from_logits_npu`](sampling_mask_from_logits_npu.md) | Pack finite-logit masks and count kept vocabulary entries. |
| [`scatter_k_scales`](scatter_k_scales.md) | Scatter quantization scales into the addressed KV cache rows. |
| [`scatter_packed_kv`](scatter_packed_kv.md) | Scatter packed payloads into the HiSparse KV cache. |
| [`scatter_ssm_states_`](scatter_ssm_states_.md) | Write updated SSM rows back into cache. |
| [`select_local_candidates`](select_local_candidates.md) | Select candidates from the local indexer shard. |
| [`sfa_dcp_exchange.exchange`](exchange.md) | Pack and exchange SFA partial attention results. |
| [`sfa_dcp_merge.fused_merge`](fused_merge.md) | Merge SFA partial results locally using log-sum-exp weights. |
| [`split_qkv_rmsnorm_rope_vnorm_impl`](split_qkv_rmsnorm_rope_vnorm_impl.md) | Split projected QKV, normalize all three heads and rotate Q/K. |
| [`store_indexer_key_scale`](store_indexer_key_scale.md) | Publish quantized indexer keys and their scales. |
| [`swiglu_quant`](swiglu_quant.md) | Apply SwiGLU with optional per-token INT8 quantization. |
| [`swiglustep_forward_triton`](swiglustep_forward_triton.md) | Apply the Step model's gated expert activation. |
| [`triton_q_rms`](triton_q_rms.md) | Normalize query rows with a runtime UB-dependent tile size. |
| [`triton_qsa_rmsnorm_mrope`](triton_qsa_rmsnorm_mrope.md) | Apply QSA weighted RMS normalization and rotary encoding. |
| [`triton_split_qkv_rmsnorm_mrope`](triton_split_qkv_rmsnorm_mrope.md) | Split QKV and optional gate, normalize Q/K and apply rotary encoding. |
| [`turboquant_finalize`](turboquant_finalize.md) | Correct TurboQuant scales and pack rows for cache storage. |
| [`update_decode`](update_decode.md) | Update Step5 region summaries for a decode token. |
| [`update_prefill`](update_prefill.md) | Update Step5 region summaries for a prefill chunk. |
| [`update_verify`](update_verify.md) | Update Step5 state in token order during speculative verification. |
| [`write_keypool_caches`](write_keypool_caches.md) | Write GLM KeyPool cache entries. |
| [`write_mxfp4_indexer_cache`](write_mxfp4_indexer_cache.md) | Quantize indexer keys and write valid cache slots. |
| [`write_pcp_ring_rows`](write_pcp_ring_rows.md) | Write gathered PCP request tails to the compressor ring. |
| [`write_recurrent_output`](write_recurrent_output.md) | Write valid recurrent KDA rows and clear destination padding. |
