# Ascend 910C vLLM Tuning Knobs Reference

Quick reference for all tunable parameters in vllm-ascend on 910C.
Use this table to decide which lever to pick next during the optimization loop.

## Tier 1 — Scheduler & Engine Shell (lowest risk, try first)

| Knob | How to set | Default | Expected impact on low-c latency | Notes |
|------|-----------|---------|----------------------------------|-------|
| `block_size` | `--block-size 128` | 128 | **High** — fewer KV pointer dereferences | 128 is recommended for 910C; required for xlite_graph |
| `enable_balance_scheduling` | `--additional-config '{"enable_balance_scheduling": true}'` | false | Medium — reduces decode stalls | Incompatible with profiling_chunk_config |
| `max_num_seqs` | `--max-num-seqs 32` | 256 | Medium at c=1–4 | Reduces scheduler overhead at low concurrency |
| `enable_chunked_prefill` | `--enable-chunked-prefill --max-num-batched-tokens 2048` | disabled | Medium — reduces head-of-line blocking | Try values: 1024 / 2048 / 4096 |
| `enable_cpu_binding` | `--additional-config '{"enable_cpu_binding": true}'` | true | Low | Disable if NUMA topology is fragmented |

## Tier 2 — Ascend Fusion & Graph Compilation

| Knob | How to set | Default | Expected impact | Notes |
|------|-----------|---------|-----------------|-------|
| `VLLM_ASCEND_ENABLE_NZ` | `VLLM_ASCEND_ENABLE_NZ=2` | 1 | **High** — reduces HBM bandwidth pressure | 0=off, 1=quant-only, 2=all ops |
| `enable_npugraph_ex` + `enable_static_kernel` | `ascend_compilation_config: {enable_static_kernel: true}` | npugraph_ex=true, static=false | **High at c=1** — eliminates JIT recompilation | Adds startup time (~minutes) |
| `fuse_norm_quant` | `ascend_compilation_config: {fuse_norm_quant: true}` | true | Medium | Fuses RMSNorm + quant into single kernel |
| `fuse_qknorm_rope` | `ascend_compilation_config: {fuse_qknorm_rope: true}` | true | Medium | Fuses QK-norm + RoPE |
| `fusion_ops_gmmswigluquant` | `ascend_fusion_config: {fusion_ops_gmmswigluquant: true}` | true | Medium (MoE) | MoE models only |
| `xlite_graph_config` | `xlite_graph_config: {enabled: true}` | disabled | **High** — Xlite backend accelerates decode | Requires block_size=128, pp=1 |

## Tier 3 — FlashComm & TP Communication Overlap (TP > 1 only)

| Knob | How to set | Default | Expected impact | Notes |
|------|-----------|---------|-----------------|-------|
| `enable_flashcomm1` | `--additional-config '{"enable_flashcomm1": true}'` | false | High — overlaps prefill allreduce | Reduces TTFT at c=1 with TP |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=<tp>` | 0 | High — overlaps decode allreduce | Reduces TPOT at low c |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1` | 0 | Medium (eager mode, TP>1) | Better in eager; less benefit in graph mode |
| `HCCL_BUFFSIZE` | `HCCL_BUFFSIZE=200` | system default | Low–Medium | Reduce HCCL fragmentation; try 100/200/400 |

## Tier 4 — Weight Prefetch & Memory Layout

| Knob | How to set | Default | Expected impact | Notes |
|------|-----------|---------|-----------------|-------|
| `VLLM_ASCEND_ENABLE_MLAPO` | `VLLM_ASCEND_ENABLE_MLAPO=1` | 1 | Medium (DeepSeek W8A8) | Disable only if memory is tight |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | 0 | Medium (MoE W8A8 + EP) | Try 0/1/2 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `=1` | 1 | Low | Fused KV cache transpose; leave enabled |

## Tier 5 — Multistream & DSA Overlap (Advanced)

| Knob | How to set | Default | Expected impact | Notes |
|------|-----------|---------|-----------------|-------|
| `multistream_overlap_shared_expert` | `additional-config` | false | Medium (MoE) | MoE shared-expert stream overlap |
| `multistream_dsv4_dsa_overlap` | `additional-config` | false | Medium | DeepSeek V4 specific |

## Incompatibility Matrix

| Lever A | Lever B | Conflict |
|---------|---------|---------|
| `profiling_chunk_config.enabled=true` | `enable_balance_scheduling=true` | Hard conflict — vllm-ascend raises ValueError |
| `xlite_graph_config.enabled=true` | `pipeline_parallel_size > 1` | Hard conflict |
| `xlite_graph_config.enabled=true` | `block_size != 128` | Warning (suboptimal) |
| `enable_static_kernel=true` | `enable_npugraph_ex=false` | Hard conflict |
| `oproj_tensor_parallel_size > 0` | `enforce_eager=true` | Hard conflict |

## Tuning order for 910C dense models (Qwen3 family)

1. `block_size=128`
2. `VLLM_ASCEND_ENABLE_NZ=2`
3. `xlite_graph_config.enabled=true` (with block_size=128)
4. `enable_static_kernel=true`
5. `enable_balance_scheduling=true`
6. `enable_flashcomm1=true` (if TP > 1)
7. `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=<tp>` (if TP > 1)

## Tuning order for 910C MoE models (DeepSeek V3/V4 family)

1. `block_size=128`
2. `VLLM_ASCEND_ENABLE_NZ=2`
3. `VLLM_ASCEND_ENABLE_MLAPO=1`
4. `fusion_ops_gmmswigluquant=true`
5. `xlite_graph_config.enabled=true`
6. `VLLM_ASCEND_ENABLE_FUSED_MC2=1` (W8A8 + EP)
7. `multistream_overlap_shared_expert=true`
8. `multistream_dsv4_dsa_overlap=true` (V4 only)
9. `enable_flashcomm1=true` (if TP > 1)
