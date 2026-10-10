# Fine-Grained Tensor Parallelism (Fine-grained TP)

## Overview

Fine-Grained Tensor Parallelism (Fine-grained TP) extends standard tensor parallelism by enabling **independent tensor-parallel sizes for different model components**. Instead of applying a single global `tensor_parallel_size` to all layers, Fine-grained TP allows users to configure separate TP sizes for key modules — such as embedding, LM head, attention output projection (o_proj), and MLP blocks — via the `finegrained_tp_config` parameter.

This capability supports heterogeneous parallelism strategies within a single model, providing finer control over weight distribution, memory layout, and communication patterns across devices. The feature is compatible with MoE transformer architectures and integrates seamlessly into vLLM’s serving pipeline.

Fine-grained TP delivers two primary performance advantages through targeted weight sharding:

- **Reduced Per-Device Memory Footprint**:  
  Fine-grained TP shards large weight matrices (e.g., LM head, o_proj) across devices, lowering peak memory usage and enabling larger batches or deployment on memory-limited hardware, all without quantization.
  
- **Faster Memory Access in GEMMs**:  
  In decode-heavy workloads, GEMM performance is often memory-bound. Weight sharding reduces per-device weight fetch volume, cutting DRAM traffic and improving bandwidth efficiency, especially for latency-sensitive layers like LM head and o_proj.

In the measured DeepSeek-R1-W8A8 deployment below, the four knobs together saved **9.72 GB per card** with a net TPOT improvement (see [Experimental Results](#experimental-results)). This guide reflects the feature on the v0.30.0 release line (current main).

## How It Works

The four components fall into two families with different communication contracts, and the family a knob belongs to determines where it can run:

- **embedding / LM head (capacity-based exchanges)**: the cross-DP exchange pads every step to a fixed capacity before the collectives and trims afterwards, so these components are independent of the per-step token count and work in eager and graph mode, in both prefill and decode.
- **o_proj / MLP (uniform-token exchanges)**: the cross-DP exchange requires every rank in the group to forward the same number of tokens on every step. This count is only guaranteed by graph dispatch in a P/D-disaggregated decode deployment, so `oproj_tensor_parallel_size > 1` and `mlp_tensor_parallel_size > 1` are only supported in graph mode during decode and carry the deployment preconditions listed [below](#preconditions-for-o_proj--mlp-tp). For `o_proj`, dummy_run in eager mode additionally never triggers o_proj.

## Supported Scenarios

The four knobs are freely combinable — on a PD decode node all four can be enabled together (this is the configuration measured in [Experimental Results](#experimental-results)); each knob only needs to satisfy its own family's constraints.

| Scenario | Components that can be enabled | Applicable Conditions |
|----------|-------------------------------|------------------------|
| All-DP MoE serving (standalone, or the decode side of PD separation) | embedding / LM head — o_proj and MLP TP additionally require a PD decode node | MoE model, `tensor_parallel_size == 1`, sizes evenly divide `data_parallel_size` |
| P/D-disaggregated decode (D) node | all four together: o_proj / MLP / embedding / LM head | The o_proj / MLP knobs require the full [preconditions for o_proj / MLP TP](#preconditions-for-o_proj--mlp-tp) |

### Component & Execution Mode Support

| TP config     | Eager | Graph | Hybrid | Prefill | Decode |
| ------------- | ----- | ----- | ------ | ------- | ------ |
| **embedding** | ✅     | ✅     | ✅      | ✅       | ✅      |
| **o_proj**    | ❌     | ✅     | ❌      | ❌       | ✅      |
| **mlp**       | ❌     | ✅     | ❌      | ❌       | ✅      |
| **LM head**   | ✅     | ✅     | ✅      | ✅       | ✅      |

> ⚠️ Note:  
>
> - Both `o_proj` TP and MLP TP additionally require `tensor_parallel_size == 1` (enforced at config load); see [Standard Tensor Parallelism Requirement](#standard-tensor-parallelism-requirement) below.
> - LM head TP can be combined with speculative decoding (EAGLE, DFlash, and DSpark draft models).

## Constraints and Limitations

| Category | Description |
|----------|-------------|
| Model | MoE models only — see [Models](#models) for how to check a checkpoint |
| Deployment Scenario | embedding / LM head TP: all-DP MoE serving or PD decode nodes; o_proj / MLP TP: P/D-disaggregated decode nodes only |
| Standard TP | Fine-grained sizes require, or are only effective under, `tensor_parallel_size == 1`; see [Standard Tensor Parallelism Requirement](#standard-tensor-parallelism-requirement) |
| Feature Mutual Exclusion | `prefill_context_parallel_size > 1` cannot be combined with o_proj / MLP TP |
| Hardware | No config-enforced hardware restriction; the performance data in this guide was measured on Atlas A2 (see [Experimental Results](#experimental-results)) |

### Models

Fine-grained TP currently supports **MoE models only**. The constraint is enforced at configuration load: a non-MoE model fails at startup with `The finegrained tp sizes can be enabled only for MOE models`.

To check whether a checkpoint qualifies, look at its `config.json`: the model counts as a MoE model when the config exposes routed experts through any of the fields `n_routed_experts` (DeepSeek-style), `num_local_experts` (Mixtral-style), `num_experts`, `moe_num_experts`, or MoE blocks under `block_configs`. Checkpoints such as DeepSeek-V3/R1, the Qwen3 MoE series, GLM MoE variants, Kimi-K2, and MiniMax-M3 qualify; dense checkpoints such as Llama or the dense Qwen series do not.

The restriction comes from the sharding axis: fine-grained TP shards weights across the data-parallel (DP) dimension, and only MoE deployments keep a cross-rank DP group — for a dense model, every DP rank runs as an independent DP=1 engine, leaving no group to shard across.

Within a qualifying MoE model, the sharded components may include the dense layers — for example, the first three dense layers of DeepSeek-R1 are sharded by `mlp_tensor_parallel_size` like the other MLP layers.

### Preconditions for o_proj / MLP TP

`oproj_tensor_parallel_size > 1` and `mlp_tensor_parallel_size > 1` are validated at startup and require all of the following (invalid configurations fail at config load with an explicit error):

- a MoE model with `tensor_parallel_size == 1` (and a `data_parallel_size` that the TP size evenly divides);
- a P/D-disaggregated deployment, on the decode (D) node only (`kv_role = kv_consumer`);
- the recompute scheduler: `scheduler_config.recompute_scheduler_enable = true` in `--additional-config`;
- `PreemptOffloadConnector` in the KV connector chain — combine the P/D transfer connector (e.g. `MooncakeConnectorV1`) and `PreemptOffloadConnector` via `MultiConnector` (see the [Preempt Offload Guide](preempt_offload_connector.md));
- a graph mode: do not start the instance with `--enforce-eager`;
- `prefill_context_parallel_size == 1`.

If the largest cudagraph capture size does not cover the largest possible step (`min(max_num_batched_tokens, max_num_seqs * (1 + num_speculative_tokens))`), both `oproj_tensor_parallel_size` and `mlp_tensor_parallel_size` are **disabled automatically at startup with a warning**, and the deployment still starts without them; raise `max_cudagraph_capture_size` to re-enable. At runtime, a step dispatched outside the captured graphs fails the affected request with an explicit error instead of hanging the cross-DP collectives.

### Configuration Limit

The Fine-Grained TP size for any component must:

- Be **≤ the data-parallel (DP) size**, and  
- **Evenly divide the DP size** (i.e., `dp_size % tp_size == 0`) to ensure valid device assignment and communication grouping.

Violating these constraints fails at configuration load with `finegrained tp sizes must divide by data_parallel_size`.

### Standard Tensor Parallelism Requirement

Fine-grained TP shards the configured layer **across the data-parallel (DP) dimension**, not the standard tensor-parallel (TP) dimension. Its interplay with the standard `tensor_parallel_size` therefore differs per component:

| Component | Behavior with `tensor_parallel_size > 1` |
|-----------|------------------------------------------|
| `o_proj` / `mlp` | **Not supported.** For `o_proj`, the DSA attention output is reshaped with `n_local_groups = n_groups // tp_size` (standard TP), while the wo_a/wo_b weights are sharded by the OTP group (DP dimension); the two axes no longer align. Both knobs are therefore rejected at config load unless `tensor_parallel_size == 1`. |
| `embedding` / LM head | **Supported, but only effective under `tensor_parallel_size == 1`.** Both components are sharded along the fine-grained (DP-realm) group, whose process group is built along the DP axis at a fixed `tp_idx` — orthogonal to the standard TP axis. When `tensor_parallel_size > 1`, the standard TP sharding and the fine-grained sharding operate on different axes of the rank grid and can no longer compose: the fine-grained group's ranks all share the same standard-TP weight shard, so fine-grained TP cannot deliver additional sharding. In other words, these components are designed for the all-DP (`tensor_parallel_size == 1`) decode scenario; under `tensor_parallel_size > 1` the standard TP dimension takes over and the fine-grained configuration does not take effect. |

---

## Usage

### Scenario 1 — embedding + LM head TP (all-DP MoE serving)

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "finegrained_tp_config": {
            "embedding_tensor_parallel_size": 8,
            "lmhead_tensor_parallel_size": 8
        }
    }'
```

### Scenario 2 — all four knobs on a P/D-disaggregated decode node

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "scheduler_config": {
            "recompute_scheduler_enable": true
        },
        "finegrained_tp_config": {
            "oproj_tensor_parallel_size": 8,
            "mlp_tensor_parallel_size": 8,
            "lmhead_tensor_parallel_size": 8,
            "embedding_tensor_parallel_size": 8
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_consumer",
                    "kv_port": "28000",
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": 1},
                        "decode": {"dp_size": 8, "tp_size": 1}
                    }
                },
                {
                    "kv_connector": "PreemptOffloadConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use_per_rank": 17179869184
                    }
                }
            ]
        }
    }'
```

> ⚠️ Note:  
>
> - Scenario 2 must run in graph mode (do not pass `--enforce-eager`), and its prefill node keeps a plain producer configuration (e.g. `MooncakeConnectorV1` with `kv_role = kv_producer`). See the [Preempt Offload Guide](preempt_offload_connector.md) for the `PreemptOffloadConnector` parameters and the P/D disaggregation tutorials for the end-to-end deployment.

## Verifying the Feature

After the instance starts, confirm which knobs actually took effect:

1. Check the startup log for the enabled-knob summary:

   ```bash
   grep "finegrained_tp_config enabled" <serve-log>
   ```

   Expected output (Scenario 1):

   ```text
   finegrained_tp_config enabled: lmhead_tensor_parallel_size=8, embedding_tensor_parallel_size=8
   ```

   Expected output (Scenario 2):

   ```text
   finegrained_tp_config enabled: oproj_tensor_parallel_size=8, mlp_tensor_parallel_size=8, lmhead_tensor_parallel_size=8, embedding_tensor_parallel_size=8
   ```

   Only the knobs that survived validation are listed; the format is `knob=size`.

2. If `oproj_tensor_parallel_size` / `mlp_tensor_parallel_size` are missing from that line, check for the capture-bound auto-disable warning:

   ```bash
   grep "Disabling oproj_tensor_parallel_size" <serve-log>
   ```

   It fires when the largest cudagraph capture size does not cover the largest possible step; raise `max_cudagraph_capture_size` to re-enable both knobs.

3. Send an inference request to confirm the service serves normally with the knobs on (any standard client works):

   ```bash
   curl http://<ip>:<port>/v1/completions \
       -H "Content-Type: application/json" \
       -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "The future of AI is", "max_tokens": 32, "temperature": 0}'
   ```

   Expected result: HTTP 200 with a `choices` field in the response.

## Configuration Parameters

All knobs live under `finegrained_tp_config` inside `--additional-config`. The default `0` disables a knob.

| Parameter | Type | Default | Required | Value Range | Description |
|-----------|------|---------|----------|-------------|-------------|
| `oproj_tensor_parallel_size` | int | 0 | No | 0, or a divisor of `data_parallel_size` | TP size of the attention output projection (`wo_a`/`wo_b`). Values > 1 require the [o_proj / MLP preconditions](#preconditions-for-o_proj--mlp-tp). |
| `lmhead_tensor_parallel_size` | int | 0 | No | 0, or a divisor of `data_parallel_size` | TP size of the LM head (shards the vocabulary dimension). |
| `embedding_tensor_parallel_size` | int | 0 | No | 0, or a divisor of `data_parallel_size` | TP size of the token embedding table. |
| `mlp_tensor_parallel_size` | int | 0 | No | 0, or a divisor of `data_parallel_size` | TP size of the MLP (feed-forward) blocks. Values > 1 require the [o_proj / MLP preconditions](#preconditions-for-o_proj--mlp-tp). |

---

## Experimental Results

To evaluate the effectiveness of fine-grained TP in large-scale service scenarios, we use the model **DeepSeek-R1-W8A8**, deploy PD separated decode instances in an environment of 32 cards Ascend Atlas A2 inference products*64GB (A2), with parallel configuration as DP32+EP32, and fine-grained TP size of 8; the performance data is as follows.

| Module           | Memory Savings | TPOT Impact (batch=24)    |
| ---------------- | -------------- | ------------------------- |
| o_proj TP = 8    | 5.8 GB         | **+1.5 ms** (degradation) |
| LM head TP = 8   | 1.51 GB        | **−1.2 ms** (improvement) |
|  FFN TP = 8 | 0.9 GB         | **−1.0 ms** (improvement) |
| Embedding TP = 8 | 1.51 GB        | **−1.0 ms** (improvement) |
| **Total**        | **9.72 GB**    | —                         |

- We achieved significant gains in terms of high memory capacity on a single card, as well as the benefits of TPOT.

---

## ✅ Deployment Recommendations  

Fine-grained TP is the **most effective** in the **decode instance** of PD separation, where models are typically deployed in all-DP mode. In this setup, sharding weight-heavy layers reduces redundant storage and memory pressure. Accordingly, `o_proj` TP and MLP TP are validated for — and limited to — P/D-disaggregated decode nodes (see [Preconditions for o_proj / MLP TP](#preconditions-for-o_proj--mlp-tp)), while embedding and LM head TP are also applicable to all-DP deployments without PD.

---

## FAQ

### Startup fails with "The finegrained tp sizes can be enabled only for MOE models"

**Problem Description**: With any `finegrained_tp_config` size set, the instance fails at configuration load with this error.

**Cause Analysis**: The checkpoint is not a MoE model — fine-grained TP shards weights across the DP dimension, which only exists as a cross-rank group for MoE deployments.

**Solution Steps**: Check the checkpoint’s `config.json` for the expert fields listed in [Models](#models). If none are present, the model is dense and fine-grained TP cannot be enabled; use standard `tensor_parallel_size` instead.

### o_proj / MLP TP are missing from the "finegrained_tp_config enabled" log line

**Problem Description**: The service starts normally, but the startup summary only lists embedding / LM head (or is empty) although `oproj_tensor_parallel_size` / `mlp_tensor_parallel_size` were configured.

**Cause Analysis**: The largest cudagraph capture size does not cover the largest possible step, so both knobs were auto-disabled with a warning.

**Solution Steps**: Raise `max_cudagraph_capture_size` (or lower `max_num_batched_tokens` / `max_num_seqs`) so the capture bound covers `min(max_num_batched_tokens, max_num_seqs * (1 + num_speculative_tokens))`, then restart.

### Startup fails on an o_proj / MLP precondition error

**Problem Description**: With `oproj_tensor_parallel_size > 1` or `mlp_tensor_parallel_size > 1`, configuration load raises an error about graph mode, the PD scenario, the recompute scheduler, or `PreemptOffloadConnector`.

**Cause Analysis**: One or more of the [preconditions for o_proj / MLP TP](#preconditions-for-o_proj--mlp-tp) are not met.

**Solution Steps**: Follow the preconditions checklist; the most commonly missed items are `scheduler_config.recompute_scheduler_enable = true` and the `PreemptOffloadConnector` entry in the `MultiConnector` chain (see Scenario 2 above).
