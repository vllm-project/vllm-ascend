# Fine-Grained Tensor Parallelism (Finegrained TP)

## Overview

Fine-Grained Tensor Parallelism (Fine-grained TP) extends standard tensor parallelism by enabling **independent tensor-parallel sizes for different model components**. Instead of applying a single global `tensor_parallel_size` to all layers, Fine-grained TP allows users to configure separate TP sizes for key modules—such as embedding, language model head (lm_head), attention output projection (o_proj), and MLP blocks—via the `finegrained_tp_config` parameter.

This capability supports heterogeneous parallelism strategies within a single model, providing finer control over weight distribution, memory layout, and communication patterns across devices. The feature is compatible with standard dense transformer architectures and integrates seamlessly into vLLM’s serving pipeline.

---

## Benefits of Finegrained TP

Fine-Grained Tensor Parallelism delivers two primary performance advantages through targeted weight sharding:

- **Reduced Per-Device Memory Footprint**:  
  Fine-grained TP shards large weight matrices(如 LM Head, o_proj)across devices, lowering peak memory usage and enabling larger batches or deployment on memory-limited hardware—without quantization.
  
- **Faster Memory Access in GEMMs**:  
  In decode-heavy workloads, GEMM performance is often memory-bound. Weight sharding reduces per-device weight fetch volume, cutting DRAM traffic and improving bandwidth efficiency—especially for latency-sensitive layers like LM Head and o_proj.

Together, these effects allow practitioners to better balance memory, communication, and compute—particularly in high-concurrency serving scenarios—while maintaining compatibility with standard dense transformer models.

---

## Supported Scenarios

### Models  

Fine-grained TP is **model-agnostic** and supports all standard dense transformer architectures, including Llama, Qwen, DeepSeek (base/dense variants), and others.

### Component & Execution Mode Support  

| TP config     | Eager | Graph | Hybrid | Prefill | Decode |
| ------------- | ----- | ----- | ------ | ------- | ------ |
| **embedding** | ✅     | ✅     | ✅      | ✅       | ✅      |
| **o_proj**    | ❌     | ✅     | ❌      | ❌       | ✅      |
| **mlp**       | ✅     | ✅     | ✅      | ✅       | ✅      |
| **LMhead**    | ✅     | ✅     | ✅      | ✅       | ✅      |

> ⚠️ Note:  
>
> - `o_proj` TP is only supported in Graph mode during Decode, because dummy_run in eager mode will not trigger o_proj.
> - `mlp` TP supports dense models, or dense layers in MoE models. For example, the first three dense layers of DeepSeek-R1.

### Configuration Limit

The Fine-Grained TP size for any component must:

- Be **≤ the data-parallel (DP) size**, and  
- **Evenly divide the DP size** (i.e., `dp_size % tp_size == 0`) to ensure valid device assignment and communication grouping.

> ⚠️ Violating these constraints will result in runtime errors or undefined behavior.

---

## How to Use Finegrained TP

### Configuration Format

Fine-grained TP is controlled via the `finegrained_tp_config` field inside `--additional-config`.

```bash
--additional-config '{
    "finegrained_tp_config": {
        "embedding_tensor_parallel_size": 8,
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8
    }
}'
```

### Example Usage

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "finegrained_tp_config": {
            "embedding_tensor_parallel_size": 8,
            "lmhead_tensor_parallel_size": 8,
            "mlp_tensor_parallel_size": 8
        }
    }'
```

---

## Experimental Results

To evaluate the effectiveness of fine-grained TP in large-scale service scenarios, we use the model **DeepSeek-R1-W8A8**, deploy PD separated decode instances in an environment of 32 cards Ascend 910B*64G (A2), with parallel configuration as DP32+EP32, and fine-grained TP size of 8; the performance data is as follows.

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

Fine-grained TP is the **most effective** in the **decode instance** of PD separation, where models are typically deployed in all-DP mode. In this setup, sharding weight-heavy layers reduces redundant storage and memory pressure.
