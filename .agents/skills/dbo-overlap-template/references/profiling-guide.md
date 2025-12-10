# DBO Profiling Analysis Guide

This document explains how to analyze the communication/computation characteristics of a model based on existing Ascend NPU Profiler output directories, verify static analysis results, and measure actual communication overhead, thereby generating DBO overlap strategies based on measured data.

> **Prerequisite**: The user has completed profiling collection and provided the profiling output directory path. This guide only covers **analysis**, not collection.

---

## 1. When to Use Profiling Analysis

| Scenario | Profiling Required? |
|----------|-------------------|
| Static analysis results are clear, model structure is standard | **Not required**, use static analysis results directly |
| Model contains conditional branches (e.g., dynamic routing, optional MoE layers) | **Required**, to verify actual hook trigger behavior |
| Need to measure communication overhead and optimize overlap benefits | **Required**, to obtain measured data |
| Static analysis cannot determine A2/A3 communication patterns | **Required**, to confirm actual communication paths via trace |
| New model structure differs significantly from known templates | **Recommended**, to avoid misjudgment |
| Need to quantify overlap effects (before/after DBO comparison) | **Required**, for comparative analysis |
| Need to identify kernel-level performance bottlenecks | **Required**, for kernel breakdown |

---

## 2. Profiling Output Directory Structure

The directory structure produced by Ascend NPU Profiler is shown below. The analysis script automatically discovers and parses the key files within:

```
<profiling_root>/
├── dp0_pp0_tp0_dcp0_ep0_rank0_<pid>_<timestamp>_ascend_pt/
│   ├── ASCEND_PROFILER_OUTPUT/
│   │   ├── op_statistic.csv          # ★ OP type aggregation statistics (primary data source for kernel breakdown)
│   │   ├── kernel_details.csv        # Detailed info for each kernel (time, stream, core type)
│   │   ├── operator_details.csv      # PyTorch operator-level details
│   │   ├── communication.json        # ★ Communication operator details (timestamp and duration for each hcom op)
│   │   ├── communication_matrix.json # ★ Communication matrix (bandwidth statistics for AllReduce/AllGather/AlltoAll)
│   │   ├── step_trace_time.csv       # ★ Step-level time breakdown (compute/comm/overlap/free)
│   │   ├── api_statistic.csv         # API call statistics
│   │   ├── trace_view.json           # Chrome trace format (viewable with Perfetto)
│   │   ├── analysis.db               # SQLite analysis database
│   │   └── ascend_pytorch_profiler_*.db
│   ├── FRAMEWORK/
│   │   └── torch.op_range            # PyTorch op range data
│   ├── PROF_<id>/
│   │   ├── device_<id>/data/         # Device-level raw data
│   │   └── host/                     # Host-side data
│   ├── profiler_info_<rank>.json     # Profiler configuration (torch_npu/CANN versions, rank_id)
│   └── profiler_metadata.json        # Environment variables and parallel group info
├── dp1_pp0_tp0_dcp0_ep8_rank0_<pid>_<timestamp>_ascend_pt/
│   └── ... (same as above, another rank)
└── cluster.db                        # Cluster-level database
```

**Key files for analysis scripts** (marked with ★):
- `op_statistic.csv`: kernel breakdown and classification
- `communication.json`: per-operator communication analysis
- `communication_matrix.json`: bandwidth and communication topology
- `step_trace_time.csv`: compute/comm overlap ratio

---

## 3. Using the Analysis Script

This skill provides a unified analysis script `scripts/analyze_ascend_profiling.py` with four subcommands:

```bash
# 1. Compact triage report (recommended default)
#    Output: Model type inference + overlap metrics + kernel breakdown + communication analysis + DBO opportunities
python3 scripts/analyze_ascend_profiling.py triage \
  --input /path/to/profiling_dir \
  --num-layers 10

# 2. Detailed kernel breakdown
python3 scripts/analyze_ascend_profiling.py breakdown \
  --input /path/to/profiling_dir \
  --top-k 50

# 3. Communication operator focused analysis
python3 scripts/analyze_ascend_profiling.py comm \
  --input /path/to/profiling_dir \
  --num-layers 10

# 4. Compare two profiling runs (e.g., different ranks or before/after DBO)
python3 scripts/analyze_ascend_profiling.py compare \
  --input-a /path/to/dir_a \
  --input-b /path/to/dir_b
```

### 3.1 Analysis Script Output Contract

#### `triage` Mode (Recommended Default)

1. **Metadata**: rank info, torch_npu/CANN versions, parallel group configuration
2. **Model Type Inference**: automatic identification of Dense/MoE/MLA+MoE, DBO template recommendation
3. **Step-Level Overlap Metrics**: time and percentage of compute/comm/overlap/free, with ASCII visualization
4. **Category Summary**: time percentage aggregated by category (gemm/norm/attention/comm/moe, etc.)
5. **Top Kernel Ops**: kernel list sorted by total time
6. **Communication Summary**: statistics by communication type (AllReduce/AllGather/AlltoAll/Broadcast)
7. **Bandwidth by Transport**: bandwidth statistics for HCCS/SIO/RDMA
8. **Per-Layer Communication**: estimated communication overhead per layer
9. **DBO Overlap Opportunities**: identified optimization opportunities and recommendations

#### `compare` Mode

1. **Overlap Metrics Comparison**: comparison of overlap metrics between two profiling runs
2. **Category Comparison**: time changes for each category

---

## 4. Interpreting Profiling Results

### 4.1 Step-Level Overlap Metrics Interpretation

```
  Total step time:           3090.03 ms
  Computing:                 1320.70 ms (42.7%)    ← NPU computation time
  Communication (total):     1270.74 ms            ← Total communication time
  Communication (exposed):   1270.67 ms (41.1%)    ← Non-overlapped communication time ★Key metric
  Overlapped:                0.07 ms (0.0%)        ← Overlapped communication time
  Free (idle):               498.66 ms (16.1%)     ← Device idle time

  [#########################!!!!!!!!!!!!!!!!!!!!!!!!~..........]
   # = compute  ! = comm(exposed)  ~ = overlap  . = free
```

**Key Judgments**:
- `Communication (exposed)` percentage > 10% → **Must implement DBO overlap**
- `Overlapped` close to 0% → No current communication/computation overlap, large DBO benefit potential
- `Free (idle)` > 10% → Possible CPU bottleneck or synchronization overhead

### 4.2 Kernel Classification System

The analysis script classifies Ascend NPU kernels into the following categories:

| Category | Typical OPs | Description |
|----------|------------|-------------|
| **gemm** | MatMulV3, GroupedMatmul, QuantBatchMatmulV3 | Matrix multiplication (AI_CORE) |
| **norm** | layer_norm_fwd_kernel, AddRmsNormBias, RmsNorm | Normalization (AI_VECTOR_CORE) |
| **attention** | _fwd_diag_kernel, _fwd_kv_parallel, FusedInferAttentionScore | Attention computation |
| **communication** | allreduceAicpuKernel, allgatherAicpuKernel | Communication operators (AI_CPU) |
| **moe** | MoeTokenPermute, MoeTokenUnpermute, MoeGatingTopK | MoE routing and token dispatch |
| **activation** | SwiGlu, _swiglu_quant_kernel, muls_add_kernel | Activation functions |
| **quantize** | DynamicQuant | Quantization operators |
| **rope** | _triton_rope, InterleaveRope | Rotary Position Embedding |
| **memory** | TensorMove, ConcatD, Transpose, Slice | Memory operations |
| **elementwise** | Cast, Add, Mul, Sub | Element-wise operations |
| **sampling** | SoftmaxV2, ArgMaxV2, ReduceSum | Sampling related |

### 4.3 Communication Operator Interpretation

#### Communication Types and DBO Mode Correspondence

| Communication Type | Meaning | DBO Mode | Corresponding Hook |
|-------------------|---------|----------|-------------------|
| **AllReduce** | ReduceScatter within TP (after o_proj/MLP) | A2/A3 universal | `dbo_linear_row_hook` |
| **AllGather** | AllGather within TP (before QKV/MLP) or MoE dispatch within EP | A2 | `dbo_linear_column_hook` / `dbo_moe_prepare_hook` |
| **AlltoAll** | MoE token dispatch/combine between EP | A3 | `dbo_moe_prepare_hook` / `dbo_moe_finalize_hook` |
| **Broadcast** | MoE routing information broadcast | — | Usually no overlap needed |

#### Communication Overhead Reference Values and Overlap Priority

```
Communication overhead > 1000μs  → Highest priority overlap, must implement
Communication overhead 500-1000μs → High priority, strongly recommended
Communication overhead 100-500μs  → Medium priority, recommended
Communication overhead < 100μs    → Low priority, limited overlap benefit
Communication overhead < 50μs     → Consider skipping (overlap overhead may exceed benefit)
```

**Typical Overhead Reference** (910B, EP=8/16, prefill bs=4):

| Communication Operator | Typical Overhead | Overlap Priority |
|----------------------|-----------------|-----------------|
| MoE AlltoAll Dispatch (A3, EP=16) | 1000-2000μs | Highest |
| MoE AlltoAll Combine (A3, EP=16) | 1000-2000μs | Highest |
| AllReduce (TP=8) | 1200-1800μs | Highest |
| AllGather (EP=8) | 600-1000μs | High |
| Broadcast (MoE routing) | 5-100μs | Low (usually skip) |

### 4.4 Automatic Model Type Inference

The analysis script automatically infers model type through the following features:

| Feature | Inference Result | Suggested Template |
|---------|-----------------|-------------------|
| Has MoeTokenPermute + AlltoAll | MoE (A3 AlltoAll) | MoEAlltoallTemplate |
| Has MoeTokenPermute + AllGather (no AlltoAll) | MoE (A2 AllGather) | MoEAllgatherTemplate |
| Has _fwd_diag_kernel + _fwd_kv_parallel | MLA attention | DeepseekAlltoallTemplate / DeepseekAllgatherTemplate |
| No MoE related OPs | Dense | DenseAllgatherTemplate |

### 4.5 Multi-Rank Comparative Analysis

When the profiling directory contains data from multiple ranks, the analysis script outputs separate reports for each rank. Key focus areas:

| Comparison Dimension | Normal Case | Abnormal Signal |
|---------------------|-------------|-----------------|
| Compute time across ranks | Difference < 5% | Difference > 10% → Load imbalance |
| Communication time across ranks | Difference < 10% | Difference > 20% → Communication topology issue |
| AlltoAll overhead between EP ranks | Similar | Certain ranks significantly higher → Network hotspot |
| Kernel distribution between DP ranks | Same | Different → Data distribution imbalance |

---

## 5. From Profiling Results to Template Generation

### 5.1 Decision Flow

```
Given profiling directory path
    │
    ├─ Run triage subcommand
    │
    ├─ Check step_trace_time.csv
    │   └─ Communication(exposed) > 10% → Confirm need for DBO overlap
    │
    ├─ Check communication.json
    │   ├─ Has AlltoAll → A3 mode
    │   ├─ Has AllGather (no AlltoAll) → A2 mode
    │   └─ Only AllReduce → Dense mode
    │
    ├─ Check op_statistic.csv
    │   ├─ Has MoeTokenPermute → MoE model
    │   ├─ Has _fwd_diag_kernel → MLA attention
    │   └─ None of above → Dense model
    │
    └─ Select template
        ├─ Dense → DenseAllgatherTemplate
        ├─ MoE + A2 → MoEAllgatherTemplate
        ├─ MoE + A3 → MoEAlltoallTemplate
        ├─ MLA+MoE + A2 → DeepseekAllgatherTemplate
        └─ MLA+MoE + A3 → DeepseekAlltoallTemplate
```

### 5.2 Template Optimization Based on Overhead Data

If profiling shows a communication operator has very short overhead (< 50μs), that hook can be skipped in the template:

```python
# Example: If Broadcast overhead is very short, no overlap needed
class OptimizedMoEAlltoallTemplate(UbatchOverlapBaseTemplate):
    def dbo_linear_column_hook(self, is_record):
        # ... normal implementation (AllGather ~1000μs, worth overlapping)

    def dbo_linear_row_hook(self, is_record):
        # ... normal implementation (AllReduce ~1500μs, worth overlapping)

    def dbo_moe_prepare_hook(self, is_record):
        # ... normal implementation (AlltoAll ~1500μs, worth overlapping)

    def dbo_moe_finalize_hook(self, is_record):
        # ... normal implementation (AlltoAll ~1500μs, worth overlapping)

    # Broadcast doesn't need overlap (< 100μs)
```

### 5.3 Before/After DBO Comparison Verification

Use the `compare` subcommand to verify DBO overlap effectiveness:

```bash
python3 scripts/analyze_ascend_profiling.py compare \
  --input-a /path/to/baseline_profiling \
  --input-b /path/to/dbo_enabled_profiling
```

**Expected Improvements**:
- `Communication (exposed)` significantly decreases
- `Overlapped` significantly increases
- `Total step time` decreases (ideally approaching max(compute, comm))

---

## 6. Verified Analysis Cases

### 6.1 DeepSeek MoE + MLA (prefill, bs=4, EP=8/16)

**Environment**: 910B × 16, torch_npu 2.9.0, CANN 8.5.0

**Rank 0 (dp=0, ep=0) Analysis Results**:
```
Model Type Inference: MoE (A3 AlltoAll) + MLA attention
Suggested Template: DeepseekAlltoallTemplate

Step-Level Overlap:
  Total: 3090ms, Compute: 1321ms (42.7%), Comm exposed: 1271ms (41.1%)
  Overlap: 0.07ms (0.0%), Free: 499ms (16.1%)

Category Distribution:
  gemm: 34.3%, norm: 21.4%, attention: 18.0%, memory: 7.3%
  communication: 5.7%, moe: 3.3%, activation: 2.8%, quantize: 2.6%

Communication Statistics:
  AlltoAll: 456 ops, avg 1.42ms → Highest priority overlap
  AllReduce: 322 ops, avg 1.28ms → Highest priority overlap
  AllGather: 230 ops, avg 0.67ms → High priority overlap
  Broadcast: 608 ops, avg 0.09ms → Low priority (skip)

Bandwidth:
  HCCS: 120.7 GB/s (AllGather), 121.3 GB/s (AllReduce)
  SIO:  123.4 GB/s (AllGather), 123.3 GB/s (AllReduce)
```

**Key Findings**:
1. **Severe Communication Exposure**: 41.1% of time spent on non-overlapped communication, huge DBO benefit potential
2. **Overlap Nearly Zero**: No current communication/computation overlap
3. **High Norm Percentage**: 21.4%, AddRmsNormBias fusion enabled but still the second largest category
4. **AlltoAll is Biggest Communication Bottleneck**: ~3 AlltoAll per layer, each ~1.4ms
5. **Device Idle 16.1%**: Possible CPU-side scheduling overhead

**Rank 8 (dp=1, ep=8) Comparison**:
- More severe communication exposure: 48.9% (vs 41.1% for rank 0)
- Higher AllGather overhead: avg 2.52ms (vs 0.67ms for rank 0)
- Longer total time: 5493ms (vs 3090ms for rank 0)
- **Conclusion**: ep=8 rank handles more MoE expert computation, higher communication overhead