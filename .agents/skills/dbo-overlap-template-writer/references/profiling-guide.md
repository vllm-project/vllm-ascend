# DBO Profiling Analysis Guide

This document explains how to analyze Ascend NPU Profiler output to characterize model communication/computation, validate static analysis results, and measure actual communication timing for data-driven DBO overlap strategy generation.

> **Prerequisite**: The user has completed profiling collection and provided the profiling output directory path. This guide covers **analysis only**, not collection.

---

## 1. When to Use Profiling Analysis

| Scenario | Profiling Needed? |
|----------|------------------|
| Static analysis is clear, model structure is standard | **Not needed** - use static analysis results directly |
| Model has conditional branches (dynamic routing, optional MoE layers) | **Needed** - verify actual hook triggering |
| Need to measure communication timing for overlap optimization | **Needed** - obtain real data |
| Cannot determine A2/A3 mode from static analysis | **Needed** - confirm actual communication path via trace |
| New model structure differs significantly from known templates | **Recommended** - avoid misclassification |
| Need to quantify overlap effect (before/after DBO comparison) | **Needed** - comparative analysis |
| Need to identify kernel-level performance bottlenecks | **Needed** - kernel breakdown |

---

## 2. Profiling Output Directory Structure

The Ascend NPU Profiler produces the following directory structure. The analysis script automatically discovers and parses key files:

```
<profiling_root>/
├── dp0_pp0_tp0_dcp0_ep0_rank0_<pid>_<timestamp>_ascend_pt/
│   ├── ASCEND_PROFILER_OUTPUT/
│   │   ├── op_statistic.csv          # ★ OP type aggregation (main data source for kernel breakdown)
│   │   ├── kernel_details.csv        # Per-kernel details (time, stream, core type)
│   │   ├── operator_details.csv      # PyTorch operator-level details
│   │   ├── communication.json        # ★ Communication operator details (timestamp and duration for each hcom op)
│   │   ├── communication_matrix.json # ★ Communication matrix (bandwidth stats for AllReduce/AllGather/AlltoAll)
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
│   ├── profiler_info_<rank>.json     # Profiler config (torch_npu/CANN version, rank_id)
│   └── profiler_metadata.json        # Environment variables and parallel group info
├── dp1_pp0_tp0_dcp0_ep8_rank0_<pid>_<timestamp>_ascend_pt/
│   └── ... (same structure, another rank)
└── cluster.db                        # Cluster-level database
```

**Key files for analysis script** (marked with ★):
- `op_statistic.csv`: Kernel breakdown and classification
- `communication.json`: Per-operator communication analysis
- `communication_matrix.json`: Bandwidth and communication topology
- `step_trace_time.csv`: Compute/comm overlap ratio

---

## 3. Using the Analysis Script

This skill provides a unified analysis script `scripts/analyze_ascend_profiling.py` with five subcommands:

```bash
# 1. Compact triage report (recommended default)
#    Output: Model type inference + overlap metrics + kernel breakdown + comm analysis + DBO opportunities
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

# 4. Communication call stack extraction (trace Python call sources from trace_view.json)
#    Output: Python call stack + code location + cross-reference with communication.json for each comm op
python3 scripts/analyze_ascend_profiling.py stack \
  --input /path/to/profiling_dir \
  [--max-depth 20]       # Maximum call stack depth, default 20
  [--full-stack]         # Show full call stack (including torch/torch_npu internal frames)
  [--no-dedup]           # No deduplication (default: dedup communication events with same call stack)
  [--top-comm-types 3]   # [OPTIMIZED] Only extract stacks for top 3 most frequent communication types (recommended)

# 5. Two profiling comparison (e.g., different ranks or before/after DBO)
python3 scripts/analyze_ascend_profiling.py compare \
  --input-a /path/to/dir_a \
  --input-b /path/to/dir_b
```

### Optimization: High-Frequency Communication Stack Extraction

The `trace_view.json` file is typically very large (hundreds of MB to GB). Loading call stacks for all communication operators at once consumes significant memory and time.

**Recommended approach**: Use `--top-comm-types` and `--max-stacks` parameters to only extract stacks for the most frequent communication types:

```bash
# Only extract stacks for top 3 most frequent communication types (recommended, default)
python3 scripts/analyze_ascend_profiling.py stack \
  --input /path/to/profiling_dir \
  --top-comm-types 3 \
  --max-stacks 50

# Output example:
# ### Filtering by top-3 communication types: ['allreduce', 'alltoallv', 'allgather']
```

**How it works**:
1. **Phase 1**: Count frequency of each communication type from `communication.json`, select top-K
2. **Phase 2**: Only extract stack info for these communication types when parsing `trace_view.json`
3. **Early termination**: Exit after finding enough unique stacks

**Performance comparison** (example with 1GB trace_view.json):

| Configuration | Time | Peak Memory |
|---------------|------|-------------|
| `--top-comm-types 0 --max-stacks 0` (extract all) | ~60s | ~2GB |
| `--top-comm-types 3 --max-stacks 50` (default) | ~15s | ~500MB |

### stack Subcommand Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-comm-types` | 3 | Only extract stacks for top K most frequent communication types, 0 means all |
| `--max-stacks` | 50 | Maximum unique stacks to collect, 0 means no limit |
| `--max-depth` | 20 | Maximum call stack depth |
| `--full-stack` | False | Show full call stack (including framework internal frames) |
| `--no-dedup` | False | Do not deduplicate communication events with same call stack |

### 3.1 Analysis Script Output Contract

#### `triage` Mode (Recommended Default)

1. **Metadata**: rank info, torch_npu/CANN version, parallel group config
2. **Model Type Inference**: Auto-detect Dense/MoE/MLA+MoE, recommend DBO template
3. **Step-Level Overlap Metrics**: compute/comm/overlap/free time and ratios, with ASCII visualization
4. **Category Summary**: Time breakdown by category (gemm/norm/attention/comm/moe etc.)
5. **Top Kernel Ops**: Kernel list sorted by total time
6. **Communication Summary**: Stats by communication type (AllReduce/AllGather/AlltoAll/Broadcast)
7. **Bandwidth by Transport**: Bandwidth stats for HCCS/SIO/RDMA
8. **Per-Layer Communication**: Estimated communication overhead per layer
9. **DBO Overlap Opportunities**: Identified optimization opportunities and recommendations

#### `compare` Mode

1. **Overlap Metrics Comparison**: Comparison of overlap metrics between two profiling runs
2. **Category Comparison**: Time changes by category

#### `stack` Mode

1. **Communication Call Stacks**: Python call stacks for each unique call site, grouped by communication type (AllReduce/AllGather/AlltoAll/ReduceScatter/Broadcast)
2. **Code Location**: Most relevant vllm/vllm-ascend source location for each comm op (file path + line number + function name)
3. **Communication Code Location Summary**: Summary table of all comm call sites (comm type, op name, code location, duration)
4. **Cross-Reference**: Cross-reference with `communication.json`, aggregate average duration and overlap priority (P0-P4) by code location

> **Use Case**: When static analysis cannot determine the source of a communication operator, use the `stack` subcommand to precisely locate the Python code that triggers it from `trace_view.json`. This is an important auxiliary tool for SKILL.md Step 3.3 (locating code positions for missing hooks).

---

## 4. Interpreting Profiling Results

### 4.1 Step-Level Overlap Metrics

```
  Total step time:           3090.03 ms
  Computing:                 1320.70 ms (42.7%)    ← NPU computation time
  Communication (total):     1270.74 ms            ← Total communication time
  Communication (exposed):   1270.67 ms (41.1%)    ← Non-overlapped communication time ★ Key metric
  Overlapped:                0.07 ms (0.0%)        ← Overlapped communication time
  Free (idle):               498.66 ms (16.1%)     ← Device idle time

  [#########################!!!!!!!!!!!!!!!!!!!!!!!!~..........] 
   # = compute  ! = comm(exposed)  ~ = overlap  . = free
```

**Key judgments**:
- `Communication (exposed)` ratio > 10% → **DBO overlap must be implemented**
- `Overlapped` close to 0% → No current communication/computation overlap, high DBO benefit potential
- `Free (idle)` > 10% → Possible CPU bottleneck or synchronization overhead

### 4.2 Kernel Classification System

The analysis script classifies Ascend NPU kernels into the following categories:

| Category | Typical OPs | Description |
|----------|-------------|-------------|
| **gemm** | MatMulV3, GroupedMatmul, QuantBatchMatmulV3 | Matrix multiplication (AI_CORE) |
| **norm** | layer_norm_fwd_kernel, AddRmsNormBias, RmsNorm | Normalization (AI_VECTOR_CORE) |
| **attention** | _fwd_diag_kernel, _fwd_kv_parallel, FusedInferAttentionScore | Attention computation |
| **communication** | allreduceAicpuKernel, allgatherAicpuKernel | Communication operators (AI_CPU) |
| **moe** | MoeTokenPermute, MoeTokenUnpermute, MoeGatingTopK | MoE routing and token dispatch |
| **activation** | SwiGlu, _swiglu_quant_kernel, muls_add_kernel | Activation functions |
| **quantize** | DynamicQuant | Quantization operators |
| **rope** | _triton_rope, InterleaveRope | Rotary position embedding |
| **memory** | TensorMove, ConcatD, Transpose, Slice | Memory operations |
| **elementwise** | Cast, Add, Mul, Sub | Elementwise operations |
| **sampling** | SoftmaxV2, ArgMaxV2, ReduceSum | Sampling related |

### 4.3 Communication Operator Interpretation

#### Communication Types and DBO Modes

| Communication Type | Meaning | DBO Mode | Corresponding Hook |
|-------------------|---------|----------|-------------------|
| **AllReduce** | ReduceScatter within TP (after o_proj/MLP) | A2/A3 common | `dbo_linear_row_hook` |
| **AllGather** | AllGather within TP (before QKV/MLP) or EP MoE dispatch | A2 | `dbo_linear_column_hook` / `dbo_moe_prepare_hook` |
| **AlltoAll** | MoE token dispatch/combine across EP | A3 | `dbo_moe_prepare_hook` / `dbo_moe_finalize_hook` |
| **Broadcast** | MoE routing info broadcast | — | Usually no overlap needed |

#### Communication Duration Reference and Overlap Priority

```
Duration > 1000μs   → Highest priority overlap, must implement
Duration 500-1000μs → High priority, strongly recommended
Duration 100-500μs  → Medium priority, recommended
Duration < 100μs    → Low priority, limited overlap benefit
Duration < 50μs     → Consider skipping (overlap overhead may exceed benefit)
```

**Typical duration reference** (910B, EP=8/16, prefill bs=4):

| Communication Operator | Typical Duration | Overlap Priority |
|-----------------------|------------------|------------------|
| MoE AlltoAll Dispatch (A3, EP=16) | 1000-2000μs | Highest |
| MoE AlltoAll Combine (A3, EP=16) | 1000-2000μs | Highest |
| AllReduce (TP=8) | 1200-1800μs | Highest |
| AllGather (EP=8) | 600-1000μs | High |
| Broadcast (MoE routing) | 5-100μs | Low (usually skip) |

### 4.4 Automatic Model Type Inference

The analysis script automatically infers model type from the following features:

| Feature | Inference Result | Suggested Template |
|---------|-----------------|-------------------|
| Has MoeTokenPermute + AlltoAll | MoE (A3 AlltoAll) | MoEAlltoallTemplate |
| Has MoeTokenPermute + AllGather (no AlltoAll) | MoE (A2 AllGather) | MoEAllgatherTemplate |
| Has _fwd_diag_kernel + _fwd_kv_parallel | MLA attention | DeepseekAlltoallTemplate / DeepseekAllgatherTemplate |
| No MoE related OPs | Dense | DenseAllgatherTemplate |

### 4.5 Multi-Rank Comparative Analysis

When the profiling directory contains data from multiple ranks, the analysis script outputs reports for each rank separately. Focus areas:

| Comparison Dimension | Normal | Anomaly Signal |
|---------------------|--------|----------------|
| Compute time across ranks | Variance < 5% | Variance > 10% → Load imbalance |
| Communication time across ranks | Variance < 10% | Variance > 20% → Communication topology issue |
| AlltoAll duration across EP ranks | Similar | Certain ranks significantly higher → Network hotspot |
| Kernel distribution across DP ranks | Same | Different → Data distribution imbalance |

---

## 5. From Profiling Results to Template Generation

### 5.1 Decision Flow

```
Given profiling directory path
    │
    ├─ Run triage subcommand
    │
    ├─ Check step_trace_time.csv
    │   └─ Communication(exposed) > 10% → Confirm DBO overlap needed
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

### 5.2 Template Optimization Based on Duration Data

#### 5.2.1 Applying Communication Scheduling Principles

When profiling provides actual durations for communication operators and computation segments, apply the two core principles:

**Step 1: Apply Principle 1 (Consecutive Communication Merging)**

Extract communication and computation sequences per layer from profiling data, mark computation durations between adjacent communications:

```
Example (extracted from profiling):
  [AllGather TP]     avg 520μs   ← QKV AllGather
  [Computation]      avg 2100μs  ← Attention
  [ReduceScatter TP] avg 450μs   ← o_proj RS
  [Computation]      avg 75μs    ← LayerNorm + MoE routing ← Very short!
  [AlltoAll EP]      avg 1350μs  ← Dispatch AlltoAll
  [Computation]      avg 850μs   ← Expert compute
  [AlltoAll EP]      avg 1280μs  ← Combine AlltoAll
  [Computation]      avg 35μs    ← token unpermute ← Very short!
  [ReduceScatter EP] avg 320μs   ← Finalize RS

Merge decision:
  o_proj RS + Dispatch AlltoAll: middle 75μs < 200μs → ✅ Merge
  Combine AlltoAll + Finalize RS: middle 35μs < 200μs → ✅ Merge
```

**Step 2: Apply Principle 2 (Compute-Communication Balance)**

Calculate efficiency for each overlap interval:

```
Communication block A: [o_proj RS + Dispatch AlltoAll] = 450 + 1350 = 1800μs
  ↔ Counterpart [Attention] = 2100μs
  Efficiency = 1800/2100 = 86% ✅ Excellent

Communication block B: [Combine AlltoAll + Finalize RS] = 1280 + 320 = 1600μs
  ↔ Counterpart [Expert compute] = 850μs
  Efficiency = 850/1600 = 53% ⚠️ Communication too long

→ Consider splitting Finalize RS from block B:
  Block B': [Combine AlltoAll] = 1280μs
  ↔ Counterpart [Expert compute] = 850μs
  Efficiency = 850/1280 = 66% → Improved

  Block C: [Finalize RS + next layer QKV AG] = 320 + 520 = 840μs
  ↔ Counterpart [Combine AlltoAll tail]
  → Need comprehensive evaluation
```

#### 5.2.2 Skipping Low-Duration Communication

If profiling shows a communication operator has very short duration (< 50μs), the hook can be skipped in the template:

```python
# Example: If Broadcast duration is very short, no overlap needed
class OptimizedMoEAlltoallTemplate(UbatchOverlapBaseTemplate):
    def dbo_linear_column_hook(self, is_record):
        # ... Normal implementation (AllGather ~1000μs, worth overlapping)
    
    def dbo_linear_row_hook(self, is_record):
        # ... Normal implementation (AllReduce ~1500μs, worth overlapping)
    
    def dbo_moe_prepare_hook(self, is_record):
        # ... Normal implementation (AlltoAll ~1500μs, worth overlapping)
    
    def dbo_moe_finalize_hook(self, is_record):
        # ... Normal implementation (AlltoAll ~1500μs, worth overlapping)
    
    # Broadcast doesn't need overlap (< 100μs)
```

### 5.3 Before/After DBO Comparison Validation

Use the `compare` subcommand to validate DBO overlap effectiveness:

```bash
python3 scripts/analyze_ascend_profiling.py compare \
  --input-a /path/to/baseline_profiling \
  --input-b /path/to/dbo_enabled_profiling
```

**Expected improvements**:
- `Communication (exposed)` significantly decreases
- `Overlapped` significantly increases
- `Total step time` decreases (ideally approaching max(compute, comm))

---

## 6. Verified Analysis Case Studies

### 6.1 DeepSeek MoE + MLA (prefill, bs=4, EP=8/16)

**Environment**: 910B × 16, torch_npu 2.9.0, CANN 8.5.0

**Rank 0 (dp=0, ep=0) Analysis Results**:
```
Model type inference: MoE (A3 AlltoAll) + MLA attention
Suggested template: DeepseekAlltoallTemplate

Step-Level Overlap:
  Total: 3090ms, Compute: 1321ms (42.7%), Comm exposed: 1271ms (41.1%)
  Overlap: 0.07ms (0.0%), Free: 499ms (16.1%)

Category distribution:
  gemm: 34.3%, norm: 21.4%, attention: 18.0%, memory: 7.3%
  communication: 5.7%, moe: 3.3%, activation: 2.8%, quantize: 2.6%

Communication stats:
  AlltoAll: 456 ops, avg 1.42ms → Highest priority overlap
  AllReduce: 322 ops, avg 1.28ms → Highest priority overlap
  AllGather: 230 ops, avg 0.67ms → High priority overlap
  Broadcast: 608 ops, avg 0.09ms → Low priority (skip)

Bandwidth:
  HCCS: 120.7 GB/s (AllGather), 121.3 GB/s (AllReduce)
  SIO:  123.4 GB/s (AllGather), 123.3 GB/s (AllReduce)
```

**Key findings**:
1. **Severe communication exposure**: 41.1% time spent on non-overlapped communication, huge DBO benefit potential
2. **Almost zero overlap**: No current communication/computation overlapping
3. **High norm ratio**: 21.4%, AddRmsNormBias fusion enabled but still second largest category
4. **AlltoAll is biggest communication bottleneck**: ~3 AlltoAll per layer, ~1.4ms each
5. **16.1% device idle**: Possible CPU-side scheduling overhead

**Rank 8 (dp=1, ep=8) Comparison**:
- More communication exposure: 48.9% (vs rank 0's 41.1%)
- Higher AllGather duration: avg 2.52ms (vs rank 0's 0.67ms)
- Longer total time: 5493ms (vs rank 0's 3090ms)
- **Conclusion**: ep=8 ranks handle more MoE expert computation, higher communication overhead

---

## 7. Detecting Missing Hooks from Profiling

This section explains how to use profiling data to identify communication locations that **exist in the model but are not covered by current hook code**. This is the detailed operation guide for SKILL.md Step 2.5.

### 7.1 Why Detect Missing Hooks

The current DBO framework defines 5 standard hooks (`dbo_linear_column_hook`, `dbo_linear_row_hook`, `dbo_mla_preprocess_hook`, `dbo_moe_prepare_hook`, `dbo_moe_finalize_hook`) covering the most common communication paths. However, new models may contain uncovered communications:

| Uncovered Communication Path | Location | Communication Type | Typical Models |
|-----------------------------|----------|-------------------|----------------|
| MLP down_proj ReduceScatter | `MLPRowParallelOp.apply_impl()` | ReduceScatter (TP) | Dense models, MoE models' shared expert |
| Shared expert AllReduce | `fused_moe/` related files | AllReduce (TP) | DeepSeek V2/V3 and other MoE models with shared expert |
| OProj all_to_all (non-SequenceRowParallelOp path) | Model-specific attention implementation | AlltoAll (EP) | Some custom attention implementations |
| Additional EP AllGather/ReduceScatter | Model-specific MoE implementation | AllGather/ReduceScatter (EP) | Multi-level MoE routing models |

### 7.2 Extract Communication Operator List from communication.json

`communication.json` contains detailed information for each communication operator. Use the following methods to extract the per-layer communication operator list:

```bash
# Use the analysis script's comm subcommand
python3 scripts/analyze_ascend_profiling.py comm \
  --input /path/to/profiling_dir \
  --num-layers <num_layers>
```

**Manual parsing method** (when analysis script unavailable):

```bash
# Directly parse communication.json to extract communication operator types and counts
cat /path/to/profiling_dir/*/ASCEND_PROFILER_OUTPUT/communication.json | \
  python3 -c "
import json, sys, collections
data = json.load(sys.stdin)
# communication.json structure varies by CANN version, typically contains hcom op list
# Count occurrences of each communication type
counter = collections.Counter()
for op in data:
    op_type = op.get('type', op.get('name', 'unknown'))
    counter[op_type] += 1
for op_type, count in counter.most_common():
    print(f'{op_type}: {count} ops')
"
```

### 7.3 Comparative Analysis: Profiling vs Static Analysis

Compare observed communication operators from profiling with static analysis hook call sequences:

**Step 1**: Count communication operators per layer from profiling

```
Assume profiling shows the following communications per layer (MoE model example):
  AllGather:      2 times (QKV AllGather + MoE Prepare AllGather)
  ReduceScatter:  3 times (o_proj RS + MoE Finalize RS + MLP down_proj RS)
  AllReduce:      1 time (shared expert AllReduce)
  AlltoAll:       0 times (A2 mode)
  Broadcast:      1 time (MoE routing)
  ─────────────────
  Total:          7 communications
```

**Step 2**: Count hook calls per layer from static analysis

```
Static analysis hook call sequence (MoE model A2):
  dbo_linear_column_hook:  2 times (True + False) → Covers 1 AllGather
  dbo_linear_row_hook:     2 times (True + False) → Covers 1 ReduceScatter
  dbo_moe_prepare_hook:    2 times (True + False) → Covers 1 AllGather
  dbo_moe_finalize_hook:   2 times (True + False) → Covers 1 ReduceScatter
  ─────────────────
  Total coverage: 4 communications (AllGather×2 + ReduceScatter×2)
```

**Step 3**: Calculate the gap

```
Missing coverage:
  ReduceScatter: 3 - 2 = 1 uncovered (MLP down_proj RS)
  AllReduce:     1 - 0 = 1 uncovered (shared expert AllReduce)
  Broadcast:     1 - 0 = 1 uncovered (MoE routing, low priority)
```

### 7.4 Prioritize Missing Hooks

Based on communication duration from profiling, determine priority for each missing hook:

```
Priority criteria:
  Duration > 1000μs  → P0 (Highest priority, must add hook)
  Duration 500-1000μs → P1 (High priority, strongly recommended to add)
  Duration 100-500μs  → P2 (Medium priority, recommended to add)
  Duration 50-100μs   → P3 (Low priority, can pass in template)
  Duration < 50μs     → P4 (Can ignore, overlap overhead may exceed benefit)
```

**Example priority assessment**:

| Missing Communication | Average Duration | Priority | Recommendation |
|----------------------|------------------|----------|----------------|
| MLP down_proj ReduceScatter | 800μs | P1 | Add hook, implement overlap in template |
| Shared expert AllReduce | 1200μs | P0 | Must add hook |
| MoE routing Broadcast | 50μs | P4 | Can ignore |

### 7.5 Notes for Generating Instrumentation Patches

1. **Hook naming convention**: New hook names should follow `dbo_<module>_<position>_hook` pattern, e.g., `dbo_mlp_row_hook`, `dbo_shared_expert_hook`
2. **forward_context check**: All hook calls must be guarded by `if forward_context.dbo_enabled:`
3. **is_record semantics**: `is_record=True` called **before** communication starts, `is_record=False` called **after** communication completes
4. **base.py default implementation**: New hooks default to `pass` in base class, ensuring no impact on existing templates
5. **Backward compatibility**: Patch code must not change any behavior when `dbo_enabled=False`

### 7.6 Validate Patch Correctness

After generating the patch, validate through:

1. **Static validation**: Re-run Step 2 static analysis, confirm new hook appears in hook call sequence
2. **Profiling validation**: Re-profile after applying patch, use `compare` subcommand:
   ```bash
   python3 scripts/analyze_ascend_profiling.py compare \
     --input-a /path/to/before_patch_profiling \
     --input-b /path/to/after_patch_profiling
   ```
3. **Coverage check**: Confirm all communications with duration > 100μs in profiling have corresponding hook coverage