---
name: dbo-overlap-template-writer
description: "Generate DBO (Dual-Batch Overlap) templates for vllm-ascend to enable compute-communication overlap optimization. Use this skill when the user wants to: (1) adapt a model for DBO support, (2) generate overlap templates, (3) optimize inference performance via dual-batch overlapping, (4) analyze communication operators and hook insertion points, or (5) mentions specific model names (LLaMA, Qwen, DeepSeek, Mixtral, GLM, Bailing, etc.) in the context of vllm-ascend performance optimization. Even if the user doesn't explicitly mention 'DBO', invoke this skill whenever the task involves compute-communication overlap optimization in vllm-ascend. This skill handles Dense, MoE, MLA+MoE, and Hybrid model architectures for both A2 (AllGather) and A3 (AlltoAll) communication modes."
---

# DBO Overlap Template Writer

## Overview

This skill generates DBO (Dual-Batch Overlap) templates for new models in `vllm_ascend/dbo/overlap_templates/` and registers them in `vllm_ascend/dbo/utils.py`, enabling the model to leverage DBO for improved inference performance.

## Inputs

### Required Inputs

The user **MUST** provide the following before proceeding:
1. **Model Name**: Architecture class name registered in vllm (e.g., `LlamaForCausalLM`, `Qwen3MoeForCausalLM`, `DeepseekV3ForCausalLM`)
2. **vllm Path**: Root path of vllm source code or installation (e.g., `/path/to/vllm`)

> If either required input is missing, **ask the user for it** before proceeding. Never guess or use defaults.

### Optional Input

3. **Profiling Path** (optional): Path to Ascend NPU Profiler output directory for data-driven optimization

**Benefits of providing profiling data**:
- Extract actual communication timing from real measurements
- Identify high-priority missing hooks based on communication duration
- Optimize overlap strategy with concrete performance data
- Verify A2/A3 communication mode from actual execution traces

**When to provide profiling data**:
- Model has conditional branches or dynamic routing
- Need precise communication timing measurements
- Static analysis cannot determine A2/A3 mode
- Want to quantify expected overlap efficiency

> If profiling data is not provided, the skill will use static analysis only. This is sufficient for most standard model architectures.

## Execution Workflow

### Step 1: Locate Model Implementation File

Search in `<vllm_path>/vllm/model_executor/models/`:

```bash
# Find filename from registry.py (preferred)
grep -n "<ModelName>" <vllm_path>/vllm/model_executor/models/registry.py
```

Extract the filename from the `_MODELS` dictionary entry:
```python
# Example: "LlamaForCausalLM": ("llama", "LlamaForCausalLM")
#                                              ↑ filename
```

If not found in `registry.py`, inform the user to verify the model name and path.

### Step 2: Static Code Analysis

Read the model file to identify model type, communication operators, and hook insertion points.

#### 2.1 Model Type Detection

Search for keywords to classify the model:

| Pattern | Model Type |
|---------|-----------|
| `MultiHeadLatentAttention` + `QKVParallelLinear` + `FusedMoE` | Hybrid(MLA+QKV)+MoE |
| `MultiHeadLatentAttention` + `FusedMoE` (no `QKVParallelLinear`) | MLA+MoE |
| `FusedMoE` / `SharedFusedMoE` (no MLA) | MoE |
| Neither MLA nor FusedMoE | Dense |

#### 2.2 Hook Identification

Five hooks are available for DBO overlap:

| Hook | Location | Communication |
|------|----------|---------------|
| `dbo_linear_column_hook` | `linear_op.py` → `MLPColumnParallelOp` / `SequenceColumnParallelOp` | QKV/MLP AllGather |
| `dbo_linear_row_hook` | `linear_op.py` → `SequenceRowParallelOp` | o_proj/down_proj ReduceScatter |
| `dbo_mla_preprocess_hook` | `mla_v1.py` | MLA AllGather |
| `dbo_moe_prepare_hook` | `prepare_finalize.py` (A2) / `token_dispatcher.py` (A3) | MoE EP AllGather / AlltoAll |
| `dbo_moe_finalize_hook` | `prepare_finalize.py` (A2) / `token_dispatcher.py` (A3) | MoE EP ReduceScatter / AlltoAll |

#### 2.3 Communication Mode (A2 vs A3)

- Uses `MoEAlltoAllTokenDispatcher` → **A3 AlltoAll mode**
- Uses `PrepareAndFinalize` → **A2 AllGather mode**

#### 2.4 Hook Call Sequence

Derive the hook call sequence by reading `DecoderLayer.forward()`:

**Dense Model (per layer):**
```
column_hook(T) → [QKV AG] → column_hook(F) → [Attn] → row_hook(T) → [o_proj RS] → row_hook(F)
→ column_hook(T) → [MLP AG] → column_hook(F) → [MLP] → row_hook(T) → [down RS] → row_hook(F)
```

**MoE Model (A2, per layer):**
```
column_hook(T) → [QKV AG] → column_hook(F) → [Attn] → row_hook(T) → [o_proj RS] → row_hook(F)
→ moe_prepare_hook(T) → [EP AG] → moe_prepare_hook(F) → [Expert] → moe_finalize_hook(T) → [EP RS] → moe_finalize_hook(F)
```

**MLA+MoE Model (A2, per layer):**
```
mla_preprocess_hook(T) → [MLA AG] → mla_preprocess_hook(F) → [MLA] → row_hook(T) → [o_proj RS]
→ moe_prepare_hook(T) → [EP AG] → moe_prepare_hook(F) → [Expert] → moe_finalize_hook(T) → [EP RS] → moe_finalize_hook(F)
```

> For detailed hook sequences and Hybrid model sequences, see `references/hook-sequences.md`.

### Step 3: Profiling-Based Analysis (Optional)

If the user provides profiling output, use the stack subcommand:

```bash
python3 scripts/analyze_ascend_profiling.py stack \
  --input /path/to/profiling_dir \
  --top-comm-types 0 \
  --max-stacks 100
```

**Priority for adding missing hooks:**
- **P0** (required): avg time > 1000μs and frequency > 10 calls/layer
- **P1** (recommended): avg time 500-1000μs or frequency > 5 calls/layer
- **P2** (optional): avg time 100-500μs
- **P3** (skip): avg time < 100μs

> See `references/profiling-guide.md` for detailed profiling analysis workflow.

### Step 4: Plan Overlap Strategy

Follow two core principles:

**Principle 1: Consecutive Communication Merging**
If the computation between two communications is < 200μs (e.g., LayerNorm, residual add), merge them into one communication block.

**Principle 2: Compute-Communication Balance**
Position `record`/`wait` to make overlapped communication time ≈ the counterpart's compute time.

#### A2 Overlap Rules (for MoE/MLA+MoE):

1. ubatch0 attention post-communication ↔ ubatch1 attention compute
2. ubatch1 attention post-communication ↔ ubatch0 MoE compute
3. ubatch0 MoE post-communication ↔ ubatch1 MoE compute

#### A3 Overlap Rules (additional for MoE):

1-3. Same as A2
4. ubatch0 MoE Dispatch AlltoAll ↔ ubatch1 pre-MoE compute
5. ubatch0 MoE Finalize AlltoAll ↔ ubatch1 Expert compute

> For detailed strategy tables and timing diagrams, see `references/overlap-strategy-guide.md`.

#### First Layer Synchronization

For models using `ATTN_PRE` event, handle the first layer specially:

```python
def dbo_linear_column_hook(self, is_record):
    if is_record:
        if get_forward_context().dbo_first_layer_sync:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
            get_forward_context().dbo_first_layer_sync = False
    else:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)
```

### Step 5: Generate Template File

Create `<model_family>.py` in `vllm_ascend/dbo/overlap_templates/`:

1. Determine the reuse group based on model type
2. Copy and adapt from the reference implementation
3. Update class names and architecture mappings

**Reuse Groups:**
| Group | Models | Reference File |
|-------|--------|----------------|
| Group 1 (QKV + MoE) | qwen3_moe, bailing_moe, glm4_moe | `qwen3_moe.py` |
| Group 2 (MLA + MoE) | deepseek, glm_moe_dsa | `deepseek.py` |
| Group 3 (Hybrid + MoE) | bailing_moe_v25 | `bailing_moe_v25.py` |
| Group 4 (Dense) | qwen3_dense | `qwen3_dense.py` |

> See `references/template-code-patterns.md` for detailed code patterns.

### Step 6: Register in utils.py

Edit `vllm_ascend/dbo/utils.py`:

```python
# Add import at top
from vllm_ascend.dbo.overlap_templates.<model_family> import (
    <ModelFamily>AllgatherTemplate,
    <ModelFamily>AlltoallTemplate,
)

# Add elif branch in select_dbo_templates()
elif "<ArchitectureName>" in architectures:
    if soc_version in {AscendDeviceType.A3}:
        return <ModelFamily>AlltoallTemplate()
    else:
        return <ModelFamily>AllgatherTemplate()
```

> Insert the new `elif` before the fallback `else` branch.

### Step 7: Validation

Check the generated code:

1. **Correctness**: Each EventKey has exactly one `record` and one `wait`
2. **First layer**: If using `ATTN_PRE`, ensure `dbo_first_layer_sync` handling exists
3. **Consecutive merging**: Verify no `record`/`wait` between communications with < 200μs compute
4. **Balance**: Estimate overlap efficiency = min(T_comm, T_compute) / max(T_comm, T_compute)
   - Efficiency < 50% indicates need for adjustment

## Key Files

| File | Purpose |
|------|---------|
| `vllm_ascend/dbo/overlap_templates/base.py` | Base class with hook interfaces |
| `vllm_ascend/dbo/overlap_templates/deepseek.py` | MLA+MoE reference (Group 2) |
| `vllm_ascend/dbo/overlap_templates/qwen3_moe.py` | MoE reference (Group 1) |
| `vllm_ascend/dbo/overlap_templates/qwen3_dense.py` | Dense reference (Group 4) |
| `vllm_ascend/dbo/overlap_templates/bailing_moe_v25.py` | Hybrid reference (Group 3) |
| `vllm_ascend/dbo/utils.py` | Template selection entry point |
| `vllm_ascend/worker/ubatching.py` | DBO primitives (`UBatchEventKey`, `dbo_record_current_stream`) |
| `vllm_ascend/ops/linear_op.py` | Hook insertion: column and row hooks |
| `vllm_ascend/attention/mla_v1.py` | Hook insertion: MLA preprocess hook |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | Hook insertion: MoE hooks (A2) |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | Hook insertion: MoE hooks (A3) |

## Outputs

1. **New template file**: `vllm_ascend/dbo/overlap_templates/<model_family>.py`
2. **Updated utils.py**: New import + elif branch
3. **Hook patches** (if Step 3 detects missing hooks): Code patches for vllm-ascend files
4. **Analysis report**: Communication sequence diagram, overlap strategy, hook explanations

## Reference Documents

- `references/hook-sequences.md` - Complete hook call sequences for all model types
- `references/overlap-strategy-guide.md` - Detailed overlap strategy planning
- `references/profiling-guide.md` - Profiling analysis workflow
- `references/template-code-patterns.md` - Code patterns and reuse guidelines
- `references/static-analysis-guide.md` - Static analysis methodology