# DBO Overlap Template Writer

A skill for generating DBO (Dual-Batch Overlap) templates to enable compute-communication overlap optimization in vllm-ascend.

## Quick Start

### Inputs

#### Required Inputs

Before using this skill, ensure you have:
1. **Model name**: The architecture class name registered in vllm (e.g., `LlamaForCausalLM`, `Qwen3MoeForCausalLM`)
2. **vllm path**: Root path of vllm source code or installation

#### Optional Input

3. **Profiling path**: Path to Ascend NPU Profiler output directory

**Benefits of providing profiling data**:
- Extract actual communication timing from real measurements
- Identify high-priority missing hooks based on duration
- Optimize overlap strategy with concrete data
- Verify A2/A3 communication mode from execution traces

**When to provide profiling data**:
- Model has conditional branches or dynamic routing
- Need precise communication timing measurements
- Static analysis cannot determine A2/A3 mode
- Want to quantify expected overlap efficiency

> If profiling data is not provided, the skill uses static analysis only, which is sufficient for most standard model architectures.

### Basic Usage

Simply tell the skill what model you want to adapt:

```
I want to adapt LlamaForCausalLM for DBO. The vllm path is /path/to/vllm.
```

Or:

```
Help me generate DBO overlap template for DeepseekV3ForCausalLM.
```

### Usage with Profiling Data

For data-driven optimization, provide profiling output:

```
Adapt Qwen3MoeForCausalLM for DBO. 
vllm path: /path/to/vllm
profiling path: /path/to/profiling_output
```

Or:

```
Generate DBO template for DeepseekV3ForCausalLM.
I have profiling data at /path/to/ascend_profiler_output.
```

The skill will automatically:
1. Locate the model implementation file
2. Analyze model architecture (Dense/MoE/MLA+MoE/Hybrid)
3. Identify hook insertion points
4. Generate appropriate overlap templates
5. Register the template in `utils.py`

## Supported Model Types

| Model Type | Examples | Template Group |
|------------|----------|----------------|
| **Dense** | LLaMA, Qwen3-Dense, Mistral | Group 4 |
| **MoE** | Qwen3-MoE, Bailing MoE, GLM-4 MoE | Group 1 |
| **MLA+MoE** | DeepSeek V2/V3, GLM-MoE-DSA | Group 2 |
| **Hybrid+MoE** | Bailing V2.5 | Group 3 |

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Locate Model File                                      │
│  ────────────────────────────────────────────────────────────── │
│  Input: Model name + vllm path                                  │
│  Output: Model implementation file path                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Static Code Analysis                                   │
│  ────────────────────────────────────────────────────────────── │
│  • Detect model type (Dense/MoE/MLA+MoE/Hybrid)                 │
│  • Identify hook insertion points                               │
│  • Determine communication mode (A2/A3)                         │
│  • Derive hook call sequence                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Profiling Analysis (Optional)                          │
│  ────────────────────────────────────────────────────────────── │
│  If profiling data provided:                                    │
│  • Extract communication operator call stacks                   │
│  • Identify missing hooks                                       │
│  • Measure communication timing                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Plan Overlap Strategy                                  │
│  ────────────────────────────────────────────────────────────── │
│  • Apply communication scheduling principles                    │
│  • Design record/wait placement                                 │
│  • Handle first layer synchronization                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Generate Template                                      │
│  ────────────────────────────────────────────────────────────── │
│  • Create <model_family>.py in overlap_templates/               │
│  • Implement AllgatherTemplate (A2) and AlltoallTemplate (A3)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Register Template                                      │
│  ────────────────────────────────────────────────────────────── │
│  • Add import to utils.py                                       │
│  • Add elif branch in select_dbo_templates()                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7: Validation                                             │
│  ────────────────────────────────────────────────────────────── │
│  • Check record/wait pairing                                    │
│  • Verify first layer sync                                      │
│  • Estimate overlap efficiency                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Output Files

After running the skill, you'll get:

1. **Template file**: `vllm_ascend/dbo/overlap_templates/<model_family>.py`
2. **Updated utils.py**: `vllm_ascend/dbo/utils.py` (with new import and registration)
3. **Hook patches** (if needed): Code patches for missing hook instrumentation

## Key Concepts

### DBO (Dual-Batch Overlap)

DBO splits one batch into two micro-batches (ubatch0 and ubatch1), executing them interleaved on different NPU streams. Through `record`/`wait` synchronization primitives, communication and computation can overlap:

```
Timeline →
ubatch0: [Compute] ─ record(KEY) ─ [Comm] ─ ...
ubatch1:                          wait(KEY)+yield ─ [Compute] ─ record(KEY) ─ [Comm] ─ ...
```

### Five Hooks

| Hook | Location | Communication |
|------|----------|---------------|
| `dbo_linear_column_hook` | `linear_op.py` | QKV/MLP AllGather |
| `dbo_linear_row_hook` | `linear_op.py` | o_proj/down_proj ReduceScatter |
| `dbo_mla_preprocess_hook` | `mla_v1.py` | MLA AllGather |
| `dbo_moe_prepare_hook` | `prepare_finalize.py` / `token_dispatcher.py` | MoE EP AllGather / AlltoAll |
| `dbo_moe_finalize_hook` | `prepare_finalize.py` / `token_dispatcher.py` | MoE EP ReduceScatter / AlltoAll |

### Communication Modes

- **A2 (AllGather)**: MoE uses EP AllGather + ReduceScatter
- **A3 (AlltoAll)**: MoE uses EP AlltoAll (dispatch + combine)

### Core Principles

1. **Consecutive Communication Merging**: Merge communications with < 200μs intermediate computation
2. **Compute-Communication Balance**: Position `record`/`wait` to balance overlapped durations

## Advanced Usage

### Analyzing Existing Model

To understand how an existing model's DBO template works:

```
Explain the DBO overlap strategy for DeepSeek V3.
```

### Checking Missing Hooks

To verify hook coverage for a model:

```
Check if there are any uncovered communication paths in LlamaForCausalLM.
```

## Reference Documentation

| Document | Content |
|----------|---------|
| `references/hook-sequences.md` | Complete hook call sequences for all model types |
| `references/overlap-strategy-guide.md` | Detailed overlap strategy planning |
| `references/profiling-guide.md` | Profiling analysis workflow |
| `references/template-code-patterns.md` | Code patterns and reuse guidelines |
| `references/static-analysis-guide.md` | Static analysis methodology |
| `references/comm-flow-analysis.md` | Communication flow analysis reference |

## Troubleshooting

### Model not found in registry

```
Error: Model architecture 'XxxForCausalLM' not found in registry.py
```

**Solution**: Verify the model name matches the architecture class name in `vllm/model_executor/models/registry.py`.

### Uncertain A2/A3 mode

If static analysis cannot determine the communication mode:

```
Warning: Cannot determine A2/A3 mode from static analysis.
Defaulting to A2 template. Please verify with profiling data.
```

**Solution**: Run with profiling data to confirm actual communication patterns.

### Overlap efficiency low

If estimated overlap efficiency < 50%:

```
Warning: Overlap efficiency for interval X is only 35%.
Consider adjusting record/wait placement.
```

**Solution**: Review the communication merging strategy or adjust `record`/`wait` positions.

## Scripts

### analyze_ascend_profiling.py

Analyze Ascend NPU profiling output:

```bash
# Triage report (recommended)
python3 scripts/analyze_ascend_profiling.py triage \
  --input /path/to/profiling_dir \
  --num-layers 10

# Communication stack extraction
python3 scripts/analyze_ascend_profiling.py stack \
  --input /path/to/profiling_dir \
  --top-comm-types 3 \
  --max-stacks 50

# Before/after DBO comparison
python3 scripts/analyze_ascend_profiling.py compare \
  --input-a /path/to/baseline \
  --input-b /path/to/dbo_enabled
```

## Contributing

When adding support for new models:

1. Run the skill to generate initial templates
2. Validate with profiling data
3. Test DBO enabled vs disabled performance
4. Submit PR with both A2 and A3 templates

## License

This skill is part of the vllm-ascend project.