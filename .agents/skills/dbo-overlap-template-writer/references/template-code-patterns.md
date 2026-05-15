# DBO Template Code Patterns

For template code, directly reference existing implementations in the repository. New models should first determine which reuse group they belong to, then reference the corresponding implementation.

## Existing Template Reference Files

| Model Type | Reference File | Reuse Group |
|------------|---------------|-------------|
| **Dense Model** | `vllm_ascend/dbo/overlap_templates/qwen3_dense.py` | Group 4 |
| **MoE Model (no MLA)** | `vllm_ascend/dbo/overlap_templates/qwen3_moe.py` | Group 1 |
| **MLA+MoE Model** | `vllm_ascend/dbo/overlap_templates/deepseek.py` | Group 2 |
| **Hybrid(MLA+QKV)+MoE Model** | `vllm_ascend/dbo/overlap_templates/bailing_moe_v25.py` | Group 3 |
| **Base Class Definition** | `vllm_ascend/dbo/overlap_templates/base.py` | — |

### Code Reuse Relationships

```
Group 1 (QKV + MoE):     qwen3_moe ≡ bailing_moe ≡ glm4_moe     (only class names differ)
Group 2 (MLA + MoE):     deepseek ≡ glm_moe_dsa                  (only class names differ)
Group 3 (Hybrid + MoE):  bailing_moe_v25                          (unique, merges Group 1 and Group 2)
Group 4 (Dense):         qwen3_dense                              (unique, uses DEFAULT event)
```

**Decision tree for selecting reference implementation**:
```
New model has MLA + QKVParallelLinear + FusedMoE simultaneously → Reference bailing_moe_v25.py (Group 3)
New model has MLA + FusedMoE (no QKVParallelLinear)            → Reference deepseek.py (Group 2)
New model has FusedMoE (no MLA)                                → Reference qwen3_moe.py (Group 1)
New model has no FusedMoE and no MLA                           → Reference qwen3_dense.py (Group 4)
```

---

## utils.py Registration Pattern

```python
# Add import at top of file
from vllm_ascend.dbo.overlap_templates.<model_family> import (
    <ModelName>AllgatherTemplate,
    <ModelName>AlltoallTemplate,
)

# Add elif branch in select_dbo_templates() function
def select_dbo_templates(vllm_config):
    model_config = vllm_config.model_config
    architectures = getattr(model_config.hf_config, "architectures", [])
    soc_version = get_ascend_device_type()

    # ... existing logic ...

    elif "<ArchClassName>" in architectures:
        if soc_version in {AscendDeviceType.A3}:
            return <ModelName>AlltoallTemplate()
        else:
            return <ModelName>AllgatherTemplate()

    else:
        return UbatchOverlapBaseTemplate()
```

**Note**: `<ArchClassName>` is the value from `hf_config.architectures`, which can be found in `registry.py`, e.g., `"LlamaForCausalLM"`, `"Qwen2MoeForCausalLM"`, etc.

### Currently Registered Architecture Names

| Architecture Name | Template | Reuse Group |
|-------------------|----------|-------------|
| `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` | `DeepseekAllgather/AlltoallTemplate` | Group 2 |
| `GlmMoeDsaForCausalLM` | `GlmMoeDsaAllgather/AlltoallTemplate` | Group 2 |
| `Qwen3MoeForCausalLM` | `QwenMoEAllgather/AlltoallTemplate` | Group 1 |
| `Glm4MoeForCausalLM` | `Glm4MoEAllgather/AlltoallTemplate` | Group 1 |
| `BailingMoeForCausalLM` / `BailingMoeV2ForCausalLM` | `BailingMoEAllgather/AlltoallTemplate` | Group 1 |
| `BailingMoeV2_5ForCausalLM` | `BailingMoEV25Allgather/AlltoallTemplate` | Group 3 |
| `Qwen3ForCausalLM` | `QwenDenseAllgather/AlltoallTemplate` | Group 4 |

---

## Multiple Architecture Names Mapping to Same Template

If multiple architecture names share the same model structure (e.g., LLaMA series), use `any()` in elif:

```python
elif any(arch in architectures for arch in [
    "LlamaForCausalLM", "MistralForCausalLM", "InternLMForCausalLM"
]):
    if soc_version in {AscendDeviceType.A3}:
        return LlamaDenseAlltoallTemplate()
    else:
        return LlamaDenseAllgatherTemplate()
```

---

## Template Code Structure

### Base Class (base.py)

All templates inherit from `UbatchOverlapBaseTemplate`:

```python
class UbatchOverlapBaseTemplate:
    """Base class for DBO overlap templates."""
    
    def dbo_mla_preprocess_hook(self, is_record: bool) -> None:
        """Hook for MLA preprocess AllGather. Default: no-op."""
        pass
    
    def dbo_linear_column_hook(self, is_record: bool) -> None:
        """Hook for QKV/MLP column parallel AllGather. Default: no-op."""
        pass
    
    def dbo_linear_row_hook(self, is_record: bool) -> None:
        """Hook for o_proj/down_proj row parallel ReduceScatter. Default: no-op."""
        pass
    
    def dbo_moe_prepare_hook(self, is_record: bool) -> None:
        """Hook for MoE prepare (AllGather/AlltoAll). Default: no-op."""
        pass
    
    def dbo_moe_finalize_hook(self, is_record: bool) -> None:
        """Hook for MoE finalize (ReduceScatter/AlltoAll). Default: no-op."""
        pass
```

### Typical Template Structure

```python
from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker ubatching import (
    dbo_record_current_stream,
    dbo_wait_current_stream_and_yield,
    UBatchEventKey,
)
from vllm_ascend.utils import get_forward_context


class ModelNameAllgatherTemplate(UbatchOverlapBaseTemplate):
    """DBO overlap template for ModelName on A2 (AllGather mode)."""
    
    def dbo_linear_column_hook(self, is_record: bool) -> None:
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)
    
    def dbo_linear_row_hook(self, is_record: bool) -> None:
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
        # is_record=False: pass (merge with MoE prepare)
    
    def dbo_moe_prepare_hook(self, is_record: bool) -> None:
        if not is_record:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)
    
    def dbo_moe_finalize_hook(self, is_record: bool) -> None:
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)


class ModelNameAlltoallTemplate(UbatchOverlapBaseTemplate):
    """DBO overlap template for ModelName on A3 (AlltoAll mode)."""
    
    def dbo_linear_column_hook(self, is_record: bool) -> None:
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)
    
    def dbo_linear_row_hook(self, is_record: bool) -> None:
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST, wait=False)
    
    def dbo_moe_prepare_hook(self, is_record: bool) -> None:
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.MOE_DISPATCH)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.MOE_DISPATCH)
    
    def dbo_moe_finalize_hook(self, is_record: bool) -> None:
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
```

---

## Event Keys

Available event keys defined in `vllm_ascend/work/ubatching.py`:

```python
class UBatchEventKey(Enum):
    DEFAULT = auto()        # For dense models
    ATTN_PRE = auto()       # Before attention computation
    ATTN_POST = auto()      # After attention computation
    MOE_DISPATCH = auto()   # For A3 MoE dispatch AlltoAll
```

**Usage guidelines**:
- `DEFAULT`: Used by dense models where no cross-layer dependency exists
- `ATTN_PRE`: Signals "attention computation is ready to start", used for first-layer synchronization
- `ATTN_POST`: Signals "attention computation completed", for attention-to-MoE transition
- `MOE_DISPATCH`: Used only in A3 mode for MoE dispatch AlltoAll synchronization

---

## First Layer Synchronization Pattern

For models using `ATTN_PRE` event, handle the first layer specially:

```python
def dbo_linear_column_hook(self, is_record: bool) -> None:
    if is_record:
        # First layer: no previous layer's MoE Finalize to wait for
        # Must actively record to unblock ubatch1
        if get_forward_context().dbo_first_layer_sync:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
            get_forward_context().dbo_first_layer_sync = False
    else:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)
```

**Why needed**: Layer 0's ubatch1 waits for `ATTN_PRE` (from previous layer's MoE Finalize), but there's no "previous layer" at layer 0. The first ubatch0 must proactively record `ATTN_PRE` to unblock ubatch1.

**When needed**: Only for templates that have `wait(ATTN_PRE)` in any hook (i.e., MoE/MLA+MoE/Hybrid models). Dense models using `DEFAULT` event don't need this.

---

## wait=False Pattern

```python
def dbo_linear_row_hook(self, is_record: bool) -> None:
    if is_record:
        dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
    else:
        # Only yield CPU, don't wait for data
        # Used in A3 mode to let ubatch1 start QKV earlier
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST, wait=False)
```

**When to use**:
- A3 AlltoAll mode only
- When ubatch1's next computation doesn't depend on ubatch0's current communication result
- Purpose: yield CPU to let ubatch0's communication execute on NPU

---

## Common Mistakes to Avoid

### 1. Missing First Layer Sync

```python
# ❌ Wrong: No first layer sync, ubatch1 hangs at layer 0
def dbo_linear_column_hook(self, is_record: bool) -> None:
    if is_record:
        pass  # Missing first layer handling
    else:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)
```

### 2. Incorrect Event Key Pairing

```python
# ❌ Wrong: Record and wait use different event keys
def dbo_linear_row_hook(self, is_record: bool) -> None:
    if is_record:
        dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
    
def dbo_moe_prepare_hook(self, is_record: bool) -> None:
    if not is_record:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)  # Wrong key!
```

### 3. Overlapping Communications That Should Be Merged

```python
# ❌ Wrong: Inserted wait between o_proj RS and MoE Prepare AG
def dbo_linear_row_hook(self, is_record: bool) -> None:
    if is_record:
        dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
    else:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)  # Should be pass

def dbo_moe_prepare_hook(self, is_record: bool) -> None:
    if not is_record:
        dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)  # Wrong: double wait
```

### 4. A3 Mode Missing MOE_DISPATCH Event

```python
# ❌ Wrong: A3 template missing MOE_DISPATCH handling
class ModelAlltoallTemplate(UbatchOverlapBaseTemplate):
    # ... other hooks ...
    
    def dbo_moe_prepare_hook(self, is_record: bool) -> None:
        # Missing record(MOE_DISPATCH) and wait(MOE_DISPATCH)
        pass
```

---

## Testing Template

After creating a new template, test with:

```bash
# Run unit tests for DBO
pytest tests/ut/dbo/test_overlap_templates.py -v

# Run integration test with DBO enabled
VLLM_ASCEND_DBO_ENABLED=1 python examples/offline_inference.py \
    --model /path/to/model \
    --trust-remote-code
```

Verify:
1. No hang or timeout (indicates correct synchronization)
2. Performance improvement vs DBO disabled (indicates effective overlap)
3. Correct output (indicates no data corruption from synchronization issues)