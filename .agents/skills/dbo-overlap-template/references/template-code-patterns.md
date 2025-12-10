# DBO Template Code Patterns Reference

This document only provides the modification logic for `utils.py`. For template code, please refer directly to the existing implementation files in the repository:

- **Dense model**: `vllm_ascend/dbo/overlap_templates/qwen3_dense.py`
- **MoE model (without MLA)**: `vllm_ascend/dbo/overlap_templates/qwen3_moe.py`
- **MLA+MoE model**: `vllm_ascend/dbo/overlap_templates/deepseek.py`
- **Base class definition**: `vllm_ascend/dbo/overlap_templates/base.py`

---

## utils.py Registration Pattern

```python
# Add import at the top of the file
from vllm_ascend.dbo.overlap_templates.<model_family> import (
    <ModelName>AllgatherTemplate,
    <ModelName>AlltoallTemplate,
)

# Add elif branch in the select_dbo_templates() function
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

**Note**: `<ArchClassName>` is the value in `hf_config.architectures`, which can be looked up from `registry.py`,
e.g., `"LlamaForCausalLM"`, `"Qwen2MoeForCausalLM"`, etc.

---

## Multiple Architecture Names Mapping to the Same Template

If multiple architecture names share the same model structure (e.g., LLaMA family), use `any()` in the elif:

```python
elif any(arch in architectures for arch in [
    "LlamaForCausalLM", "MistralForCausalLM", "InternLMForCausalLM"
]):
    if soc_version in {AscendDeviceType.A3}:
        return LlamaDenseAlltoallTemplate()
    else:
        return LlamaDenseAllgatherTemplate()
```
