# Adding a New Model

This guide demonstrates how to integrate a novel or customized model into vllm-ascend. For foundational concepts, it is highly recommended to refer to
[vllm official doc: Adding a New Model](https://docs.vllm.ai/en/stable/contributing/model/) first.

## Step 1: Implementing Models with `torch` and `torch_npu`

This section provides instructions for implementing new models compatible with vllm and vllm-ascend.

**Before starting:**

- Verify whether your model already exists in vllm's [models](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) directory.
- Use existing models' implementation as templates to accelerate your development.

### Method 1: Implementing New Models from Scratch

Follow vllm's [OPT model adaptation](https://docs.vllm.ai/en/stable/contributing/model/basic.html) example for guidance.

**Key implementation requirements:**

1. Place model files in `vllm_ascend/models/` directory.

2. Standard module structure for decoder-only LLMs (please checkout vllm's implementations for other kinds of model):

- `*ModelForCausalLM` (top-level wrapper)
- `*Model` (main architecture)
- `*DecoderLayer` (transformer block)
- `*Attention` and `*MLP` (specific computation unit)

:::{note}
`*` denotes your model's unique identifier.
:::

3. Critical Implementation Details:

All modules must include a `prefix` argument in `__init__()`.

**Required interfaces:**

| Module Type          | Required Methods                          |
| :------------------- | :---------------------------------------- |
| `*ModelForCausalLM`  | `get_input_embeddings`, `compute_logits`, `load_weights` |
| `*Model`             | `get_input_embeddings`, `load_weights`    |

4. Attention Backend Integration:

Importing attention via `from vllm.attention import Attention` can automatically leverage the attention backend routing of vllm-ascend (see: `get_attn_backend_cls()` in `vllm_ascend/platform.py`).

5. Tensor Parallelism:

Use vllm's parallel layers (`ColumnParallelLinear`, `VocabParallelEmbedding`, etc.) to implement models supporting tensor parallelism. Note that Ascend-specific customizations are implemented in `vllm_ascend/ops/` directory (RMSNorm, VocabParallelEmbedding, etc.).

**Reference Implementation Template** (assumed path: `vllm_ascend/models/custom_model.py`):

```python
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata

class CustomAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.attn = Attention(prefix=f"{prefix}.attn")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Implement attention logic
        ...

class CustomDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = CustomAttention(vllm_config, prefix=f"{prefix}.self_attn")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Implement decoder layer
        ...

class CustomModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") 
            for i in range(vllm_config.model_config.hf_config.num_hidden_layers)
        ])

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ...

    def load_weights(self, 
                    weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ...

class CustomModelForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = CustomModel(vllm_config, prefix=f"{prefix}.model")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ...

    def compute_logits(self,
                      hidden_states: torch.Tensor,
                      sampling_metadata: SamplingMetadata) -> torch.Tensor:
        ...

    def load_weights(self, 
                    weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ...
```

### Method 2: Customizing Existing vLLM Models

For most use cases, extending existing implementations is preferable. We demonstrate an example to inherit from base classes and implement a custom deepseek model below (assumed path: `vllm_ascend/models/deepseek_v2.py`).

```python
from typing import List, Optional
import torch
from vllm.attention import AttentionMetadata
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.sequence import IntermediateTensors

class CustomDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):
    # Define merged weights for quantization/efficiency
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Custom forward logic
        hidden_states = self.model(
            input_ids, 
            positions, 
            kv_caches,
            attn_metadata, 
            intermediate_tensors,
            inputs_embeds
        )
        return hidden_states
```

:::{note}
For a complete implementation reference, see: `vllm_ascend/models/deepseek_v2.py`.
:::

## Step 2: Registering Custom Models using ModelRegistry Plugins in vLLM

vllm provides a plugin mechanism for registering externally implemented models without modifying its codebase.

To integrate your implemented model from `vllm_ascend/models/` directory:

1. Import your model implementation in `vllm_ascend/models/__init__.py` using relative imports.
2. Register the model wrapper class via `vllm.ModelRegistry.register_model()` function.

**Reference Registration Template** (an example of registering new models in `vllm_ascend/models/__init__.py`):

```python
from vllm import ModelRegistry

def register_model():
    from .custom_model import CustomModelForCausalLM        # New custom model
    from .deepseek_v2 import ModifiedDeepseekV2ForCausalLM  # Customized Deepseek

    # For NEW architectures: Register with unique name
    ModelRegistry.register_model(
        "CustomModelForCausalLM",  # Must match config.json's 'architectures'
        "vllm_ascend.models.custom_model:CustomModelForCausalLM"
    )

    # For MODIFIED architectures: Use original name
    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",   # Original architecture identifier in vLLM
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV2ForCausalLM  "
    )
```

:::{note}
The first argument of `vllm.ModelRegistry.register_model()` indicates the unique architecture identifier which must match `architectures` in `config.json` of the model.

```json
{
  "architectures": [
    "CustomModelForCausalLM"
  ],
}
```
:::

## Step 3: Verification

### Case 1: Overriding Existing vLLM Model Architecture

If you're registering a customized model architecture based on vllm's existing implementation (overriding vllm's original class), when executing vllm offline/online inference (using any model), you'll observe warning logs similar to the following output from `vllm/models_executor/models/registry.py`.

```bash
Model architecture DeepseekV2ForCausalLM is already registered, and will be overwritten by the new model class vllm_ascend/models/deepseek_v2:CustomDeepseekV2ForCausalLM.
```

### Case 2: Registering New Model Architecture

If you're registering a novel model architecture not present in vllm (creating a completely new class), current logs won't provide explicit confirmation by default. It's recommended to add the following logging statement at the end of the `register_model` method in `vllm/models_executor/models/registry.py`.

```python
logger.info(f"model_arch: {model_arch} has been registered here!")
```

After adding this line, you will see confirmation logs shown below when running vllm offline/online inference (using any model).

```bash
model_arch: CustomModelForCausalLM has been registered here!
```

This log output confirms your novel model architecture has been successfully registered in vllm.

## Step 4: Testing

After adding a new model, we should do basic functional test (offline/online inference), accuracy test and performance benchmark for the model.

Find more details at:

- [Accuracy test guide](https://vllm-ascend.readthedocs.io/en/latest/developer_guide/evaluation/index.html)
- [Performance benchmark guide](https://vllm-ascend.readthedocs.io/en/latest/developer_guide/performance/performance_benchmark.html)

## Step 5: Updating Supported Models Doc

At last, if all the steps above are completed, you should add the new model into our [Supported Models](https://vllm-ascend.readthedocs.io/en/latest/user_guide/supported_models.html) doc.
