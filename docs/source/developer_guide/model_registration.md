# Integrating New Models into vLLM-Ascend

This guide demonstrates how to integrate novel or customized models into vLLM-Ascend. For foundational concepts, it is highly recommended to refer to:
[Adding a New Model - vLLM Documentation](https://docs.vllm.ai/en/stable/contributing/model/)

### 1. Implementing Models Using PyTorch and Ascend Extension for PyTorch

This section provides instructions for implementing new models compatible with vLLM and vLLM-Ascend. Before starting:

1. Verify whether your model already exists in [vLLM's Model Executor](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) directory
2. Use existing implementations as templates to accelerate development

#### 1.1 Implementing New Models from Scratch

Follow vLLM's OPT model adaptation example for guidance:
[Implementing a Basic Model - vLLM Documentation](https://docs.vllm.ai/en/stable/contributing/model/basic.html)

Key implementation requirements:

(1) Place model files in [vllm_ascend/models/](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/models) directory

(2) Standard module structure for decoder-only LLMs (please checkout vllm's implementations for other kinds of model):
   
   - `*ModelForCausalLM` (top-level wrapper)
   - `*Model` (main architecture)
   - `*DecoderLayer` (transformer block)
   - `*Attention` & `*MLP` (specific computation unit)
     `*` denotes your model's unique identifier

(3) **Critical Implementation Details**:
- All modules **must** include a `prefix` argument in `__init__()`
- Required interfaces:

  | Module Type          | Required Methods                          |
  | :------------------- | :---------------------------------------- |
  | `*ModelForCausalLM`  | `get_input_embeddings`, `compute_logits`, `load_weights` |
  | `*Model`             | `get_input_embeddings`, `load_weights`    |
     
     
(4) **Attention Backend Integration**:
   Import attention via `from vllm.attention import Attention` can automatically leverage vLLM-Ascend's attention backend routing (see: `get_attn_backend_cls()` in [vllm_ascend/platform.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/platform.py))

(5) **Tensor Parallelism**:
   Use vLLM's parallel layers (`ColumnParallelLinear`, `VocabParallelEmbedding`, etc.), but note Ascend-specific customizations implemented in [vllm_ascend/ops/](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops) directory (RMSNorm, VocabParallelEmbedding, etc.).

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

#### 1.2 Customizing Existing vLLM Models

For most use cases, extending existing implementations is preferable. We demonstrate an example to inherit from base classes and implement a custom deepseek model below (assumed path: `vllm_ascend/models/deepseek_v2.py`):

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

For a complete implementation reference, see: [vllm_ascend/models/deepseek_v2.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/deepseek_v2.py)

### 2. Registering Custom Models as Out-of-Tree Plugins in vLLM

vLLM provides a plugin mechanism for registering externally implemented models without modifying its codebase. To integrate your implemented model from [vllm\_ascend/models/](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models) directory:

(1) **Import your model implementation** in [vllm\_ascend/models/\_\_init\_\_.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/__init__.py) using relative imports

(2) **Register the model wrapper class** via `vllm.ModelRegistry.register_model()` function

**Reference Registration Template** (an example of registering new models in [vllm_ascend/models/\_\_init\_\_.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/__init__.py)
)

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

**Key Note**
The first argument of  `vllm.ModelRegistry.register_model()` indicates the unique architecture identifier which must match 'architectures' in `config.json` of the model.

```json
{
  "architectures": [
    "CustomModelForCausalLM"
  ],
}
```

```json
{
  "architectures": [
    "DeepseekV2ForCausalLM"
  ],
}
```

### 3.  Verifying Model Registration

#### 3.1 Overriding Existing vLLM Model Architecture

If you're registering a customized model architecture based on vLLM's existing implementation (overriding vLLM's original class), when executing vLLM offline/online inference (using any model), you'll observe warning logs similar to the following output from  [vllm/models\_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py) :

```
Model architecture DeepseekV2ForCausalLM is already registered, and will be overwritten by the new model class vllm_ascend/models/deepseek_v2:CustomDeepseekV2ForCausalLM.
```

#### 3.2 Registering New Model Architecture

If you're registering a novel model architecture not present in vLLM (creating a completely new class), current logs won't provide explicit confirmation by default. It's recommended to add the following logging statement at the end of the `register_model` method in [vllm/models\_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py) :

```
logger.warning(f"model_arch: {model_arch} has been registered here!")
```

When running vLLM offline/online inference (using any model), you should then see confirmation logs similar to:

```
model_arch: CustomModelForCausalLM has been registered here!
```

This log output confirms your novel model architecture has been successfully registered in vLLM.

### 4. Using Quantized Model Weights

Computation modules (e.g., variants of Linear layer) in vLLM and vLLM-Ascend keep an attribute namely `quant_method` indicating weights for a layer. It means that vLLM-Ascend will automatically load quantized weights and apply forward process according to different quantization schemes. vLLM-Ascend now delivers both static and dynamic W8A8 quantization solutions on Ascend platforms, achieving optimized inference speed with significant memory savings.​ Please refer to [msit user guidance for W8A8 quantization and accuracy calibration](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/w8a8%E7%B2%BE%E5%BA%A6%E8%B0%83%E4%BC%98%E7%AD%96%E7%95%A5.md) to quantize your own models. It is recommended to quantize models with scripts in [msit quantization examples for prevailing models](https://gitee.com/ascend/msit/tree/master/msmodelslim/example) for common models.
