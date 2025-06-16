# How to Generate Weights of Model with Reduced Layer

1. Fork the original model repo in modelscope, we need all the files in the repo except for weights.
2. Set `num_hidden_layers` to the expected number of layers, e.g., `{"num_hidden_layers": 2,}`
3. Copy the following python script as `generate_random_weight.py`. Set the relevant parameters `MODEL_LOCAL_PATH`, `DIST_DTYPE` and `DIST_MODEL_PATH` as needed:

```python
import torch
from transformers import AutoTokenizer, AutoConfig
from modeling_deepseek import DeepseekV3ForCausalLM
from modelscope import snapshot_download

MODEL_LOCAL_PATH = "~/.cache/modelscope/models/vllm-ascend/DeepSeek-V3-Pruning"
DIST_DTYPE = torch.bfloat16
DIST_MODEL_PATH = "./random_deepseek_v3_with_2_hidden_layer"

config = AutoConfig.from_pretrained(MODEL_LOCAL_PATH, trust_remote_code=True)
model = DeepseekV3ForCausalLM(config)
model = model.to(DIST_DTYPE)
model.save_pretrained(DIST_MODEL_PATH)
```
