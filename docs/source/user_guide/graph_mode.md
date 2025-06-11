# Graph Mode Guide


This feature is currently experimental. In future versions, there may be behavioral changes around configuration, coverage, performance improvement.

This guide provides instructions for using Ascend Graph Mode with vLLM Ascend. Please note that graph mode is only available on V1 Engine. And only Qwen, DeepSeek series models are well tested in 0.9.0rc1. We'll make it stable and generalize in the next release.

## Getting Started

From v0.9.0rc1 with V1 Engine, vLLM Ascend will run models in graph mode by default to keep the same behavior with vLLM. If you hit any issues, please feel free to open an issue on GitHub and fallback to eager mode temporarily by set `enforce_eager=True` when initializing the model.

There are two kinds for graph mode supported by vLLM Ascend:
- **ACLGraph**: This is the default graph mode supported by vLLM Ascend. In v0.9.0rc1, only Qwen series models are well tested.
- **TorchAirGraph**: This is the GE graph mode. In v0.9.0rc1, only DeepSeek series models are supported.

## Using ACLGraph
ACLGraph is enabled by default. Take Qwen series models as an example, just set to use V1 Engine is enough.

offline example:

```python
import os

from vllm import LLM

os.environ["VLLM_USE_V1"] = 1

model = LLM(model="Qwen/Qwen2-7B-Instruct")
outputs = model.generate("Hello, how are you?")
```

online example:

```shell
vllm serve Qwen/Qwen2-7B-Instruct
```

## Using TorchAirGraph

If you want to run DeepSeek series models with graph mode, you should use [TorchAirGraph](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/torchairuseguide/torchair_0002.html). In this case, additional config is required.

offline example:

```python
import os
from vllm import LLM

os.environ["VLLM_USE_V1"] = 1

model = LLM(model="deepseek-ai/DeepSeek-R1-0528", additional_config={"torchair_graph_config": {"enabled": True}})
outputs = model.generate("Hello, how are you?")
```

online example:

```shell
vllm serve Qwen/Qwen2-7B-Instruct --additional-config='{"torchair_graph_config": {"enabled": true}}'
```

You can find more detail about additional config [here](./additional_config.md)

## Fallback to Eager Mode

If both `ACLGraph` and `TorchAirGraph` fail to run, you should fallback to eager mode.

offline example:

```python
import os
from vllm import LLM

os.environ["VLLM_USE_V1"] = 1

model = LLM(model="someother_model_weight", enforce_eager=True)
outputs = model.generate("Hello, how are you?")
```

online example:

```shell
vllm serve Qwen/Qwen2-7B-Instruct --enforce-eager
```
