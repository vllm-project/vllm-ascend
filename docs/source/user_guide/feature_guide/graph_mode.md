# Graph Mode Guide

```{note}
This feature is currently experimental. In future versions, there may be behavioral changes around configuration, coverage, performance improvement.
```

This guide provides instructions for using Ascend Graph Mode with vLLM Ascend. Please note that graph mode is only available on V1 Engine. And only Qwen, DeepSeek series models are well tested from 0.9.0rc1. We will make it stable and generalized in the next release.

## Getting Started

From v0.9.1rc1 with V1 Engine, vLLM Ascend will run models in graph mode by default to keep the same behavior with vLLM. If you hit any issues, please feel free to open an issue on GitHub and fallback to the eager mode temporarily by setting `enforce_eager=True` when initializing the model.

There are three kinds for graph mode supported by vLLM Ascend:
- **ACLGraph**: This is the default graph mode supported by vLLM Ascend. In v0.9.1rc1, Qwen and Deepseek series models are well tested.
- **TorchAirGraph**: This is the GE graph mode. In v0.9.1rc1, only DeepSeek series models are supported.
- **XliteGraph**: This is the euler xlite graph mode. In v0.11.0, only Llama and Qwen dense serise models are supported.

## Using ACLGraph
ACLGraph is enabled by default. Take Qwen series models as an example, just set to use V1 Engine is enough.

Offline example:

```python
import os

from vllm import LLM

model = LLM(model="path/to/Qwen2-7B-Instruct")
outputs = model.generate("Hello, how are you?")
```

Online example:

```shell
vllm serve Qwen/Qwen2-7B-Instruct
```

## Using TorchAirGraph

If you want to run DeepSeek series models with the graph mode, you should use [TorchAirGraph](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/torchairuseguide/torchair_0002.html). In this case, additional configuration is required.

Offline example:

```python
import os
from vllm import LLM

# TorchAirGraph only works without chunked-prefill now
model = LLM(model="path/to/DeepSeek-R1-0528", additional_config={"torchair_graph_config": {"enabled": True}})
outputs = model.generate("Hello, how are you?")
```

Online example:

```shell
vllm serve path/to/DeepSeek-R1-0528 --additional-config='{"torchair_graph_config": {"enabled": true}}'
```

You can find more details about additional configuration [here](../configuration/additional_config.md).

## Using XliteGraph

If you want to run Llama or Qwen dense series models with xlite graph mode, please install xlite, and set xlite_graph_config.

```bash
pip install xlite
```

Offline example:

```python
import os
from vllm import LLM

# xlite supports the decode-only mode by default, and the full mode can be enabled by setting: "full_mode": True
model = LLM(model="path/to/Qwen3-32B", tensor_parallel_size=8, additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}})
outputs = model.generate("Hello, how are you?")
```

Online example:

```shell
vllm serve path/to/Qwen3-32B --tensor-parallel-size 8 --additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

You can find more details abort xlite [here](https://gitee.com/openeuler/GVirt/blob/master/xlite/README.md)

## Fallback to the Eager Mode

If `ACLGraph`, `TorchAirGraph` and `XliteGraph` all fail to run, you should fallback to the eager mode.

Offline example:

```python
import os
from vllm import LLM

model = LLM(model="someother_model_weight", enforce_eager=True)
outputs = model.generate("Hello, how are you?")
```

Online example:

```shell
vllm serve someother_model_weight --enforce-eager
```
