# Graph Mode Guide

```{note}
This feature is currently experimental. In future versions, there may be behavioral changes around configuration, coverage, performance improvement.
```

This guide provides instructions for using Ascend Graph Mode with vLLM Ascend. Please note that graph mode is only available on V1 Engine. And only Qwen, DeepSeek series models are well tested from 0.9.0rc1. We will make it stable and generalized in the next release.

## Getting Started

From v0.9.1rc1 with V1 Engine, vLLM Ascend will run models in graph mode by default to keep the same behavior with vLLM. If you hit any issues, please feel free to open an issue on GitHub and fall back to the eager mode temporarily by setting `enforce_eager=True` when initializing the model.

There are two kinds of graph mode supported by vLLM Ascend:

- **ACLGraph**: This is the default graph mode supported by vLLM Ascend. In v0.9.1rc1, Qwen and DeepSeek series models are well tested.
- **XliteGraph**: This is the OpenEuler Xlite graph mode. In v0.11.0, only Llama, Qwen dense series models, Qwen MoE series models, and Qwen3-VL are supported.

## Using ACLGraph

ACLGraph is enabled by default. Take Qwen series models as an example, just set to use V1 Engine.

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

## Using XliteGraph

If you want to run Llama, Qwen dense series models, Qwen MoE series models, or Qwen3-VL with Xlite graph mode, please install xlite, and set xlite_graph_config.

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

You can find more details about Xlite [here](https://atomgit.com/openeuler/GVirt/blob/master/xlite/README.md)

## Fallback to the Eager Mode

If `ACLGraph` and `XliteGraph` all fail to run, you should fall back to the eager mode.

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
