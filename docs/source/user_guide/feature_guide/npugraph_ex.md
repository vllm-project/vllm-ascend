# Npugraph_ex

## Introduction

As introduced in the [RFC](https://github.com/vllm-project/vllm-ascend/issues/4715), this is a simple ACLGraph graph mode acceleration solution based on Fx graphs.

## Using npugraph_ex

Npugraph_ex will be enabled by default in the future, Take Qwen series models as an example to show how to configure it.

Offline example:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "npugraph_ex_config": {
            "enable": True,
            "enable_static_kernel": False,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

Online example:

```shell
vllm serve Qwen/Qwen2-7B-Instruct
--additional-config '{"npugraph_ex_config":{"enable":true, "enable_static_kernel":false}}'
```

You can find more details about npugraph_ex [here](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00021.html)
