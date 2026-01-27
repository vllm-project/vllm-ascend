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

## Fx Graph Optimization

### Fx Graph pass

- For the intermediate nodes of the model, replace the non-in-place operators contained in the nodes with in-place operators to reduce memory movement during computation and improve performance.
- For the original input parameters of the model, if they include in-place operators, Dynamo's Functionalize process will replace the in-place operators with a form of non-in-place operators + copy operators. npugraph_ex will reverse this process, restoring the in-place operators and reducing memory movement.

### Fx default fusion pass

npugraph_ex provides three default operator fusion passes. Operator combinations that meet the replacement rules can be replaced with the corresponding fused operators.

| replacement pattern                                                                                                 | Corresponding fusion operator                                                    |
|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| The output of npu_add_rms_norm is directly used as the input for npu_dynamic_quant (including the smooth_scales parameter) | npu_add_rms_norm_dynamic_quant                                                   |
| The output of npu_add_rms_norm, after being flattened (0,1), is used as input to npu_dynamic_quant (without the smooth_scales parameter), and the scaleOut output of npu_dynamic_quant is then executed with view(-1,1) | npu_add_rms_norm_dynamic_quant(Automatically handle flatten and view operations) |
| The output of npu_add_rms_norm first obtains the size of the last dimension h, then reshapes it using view(-1, h) and converts it to torch.float32 type. | npu_add_rms_norm_cast(Automatically handle view)                                 |

### Custom fusion pass

Users can register a custom graph fusion pass in TorchAir to modify PyTorch FX graphs. The registration relies on the replacement API. Below is the declaration of this API and a demo of its usage.

```python
register_replacement(search_fn, replace_fn, example_inputs, trace_fn=fwd_only, extra_check=_return_true, search_fn_pattern=None)
```

|Parameter Name| Input/Output |Explanation|Is necessary|
|--|--------------|---|-------|
|search_fn|Input|This function is the operator combination or calculation logic that you want to recognize in the FX graph, such as the operator combination that needs to be fused|Yes|
|replace_fn|Input|When the combination corresponding to search_fn is found in the target graph, this function's computation logic will replace the original subgraph to achieve operator fusion or optimization.|Yes|
|example_inputs|Input|Example input tensors used to track search_fn and replace_fn. The shape and dtype of the input should match the actual scenario.|Yes|
|trace_fn|Input|By default, only the forward computation graph is tracked, which is suitable for optimization during the inference phase; if training scenarios need to be supported, a function that supports backward tracking can be provided.|No|
|extra_check|Input|Find the extra verification function after operator fusion. The function's input parameter must be a Match object from torch._inductor.pattern_matcher, and it is used for further custom checks on the matching result, such as checking whether the fused operators are on the same stream, checking the device type, checking the input shapes, and so on.|No|
|search_fn_pattern|Input|A custom pattern object is generally unnecessary to provide. Its definition follows the rules of the native PyTorch MultiOutputPattern object. After passing this parameter, search_fn will no longer be used to match operator combinations; instead, this parameter will be used directly as the matching rule.|No|

Usage Example

```python
import functools
import torch, torch_npu, torchair

from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode
from torchair.core.utils import logger

# Assume fusing the add operator and the npu_rms_norm operator into the npu_add_rms_norm operator
# Define a search_fn to find the operator combinations in the original FX graph before fusion.
def search_fn(x1, x2, gamma):
    xOut = torch.add(x1, x2)
    y, _ = torch_npu.npu_rms_norm(xOut, gamma)
    return y, xOut

# Define a replace_fn, that is, a fusion operator, used to replace operator combinations in the FX graph
def replace_fn(x1, x2, gamma):
    y, _, xOut = torch_npu.npu_add_rms_norm(
        x1, x2, gamma
    )
    return y, xOut

# extra_check can pass in additional validation logic. Here, it is used to check whether the last dimension of the first input parameter x1 is a specific value; if it is not the specific value, fusion is not allowed.
def extra_check(match: Match):
    x1 = match.kwargs.get("x1")

    if x1 is None:
        return False 
    if not hasattr(x1, "meta") or "val" not in x1.meta:
        return False

    a_shape = x1.meta["val"].shape
    return a_shape[-1] == 7168 


# Define some sample inputs to trace search_fn and replace_fn into an FX graph
fake_mode = FakeTensorMode()
with fake_mode:
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu", dtype=torch.float16)
    kwargs_tensor = functools.partial(torch.empty, 2, device="npu", dtype=torch.float16)

    # Call the torchair.register_replacement API with search_fn, replace_fn, and example_inputs. If there are additional validations, you can pass them in as extra_check.
    torchair.register_replacement(
        search_fn=search_fn,
        replace_fn=replace_fn,
        example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
        extra_check=extra_check
    )
```

### DFX

By reusing the TORCH_COMPILE_DEBUG environment variable from the PyTorch community, when TORCH_COMPILE_DEBUG=1 is set, it will output the FX graphs throughout the entire process.
