# Npugraph_ex

## How Does It Work?

This is an optimization based on Fx graphs, which can be considered an acceleration solution for the aclgraph mode.

You can get its code [here](https://gitcode.com/Ascend/torchair)

## Default Fx Graph Optimization

### Fx Graph pass

- For the intermediate nodes of the model, replace the non-in-place operators contained in the nodes with in-place operators to reduce memory movement during computation and improve performance.
- For the original input parameters of the model, if they include in-place operators, Dynamo's Functionalize process will replace the in-place operators with a form of non-in-place operators + copy operators. npugraph_ex will reverse this process, restoring the in-place operators and reducing memory movement.

### Fx fusion pass

npugraph_ex now provides three default operator fusion passes, and more will be added in the future.

Operator combinations that meet the replacement rules can be replaced with the corresponding fused operators.

You can get the default fusion pass list [here](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00017.html)

## Custom fusion pass

Users can register a custom graph fusion pass in TorchAir to modify PyTorch FX graphs. The registration relies on the register_replacement API.

Below is the declaration of this API and a demo of its usage.

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

The default fusion pass in npugraph_ex is also implemented based on this API. You can see more examples of using this API in the vllm-ascend and npugraph_ex code repositories.

### DFX

By reusing the TORCH_COMPILE_DEBUG environment variable from the PyTorch community, when TORCH_COMPILE_DEBUG=1 is set, it will output the FX graphs throughout the entire process.
